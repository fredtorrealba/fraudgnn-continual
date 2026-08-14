"""
La cabeza XGBoost del sistema híbrido: ensamblado de columnas e IO.

NO importa torch a propósito. Es lo que permite que `train_head.py` (que hace
Optuna y quiere todos los núcleos) nunca cargue PyTorch y XGBoost en el mismo
proceso — la combinación que segfaultea en macOS.

LAS TRES VARIANTES
    431  las columnas originales                      -> referencia
    439  + las 8 estructurales del grafo              -> ¿aporta la estructura?
    440  + gnn_score                                  -> ¿aporta además la red?

Entrenadas con los mismos datos y los mismos hiperparámetros, las diferencias
entre ellas son atribuibles SOLO a las columnas añadidas.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.hybrid.features import COLUMNAS as COLS_ESTRUCTURALES
from src.utils.common import get_logger, resolve

log = get_logger("hybrid.head")

VARIANTES = (431, 439, 440)


def nombre_modelo(variant: int) -> str:
    return f"hybrid_head_{variant}.json"


# Variante con EMBEDDING: en vez del escalar `gnn_score`, las `dim` columnas
# del vector de la última capa de la GNN. La red pasa de "asesor que opina un
# número" a "codificador de grafo": sin ella desaparecen 256 de las 695
# columnas, no una. El número de variante es 431 + 8 + dim, así que depende de
# `gnn.hidden_dims`; con [256] son 695.
def variante_embedding(cols_base: list[str], dim: int) -> int:
    return len(cols_base) + len(COLS_ESTRUCTURALES) + dim


def cols_embedding(df) -> list[str]:
    """Las columnas `emb_*` presentes en la tabla, en orden."""
    return sorted((c for c in df.columns if c.startswith("emb_")),
                  key=lambda c: int(c.split("_")[1]))


def columnas(variant: int, cols_base: list[str],
             cols_emb: list[str] | None = None) -> list[str]:
    if variant == 431:
        return list(cols_base)
    if variant == 439:
        return list(cols_base) + list(COLS_ESTRUCTURALES)
    if variant == 440:
        return list(cols_base) + list(COLS_ESTRUCTURALES) + ["gnn_score"]
    if cols_emb and variant == variante_embedding(cols_base, len(cols_emb)):
        return list(cols_base) + list(COLS_ESTRUCTURALES) + list(cols_emb)
    raise ValueError(
        f"Variante {variant} desconocida. Con {len(cols_base)} columnas base "
        f"las válidas son 431, 439, 440"
        + (f" y {variante_embedding(cols_base, len(cols_emb))} (embedding de "
           f"{len(cols_emb)})" if cols_emb else
           " (la de embedding exige gnn_oof_*.parquet con columnas emb_*)"))


def cargar_tabla(cfg, oof_window: str | None) -> tuple[pd.DataFrame, list[str]]:
    """
    `full.parquet` + las 8 estructurales + `gnn_score`, unidas POR POSICIÓN.

    El índice de fila del parquet es el índice de nodo del grafo, así que la
    unión es un alineado posicional, no un join por clave. Se comprueba.

    El parquet del OOF cubre TODAS las filas: dentro de la ventana con el score
    out-of-fold, y fuera con el modelo real que no las vio (ver `oof.py`). No
    siempre fue así: cuando los meses de fuera llegaban en NaN, el paso
    `hybrid` entrenaba las variantes con `gnn_score`/embedding y las validaba
    sobre el mes 5, donde esas columnas no existían — la variante del embedding
    cortaba por early stopping en 6 árboles.

    Si aun así quedara algún NaN, XGBoost lo trata nativamente, así que un
    fallo de relleno no se disfraza de cero silencioso.
    """
    proc = resolve(cfg, "processed_dir")
    df = pd.read_parquet(proc / "full.parquet")
    with open(proc / "feature_cols.json") as f:
        cols_base = json.load(f)["feature_cols"]

    est = pd.read_parquet(proc / "graph_features.parquet")
    assert len(est) == len(df), "graph_features.parquet no cuadra con full.parquet"
    for c in COLS_ESTRUCTURALES:
        df[c] = est[c].values

    df["gnn_score"] = np.nan
    if oof_window:
        oof = pd.read_parquet(proc / f"gnn_oof_{oof_window}.parquet")
        df.loc[oof["node_idx"].values, "gnn_score"] = oof["gnn_score"].values
        # Embedding, si la corrida del OOF lo dejó.
        emb = cols_embedding(oof)
        if emb:
            # De golpe con concat, NO en un bucle de `df[c] = ...`: cada
            # asignación inserta una columna y recopia las 582.429 filas del
            # DataFrame. Con 256 columnas eso son 256 recopiados y pandas avisa
            # con PerformanceWarning ("DataFrame is highly fragmented").
            bloque = pd.DataFrame(np.float32("nan"), index=df.index,
                                  columns=emb, dtype=np.float32)
            bloque.iloc[oof["node_idx"].values] = oof[emb].values
            df = pd.concat([df, bloque], axis=1)
        faltan = int(df["gnn_score"].isna().sum())
        log.info("gnn_score (%s): %d filas%s%s", oof_window, len(oof),
                 f" + embedding de {len(emb)} dims" if emb else "",
                 f" | AVISO: {faltan} filas sin score" if faltan else "")
    return df, cols_base


def matriz(df: pd.DataFrame, filas: np.ndarray, variant: int,
           cols_base: list[str], cols_emb: list[str] | None = None) -> np.ndarray:
    """Matriz de diseño de una variante, en float32 (SMOTE devuelve float64)."""
    return df.loc[filas, columnas(variant, cols_base, cols_emb)].values.astype(
        np.float32)


def guardar(booster, cfg, nombre: str) -> Path:
    ruta = resolve(cfg, "models_dir") / nombre
    booster.save_model(str(ruta))
    return ruta


def cargar(cfg, nombre: str):
    """
    Booster nativo (no XGBClassifier) con nthread=1.

    `inplace_predict` sobre el Booster evita el overhead del wrapper sklearn en
    llamadas pequeñas, y `nthread=1` es el cinturón extra contra el conflicto
    de runtimes de OpenMP cuando este proceso ya tiene torch cargado.
    """
    import xgboost as xgb
    booster = xgb.Booster()
    booster.load_model(str(resolve(cfg, "models_dir") / nombre))
    # device="cpu": si se entrenó en GPU, el booster queda en cuda:0 y cada
    # inplace_predict sobre numpy avisa de "mismatched devices" y cae a DMatrix.
    # En el CL se predice muchas veces sobre conjuntos pequeños, así que ese
    # rodeo se paga en cada ciclo. La predicción es idéntica (ver inferir_en_cpu).
    booster.set_param({"nthread": 1, "device": "cpu"})
    return booster


def umbral_por_presupuesto(scores: np.ndarray, pct: float) -> float:
    """
    Umbral que produce `pct`% de alertas.

    Un umbral fijo no sirve para este sistema: la GNN entrena con pos_weight
    ~27 y sus scores están inflados, mientras la cabeza devuelve probabilidades
    calibradas en torno a la tasa base (~3%). Con 0.5 el híbrido no alertaría
    casi nada y NINGÚN ciclo de continual learning llegaría a desplegar.
    Fijar el umbral por volumen de alertas hace comparables ambos sistemas y es
    lo que de verdad restringe a un equipo de revisión.
    """
    return float(np.quantile(scores, 1.0 - pct / 100.0))
