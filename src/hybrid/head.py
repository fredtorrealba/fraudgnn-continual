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


def columnas(variant: int, cols_base: list[str]) -> list[str]:
    if variant == 431:
        return list(cols_base)
    if variant == 439:
        return list(cols_base) + list(COLS_ESTRUCTURALES)
    return list(cols_base) + list(COLS_ESTRUCTURALES) + ["gnn_score"]


def cargar_tabla(cfg, oof_window: str | None) -> tuple[pd.DataFrame, list[str]]:
    """
    `full.parquet` + las 8 estructurales + `gnn_score`, unidas POR POSICIÓN.

    El índice de fila del parquet es el índice de nodo del grafo, así que la
    unión es un alineado posicional, no un join por clave. Se comprueba.

    `gnn_score` queda NaN fuera de la ventana OOF (meses 5-6 en `train`). Para
    esos meses lo rellena quien puntúa con el modelo real; XGBoost trata los
    NaN nativamente, así que un fallo de relleno no pasa desapercibido como un
    cero silencioso.
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
        log.info("gnn_score OOF (%s): %d filas", oof_window, len(oof))
    return df, cols_base


def matriz(df: pd.DataFrame, filas: np.ndarray, variant: int,
           cols_base: list[str]) -> np.ndarray:
    """Matriz de diseño de una variante, en float32 (SMOTE devuelve float64)."""
    return df.loc[filas, columnas(variant, cols_base)].values.astype(np.float32)


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
