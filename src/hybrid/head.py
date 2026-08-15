"""
La cabeza XGBoost del sistema híbrido: qué columnas recibe y cómo se arma.

TRES VARIANTES, con nombre en vez de número. Antes se llamaban 431/439/440 y el
número dependía de `in_dim`, así que cambiaba solo al tocar el dataset.

    control          las N features tabulares      ¿cuánto vale lo tabular solo?
    solo_gnn         SOLO el embedding de la GNN   ¿basta el grafo por sí mismo?
    gnn_mas_tabular  tabular + embedding           ¿APORTA el grafo sobre lo tabular?

La pregunta del capstone la responde `control` vs `gnn_mas_tabular`, con
idéntica ventana de entrenamiento. `solo_gnn` acota el techo del grafo aislado:
si sale muy por debajo, el grafo no basta; si sale cerca, el grafo lleva casi
toda la señal.

POR QUÉ NO HAY COLUMNAS ESTRUCTURALES
Las 8 de `features.py` (n_vecinos, monto_medio_vecinos...) se midieron y dieron
-0,0013 de PR-AUC: son una versión pobre de lo que ya calcula la GNN y duplican
las columnas C1-C14 y D1-D15 del propio dataset. Se retiran para que la
comparación tenga UNA sola diferencia entre `control` y `gnn_mas_tabular`.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.common import get_logger, resolve

log = get_logger("hybrid.head")

VARIANTES = ("control", "solo_gnn", "gnn_mas_tabular")


def nombre_modelo(variante: str, sufijo: str = "") -> str:
    return f"hybrid_head_{variante}{sufijo}.json"


def cols_embedding(df, modo: str = "completo") -> list[str]:
    """
    Las columnas del embedding, ordenadas por su índice numérico.

        completo  emb_*   "yo + mi vecindario"
        vecinos   embv_*  solo el vecindario, sin las features propias

    Cada cabeza necesita una distinta y NO es intercambiable:
      · `solo_gnn` recibe el embedding y nada más. Con la versión de vecinos no
        sabría nada de la propia transacción.
      · `gnn_mas_tabular` ya trae las features propias en su bloque tabular, así
        que con la versión completa las recibiría DOS veces.
    """
    pref = "embv_" if modo == "vecinos" else "emb_"
    return sorted((c for c in df.columns
                   if c.startswith(pref) and c[len(pref):].isdigit()),
                  key=lambda c: int(c[len(pref):]))


def filtrar_prefijos(cols: list[str], prefijos) -> list[str]:
    """
    Quita las columnas que empiecen por alguno de esos prefijos.

    La ablación interesante es ["V", "C", "D"]: son los agregados relacionales
    que Vesta precalculó sobre historiales de entidad, y son la explicación de
    por qué el grafo no aportaba. Quitarlas prueba esa hipótesis directamente.

    Se aplica a TODAS las cabezas por igual: si `control` corriera con menos
    columnas que `gnn_mas_tabular`, la diferencia entre ellas dejaría de ser el
    aporte del grafo.
    """
    pref = tuple(str(x) for x in (prefijos or ()))
    if not pref:
        return list(cols)
    # se compara con el prefijo seguido de dígito para no llevarse por delante
    # columnas como DeviceType o card1 al excluir "D" o "C"
    import re
    patron = re.compile(rf"^({'|'.join(re.escape(x) for x in pref)})\d")
    return [c for c in cols if not patron.match(c)]


def columnas(variante: str, cols_base: list[str],
             cols_emb: list[str] | None = None,
             cols_embv: list[str] | None = None) -> list[str]:
    cols_emb, cols_embv = list(cols_emb or []), list(cols_embv or [])
    if variante == "control":
        return list(cols_base)
    if variante == "solo_gnn":
        # El COMPLETO: es lo único que recibe, necesita saber de sí misma
        if not cols_emb:
            raise ValueError("'solo_gnn' necesita columnas emb_* en el parquet "
                             "del OOF. ¿Corriste la etapa `oof`?")
        return cols_emb
    if variante == "gnn_mas_tabular":
        # El de VECINOS: las features propias ya vienen en cols_base
        usar = cols_embv or cols_emb
        if not usar:
            raise ValueError("'gnn_mas_tabular' necesita columnas embv_* en el "
                             "parquet del OOF. ¿Corriste la etapa `oof`?")
        return list(cols_base) + usar
    raise ValueError(f"Variante desconocida: {variante!r}. "
                     f"Las válidas son {VARIANTES}")


def cargar_tabla(cfg, oof_window: str | None) -> tuple[pd.DataFrame, list[str]]:
    """
    `full.parquet` + el embedding del OOF, unidos POR POSICIÓN.

    El índice de fila del parquet ES el índice de nodo del grafo, así que la
    unión es un alineado posicional, no un join por clave. Se comprueba con un
    assert: si `build_graph` reordenara nodos, todo quedaría desalineado en
    silencio y esto es la única red.

    El parquet del OOF cubre TODAS las filas: dentro de la ventana con el
    embedding out-of-fold y fuera con el modelo real que no las vio (ver
    `oof.py`). Si quedara alguna fila sin embedding se avisa — cuando los meses
    de fuera llegaban en NaN, el paso `heads` entrenaba con esas columnas y las
    validaba donde no existían.
    """
    proc = resolve(cfg, "processed_dir")
    df = pd.read_parquet(proc / "full.parquet")
    with open(proc / "feature_cols.json") as f:
        cols_base = json.load(f)["feature_cols"]

    if oof_window:
        oof = pd.read_parquet(proc / f"gnn_oof_{oof_window}.parquet")
        # EL CONTRATO: `node_idx` es el índice de FILA de full.parquet, que a su
        # vez es el índice de nodo del grafo. La unión de abajo es posicional,
        # no un join por clave, así que si `build_graph` reordenara nodos todo
        # quedaría desalineado EN SILENCIO. Esto es la única red.
        ni = oof["node_idx"].values
        assert ni.min() >= 0 and ni.max() < len(df), (
            f"node_idx fuera de rango: [{ni.min()}, {ni.max()}] contra "
            f"{len(df)} filas en full.parquet. El grafo y el parquet no "
            f"corresponden: reconstruye el grafo.")
        assert len(np.unique(ni)) == len(ni), (
            "node_idx tiene duplicados: el parquet del OOF está corrupto.")
        emb = cols_embedding(oof, "completo") + cols_embedding(oof, "vecinos")
        if emb:
            # De golpe con concat, NO en un bucle de `df[c] = ...`: cada
            # asignación inserta una columna y recopia el DataFrame entero.
            bloque = pd.DataFrame(np.float32("nan"), index=df.index,
                                  columns=emb, dtype=np.float32)
            bloque.iloc[oof["node_idx"].values] = oof[emb].values
            df = pd.concat([df, bloque], axis=1)
        if "gnn_score" in oof.columns:
            df["gnn_score"] = np.nan
            df.loc[oof["node_idx"].values, "gnn_score"] = oof["gnn_score"].values
        faltan = int(df[emb[0]].isna().sum()) if emb else 0
        log.info("embedding (%s): %d filas, %d dims%s", oof_window, len(oof),
                 len(emb),
                 f" | AVISO: {faltan} filas sin embedding" if faltan else "")
    return df, cols_base


def matriz(df: pd.DataFrame, filas: np.ndarray, variante: str,
           cols_base: list[str], cols_emb: list[str] | None = None,
           cols_embv: list[str] | None = None) -> np.ndarray:
    """Matriz de diseño en float32 (SMOTE devuelve float64)."""
    cols = columnas(variante, cols_base, cols_emb, cols_embv)
    return df.loc[filas, cols].values.astype(np.float32)


def guardar(booster, cfg, nombre: str) -> Path:
    ruta = resolve(cfg, "models_dir") / nombre
    booster.save_model(str(ruta))
    return ruta


def cargar(cfg, nombre: str):
    """
    Booster nativo, NO XGBClassifier.

    El wrapper de sklearn consulta `self._estimator_type`, que las versiones
    nuevas de scikit-learn ya no definen, y `load_model` revienta con
    "TypeError: _estimator_type undefined". El Booster no depende de sklearn.

    `device=cpu` porque recorrer árboles es ramificación pura y accesos
    dispersos — lo que una GPU hace mal. Es exacto: lo que difiere entre GPU y
    CPU es la CONSTRUCCIÓN de histogramas al entrenar, no la predicción.
    """
    import xgboost as xgb
    booster = xgb.Booster()
    booster.load_model(str(resolve(cfg, "models_dir") / nombre))
    booster.set_param({"nthread": 1, "device": "cpu"})
    return booster


def umbral_por_presupuesto(scores: np.ndarray, pct: float) -> float:
    """
    Umbral que produce `pct`% de alertas.

    Un umbral fijo no sirve para comparar estos sistemas: la GNN entrena con
    pos_weight y sus scores están inflados, mientras la cabeza devuelve
    probabilidades calibradas en torno a la tasa base. Con 0.5 uno alertaría de
    todo y el otro casi nada. Y un equipo de revisión tiene capacidad constante,
    no un corte de probabilidad constante: el presupuesto es lo que de verdad
    restringe.
    """
    return float(np.quantile(scores, 1.0 - pct / 100.0))
