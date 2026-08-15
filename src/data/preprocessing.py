"""
Paso 1 — Preprocesamiento del IEEE-CIS.

Hace tres cosas:
1. Merge de train_transaction + train_identity (left join por TransactionID).
2. Limpieza: encoding de categóricas, imputación de NAs, matriz final de
   432 features numéricas por transacción.
3. División temporal ESTRICTA (walk-forward, sin data leakage):
   meses 1-4 = TRAIN | mes 5 = VAL | mes 6 = TEST (dividido en semanas).

El mes se deriva de TransactionDT (segundos desde t0): mes = DT // (86400*30).
Nada del futuro se usa para transformar el pasado: los encoders y la
imputación se AJUSTAN solo con TRAIN y se aplican a VAL/TEST.

Salidas (data/processed/):
  full.parquet          dataset completo procesado + columnas month/week
  feature_cols.json     lista de columnas de features (orden fijo)
  split_masks.parquet   TransactionID -> split (train/val/test) y semana
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.common import ensure_dirs, get_logger, load_config, resolve

log = get_logger("preprocessing")

# Columnas que se usan para construir ARISTAS del grafo: se conservan crudas
# (además de su versión codificada) porque el builder del grafo las necesita.
def edge_raw_cols(cfg: dict) -> list[str]:
    """
    Columnas que hay que preservar SIN codificar, para construir las entidades.

    Se derivan de `graph.entidades` en vez de escribirse a mano. La lista fija
    que había aquí se quedó atrás cuando se añadieron las entidades `device` y
    `net`: faltaban id_33 e id_13/17/19/20, y `build_graph` las omitía enteras
    con un WARNING fácil de pasar por alto. Dos de las cinco entidades no
    existían y el grafo se construía igualmente.

    Derivarlas del config significa que añadir una entidad nueva no obliga a
    tocar este archivo: si está en `graph.entidades`, su columna se preserva.
    """
    cols = []
    for spec in (cfg.get("graph", {}).get("entidades") or {}).values():
        for c in spec.get("cols", []):
            if c not in cols:
                cols.append(c)
    return cols


def load_raw(raw_dir: Path) -> pd.DataFrame:
    tx = pd.read_csv(raw_dir / "train_transaction.csv")
    ident = pd.read_csv(raw_dir / "train_identity.csv")
    df = tx.merge(ident, on="TransactionID", how="left")
    log.info("Merge: %d filas, %d columnas", *df.shape)
    # El merge deja el DataFrame repartido en muchos bloques internos. Con
    # 400+ columnas, CUALQUIER inserción posterior dispara PerformanceWarning.
    # Una copia aquí lo consolida de una vez y evita el aviso aguas abajo.
    return df.copy()


def add_temporal_columns(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    spm = cfg["data"]["seconds_per_month"]
    # assign() en vez de dos asignaciones sueltas: insertar columnas una a una
    # en un DataFrame de 400+ columnas lo fragmenta y dispara PerformanceWarning.
    dt = df["TransactionDT"].astype("float64")
    return df.assign(
        # El reloj. Va AQUÍ y no en build_graph porque no tiene nada de grafo, y
        # porque calculado allí no llegaba al parquet: la GNN sabía la hora y las
        # cabezas tabulares no. Esa información entraba en `gnn_mas_tabular`
        # dentro del embedding, así que parte del "aporte del grafo" era el reloj.
        __hora_dia=(dt % 86400) / 86400.0,
        __pos_temporal=(dt - dt.min()) / max(dt.max() - dt.min(), 1),
        month=(df["TransactionDT"] // spm).astype(int) + 1,
        # Semana relativa dentro del mes (simula el "futuro que llega" en test)
        week_in_month=(((df["TransactionDT"] % spm) // (spm // 4)) + 1)
                      .clip(1, 4).astype(int),
    )


def encode_and_impute(df: pd.DataFrame, train_mask: pd.Series, cfg: dict):
    """Label encoding de categóricas + imputación, ajustados SOLO con train."""
    id_like = {"TransactionID", "isFraud", "TransactionDT", "month", "week_in_month"}
    feature_cols = [c for c in df.columns if c not in id_like]

    cat_cols = [c for c in feature_cols
                if not pd.api.types.is_numeric_dtype(df[c])]
    num_cols = [c for c in feature_cols if c not in cat_cols]

    # Guardar crudas para las aristas ANTES de codificar.
    # Se añaden TODAS de una vez con concat: en bucle serían ~10 inserciones
    # sucesivas sobre un DataFrame ancho, que es justo lo que pandas penaliza
    # con PerformanceWarning ("DataFrame is highly fragmented").
    pedidas = edge_raw_cols(cfg)
    crudas = {f"raw__{c}": df[c].astype(str).replace("nan", np.nan)
              for c in pedidas if c in df.columns}
    if crudas:
        df = pd.concat([df, pd.DataFrame(crudas, index=df.index)], axis=1)
    # Avisar AQUÍ, donde se puede arreglar, y no dos etapas más tarde cuando
    # build_graph descarta la entidad entera sin decir de dónde venía.
    ausentes = [c for c in pedidas if c not in df.columns]
    if ausentes:
        log.warning("Estas columnas las piden las entidades de config y NO "
                    "están en el dataset: %s. Las entidades que dependan de "
                    "ellas se omitirán en `graph`.", ausentes)

    # Categóricas: mapa de categorías aprendido en train; lo no visto -> -1
    for c in cat_cols:
        cats = df.loc[train_mask, c].astype(str).fillna("__NA__").unique()
        mapping = {v: i for i, v in enumerate(sorted(cats))}
        df[c] = df[c].astype(str).fillna("__NA__").map(mapping).fillna(-1).astype(np.int32)

    # Numéricas: mediana de train; flag adicional de NA para columnas muy vacías
    for c in num_cols:
        med = df.loc[train_mask, c].median()
        df[c] = df[c].fillna(0.0 if np.isnan(med) else med)

    df[num_cols] = df[num_cols].astype(np.float32)
    # Los dos bucles de arriba reescriben cientos de columnas una a una y
    # vuelven a fragmentar el frame; se consolida antes de devolverlo.
    return df.copy(), feature_cols


def main():
    cfg = load_config()
    ensure_dirs(cfg)
    raw_dir, out_dir = resolve(cfg, "raw_dir"), resolve(cfg, "processed_dir")

    df = load_raw(raw_dir)
    df = add_temporal_columns(df, cfg)

    train_months = set(cfg["data"]["train_months"])
    train_mask = df["month"].isin(train_months)
    df, feature_cols = encode_and_impute(df, train_mask, cfg)

    # Asignación de split, en una sola pasada (np.select en vez de 4
    # asignaciones sucesivas, que volverían a fragmentar el frame).
    df["split"] = np.select(
        [df["month"].isin(train_months),
         df["month"] == cfg["data"]["val_month"],
         df["month"] == cfg["data"]["test_month"]],
        ["train", "val", "test"], default="other")
    df = df[df["split"] != "other"].reset_index(drop=True)

    ratio = (df.loc[df.split == "train", "isFraud"] == 0).sum() / max(
        1, (df.loc[df.split == "train", "isFraud"] == 1).sum())
    log.info("Desbalance en TRAIN: %.1f legítimas por fraude (pos_weight)", ratio)

    df.to_parquet(out_dir / "full.parquet", index=False)
    with open(out_dir / "feature_cols.json", "w") as f:
        json.dump({"feature_cols": feature_cols, "pos_weight_train": float(ratio)}, f, indent=2)
    df[["TransactionID", "split", "month", "week_in_month", "isFraud"]].to_parquet(
        out_dir / "split_masks.parquet", index=False)

    log.info("Split -> train: %d | val: %d | test: %d",
             (df.split == "train").sum(), (df.split == "val").sum(), (df.split == "test").sum())


if __name__ == "__main__":
    main()
