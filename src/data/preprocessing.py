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
EDGE_RAW_COLS = ["card1", "card2", "card3", "card5", "addr1",
                 "P_emaildomain", "DeviceInfo", "id_30", "id_31"]


def load_raw(raw_dir: Path) -> pd.DataFrame:
    log.info("Cargando CSVs crudos...")
    tx = pd.read_csv(raw_dir / "train_transaction.csv")
    ident = pd.read_csv(raw_dir / "train_identity.csv")
    df = tx.merge(ident, on="TransactionID", how="left")
    log.info("Merge: %d filas, %d columnas", *df.shape)
    return df


def add_temporal_columns(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    spm = cfg["data"]["seconds_per_month"]
    df["month"] = (df["TransactionDT"] // spm).astype(int) + 1
    # Semana relativa dentro del mes (para simular "futuro que llega" en test)
    df["week_in_month"] = (((df["TransactionDT"] % spm) // (spm // 4)) + 1).clip(1, 4).astype(int)
    return df


def encode_and_impute(df: pd.DataFrame, train_mask: pd.Series):
    """Label encoding de categóricas + imputación, ajustados SOLO con train."""
    id_like = {"TransactionID", "isFraud", "TransactionDT", "month", "week_in_month"}
    feature_cols = [c for c in df.columns if c not in id_like]

    cat_cols = [c for c in feature_cols
                if not pd.api.types.is_numeric_dtype(df[c])]
    num_cols = [c for c in feature_cols if c not in cat_cols]

    # Guardar crudas para las aristas ANTES de codificar
    for c in EDGE_RAW_COLS:
        if c in df.columns:
            df[f"raw__{c}"] = df[c].astype(str).replace("nan", np.nan)

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
    return df, feature_cols


def main():
    cfg = load_config()
    ensure_dirs(cfg)
    raw_dir, out_dir = resolve(cfg, "raw_dir"), resolve(cfg, "processed_dir")

    df = load_raw(raw_dir)
    df = add_temporal_columns(df, cfg)

    train_months = set(cfg["data"]["train_months"])
    train_mask = df["month"].isin(train_months)
    df, feature_cols = encode_and_impute(df, train_mask)

    # Asignación de split
    df["split"] = "other"
    df.loc[df["month"].isin(train_months), "split"] = "train"
    df.loc[df["month"] == cfg["data"]["val_month"], "split"] = "val"
    df.loc[df["month"] == cfg["data"]["test_month"], "split"] = "test"
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
    log.info("Guardado en %s", out_dir)


if __name__ == "__main__":
    main()
