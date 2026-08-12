"""
SMOTE — Generación de fraudes sintéticos para el BASELINE TABULAR.

IMPORTANTE (decisión de diseño del proyecto):
- SMOTE actúa sobre los DATOS: interpola entre fraudes reales vecinos para
  crear fraudes sintéticos.
- Se aplica SOLO al set de ENTRENAMIENTO del baseline XGBoost. Jamás en
  validación ni test (data leakage).
- NO se usa en la GNN: los ejemplos sintéticos son filas tabulares SIN
  aristas — no existen en el grafo, no aportan estructura. En la GNN el
  desbalance se maneja con weighted loss (pos_weight).
- Tampoco se usa en el fine-tuning del continual learning: ahí el balanceo
  es POR COMPOSICIÓN del batch (40/60 con buffer 50/50 de casos reales).

Este módulo también sirve como utilidad standalone para "producir más datos"
tabulares si se necesita experimentar con el baseline.
"""
import sys
from pathlib import Path

import numpy as np
from imblearn.over_sampling import SMOTE

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.common import get_logger, load_config

log = get_logger("smote")


def apply_smote(X_train: np.ndarray, y_train: np.ndarray, cfg: dict | None = None,
                seed: int = 42):
    """
    Devuelve (X_res, y_res) con fraudes sintéticos añadidos.
    sampling_strategy=0.5 -> los fraudes llegan al 50% de las legítimas (1:2).
    """
    cfg = cfg or load_config()
    scfg = cfg["xgboost"]["smote"]
    before = int(y_train.sum())
    sm = SMOTE(sampling_strategy=scfg["sampling_strategy"],
               k_neighbors=scfg["k_neighbors"], random_state=seed)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    log.info("SMOTE: fraudes %d -> %d (total %d -> %d filas)",
             before, int(y_res.sum()), len(y_train), len(y_res))
    return X_res, y_res


if __name__ == "__main__":
    # Demo standalone sobre el split de train procesado
    import json
    import pandas as pd
    from src.utils.common import resolve

    cfg = load_config()
    proc = resolve(cfg, "processed_dir")
    df = pd.read_parquet(proc / "full.parquet")
    with open(proc / "feature_cols.json") as f:
        cols = json.load(f)["feature_cols"]
    tr = df[df.split == "train"]
    Xr, yr = apply_smote(tr[cols].values, tr["isFraud"].values)
    out = proc / "train_smote.npz"
    np.savez_compressed(out, X=Xr, y=yr)
    log.info("Set aumentado guardado en %s", out)
