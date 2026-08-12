"""
Paso 8 — COMPARACIÓN FINAL (OE4): GNN + Continual Learning vs XGBoost congelado.

Protocolo:
- Mismo mes 6, mismo threshold >= 0.5.
- XGBoost representa el "mundo actual" estático: entrenado en meses 1-4,
  jamás reentrenado.
- La GNN+CL se evaluó semana a semana adaptándose (scores generados por el
  orquestador de CL).
- KPI: recall de la GNN+CL >= 20% superior sobre PATRONES EMERGENTES.
  Patrón emergente = fraude del mes 6 que el modelo GNN ORIGINAL (antes de
  cualquier ciclo de CL) scoreó bajo (<0.5) — es decir, lo que el mundo
  estático no tenía cómo ver.
- Impacto económico: fraudes adicionales detectados x monto promedio (USD).
  Si el dataset tiene TransactionAmt se usa el monto real de cada fraude
  adicional en vez del promedio de config.

Requiere haber corrido antes:
  1. train_xgboost.py            (baseline congelado)
  2. compare_gnns.py             (modelo GNN seleccionado)
  3. cl_orchestrator.py          (scores GNN+CL del mes 6)

Uso: python -m src.comparison.final_comparison
"""
import json
import os
import sys
from pathlib import Path

# macOS: este es el único módulo que carga PyTorch y XGBoost en el mismo
# proceso, y cada uno trae su propio runtime de OpenMP (torch empaqueta el
# suyo; XGBoost usa el libomp de Homebrew). Con los dos multihilo a la vez el
# intérprete muere con SIGSEGV al llamar a load_model(). Limitar OpenMP a un
# hilo es el único workaround que funciona (KMP_DUPLICATE_LIB_OK no basta), y
# tiene que definirse ANTES de importar torch/xgboost. En Linux hay un solo
# runtime, así que no se toca nada.
if sys.platform == "darwin":
    os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import torch
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.continual_learning.validate import score_nodes
from src.gnn.models import build_model
from src.utils.common import ensure_dirs, get_logger, load_config, resolve
from src.utils.metrics import full_report

log = get_logger("comparison")


def xgboost_scores_on_test(cfg) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    proc = resolve(cfg, "processed_dir")
    df = pd.read_parquet(proc / "full.parquet")
    with open(proc / "feature_cols.json") as f:
        cols = json.load(f)["feature_cols"]
    test = df[df.split == "test"].reset_index(drop=True)
    model = xgb.XGBClassifier()
    model.load_model(resolve(cfg, "models_dir") / "xgboost_baseline.json")
    return (model.predict_proba(test[cols].values)[:, 1],
            test["isFraud"].values.astype(int), test)


def original_gnn_scores_on_test(cfg, data, test_idx) -> np.ndarray:
    """Scores del GNN ORIGINAL (pre-CL) para identificar patrones emergentes."""
    models_dir = resolve(cfg, "models_dir")
    with open(models_dir / "selected_model.json") as f:
        sel = json.load(f)["selection"]
    ckpt = torch.load(models_dir / sel["checkpoint"], weights_only=False)
    cfg["gnn"]["in_dim"] = ckpt["in_dim"]
    model = build_model(ckpt["model_name"], cfg)
    model.load_state_dict(ckpt["state_dict"])
    return score_nodes(model, data, test_idx, cfg)


def main():
    cfg = load_config()
    ensure_dirs(cfg)
    thr = cfg["gnn"]["threshold"]
    reports_dir = resolve(cfg, "reports_dir")

    # --- scores de los tres actores sobre el mes 6 ---
    xgb_scores, y_xgb, test_df = xgboost_scores_on_test(cfg)

    cl_pack = np.load(reports_dir / "gnn_cl_test_scores.npz")
    gnn_cl_scores, y_gnn, test_idx = cl_pack["scores"], cl_pack["y"], cl_pack["node_idx"]

    data = torch.load(resolve(cfg, "graph_dir") / "graph.pt", weights_only=False)
    gnn_orig_scores = original_gnn_scores_on_test(cfg, data, test_idx)

    assert len(y_xgb) == len(y_gnn), \
        "El test de XGBoost y de la GNN no coinciden en tamaño"

    # --- métricas globales mes 6 ---
    rep_xgb = full_report(y_xgb, xgb_scores, thr)
    rep_cl = full_report(y_gnn, gnn_cl_scores, thr)

    # --- patrones emergentes: fraudes que el GNN ORIGINAL no veía ---
    fraud = y_gnn == 1
    emerging = fraud & (gnn_orig_scores < thr)
    n_emerging = int(emerging.sum())
    recall_cl_emerging = float((gnn_cl_scores[emerging] >= thr).mean()) if n_emerging else 0.0
    recall_xgb_emerging = float((xgb_scores[emerging] >= thr).mean()) if n_emerging else 0.0
    gap = recall_cl_emerging - recall_xgb_emerging
    kpi_ok = gap >= cfg["comparison"]["kpi_recall_gap"]

    # --- impacto económico ---
    extra_detected = emerging & (gnn_cl_scores >= thr) & (xgb_scores < thr)
    if "TransactionAmt" in test_df.columns:
        usd = float(test_df.loc[np.where(extra_detected)[0], "TransactionAmt"].sum())
        usd_note = "suma de TransactionAmt de los fraudes adicionales detectados"
    else:
        usd = float(extra_detected.sum() * cfg["comparison"]["avg_fraud_amount_usd"])
        usd_note = f"n x monto promedio (USD {cfg['comparison']['avg_fraud_amount_usd']})"

    result = {
        "month6_overall": {"xgboost_frozen": rep_xgb, "gnn_continual_learning": rep_cl},
        "emerging_patterns": {
            "definition": "fraudes del mes 6 con score <0.5 del GNN original (pre-CL)",
            "n_emerging_frauds": n_emerging,
            "recall_gnn_cl": recall_cl_emerging,
            "recall_xgboost": recall_xgb_emerging,
            "recall_gap": round(gap, 4),
            "kpi_gap_target": cfg["comparison"]["kpi_recall_gap"],
            "kpi_ok": bool(kpi_ok),
        },
        "economic_impact": {
            "extra_frauds_detected_vs_xgboost": int(extra_detected.sum()),
            "estimated_usd_saved": round(usd, 2),
            "method": usd_note,
        },
    }

    with open(reports_dir / "final_comparison.json", "w") as f:
        json.dump(result, f, indent=2)

    log.info("=========== COMPARACIÓN FINAL (mes 6, threshold %.2f) ===========", thr)
    log.info("Global    -> XGBoost recall %.4f | GNN+CL recall %.4f",
             rep_xgb["recall"], rep_cl["recall"])
    log.info("Emergentes-> XGBoost %.4f | GNN+CL %.4f | gap %+.4f (KPI >= %.2f: %s)",
             recall_xgb_emerging, recall_cl_emerging, gap,
             cfg["comparison"]["kpi_recall_gap"], "OK" if kpi_ok else "NO")
    log.info("Impacto   -> %d fraudes adicionales ~ USD %.0f",
             int(extra_detected.sum()), usd)
    log.info("Guardado en %s", reports_dir / "final_comparison.json")


if __name__ == "__main__":
    main()
