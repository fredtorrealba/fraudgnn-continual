"""
Paso 3 — Baseline XGBoost (CAJA AISLADA).

Representa el "mundo actual" estático (el enfoque tipo ganador de la
competencia IEEE-CIS: gradient boosting sobre features tabulares):
- Tabular, sin grafo.
- SMOTE en train + búsqueda bayesiana de hiperparámetros (Optuna/TPE).
- Entrenado SOLO con meses 1-4. Validado en mes 5.
- CONGELADO: nunca se reentrena, sin continual learning.
- No interactúa con el sistema GNN — su único rol es la comparación final
  (OE4) sobre el mes 6 con el mismo threshold >= 0.5.

Salidas:
  models/xgboost_baseline.json      modelo congelado
  reports/xgboost_val_metrics.json  métricas en validación (mes 5)
"""
import json
import sys
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.baseline_xgboost.smote_pipeline import apply_smote
from src.utils.common import (ensure_dirs, get_logger, load_config, n_jobs,
                              resolve, set_seed)
from src.utils.metrics import full_report

log = get_logger("xgboost")


def load_splits(cfg):
    proc = resolve(cfg, "processed_dir")
    df = pd.read_parquet(proc / "full.parquet")
    with open(proc / "feature_cols.json") as f:
        cols = json.load(f)["feature_cols"]
    tr, va = df[df.split == "train"], df[df.split == "val"]
    return (tr[cols].values, tr["isFraud"].values.astype(int),
            va[cols].values, va["isFraud"].values.astype(int), cols)


def xgb_device(cfg: dict) -> str:
    """
    Dispositivo para XGBoost desde `xgboost.device`: auto | cuda | cpu.
    "auto" usa CUDA solo si torch la ve; sin torch o sin GPU, cae a CPU.
    """
    d = str(cfg["xgboost"].get("device", "auto")).lower()
    if d != "auto":
        return d
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def inferir_en_cpu(modelo):
    """
    Pasa el booster a CPU para INFERENCIA y devuelve el mismo objeto.

    Entrenar con device="cuda" deja el booster en cuda:0. Al predecir sobre
    arrays de numpy —que viven en CPU— XGBoost avisa de "mismatched devices" y
    cae a construir un DMatrix intermedio: más lento y con más memoria.

    Es seguro: el recorrido de los árboles es exacto y da el mismo resultado en
    cualquier dispositivo. Lo que difiere entre GPU y CPU es la CONSTRUCCIÓN de
    histogramas durante el entrenamiento, no la predicción sobre un árbol ya
    construido. Así que esto no cambia ni un decimal de los scores.
    """
    booster = modelo.get_booster() if hasattr(modelo, "get_booster") else modelo
    booster.set_param({"device": "cpu"})
    return modelo


def objective_factory(X_tr, y_tr, X_va, y_va, cfg):
    """Objetivo Optuna: maximizar AUC en validación (mes 5, distribución real)."""
    def objective(trial: optuna.Trial) -> float:
        params = {
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "tree_method": "hist",
            "max_depth": trial.suggest_int("max_depth", 4, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "n_estimators": 1000,
            "early_stopping_rounds": cfg["xgboost"]["early_stopping_rounds"],
            "random_state": 42,
            "n_jobs": n_jobs(cfg),
            "device": xgb_device(cfg),
        }
        model = xgb.XGBClassifier(**params)
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        from sklearn.metrics import roc_auc_score
        inferir_en_cpu(model)
        return roc_auc_score(y_va, model.predict_proba(X_va)[:, 1])
    return objective


def main():
    cfg = load_config()
    ensure_dirs(cfg)
    set_seed(42)
    models_dir, reports_dir = resolve(cfg, "models_dir"), resolve(cfg, "reports_dir")

    X_tr, y_tr, X_va, y_va, cols = load_splits(cfg)
    log.info("Train: %d filas (%.2f%% fraude) | Val: %d filas",
             len(y_tr), 100 * y_tr.mean(), len(y_va))

    # SMOTE SOLO sobre train — validación queda con la distribución real
    X_tr_res, y_tr_res = apply_smote(X_tr, y_tr, cfg)

    log.info("XGBoost en %s | n_jobs=%d", xgb_device(cfg), n_jobs(cfg))
    log.info("Búsqueda bayesiana (%d trials)...", cfg["xgboost"]["optuna_trials"])
    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective_factory(X_tr_res, y_tr_res, X_va, y_va, cfg),
                   n_trials=cfg["xgboost"]["optuna_trials"], show_progress_bar=True)
    log.info("Mejor AUC val: %.4f | params: %s", study.best_value, study.best_params)

    # Modelo final con los mejores hiperparámetros
    best = dict(study.best_params)
    best.update({"objective": "binary:logistic", "eval_metric": "auc",
                 "tree_method": "hist", "n_estimators": 1000,
                 "early_stopping_rounds": cfg["xgboost"]["early_stopping_rounds"],
                 "random_state": 42, "n_jobs": n_jobs(cfg),
                 "device": xgb_device(cfg)})
    model = xgb.XGBClassifier(**best)
    model.fit(X_tr_res, y_tr_res, eval_set=[(X_va, y_va)], verbose=False)

    report = full_report(y_va, model.predict_proba(X_va)[:, 1],
                         cfg["gnn"]["threshold"])
    log.info("Validación mes 5 -> %s", report)

    model.save_model(models_dir / "xgboost_baseline.json")
    with open(reports_dir / "xgboost_val_metrics.json", "w") as f:
        json.dump({"best_params": study.best_params, "val": report}, f, indent=2)
    log.info("Modelo CONGELADO en %s — no se vuelve a tocar.",
             models_dir / "xgboost_baseline.json")


if __name__ == "__main__":
    main()
