"""
Métricas compartidas por todos los módulos (baseline, GNN, CL, comparación).
La validación SIEMPRE se hace sobre la distribución real, sin balanceo.
"""
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def full_report(y_true, y_score, threshold: float = 0.5) -> dict:
    """Reporte estándar: AUC-ROC, PR-AUC, recall, precision, F1 al threshold."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    y_pred = (y_score >= threshold).astype(int)
    out = {
        "n": int(len(y_true)),
        "n_fraud": int(y_true.sum()),
        "threshold": threshold,
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    # AUC solo si hay ambas clases
    if len(np.unique(y_true)) == 2:
        out["auc_roc"] = float(roc_auc_score(y_true, y_score))
        out["pr_auc"] = float(average_precision_score(y_true, y_score))
    return out


def recall_at_threshold(y_true, y_score, threshold: float = 0.5) -> float:
    y_pred = (np.asarray(y_score) >= threshold).astype(int)
    return float(recall_score(np.asarray(y_true), y_pred, zero_division=0))
