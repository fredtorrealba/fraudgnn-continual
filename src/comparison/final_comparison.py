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
    # Asignación directa, NO setdefault: el pipeline padre exporta
    # OMP_NUM_THREADS desde compute.n_jobs y el subproceso lo hereda, así que
    # un setdefault no llegaría a aplicarse nunca. En macOS esto no es un valor
    # por defecto sino un requisito para no segfaultear.
    os.environ["OMP_NUM_THREADS"] = "1"

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


# Puntos de operación para la comparación justa. No son hiperparámetros del
# modelo (por eso no van a config.yaml): son los cortes con que se lee el
# resultado. Porcentajes del mes y objetivos de precisión.
BUDGETS_PCT = (0.5, 1.0, 2.0, 5.0, 10.0, 25.0)
PRECISION_TARGETS = (0.90, 0.80, 0.70, 0.50)


def _curva(y: np.ndarray, s: np.ndarray):
    """Ordenado por score descendente: TP acumulados y precisión en cada K."""
    orden = np.argsort(-s)
    tp = np.cumsum(y[orden])
    return tp, tp / np.arange(1, len(y) + 1)


def recall_at_budget(y, s, k: int) -> float:
    """Recall si solo se pueden revisar las K transacciones de mayor score."""
    k = max(1, min(int(k), len(y)))
    tp, _ = _curva(y, s)
    return float(tp[k - 1] / max(y.sum(), 1))


def recall_at_precision(y, s, objetivo: float):
    """Recall máximo alcanzable sin bajar de `objetivo` de precisión."""
    tp, prec = _curva(y, s)
    ok = np.where(prec >= objetivo)[0]
    if len(ok) == 0:
        return None, 0
    k = int(ok[-1]) + 1
    return float(tp[k - 1] / max(y.sum(), 1)), k


def cl_desplego(cfg) -> bool:
    """¿Algún ciclo de CL llegó a desplegar? Si no, el modelo nunca cambió."""
    ruta = resolve(cfg, "reports_dir") / "cl_cycles.json"
    if not ruta.exists():
        return True                      # sin datos, no se afirma nada
    with open(ruta) as f:
        ciclos = json.load(f)
    return any(c.get("verdict", {}).get("deploy") for c in ciclos)


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
    from src.gnn.train_gnn import ruta_modelo_operativo
    ruta, etiqueta = ruta_modelo_operativo(cfg)
    ckpt = torch.load(ruta, weights_only=False)
    cfg["gnn"]["in_dim"] = ckpt["in_dim"]
    model = build_model(ckpt["model_name"], cfg)
    model.load_state_dict(ckpt["state_dict"])
    log.info("GNN de referencia (pre-CL): %s", etiqueta)
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

    # --- comparación JUSTA: mismo coste operativo -------------------------
    # Comparar recall a un threshold fijo entre dos modelos con calibraciones
    # distintas no mide detección, mide agresividad: la GNN entrena con
    # pos_weight (~27) y sus scores están desplazados hacia arriba, así que al
    # mismo 0.5 alerta muchísimo más. Estas dos vistas eliminan ese sesgo.
    n_alertas_xgb = int((xgb_scores >= thr).sum())
    presupuestos = sorted({int(len(y_gnn) * p / 100) for p in BUDGETS_PCT}
                          | {n_alertas_xgb})
    por_presupuesto = [
        {"n_alertas": k,
         "pct_del_mes": round(100 * k / len(y_gnn), 2),
         "recall_xgboost": recall_at_budget(y_xgb, xgb_scores, k),
         "recall_gnn_cl": recall_at_budget(y_gnn, gnn_cl_scores, k)}
        for k in presupuestos
    ]
    por_precision = []
    for objetivo in PRECISION_TARGETS:
        rx, kx = recall_at_precision(y_xgb, xgb_scores, objetivo)
        rg, kg = recall_at_precision(y_gnn, gnn_cl_scores, objetivo)
        por_precision.append({
            "precision_objetivo": objetivo,
            "xgboost": {"recall": rx, "n_alertas": kx},
            "gnn_cl": {"recall": rg, "n_alertas": kg},
        })

    # A igual presupuesto (el de XGBoost en su threshold), ¿quién gana?
    r_xgb_ref = recall_at_budget(y_xgb, xgb_scores, n_alertas_xgb)
    r_gnn_ref = recall_at_budget(y_gnn, gnn_cl_scores, n_alertas_xgb)

    # El impacto económico se recalcula al MISMO presupuesto, no al mismo
    # threshold: si no, se contabiliza como "ganancia" el alertar 16x más.
    k_ref = n_alertas_xgb
    top_gnn = np.zeros(len(y_gnn), dtype=bool)
    top_gnn[np.argsort(-gnn_cl_scores)[:k_ref]] = True
    top_xgb = np.zeros(len(y_xgb), dtype=bool)
    top_xgb[np.argsort(-xgb_scores)[:k_ref]] = True
    extra_iso = (y_gnn == 1) & top_gnn & ~top_xgb
    if "TransactionAmt" in test_df.columns:
        usd_iso = float(test_df.loc[np.where(extra_iso)[0], "TransactionAmt"].sum())
    else:
        usd_iso = float(extra_iso.sum() * cfg["comparison"]["avg_fraud_amount_usd"])

    desplego = cl_desplego(cfg)

    result = {
        "month6_overall": {"xgboost_frozen": rep_xgb, "gnn_continual_learning": rep_cl},
        "matched_budget": {
            "nota": ("Recall cuando ambos modelos emiten el MISMO número de "
                     "alertas. Es la comparación con sentido operativo: el "
                     "coste de revisión es el mismo para los dos."),
            "presupuesto_referencia": n_alertas_xgb,
            "recall_xgboost_ref": r_xgb_ref,
            "recall_gnn_cl_ref": r_gnn_ref,
            "gana_en_referencia": "xgboost" if r_xgb_ref > r_gnn_ref else "gnn_cl",
            "barrido": por_presupuesto,
        },
        "matched_precision": {
            "nota": ("Recall máximo de cada modelo sin bajar de la precisión "
                     "objetivo. Responde: a igual calidad de alerta, ¿quién "
                     "recupera más fraude?"),
            "barrido": por_precision,
        },
        "emerging_patterns": {
            "definition": "fraudes del mes 6 con score <0.5 del GNN original (pre-CL)",
            "n_emerging_frauds": n_emerging,
            "recall_gnn_cl": recall_cl_emerging,
            "recall_xgboost": recall_xgb_emerging,
            "recall_gap": round(gap, 4),
            "kpi_gap_target": cfg["comparison"]["kpi_recall_gap"],
            "kpi_ok": bool(kpi_ok),
            "cl_desplego_alguna_vez": desplego,
            "advertencia": (None if desplego else
                "NINGÚN ciclo de CL llegó a desplegar: el modelo 'GNN+CL' y el "
                "'GNN original' son EL MISMO. La diferencia entre sus scores es "
                "solo ruido del neighbor sampling (cada pasada elige vecinos "
                "distintos), así que este recall NO mide adaptación."),
        },
        "economic_impact": {
            "a_threshold_fijo": {
                "extra_frauds_detected_vs_xgboost": int(extra_detected.sum()),
                "estimated_usd_saved": round(usd, 2),
                "method": usd_note,
                "sesgo": ("Cuenta como ganancia alertar más. Solo es "
                          "comparable si ambos modelos emiten alertas "
                          "similares; ver a_igual_presupuesto."),
            },
            "a_igual_presupuesto": {
                "presupuesto": k_ref,
                "extra_frauds_detected_vs_xgboost": int(extra_iso.sum()),
                "estimated_usd_saved": round(usd_iso, 2),
                "method": ("fraudes en el top-K de la GNN que NO están en el "
                           "top-K de XGBoost, con el mismo K"),
            },
        },
    }

    with open(reports_dir / "final_comparison.json", "w") as f:
        json.dump(result, f, indent=2)

    log.info("=========== COMPARACIÓN FINAL (mes 6) ===========")
    log.info("-- A threshold fijo %.2f (NO comparable: distinta calibración) --", thr)
    log.info("  XGBoost  alertas %6d (%5.2f%%) | recall %.4f | precisión %.4f",
             n_alertas_xgb, 100 * n_alertas_xgb / len(y_xgb),
             rep_xgb["recall"], rep_xgb["precision"])
    log.info("  GNN+CL   alertas %6d (%5.2f%%) | recall %.4f | precisión %.4f",
             int((gnn_cl_scores >= thr).sum()),
             100 * (gnn_cl_scores >= thr).sum() / len(y_gnn),
             rep_cl["recall"], rep_cl["precision"])
    log.info("-- A IGUAL presupuesto de alertas (comparación válida) --")
    for fila in por_presupuesto:
        log.info("  %6d alertas (%5.2f%%) | XGBoost %.4f | GNN+CL %.4f | gana %s",
                 fila["n_alertas"], fila["pct_del_mes"],
                 fila["recall_xgboost"], fila["recall_gnn_cl"],
                 "XGBoost" if fila["recall_xgboost"] > fila["recall_gnn_cl"] else "GNN+CL")
    log.info("-- A IGUAL precisión --")
    for fila in por_precision:
        fmt = lambda d: ("—" if d["recall"] is None
                         else f"{d['recall']:.4f} ({d['n_alertas']} alertas)")
        log.info("  precisión %3.0f%% | XGBoost %-22s | GNN+CL %s",
                 100 * fila["precision_objetivo"], fmt(fila["xgboost"]),
                 fmt(fila["gnn_cl"]))
    log.info("-- Threshold-independiente --")
    log.info("  XGBoost  ROC-AUC %.4f | PR-AUC %.4f",
             rep_xgb.get("auc_roc", float("nan")), rep_xgb.get("pr_auc", float("nan")))
    log.info("  GNN+CL   ROC-AUC %.4f | PR-AUC %.4f",
             rep_cl.get("auc_roc", float("nan")), rep_cl.get("pr_auc", float("nan")))
    log.info("-- Emergentes --")
    log.info("  XGBoost %.4f | GNN+CL %.4f | gap %+.4f (KPI >= %.2f: %s)",
             recall_xgb_emerging, recall_cl_emerging, gap,
             cfg["comparison"]["kpi_recall_gap"], "OK" if kpi_ok else "NO")
    if not desplego:
        log.warning("  NINGÚN ciclo de CL desplegó: 'GNN+CL' y 'GNN original' son "
                    "el MISMO modelo. Ese gap es ruido de muestreo, no adaptación.")
    log.info("-- Impacto económico --")
    log.info("  a threshold fijo   : %d fraudes ~ USD %.0f  (sesgado: alerta 16x más)",
             int(extra_detected.sum()), usd)
    log.info("  a igual presupuesto: %d fraudes ~ USD %.0f",
             int(extra_iso.sum()), usd_iso)


if __name__ == "__main__":
    main()
