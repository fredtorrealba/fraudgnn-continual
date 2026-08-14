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

# Este módulo carga PyTorch y XGBoost en el mismo proceso: en macOS hay que
# limitar OpenMP ANTES de importarlos o el intérprete muere con SIGSEGV.
# Ver src/utils/omp.py.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.omp import guard_omp

guard_omp()

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


def por_semana(test_df, y, scores_xgb, scores_gnn, thr):
    """
    AUC y PR-AUC SEMANA A SEMANA dentro del mes 6.

    Existe porque el AUC del mes agregado puede mentir. Medido en la ablación
    sin aristas: el modelo daba 0.8524 sobre el mes completo pero 0.6075 de
    media semanal — separaba transacciones por PERIODO TEMPORAL, no por fraude.
    Al agrupar el mes entero esa correlación con el tiempo infla la métrica;
    dentro de una semana, donde apenas hay variación temporal, no queda nada.

    Un sistema de fraude decide semana a semana, así que esta es la métrica con
    sentido operativo. La misma lógica que ya usa compare_gnns para elegir
    arquitectura (AUC walk-forward), aplicada ahora al mes de test.
    """
    if "week_in_month" not in test_df.columns:
        return None
    semanas = test_df["week_in_month"].values
    filas = []
    for w in sorted(set(semanas.tolist())):
        m = semanas == w
        if len(np.unique(y[m])) < 2:
            continue
        rx = full_report(y[m], scores_xgb[m], thr)
        rg = full_report(y[m], scores_gnn[m], thr)
        filas.append({"semana": int(w), "n": int(m.sum()),
                      "n_fraude": int(y[m].sum()),
                      "xgboost": {"auc_roc": rx.get("auc_roc"),
                                  "pr_auc": rx.get("pr_auc")},
                      "gnn_cl": {"auc_roc": rg.get("auc_roc"),
                                 "pr_auc": rg.get("pr_auc")}})
    if not filas:
        return None
    med = lambda k, c: float(np.mean([f[k][c] for f in filas]))
    return {"nota": ("AUC dentro de cada semana. Si cae mucho respecto al mes "
                     "agregado, la métrica mensual estaba inflada por "
                     "correlación temporal, no por capacidad de detección."),
            "semanas": filas,
            "media_semanal": {
                "xgboost": {"auc_roc": med("xgboost", "auc_roc"),
                            "pr_auc": med("xgboost", "pr_auc")},
                "gnn_cl": {"auc_roc": med("gnn_cl", "auc_roc"),
                           "pr_auc": med("gnn_cl", "pr_auc")}}}


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
    # Booster nativo, NO XGBClassifier: el wrapper de sklearn comprueba
    # `self._estimator_type`, que las versiones nuevas de scikit-learn ya no
    # definen, y load_model revienta con "TypeError: _estimator_type undefined".
    # El Booster no depende de sklearn, así que es inmune a ese vaivén de
    # versiones. Es además lo que ya hace src/hybrid/head.py:cargar().
    # device=cpu: si el modelo se entrenó en GPU quedaría en cuda:0 y predecir
    # sobre numpy caería a DMatrix (ver train_xgboost.inferir_en_cpu).
    booster = xgb.Booster()
    booster.load_model(str(resolve(cfg, "models_dir") / "xgboost_baseline.json"))
    booster.set_param({"nthread": 1, "device": "cpu"})
    # binary:logistic -> inplace_predict devuelve ya P(fraude) en 1-D
    scores = booster.inplace_predict(test[cols].values.astype(np.float32))
    return (np.asarray(scores, dtype=np.float64),
            test["isFraud"].values.astype(int), test)


def control_variantes_on_test(cfg) -> dict:
    """
    Scores del mes 6 de las cabezas entrenadas con meses 1-5 que NO usan
    `gnn_score` (431 y 439). Son el CONTROL de atribución.

    Sin ellas, la comparación "híbrido 0.61 vs baseline 0.55" cambia dos cosas
    a la vez —las columnas (440 vs 431) y los datos (meses 1-5 vs 1-4)—, así
    que la diferencia no es atribuible. Con el control, cada efecto se aísla:

        baseline        -> 431(1-5)   = lo que aporta el mes extra de datos
        431(1-5)        -> hibrido    = lo que aporta el gnn_score

    La variante completa NO se recalcula aquí: su `gnn_score` del mes 6 exige
    correr la GNN, y ya viene resuelto en `hybrid_cl_test_scores.npz`.
    Devuelve {} si la corrida no dejó esas cabezas (compatibilidad hacia atrás).
    """
    from src.hybrid.head import cargar_tabla, columnas, matriz

    models_dir = resolve(cfg, "models_dir")
    variantes = [v for v in (cfg.get("hybrid") or {}).get("variants", [])
                 if (models_dir / f"hybrid_head_prod_{int(v)}.json").exists()]
    if not variantes:
        return {}

    df, cols_base = cargar_tabla(cfg, None)      # sin OOF: estas no lo usan
    filas_te = np.where(df["split"].values == "test")[0]
    salida = {}
    for v in sorted(int(x) for x in variantes):
        if "gnn_score" in columnas(v, cols_base):
            continue                              # necesita la GNN en ejecución
        booster = xgb.Booster()
        booster.load_model(str(models_dir / f"hybrid_head_prod_{v}.json"))
        booster.set_param({"nthread": 1, "device": "cpu"})
        salida[f"control_{v}"] = np.asarray(
            booster.inplace_predict(matriz(df, filas_te, v, cols_base)),
            dtype=np.float64)
        log.info("Control %d columnas (meses 1-5) cargado", v)
    return salida


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

    # El sistema híbrido, si la corrida lo produjo. Sin él todo lo demás sigue
    # funcionando igual: la comparación GNN vs baseline no depende de esto.
    ruta_hib = reports_dir / "hybrid_cl_test_scores.npz"
    hib_scores = None
    if ruta_hib.exists():
        hib_pack = np.load(ruta_hib)
        hib_scores = hib_pack["scores"]
        assert np.array_equal(hib_pack["node_idx"], test_idx), \
            "los scores del híbrido no están alineados con los de la GNN"
        log.info("Sistema híbrido cargado (%d scores)", len(hib_scores))

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
    # --- el sistema híbrido, en los mismos barridos ---
    sistemas = {"gnn_cl": gnn_cl_scores}
    if hib_scores is not None:
        sistemas["hibrido"] = hib_scores
    sistemas.update(control_variantes_on_test(cfg))
    # el barrido recorre todo lo que haya, no solo el híbrido
    for etq, sc in sistemas.items():
        if etq == "gnn_cl":
            continue                              # ya va como recall_gnn_cl
        for fila, k in zip(por_presupuesto, presupuestos):
            fila[f"recall_{etq}"] = recall_at_budget(y_gnn, sc, k)

    por_precision = []
    for objetivo in PRECISION_TARGETS:
        rx, kx = recall_at_precision(y_xgb, xgb_scores, objetivo)
        fila = {"precision_objetivo": objetivo,
                "xgboost": {"recall": rx, "n_alertas": kx}}
        for etq, sc in sistemas.items():
            r, k = recall_at_precision(y_gnn, sc, objetivo)
            fila[etq] = {"recall": r, "n_alertas": k}
        por_precision.append(fila)

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

    semanal = por_semana(test_df, y_gnn, xgb_scores, gnn_cl_scores, thr)
    desplego = cl_desplego(cfg)

    globales = {"xgboost_frozen": rep_xgb, "gnn_continual_learning": rep_cl}
    if hib_scores is not None:
        # Umbral propio: el híbrido no comparte escala con la GNN (ver
        # hybrid/head.py: umbral_por_presupuesto).
        # Punto de operación por PRESUPUESTO sobre el propio mes 6, no el
        # umbral congelado en el mes 5. Medido: ese umbral produce un 1,07% de
        # alertas en el mes 6 en vez del 2% configurado (los scores se
        # desplazan entre meses), lo que deja al híbrido artificialmente
        # conservador —842 TP y solo 81 FP— y no refleja lo que un equipo con
        # capacidad fija revisaría. Se reporta también el congelado para que la
        # magnitud del drift quede a la vista.
        from src.hybrid.head import umbral_por_presupuesto
        pct = float((cfg.get("hybrid") or {}).get("alert_budget_pct", 2.0))
        thr_hib = umbral_por_presupuesto(hib_scores, pct)
        thr_frio = float(hib_pack["umbral"][0]) if "umbral" in hib_pack else thr
        globales["hibrido"] = full_report(y_gnn, hib_scores, thr_hib)
        globales["hibrido"]["umbral_usado"] = thr_hib
        globales["hibrido"]["umbral_congelado_mes5"] = thr_frio
        globales["hibrido"]["alertas_con_umbral_congelado"] = int(
            (hib_scores >= thr_frio).sum())
        globales["hibrido"]["nota_umbral"] = (
            f"Umbral recalculado sobre el mes 6 para un {pct}% de alertas. "
            f"El congelado del mes 5 ({thr_frio:.4f}) solo produce "
            f"{100 * (hib_scores >= thr_frio).mean():.2f}% aquí.")
    for etq, sc in sistemas.items():
        if etq.startswith("control_"):
            globales[etq] = full_report(y_gnn, sc, thr)

    result = {
        "month6_overall": globales,
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
        "month6_weekly": semanal,
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
    hay_hib = hib_scores is not None
    for fila in por_presupuesto:
        cand = {"XGBoost": fila["recall_xgboost"], "GNN+CL": fila["recall_gnn_cl"]}
        if hay_hib:
            cand["híbrido"] = fila["recall_hibrido"]
        log.info("  %6d alertas (%5.2f%%) | XGBoost %.4f | GNN+CL %.4f%s | gana %s",
                 fila["n_alertas"], fila["pct_del_mes"],
                 fila["recall_xgboost"], fila["recall_gnn_cl"],
                 f" | híbrido {fila['recall_hibrido']:.4f}" if hay_hib else "",
                 max(cand, key=cand.get))
    log.info("-- A IGUAL precisión --")
    fmt = lambda d: ("—" if d["recall"] is None
                     else f"{d['recall']:.4f} ({d['n_alertas']} alertas)")
    for fila in por_precision:
        log.info("  precisión %3.0f%% | XGBoost %-22s | GNN+CL %-22s%s",
                 100 * fila["precision_objetivo"], fmt(fila["xgboost"]),
                 fmt(fila["gnn_cl"]),
                 f" | híbrido {fmt(fila['hibrido'])}" if hay_hib else "")
    if semanal:
        log.info("-- SEMANA A SEMANA (el mes agregado puede inflar el AUC) --")
        for f in semanal["semanas"]:
            log.info("  semana %d (%5d txn, %3d fraudes) | XGBoost %.4f | GNN+CL %.4f",
                     f["semana"], f["n"], f["n_fraude"],
                     f["xgboost"]["auc_roc"], f["gnn_cl"]["auc_roc"])
        ms = semanal["media_semanal"]
        log.info("  MEDIA SEMANAL             | XGBoost %.4f | GNN+CL %.4f",
                 ms["xgboost"]["auc_roc"], ms["gnn_cl"]["auc_roc"])
        log.info("  (contra el mes agregado   | XGBoost %.4f | GNN+CL %.4f)",
                 rep_xgb.get("auc_roc", float("nan")),
                 rep_cl.get("auc_roc", float("nan")))
    log.info("-- Threshold-independiente --")
    for etq, rep in (("XGBoost", rep_xgb), ("GNN+CL", rep_cl),
                     *((("híbrido", globales["hibrido"]),) if hay_hib else ())):
        log.info("  %-8s ROC-AUC %.4f | PR-AUC %.4f", etq,
                 rep.get("auc_roc", float("nan")), rep.get("pr_auc", float("nan")))
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
