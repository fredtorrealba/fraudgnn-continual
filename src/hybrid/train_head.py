"""
Paso 6 / 9 — Entrenamiento de la cabeza XGBoost del sistema híbrido.

    --window train      entrena con meses 1-4, valida en el 5, tres variantes
    --window trainval   reentrena la ganadora con meses 1-5 -> modelo de producción

OPTUNA SE EJECUTA UNA SOLA VEZ, Y SOBRE LA VARIANTE DE 431 COLUMNAS
Las tres variantes existen para responder "¿qué aportan las columnas del
grafo?", que es una afirmación sobre el CONJUNTO DE FEATURES. Si cada una
buscara sus propios hiperparámetros, una diferencia de PR-AUC confundiría dos
causas —más información contra un sorteo más afortunado— y con 30 trials la
dispersión es del orden del efecto que se quiere medir.

Se afina sobre la variante MÁS PEQUEÑA a propósito: si se afinara sobre 440,
cualquier ventaja suya sería atacable ("los hiperparámetros estaban hechos a su
medida"). Afinando sobre 431 se le regala la ventaja a la referencia, y el
resultado de 440 pasa a ser una COTA INFERIOR conservadora.

Lo único que varía por variante es `n_estimators`, fijado por early stopping
sobre el mes 5: es una regla de parada, no un hiperparámetro.

El desbalance se trata con SMOTE, igual que el baseline, para que la única
diferencia entre ambos sean las columnas. Se midió la alternativa
(`scale_pos_weight`) y cuesta 0.024 de PR-AUC, así que se mantiene SMOTE; el
coste asumido es que interpola también las columnas del grafo.

`--window trainval` NO corre Optuna — hereda, igual que `src/gnn/refit.py`
hereda `best_epoch`. Sin mes de validación no hay con qué buscar.

No importa torch: este proceso solo carga XGBoost.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import optuna

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.baseline_xgboost.smote_pipeline import apply_smote
from src.baseline_xgboost.train_xgboost import (inferir_en_cpu,
                                                objective_factory, xgb_device)
from src.hybrid.head import (VARIANTES, cargar_tabla, columnas, guardar, matriz,
                             nombre_modelo, umbral_por_presupuesto)
from src.utils.common import (ensure_dirs, get_logger, load_config, n_jobs,
                              resolve, set_seed)
from src.utils.metrics import full_report

log = get_logger("hybrid.head")


def _params_base(cfg, best_params: dict) -> dict:
    return dict(best_params, objective="binary:logistic", eval_metric="auc",
                tree_method="hist", n_estimators=1000,
                early_stopping_rounds=cfg["xgboost"]["early_stopping_rounds"],
                random_state=42, n_jobs=n_jobs(cfg), device=xgb_device(cfg))


def _entrenar(X_tr, y_tr, X_va, y_va, cfg, best_params, n_est=None):
    import xgboost as xgb
    p = _params_base(cfg, best_params)
    if n_est is not None:                     # sin validación: rondas heredadas
        p["n_estimators"] = int(n_est)
        p.pop("early_stopping_rounds", None)
    m = xgb.XGBClassifier(**p)
    if X_va is not None:
        m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    else:
        m.fit(X_tr, y_tr, verbose=False)
    return m


def main():
    ap = argparse.ArgumentParser(description="Cabeza XGBoost del sistema híbrido")
    ap.add_argument("--window", choices=["train", "trainval"], default="train")
    args = ap.parse_args()

    cfg = load_config()
    ensure_dirs(cfg)
    set_seed(42)
    reports_dir = resolve(cfg, "reports_dir")
    hcfg = cfg.get("hybrid") or {}

    oof_window = "train" if args.window == "train" else "trainval"
    df, cols_base = cargar_tabla(cfg, oof_window)
    log.info("Tabla: %d filas | %d columnas base + %d estructurales + gnn_score",
             len(df), len(cols_base), 8)

    if args.window == "train":
        filas_tr = np.where(df["split"].values == "train")[0]
        filas_va = np.where(df["split"].values == "val")[0]
    else:
        filas_tr = np.where(np.isin(df["split"].values, ["train", "val"]))[0]
        filas_va = None
    y_tr = df["isFraud"].values[filas_tr].astype(int)
    log.info("%s: entrena %d filas (%.2f%% fraude)%s", args.window, len(filas_tr),
             100 * y_tr.mean(),
             f" | valida {len(filas_va)}" if filas_va is not None else " | sin validación")

    # ---- ventana de entrenamiento inicial: Optuna + las tres variantes ----
    if args.window == "train":
        v_opt = int(hcfg.get("optuna_on_variant", 431))
        y_va = df["isFraud"].values[filas_va].astype(int)
        Xo_tr = matriz(df, filas_tr, v_opt, cols_base)
        Xo_va = matriz(df, filas_va, v_opt, cols_base)
        Xo_res, yo_res = apply_smote(Xo_tr, y_tr, cfg)
        Xo_res = Xo_res.astype(np.float32)

        log.info("Optuna (%d trials) sobre la variante de %d columnas — "
                 "los hiperparámetros se reutilizan en las tres",
                 cfg["xgboost"]["optuna_trials"], v_opt)
        estudio = optuna.create_study(direction="maximize",
                                      sampler=optuna.samplers.TPESampler(seed=42))
        estudio.optimize(objective_factory(Xo_res, yo_res, Xo_va, y_va, cfg),
                         n_trials=cfg["xgboost"]["optuna_trials"],
                         show_progress_bar=True)
        best = dict(estudio.best_params)
        log.info("Mejor AUC val: %.4f", estudio.best_value)
        del Xo_tr, Xo_va, Xo_res

        resultados, t0 = {}, time.time()
        for v in [int(x) for x in hcfg.get("variants", VARIANTES)]:
            X_tr = matriz(df, filas_tr, v, cols_base)
            X_va = matriz(df, filas_va, v, cols_base)
            X_res, y_res = apply_smote(X_tr, y_tr, cfg)
            m = _entrenar(X_res.astype(np.float32), y_res, X_va, y_va, cfg, best)
            inferir_en_cpu(m)
            rep = full_report(y_va, m.predict_proba(X_va)[:, 1], cfg["gnn"]["threshold"])
            rep["n_estimators"] = int(getattr(m, "best_iteration", 0) or 0) + 1
            rep["n_columnas"] = len(columnas(v, cols_base))
            resultados[str(v)] = rep
            guardar(m.get_booster(), cfg, nombre_modelo(v))
            log.info("  variante %d: PR-AUC %.4f | ROC %.4f | %d árboles",
                     v, rep.get("pr_auc", float("nan")),
                     rep.get("auc_roc", float("nan")), rep["n_estimators"])
            del X_tr, X_va, X_res

        with open(reports_dir / "hybrid_variants.json", "w") as f:
            json.dump({"best_params": best, "optuna_on_variant": v_opt,
                       "variantes": resultados,
                       "minutos": round((time.time() - t0) / 60, 1),
                       "nota": ("Optuna corrió UNA vez sobre la variante de "
                                f"{v_opt} columnas; las tres comparten "
                                "hiperparámetros para que la diferencia sea "
                                "atribuible solo a las columnas.")}, f,
                      indent=2, ensure_ascii=False)
        log.info("Variantes -> hybrid_variants.json")
        return resultados

    # ---- refit: la variante completa con meses 1-5, sin Optuna ----
    with open(reports_dir / "hybrid_variants.json") as f:
        prev = json.load(f)
    todas = sorted(int(x) for x in prev["variantes"])
    v = max(todas)                       # la completa: la que va a producción
    t0 = time.time()

    # Se entrenan TODAS con meses 1-5, no solo la completa. Las que no usan
    # gnn_score (431, 439) son el CONTROL de atribución: sin ellas, comparar el
    # híbrido (440 col, meses 1-5) contra el baseline congelado (431 col, meses
    # 1-4) cambia las columnas Y los datos a la vez, y la diferencia no se
    # puede atribuir a ninguna de las dos cosas. `final` las mide en el mes 6.
    for vi in todas:
        n_i = prev["variantes"][str(vi)]["n_estimators"]
        log.info("Refit variante %d con %d árboles heredados (sin Optuna)",
                 vi, n_i)
        X_res, y_res = apply_smote(matriz(df, filas_tr, vi, cols_base), y_tr, cfg)
        mi = _entrenar(X_res.astype(np.float32), y_res, None, None, cfg,
                       prev["best_params"], n_est=n_i)
        inferir_en_cpu(mi)
        guardar(mi.get_booster(), cfg, f"hybrid_head_prod_{vi}.json")
        if vi == v:
            m, n_est = mi, n_i
            # Nombre estable para la cabeza de producción: lo consumen
            # cl_orchestrator y HybridSystem, que no deben saber de variantes.
            guardar(mi.get_booster(), cfg, "hybrid_head_prod.json")
        del X_res

    # Umbral operativo: el cuantil que produce el presupuesto de alertas
    # configurado, medido sobre el mes 5 (ver head.umbral_por_presupuesto).
    filas_v5 = np.where(df["split"].values == "val")[0]
    s_v5 = m.predict_proba(matriz(df, filas_v5, v, cols_base))[:, 1]
    pct = float(hcfg.get("alert_budget_pct", 2.0))
    thr = umbral_por_presupuesto(s_v5, pct)
    with open(reports_dir / "hybrid_thresholds.json", "w") as f:
        json.dump({"variante": v, "alert_budget_pct": pct, "umbral": thr,
                   "alertas_mes5": int((s_v5 >= thr).sum()),
                   "n_mes5": int(len(s_v5)),
                   "nota": ("Umbral por volumen de alertas, no fijo: la GNN "
                            "entrena con pos_weight ~27 y la cabeza devuelve "
                            "probabilidades calibradas. Con 0.5 el sistema no "
                            "alertaría casi nada.")}, f, indent=2, ensure_ascii=False)

    informe = {"variante": v, "n_estimators": n_est, "n_filas": int(len(filas_tr)),
               "umbral": thr, "variantes_entrenadas": todas,
               "minutos": round((time.time() - t0) / 60, 1)}
    with open(reports_dir / "hybrid_refit.json", "w") as f:
        json.dump(informe, f, indent=2, ensure_ascii=False)
    log.info("Cabeza de producción lista | umbral %.4f -> %d alertas en el mes 5 (%.1f%%)",
             thr, int((s_v5 >= thr).sum()), 100 * (s_v5 >= thr).mean())
    return informe


if __name__ == "__main__":
    main()
