"""
Etapas `heads` y `heads_refit` — las tres cabezas XGBoost.

    control          las N features tabulares      ¿cuánto vale lo tabular solo?
    solo_gnn         SOLO el embedding de la GNN   ¿basta el grafo por sí mismo?
    gnn_mas_tabular  tabular + embedding           ¿APORTA el grafo sobre lo tabular?

Las tres comparten hiperparámetros, ventana de entrenamiento y tratamiento de
desbalance. Así la diferencia entre ellas es atribuible SOLO a las columnas.

OPTUNA CORRE UNA SOLA VEZ, sobre `control`
Si cada variante buscara por su cuenta, una diferencia de PR-AUC confundiría dos
causas: más información contra un sorteo de hiperparámetros más afortunado. Se
afina sobre la MÁS PEQUEÑA a propósito, así cualquier ventaja de las otras es
una cota inferior conservadora y no es atacable con "los hiperparámetros estaban
hechos a su medida".

DOS VENTANAS
    --window train      meses 1-4, valida en el mes 5  -> aquí se elige
    --window trainval   meses 1-5, DESDE CERO          -> esta va al mes 6

"Desde cero" es literal: en `trainval` no se hereda `n_estimators`, no hay warm
start y no se reutiliza ningún booster. Lo único que se hereda son los
hiperparámetros de Optuna, que es lo que hace comparables a las tres.

No importa torch a propósito: en macOS torch y XGBoost traen runtimes de OpenMP
distintos y cargarlos juntos mata el intérprete (ver utils/omp.py).
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
from src.hybrid.head import (VARIANTES, cargar_tabla, cols_embedding, columnas,
                             filtrar_prefijos,
                             guardar, matriz, nombre_modelo,
                             umbral_por_presupuesto)
from src.utils.common import (ensure_dirs, get_logger, load_config, n_jobs,
                              resolve, set_seed)
from src.utils.metrics import full_report

log = get_logger("hybrid.head")

# Los hiperparámetros que Optuna explora (ver
# baseline_xgboost/train_xgboost.py:objective_factory). Solo estos se pueden
# sembrar; `n_estimators` y `tree_method` los fija el objetivo y no se buscan.
_ESPACIO_OPTUNA = {"max_depth", "learning_rate", "subsample", "colsample_bytree",
                   "min_child_weight", "reg_lambda", "reg_alpha"}


def _params(cfg, best, variante: str | None = None):
    """
    Hiperparámetros de una cabeza: los de Optuna, más la ablación de capacidad
    si está activa, más el override por cabeza si lo hay.
    """
    x = cfg["xgboost"]
    p = dict(**best, n_estimators=1000, objective="binary:logistic",
             eval_metric="auc", tree_method="hist", random_state=42,
             n_jobs=n_jobs(cfg), device=xgb_device(cfg))
    cap = x.get("capacidad_limitada") or {}
    if cap.get("activo"):
        p.update({k: v for k, v in cap.items() if k != "activo"})
    if variante:
        p.update((x.get("por_cabeza") or {}).get(variante, {}) or {})
    return p


def _entrenar(X_tr, y_tr, X_va, y_va, cfg, best, n_est=None, variante=None):
    import xgboost as xgb
    p = _params(cfg, best, variante)
    if n_est is not None:                     # sin validación: rondas fijas
        p["n_estimators"] = int(n_est)
    else:
        p["early_stopping_rounds"] = cfg["xgboost"]["early_stopping_rounds"]
    m = xgb.XGBClassifier(**p)
    if X_va is not None:
        m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    else:
        m.fit(X_tr, y_tr, verbose=False)
    inferir_en_cpu(m)
    return m


def main():
    ap = argparse.ArgumentParser(description="Cabezas XGBoost del híbrido")
    ap.add_argument("--window", choices=["train", "trainval"], default="train")
    args = ap.parse_args()

    cfg = load_config()
    ensure_dirs(cfg)
    set_seed(42)
    reports_dir = resolve(cfg, "reports_dir")
    hcfg = cfg.get("hybrid") or {}
    variantes = [str(v) for v in hcfg.get("variantes", VARIANTES)]

    df, cols_base = cargar_tabla(cfg, args.window)
    cols_emb = cols_embedding(df, "completo")
    cols_embv = cols_embedding(df, "vecinos")

    # Ablación de columnas, aplicada a TODAS las cabezas por igual.
    prefijos = cfg["xgboost"].get("excluir_prefijos") or []
    if prefijos:
        antes = len(cols_base)
        cols_base = filtrar_prefijos(cols_base, prefijos)
        log.info("ABLACIÓN: se excluyen los prefijos %s -> %d columnas "
                 "tabulares en vez de %d, para las TRES cabezas",
                 prefijos, len(cols_base), antes)
    if (cfg["xgboost"].get("capacidad_limitada") or {}).get("activo"):
        log.info("ABLACIÓN: capacidad limitada activa para las TRES cabezas: %s",
                 {k: v for k, v in cfg["xgboost"]["capacidad_limitada"].items()
                  if k != "activo"})
    # Los overrides por cabeza rompen la atribución: si están puestos, se avisa.
    con_override = {k: v for k, v in (cfg["xgboost"].get("por_cabeza") or {}).items() if v}
    if con_override:
        log.warning("Hay overrides por cabeza (%s). La diferencia entre cabezas "
                    "YA NO es atribuible solo a las columnas: decláralo en la "
                    "memoria.", con_override)
    split = df["split"].values
    y_all = df["isFraud"].values.astype(int)

    if args.window == "train":
        filas_tr = np.where(split == "train")[0]
        filas_va = np.where(split == "val")[0]
    else:
        filas_tr = np.where(np.isin(split, ["train", "val"]))[0]
        filas_va = None
    y_tr = y_all[filas_tr]

    log.info("%s: entrena %d filas (%.2f%% fraude)%s | %d tabulares + %d del embedding",
             args.window, len(filas_tr), 100 * y_tr.mean(),
             f" | valida {len(filas_va)}" if filas_va is not None else "",
             len(cols_base), len(cols_emb))

    # ---------- ventana train: Optuna una vez + las tres variantes ----------
    if args.window == "train":
        y_va = y_all[filas_va]
        n_trials = int(cfg["xgboost"]["optuna_trials"])
        modo = str(cfg["xgboost"].get("optuna_modo", "compartido"))

        def buscar(variante):
            """Optuna sobre una cabeza concreta. Mismo sampler y mismos trials
            para todas: es lo que hace comparables los resultados."""
            Xt = matriz(df, filas_tr, variante, cols_base, cols_emb, cols_embv)
            Xv = matriz(df, filas_va, variante, cols_base, cols_emb, cols_embv)
            Xr, yr = apply_smote(Xt, y_tr, cfg)
            est = optuna.create_study(direction="maximize",
                                      sampler=optuna.samplers.TPESampler(seed=42))
            # Si hay una configuración propuesta para esta cabeza, se evalúa
            # como PRIMER trial. Así la búsqueda arranca de algo conocido y solo
            # puede mejorarlo: si tu configuración es la mejor, gana ella.
            semilla = (cfg["xgboost"].get("por_cabeza") or {}).get(variante) or {}
            sembrado = {k: v for k, v in semilla.items()
                        if k in _ESPACIO_OPTUNA}
            if sembrado:
                est.enqueue_trial(sembrado)
                log.info("  '%s': se siembra la búsqueda con %s", variante, sembrado)
            est.optimize(objective_factory(Xr.astype(np.float32), yr, Xv, y_va, cfg),
                         n_trials=n_trials, show_progress_bar=True)
            log.info("  '%s': mejor AUC val %.4f", variante, est.best_value)
            del Xt, Xv, Xr
            return dict(est.best_params), float(est.best_value)

        best_por_variante, valores = {}, {}
        if modo == "por_cabeza":
            # Cada cabeza recibe su propia búsqueda con el MISMO presupuesto.
            # Es más justo cuando las entradas son tan distintas (431 columnas
            # dispersas contra 32 densas), a cambio de que la diferencia entre
            # cabezas incluya algo de varianza de búsqueda.
            log.info("Optuna POR CABEZA: %d trials para cada una de %s",
                     n_trials, variantes)
            for v in variantes:
                best_por_variante[v], valores[v] = buscar(v)
            best = best_por_variante[variantes[0]]
        else:
            v_opt = str(hcfg.get("optuna_on_variant", "control"))
            log.info("Optuna COMPARTIDO: %d trials sobre '%s', las demás heredan",
                     n_trials, v_opt)
            best, valores[v_opt] = buscar(v_opt)
            best_por_variante = {v: best for v in variantes}

        resultados, t0 = {}, time.time()
        for v in variantes:
            X_tr = matriz(df, filas_tr, v, cols_base, cols_emb, cols_embv)
            X_va = matriz(df, filas_va, v, cols_base, cols_emb, cols_embv)
            X_res, y_res = apply_smote(X_tr, y_tr, cfg)
            m = _entrenar(X_res.astype(np.float32), y_res, X_va, y_va, cfg,
                          best_por_variante[v], variante=v)
            s_va = m.predict_proba(X_va)[:, 1]
            # DOS puntos de operación. El fijo (0.5) queda como referencia
            # histórica, pero NO compara: las tres cabezas tienen calibraciones
            # distintas y a umbral fijo se mide agresividad, no detección. El
            # mismo modelo de este proyecto pasó de F1 0.4356 a 0.5785 solo por
            # corregir el punto de operación.
            pct = float(hcfg.get("alert_budget_pct", 2.0))
            thr = umbral_por_presupuesto(s_va, pct)
            rep = full_report(y_va, s_va, thr)
            rep["umbral"] = thr
            rep["alertas_pct"] = round(100 * float((s_va >= thr).mean()), 2)
            rep["a_umbral_fijo_0.5"] = full_report(y_va, s_va,
                                                   cfg["gnn"]["threshold"])
            rep["n_estimators"] = int(getattr(m, "best_iteration", 0) or 0) + 1
            rep["n_columnas"] = len(columnas(v, cols_base, cols_emb, cols_embv))
            resultados[v] = rep
            guardar(m.get_booster(), cfg, nombre_modelo(v))
            log.info("  %-16s %4d col | PR-AUC %.4f | ROC %.4f | recall %.4f "
                     "@%.1f%% alertas | %d árboles",
                     v, rep["n_columnas"], rep.get("pr_auc", float("nan")),
                     rep.get("auc_roc", float("nan")), rep["recall"], pct,
                     rep["n_estimators"])
            del X_tr, X_va, X_res

        with open(reports_dir / "heads_variantes.json", "w") as f:
            json.dump({"best_params": best,
                       "optuna_modo": modo,
                       "best_por_variante": best_por_variante,
                       "auc_optuna": valores,
                       "variantes": resultados,
                       "minutos": round((time.time() - t0) / 60, 1),
                       "nota": ("optuna_modo='por_cabeza': cada cabeza "
                                "recibió su propia búsqueda con el MISMO número "
                                "de trials y el mismo sampler. "
                                "optuna_modo='compartido': Optuna corrió una vez "
                                "sobre la más pequeña y las demás heredaron, lo "
                                "que hace de cualquier ventaja una cota inferior "
                                "conservadora.")}, f, indent=2, ensure_ascii=False)
        log.info("Variantes -> heads_variantes.json")
        return resultados

    # ---------- ventana trainval: las tres DESDE CERO con meses 1-5 ---------
    with open(reports_dir / "heads_variantes.json") as f:
        prev = json.load(f)
    best_por_variante = prev.get("best_por_variante") or {}
    best_global = prev["best_params"]
    t0 = time.time()

    informe = {"n_filas": int(len(filas_tr)), "variantes": {}}
    for v in variantes:
        # n_estimators heredado del early stopping que ESA variante hizo sobre
        # el mes 5: aquí ya no hay conjunto de validación con que pararse.
        n_est = prev["variantes"].get(v, {}).get("n_estimators")
        if not n_est:
            log.warning("Sin n_estimators previo para '%s': se omite", v)
            continue
        log.info("Reentrenando '%s' DESDE CERO con meses 1-5 (%d árboles)", v, n_est)
        X_res, y_res = apply_smote(matriz(df, filas_tr, v, cols_base, cols_emb, cols_embv),
                                   y_tr, cfg)
        m = _entrenar(X_res.astype(np.float32), y_res, None, None, cfg,
                      best_por_variante.get(v, best_global),
                      n_est=n_est, variante=v)
        guardar(m.get_booster(), cfg, nombre_modelo(v, "_prod"))
        informe["variantes"][v] = {"n_estimators": int(n_est),
                                   "n_columnas": len(columnas(v, cols_base, cols_emb, cols_embv))}
        del X_res

        # umbral operativo por presupuesto de alertas, medido en el mes 5
        filas_v5 = np.where(split == "val")[0]
        s_v5 = m.predict_proba(matriz(df, filas_v5, v, cols_base, cols_emb, cols_embv))[:, 1]
        pct = float(hcfg.get("alert_budget_pct", 2.0))
        thr = umbral_por_presupuesto(s_v5, pct)
        informe["variantes"][v]["umbral"] = thr
        informe["variantes"][v]["alertas_mes5_pct"] = round(
            100 * float((s_v5 >= thr).mean()), 2)

    informe["minutos"] = round((time.time() - t0) / 60, 1)
    with open(reports_dir / "heads_refit.json", "w") as f:
        json.dump(informe, f, indent=2, ensure_ascii=False)
    log.info("Cabezas de producción listas -> heads_refit.json")
    return informe


if __name__ == "__main__":
    main()
