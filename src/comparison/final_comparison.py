"""
Etapa `final` — métricas de las tres cabezas sobre TODOS los meses.

QUÉ SE COMPARA

    control          las N features tabulares      ¿cuánto vale lo tabular solo?
    solo_gnn         SOLO el embedding de la GNN   ¿basta el grafo por sí mismo?
    gnn_mas_tabular  tabular + embedding           ¿APORTA el grafo sobre lo tabular?

Las tres se entrenaron con la MISMA ventana (meses 1-5), los mismos
hiperparámetros y el mismo SMOTE, así que la diferencia es atribuible solo a las
columnas. Esa es toda la razón de ser de `control`: sin él, comparar el híbrido
contra un baseline entrenado con menos meses cambia dos cosas a la vez y la
diferencia no significa nada.

TRES CORTES DE MÉTRICAS
    por mes (1-6)  los meses 1-5 van marcados IN-SAMPLE: el modelo entrenó con
                   ellos y sus números son optimistas por construcción
    mes 5          validación — donde se eligió arquitectura y cabeza
    mes 6          test — no se toca hasta aquí

EL UMBRAL NO ES 0.5
Cada cabeza tiene su propia calibración, así que un corte fijo compara
volúmenes de alerta distintos y no dice nada. Se usa el cuantil que produce
`hybrid.alert_budget_pct` de alertas sobre el mes que se mide, y las
comparaciones de fondo van a IGUAL PRESUPUESTO y a IGUAL PRECISIÓN.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.omp import guard_omp

guard_omp()   # antes de importar xgboost (SIGSEGV en macOS si torch ya está)

from src.hybrid.head import (cargar, cargar_tabla, cols_embedding,  # noqa: E402
                             columnas, nombre_modelo,
                             umbral_por_presupuesto)
from src.utils.common import (ensure_dirs, get_logger, load_config,  # noqa: E402
                              resolve)
from sklearn.metrics import average_precision_score  # noqa: E402
from src.utils.metrics import full_report  # noqa: E402
from src.utils.ventanas import verificar  # noqa: E402

log = get_logger("comparison")

PRESUPUESTOS_PCT = (0.5, 1.0, 2.0, 5.0, 10.0, 25.0)
PRECISIONES = (0.9, 0.8, 0.7, 0.5)


def _curva(y, s):
    """(aciertos acumulados, precisión acumulada) recorriendo de mayor a menor."""
    orden = np.argsort(-np.asarray(s))
    tp = np.cumsum(np.asarray(y)[orden])
    return tp, tp / np.arange(1, len(tp) + 1)


def recall_at_budget(y, s, k: int) -> float:
    """Recall si solo se pueden revisar las K transacciones de mayor score."""
    k = max(1, min(int(k), len(y)))
    tp, _ = _curva(y, s)
    return float(tp[k - 1] / max(np.sum(y), 1))


def recall_at_precision(y, s, objetivo: float):
    """Recall máximo alcanzable sin bajar de `objetivo` de precisión."""
    tp, prec = _curva(y, s)
    ok = np.where(prec >= objetivo)[0]
    if len(ok) == 0:
        return None, 0
    k = int(ok[-1]) + 1
    return float(tp[k - 1] / max(np.sum(y), 1)), k


def _confusion(y, s, thr):
    """Reporte completo + matriz de confusión al umbral dado."""
    rep = full_report(y, s, thr)
    pred = np.asarray(s) >= thr
    y = np.asarray(y).astype(bool)
    rep.update(TP=int((pred & y).sum()), FP=int((pred & ~y).sum()),
               FN=int((~pred & y).sum()), TN=int((~pred & ~y).sum()),
               alertas=int(pred.sum()))
    n = len(y)
    rep["accuracy"] = float((rep["TP"] + rep["TN"]) / n) if n else 0.0
    # El accuracy va con su referencia al lado a propósito: con 3,4% de fraude,
    # un modelo que no alerte nunca saca 96,6%. Sin esa comparación el número
    # engaña.
    rep["accuracy_sin_alertar"] = float((~y).sum() / n) if n else 0.0
    return rep


def bootstrap_delta(y, s_a, s_b, n_rep: int = 1000, seed: int = 42) -> dict:
    """
    ¿La diferencia de PR-AUC entre dos cabezas es real o es ruido del mes?

    Sin esto, un +0.004 y un +0.04 se leen igual en el informe, y en un mes con
    ~2.900 fraudes el primero cabe de sobra dentro del error de muestreo.

    Bootstrap EMPAREJADO: se remuestrean FILAS (las mismas para ambas cabezas)
    y se recalcula la diferencia en cada réplica. Emparejar importa —las dos
    puntúan las mismas transacciones y sus errores están correlacionados—:
    tratarlas como independientes ensancharía el intervalo sin motivo.

    Mide el error de MUESTREO del mes de evaluación, no la varianza de
    entrenamiento (eso exigiría reentrenar con varias semillas). Es la cota
    optimista: si el intervalo ya cruza el cero, la varianza de entrenamiento
    solo puede empeorarlo.
    """
    y = np.asarray(y); s_a = np.asarray(s_a); s_b = np.asarray(s_b)
    rng = np.random.default_rng(seed)
    n = len(y)
    obs = average_precision_score(y, s_b) - average_precision_score(y, s_a)
    deltas = []
    for _ in range(n_rep):
        i = rng.integers(0, n, n)
        if y[i].sum() == 0 or y[i].sum() == len(i):
            continue                       # réplica sin ambas clases: no hay curva
        deltas.append(average_precision_score(y[i], s_b[i]) -
                      average_precision_score(y[i], s_a[i]))
    d = np.asarray(deltas)
    lo, hi = np.percentile(d, [2.5, 97.5])
    return {"delta_observado": round(float(obs), 5),
            "ic95": [round(float(lo), 5), round(float(hi), 5)],
            "p_delta_mayor_que_cero": round(float((d > 0).mean()), 4),
            "significativo": bool(lo > 0 or hi < 0),
            "n_replicas": int(len(d))}


def importancia_por_bloque(booster, cols: list[str]) -> dict:
    """
    Cuánta GANANCIA saca la cabeza de cada bloque de columnas.

    Responde algo que ninguna métrica de rendimiento contesta: si el embedding
    aporta +0.000, ¿es que el grafo no sirve, o que XGBoost ni lo miró? Con la
    ganancia por bloque se distingue: importancia ~0 significa que los árboles
    no encontraron un corte útil en esas 32 columnas.

    La ganancia se reparte por columna; se agrega por bloque y se normaliza,
    así que las cifras son porcentajes del total y suman 100.
    """
    g = booster.get_score(importance_type="gain")   # solo columnas USADAS
    # XGBoost nombra f0, f1... por POSICIÓN cuando entrena desde numpy.
    por_col = {}
    for k, v in g.items():
        i = int(k[1:]) if k.startswith("f") and k[1:].isdigit() else None
        if i is not None and i < len(cols):
            por_col[cols[i]] = float(v)
    total = sum(por_col.values()) or 1.0
    bloques = {"tabular": 0.0, "embedding": 0.0}
    for c, v in por_col.items():
        bloques["embedding" if c.startswith(("emb_", "embv_")) else "tabular"] += v
    n_emb = sum(1 for c in cols if c.startswith(("emb_", "embv_")))
    return {
        "ganancia_pct": {k: round(100 * v / total, 2) for k, v in bloques.items()},
        "columnas_usadas": len(por_col),
        "columnas_totales": len(cols),
        "embedding_usadas": sum(1 for c in por_col
                                if c.startswith(("emb_", "embv_"))),
        "embedding_totales": n_emb,
        "top10": sorted(por_col.items(), key=lambda kv: -kv[1])[:10],
    }


def main():
    cfg = load_config()
    ensure_dirs(cfg)
    reports_dir = resolve(cfg, "reports_dir")
    hcfg = cfg.get("hybrid") or {}
    pct = float(hcfg.get("alert_budget_pct", 2.0))

    # MODO. Mientras se decide qué enfoque gana, las etapas `refit`/`oof_refit`/
    # `heads_refit` no se corren: cuestan ~38 min y solo sirven para medir el
    # mes 6, que en esta fase debe seguir sellado. Si no hay cabezas de
    # producción se cae al modo exploración, que mide el MES 5 y no mira el 6.
    modelos_dir = resolve(cfg, "models_dir")
    variantes = [str(x) for x in hcfg.get("variantes", ())]
    sufijo, corte = "", "examen"     # el veredicto sale del bloque SELLADO
    log.info("=" * 62)
    log.info("Veredicto sobre la ventana `examen`, que no entrenó ni validó nada.")
    log.info("  `cabezas_validan` se informa aparte: eligió nº de árboles y")
    log.info("  umbral, así que sus cifras son de SELECCIÓN, no finales.")

    df, cols_base = cargar_tabla(cfg, "train")
    cols_emb = cols_embedding(df, "completo")
    cols_embv = cols_embedding(df, "vecinos")
    # VENTANAS: el veredicto sale de `examen`, un bloque que no entrenó ni
    # validó nada. `cabezas_validan` se informa aparte porque eligió el nº de
    # árboles y el umbral: sus cifras son de SELECCIÓN, no finales.
    _v = verificar(cfg, df["month"].values, df["week_in_month"].values)
    y_all = df["isFraud"].values.astype(int)

    # --- puntuar TODO el dataset con cada cabeza de producción -------------
    scores, imp = {}, {}
    for v in variantes:
        ruta = modelos_dir / nombre_modelo(v, sufijo)
        if not ruta.exists():
            log.warning("Falta %s — se omite '%s'", ruta.name, v)
            continue
        booster = cargar(cfg, ruta.name)
        cols_v = columnas(v, cols_base, cols_emb, cols_embv)
        # EL ANCHO LO DICTA EL BOOSTER, nunca una constante (regla 1). Sin este
        # assert el fallo llega como "Feature shape mismatch, expected: 63, got
        # 431" desde las tripas de XGBoost, sin decir qué cabeza ni por qué.
        esperado = booster.num_features()
        assert len(cols_v) == esperado, (
            f"La cabeza '{v}' se entrenó con {esperado} columnas y aquí se le "
            f"arman {len(cols_v)}. Suele significar que la ablación "
            f"(xgboost.excluir_prefijos) cambió DESPUÉS de entrenarla: vuelve a "
            f"correr `heads`.")
        X = df[cols_v].values.astype(np.float32)
        scores[v] = np.asarray(booster.inplace_predict(X), dtype=np.float64)
        imp[v] = importancia_por_bloque(booster, cols_v)
        log.info("'%s' cargada (%d columnas) | ganancia: tabular %.1f%% / "
                 "embedding %.1f%% | usa %d de %d columnas del embedding",
                 v, X.shape[1], imp[v]["ganancia_pct"]["tabular"],
                 imp[v]["ganancia_pct"]["embedding"],
                 imp[v]["embedding_usadas"], imp[v]["embedding_totales"])
        del X
    if not scores:
        raise SystemExit(
            "No hay ninguna cabeza entrenada. Corre `heads` (exploración) o "
            "`heads_refit` (producción).")

    # La GNN SOLA, como referencia. Es el `gnn_score` que ya guarda el OOF:
    # sigmoid(clasificador(embedding)), o sea la red decidiendo por su cuenta
    # sin XGBoost de por medio. No cuesta nada —ya está calculado— y responde
    # una pregunta propia: ¿XGBoost extrae del embedding MÁS de lo que la propia
    # red sacaba de él? Si `solo_gnn` supera a `gnn_sola`, la respuesta es sí.
    if "gnn_score" in df.columns and df["gnn_score"].notna().any():
        scores["gnn_sola"] = df["gnn_score"].fillna(0.0).values.astype(np.float64)
        log.info("'gnn_sola' añadida como referencia (el gnn_score del OOF)")

    meses_txt = "la ventana cabezas_entrenan"
    resultado = {"modo": "ventanas",
                 "corte_del_veredicto": corte,
                 "nota": (f"Las tres cabezas comparten ventana ({meses_txt}), "
                          "hiperparámetros y SMOTE: la diferencia es atribuible "
                          "solo a las columnas. 'gnn_sola' es la red decidiendo "
                          "por su cuenta, sin XGBoost: va como referencia, no "
                          "como competidora."),
                 "cabezas_validan": {}, "examen": {}}
    resultado["aviso"] = (
        "El bloque `examen` no entrenó ni validó nada: su cifra es limpia. "
        "El bloque `cabezas_validan` sí eligió el nº de árboles y el umbral, "
        "así que sus cifras son OPTIMISTAS en términos absolutos — se informan "
        "para diagnóstico, no como resultado. La COMPARACIÓN entre cabezas es "
        "válida en ambos: las tres reciben las mismas filas, las mismas "
        "ventanas y el mismo presupuesto de búsqueda.")

    # Las métricas POR MES se eliminaron: con las `ventanas` el experimento vive
    # en dos meses, y una tabla de dos filas donde los bloques de entrenamiento
    # salen inflados por construcción no informa de nada. Lo que importa son las
    # dos ventanas que no entrenaron: `cabezas_validan` (diagnóstico) y `examen`
    # (el veredicto).

    cortes = (("cabezas_validan", "cabezas_validan"), ("examen", "examen"))
    for etiqueta, clave in cortes:
        sel = _v[clave]
        y = y_all[sel]
        if not sel.any():
            continue
        bloque = {"n": int(sel.sum()), "n_fraud": int(y.sum()),
                  "modelos": {}, "presupuesto": [], "precision": []}
        log.info("=" * 62)
        log.info("%s — %d transacciones, %d fraudes (%.2f%%)",
                 etiqueta.upper(), sel.sum(), y.sum(), 100 * y.mean())

        for v, s in scores.items():
            thr = umbral_por_presupuesto(s[sel], pct)
            rep = _confusion(y, s[sel], thr)
            rep["umbral_usado"] = thr
            bloque["modelos"][v] = rep
            log.info("  %-16s PR %.4f | ROC %.4f | recall %.4f | prec %.4f | "
                     "F1 %.4f | acc %.4f", v, rep.get("pr_auc", 0),
                     rep.get("auc_roc", 0), rep["recall"], rep["precision"],
                     rep["f1"], rep["accuracy"])
        log.info("  %-16s (no alertar nunca da accuracy %.4f — por eso el "
                 "accuracy no compara)", "",
                 next(iter(bloque["modelos"].values()))["accuracy_sin_alertar"])

        log.info("  -- a IGUAL presupuesto de alertas --")
        for p in PRESUPUESTOS_PCT:
            k = int(round(sel.sum() * p / 100))
            fila = {"pct": p, "n_alertas": k}
            for v, s in scores.items():
                fila[v] = round(recall_at_budget(y, s[sel], k), 4)
            bloque["presupuesto"].append(fila)
            log.info("     %6d alertas (%5.1f%%) | %s", k, p,
                     " | ".join(f"{v} {fila[v]:.4f}" for v in scores))

        log.info("  -- a IGUAL precisión --")
        for objetivo in PRECISIONES:
            fila = {"precision_objetivo": objetivo}
            for v, s in scores.items():
                r, k = recall_at_precision(y, s[sel], objetivo)
                fila[v] = {"recall": r, "n_alertas": k}
            bloque["precision"].append(fila)
            log.info("     precisión %3.0f%% | %s", 100 * objetivo,
                     " | ".join(f"{v} {fila[v]['recall'] or 0:.4f}" for v in scores))

        # semana a semana: el AUC del mes agregado puede inflarse por
        # correlación temporal; dentro de una semana esa correlación no está.
        if "week_in_month" in df.columns:
            semanas = df["week_in_month"].values[sel]
            bloque["semanal"] = []
            for w in sorted(np.unique(semanas)):
                sw = semanas == w
                if y[sw].sum() == 0:
                    continue
                fila = {"semana": int(w), "n": int(sw.sum()),
                        "n_fraude": int(y[sw].sum())}
                for v, s in scores.items():
                    r = full_report(y[sw], s[sel][sw], 0.5)
                    fila[v] = {"auc_roc": r.get("auc_roc"), "pr_auc": r.get("pr_auc")}
                bloque["semanal"].append(fila)

        resultado[etiqueta] = bloque

    # --- atribución --------------------------------------------------------
    if "control" in scores and "gnn_mas_tabular" in scores and resultado[corte]:
        c = resultado[corte]["modelos"]["control"].get("pr_auc")
        g = resultado[corte]["modelos"]["gnn_mas_tabular"].get("pr_auc")
        s_ = resultado[corte]["modelos"].get("solo_gnn", {}).get("pr_auc")
        resultado["atribucion"] = {
            "control_pr_auc": c, "gnn_mas_tabular_pr_auc": g,
            "solo_gnn_pr_auc": s_,
            "aporte_del_grafo": round(g - c, 4) if (c and g) else None,
            "medido_en": corte,
            "importancia": imp,
            "nota": (f"Ambas con {meses_txt} y mismos hiperparámetros: la "
                     "diferencia es el aporte del grafo, sin confundirlo con "
                     "la ventana de entrenamiento."
                     ),
        }
        log.info("=" * 62)
        log.info("APORTE DEL GRAFO (%s): %.4f - %.4f = %+.4f",
                 corte, g, c, g - c)
        if s_ is not None:
            log.info("  solo_gnn: %.4f  (techo del grafo aislado)", s_)

        # ¿Ese delta sobrevive al ruido del mes? Sin esto, un +0.004 y un
        # +0.04 se leen igual en el informe.
        sel_v = _v["examen"]
        bs = bootstrap_delta(y_all[sel_v], scores["control"][sel_v],
                             scores["gnn_mas_tabular"][sel_v])
        resultado["atribucion"]["bootstrap"] = bs
        log.info("  bootstrap emparejado (1000 réplicas sobre %s):", corte)
        log.info("    delta %+.5f | IC95 [%+.5f, %+.5f] | P(delta>0) = %.3f",
                 bs["delta_observado"], bs["ic95"][0], bs["ic95"][1],
                 bs["p_delta_mayor_que_cero"])
        if bs["significativo"]:
            log.info("    -> el intervalo NO cruza el cero: la diferencia se "
                     "sostiene")
        else:
            log.info("    -> el intervalo CRUZA EL CERO: con estos datos no se "
                     "puede afirmar que haya diferencia")

        # Y si el aporte es ~0, ¿es que el grafo no sirve o que XGBoost ni lo
        # miró? La ganancia por bloque separa las dos explicaciones.
        e = imp.get("gnn_mas_tabular", {})
        if e:
            log.info("  la cabeza mixta saca el %.1f%% de su ganancia del "
                     "embedding (%d de %d columnas usadas)",
                     e["ganancia_pct"]["embedding"], e["embedding_usadas"],
                     e["embedding_totales"])

    # Fichero distinto: un informe de exploración NO debe pisar el definitivo.
    salida = reports_dir / "final_comparison.json"
    with open(salida, "w") as f:
        json.dump(resultado, f, indent=2, ensure_ascii=False)
    log.info("-> %s", salida)

    # El resumen se genera AQUÍ y no como etapa aparte: es derivado, cuesta
    # milisegundos, y una etapa con salida propia se saltaría por "ya hecha"
    # dejando un resumen viejo junto a métricas nuevas.
    try:
        from src.comparison.resumen import main as _resumen
        _resumen(escribir_json=True)
    except Exception as e:                       # nunca debe tumbar la corrida
        log.warning("El resumen falló (%s). Las métricas están en %s",
                    e, salida)


if __name__ == "__main__":
    main()
