"""
Paso 5 — COMPARACIÓN GraphSAGE vs GAT (OE2).

Protocolo (walk-forward × 3):
- Cada arquitectura se entrena 3 veces (seeds 42/123/2026) con el mismo
  split temporal (entrena meses 1-4, valida mes 5).
- Además del AUC sobre el mes 5 completo, cada modelo se evalúa SEMANA A
  SEMANA dentro del mes de validación (walk-forward: semanas 1→4, el
  "futuro que va llegando"). La selección usa el AUC promedio de las
  semanas × seeds — así la comparación premia consistencia temporal y no
  solo el promedio del mes.
- También se reportan recall/PR-AUC como métricas de apoyo.
- KPI del objetivo: AUC-ROC > 0.93.
- Se SELECCIONA la mejor y se registra en models/selected_model.json — a
  partir de ahí, esa es la red que entra en operación y en el ciclo de
  continual learning.

Criterio de desempate: si la diferencia de AUC promedio es < 0.005, gana
GraphSAGE por costo fijo de inferencia + carácter inductivo (producción).

REANUDABLE: el avance vive en artifacts/pipeline_state.json (se crea solo en la
primera corrida, con las 6 marcadas "pending"). Si el proceso muere, vuelve a
lanzar EL MISMO comando: salta las corridas terminadas y retoma la que quedó a
medias desde su última época. No hay que pasar ningún flag.

Uso:
  python -m src.gnn.compare_gnns              # entrena lo que falte + compara
  python -m src.gnn.compare_gnns --skip-train # solo compara reportes ya generados
  python -m src.gnn.compare_gnns --force      # reentrena las 6 desde cero
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.common import (ensure_dirs, get_device, get_logger, load_config,
                              load_state, resolve, state_path, update_state)

log = get_logger("compare_gnns")

TIE_MARGIN = 0.005  # si el AUC difiere menos que esto, decide producción
# GAT va PRIMERO a propósito: es la arquitectura cara (atención con 4 cabezas
# sobre ~1.8M aristas por batch, ~12 GB de activaciones). Si va a fallar por
# memoria o a ir demasiado lenta, mejor saberlo en la primera corrida que tras
# hora y media de GraphSAGE. El orden NO afecta los resultados: train() llama a
# set_seed(seed) al inicio de cada corrida, así que cada una es independiente.
# Las arquitecturas salen del config (gnn.arquitecturas), no de una
# constante: cambiar de GAT a GATv2 no debería exigir tocar código.
MODELS = ("graphsage", "gatv2")   # valor por defecto si falta en config


def plan(cfg) -> list[tuple[str, int]]:
    """Las 6 corridas del protocolo, en orden fijo."""
    arqs = cfg["gnn"].get("arquitecturas", MODELS)
    return [(m, s) for m in arqs for s in cfg["gnn"]["seeds"]]


def init_state(cfg, force: bool = False):
    """Crea artifacts/pipeline_state.json si no existe (primera corrida) y deja
    cada corrida marcada según lo que YA hay en disco: done o pending."""
    from src.gnn.train_gnn import is_done, run_key

    fresh = not state_path(cfg).exists()
    if fresh:
        log.info("Sin archivo de estado — se crea %s", state_path(cfg))
    known = load_state(cfg)["runs"]
    for model, seed in plan(cfg):
        key = run_key(model, seed)
        if not force and is_done(model, seed, cfg):
            status = "done"
        elif not force and known.get(key, {}).get("status") == "running":
            status = "running"                     # quedó a medias: se retoma
        else:
            status = "pending"
        update_state(cfg, key, status=status, model=model, seed=seed)


def _seeds(lista) -> str:
    return ", ".join(str(s) for s in lista) if lista else "ninguna"


def show_runs(cfg, force: bool = False):
    """Resumen corto: por arquitectura, qué seeds están listas y cuáles faltan."""
    from src.gnn.train_gnn import is_done, resume_info

    pending = []
    hd = cfg["gnn"]["hidden_dims"]
    log.info("--- Corridas GNN (%d arquitecturas x %d seeds) ---",
             len(cfg["gnn"].get("arquitecturas", MODELS)),
             len(cfg["gnn"]["seeds"]))
    log.info("    arquitectura: %d capa(s) [%s] = %d salto(s) en el grafo",
             len(hd), ", ".join(map(str, hd)), len(hd))
    if cfg["gnn"].get("sin_aristas"):
        log.warning("    ABLACIÓN ACTIVA: sin_aristas=true -> el grafo se anula, "
                    "cada nodo queda aislado y el modelo es una MLP")
    for model in cfg["gnn"].get("arquitecturas", MODELS):
        listas, faltan = [], []
        for seed in cfg["gnn"]["seeds"]:
            (listas if is_done(model, seed, cfg) and not force
             else faltan).append(seed)
        log.info("%-10s listas: %-16s | faltan: %s",
                 model, _seeds(listas), _seeds(faltan))
        pending += [(model, s) for s in faltan]

    # ¿alguna quedó a mitad de camino?
    for model, seed in pending:
        info = None if force else resume_info(model, seed, cfg)
        if info:
            log.info("%s seed=%d quedó a medias en la época %d — se retoma.",
                     model, seed, info["epoch"])
    return pending


def run_all(cfg, force: bool = False):
    init_state(cfg, force)
    from src.gnn.train_gnn import resume_info, train

    pending = show_runs(cfg, force)
    total = len(plan(cfg))
    if not pending:
        log.info("Las %d corridas están listas — a la comparación.", total)
        return
    log.info("Faltan %d de %d corridas.", len(pending), total)

    # La búsqueda va ANTES de las semillas: las 3 corridas de cada arquitectura
    # tienen que compartir hiperparámetros o no serían réplicas de lo mismo.
    if int(cfg["gnn"].get("optuna_trials", 0)) > 0:
        for arq in sorted({m for m, _ in pending}):
            aplicar_hiperparametros(cfg, buscar_hiperparametros(arq, cfg))

    for i, (model, seed) in enumerate(pending, 1):
        info = None if force else resume_info(model, seed, cfg)
        desde = (f"retoma en la época {info['epoch'] + 1} de {cfg['gnn']['epochs']}"
                 if info else f"desde cero (época 1 de {cfg['gnn']['epochs']})")
        log.info("=== [%d/%d] %s seed=%d — %s ===",
                 i, len(pending), model, seed, desde)
        train(model, seed, cfg, force=force)


def buscar_hiperparametros(model_name: str, cfg) -> dict:
    """
    Búsqueda bayesiana para la GNN — lo que hasta ahora solo tenía XGBoost.

    La comparación era injusta en esfuerzo de ajuste: XGBoost recibía 30 trials
    de Optuna y la GNN corría con valores puestos a mano. No es paridad de
    cómputo (un trial de XGBoost cuesta ~13 s y uno de GNN ~5 min, 23x más) pero
    sí de protocolo, que es lo que se defiende: AMBOS modelos recibieron búsqueda
    bayesiana con el mismo número de trials y el mismo sampler.

    Cada trial entrena con meses 1-4 y se puntúa por PR-AUC sobre el mes 5 —
    la misma métrica y el mismo conjunto con que se elige la cabeza XGBoost.
    `MedianPruner` mata pronto los trials que van claramente peor que la mediana,
    que es lo que hace viable el presupuesto.

    El resultado se cachea en reports/optuna_{modelo}.json: la búsqueda es la
    parte cara y no debe repetirse al relanzar el pipeline.
    """
    import optuna
    import torch
    from sklearn.metrics import average_precision_score

    from src.gnn.models import build_model
    from src.gnn.train_gnn import evaluate, make_loader

    cache = resolve(cfg, "reports_dir") / f"optuna_{model_name}.json"
    if cache.exists():
        with open(cache) as f:
            prev = json.load(f)
        log.info("[%s] hiperparámetros ya buscados (PR-AUC %.4f) — se reutilizan",
                 model_name, prev.get("mejor_valor", float("nan")))
        return prev["mejores_params"]

    n_trials = int(cfg["gnn"].get("optuna_trials", 30))
    data = torch.load(resolve(cfg, "graph_dir") / "graph.pt", weights_only=False)
    device = get_device()
    y_val = data["transaction"].y[data["transaction"].val_mask].numpy()

    def objetivo(trial):
        c = json.loads(json.dumps(cfg))          # copia profunda por trial
        g = c["gnn"]
        g["lr"] = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        g["dropout"] = trial.suggest_float("dropout", 0.0, 0.5)
        ancho = trial.suggest_categorical("ancho", [64, 128, 256])
        # MÍNIMO 2 capas: con una sola, los nodos de entidad llegan a la
        # transacción todavía en ceros y el grafo no aporta nada (models.py lo
        # rechaza con un ValueError). El rango 1-2 venía del grafo homogéneo,
        # donde una capa sí tenía sentido; aquí hacía fallar el primer trial.
        g["hidden_dims"] = [ancho] * trial.suggest_int("capas", 2, 3)
        g["mlp_head_dim"] = trial.suggest_categorical("mlp_head_dim", [16, 32, 64])
        g["in_dim"] = data["transaction"].x.shape[1]
        wd = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
        # épocas cortas: la búsqueda compara configuraciones, no exprime cada una
        epocas = max(3, cfg["gnn"]["epochs"] // 4)

        balancear = bool(g.get("balanceo_semillas", False))
        pw = (float(g.get("pos_weight_con_balanceo", 1.0)) if balancear else
              float((y_val == 0).sum() / max(1, (y_val == 1).sum())))
        modelo = build_model(model_name, c, data.metadata()).to(device)
        opt = torch.optim.Adam(modelo.parameters(), lr=g["lr"], weight_decay=wd)
        crit = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pw, device=device))

        tr = make_loader(data, data["transaction"].train_mask, c, True, balancear)
        va = make_loader(data, data["transaction"].val_mask, c, False)
        mejor = 0.0
        for ep in range(1, epocas + 1):
            modelo.train()
            for batch in tr:
                batch = batch.to(device)
                n = batch["transaction"].batch_size
                opt.zero_grad()
                loss = crit(modelo(batch.x_dict, batch.edge_index_dict, batch)[:n],
                            batch["transaction"].y[:n])
                loss.backward()
                opt.step()
            yv, sv = evaluate(modelo, va, device)
            mejor = max(mejor, float(average_precision_score(yv, sv)))
            trial.report(mejor, ep)
            if trial.should_prune():
                raise optuna.TrialPruned()
        del modelo
        return mejor

    log.info("[%s] Optuna: %d trials (PR-AUC sobre el mes 5)", model_name, n_trials)
    estudio = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),   # mismo sampler que XGBoost
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2))
    estudio.optimize(objetivo, n_trials=n_trials, show_progress_bar=True)

    best = dict(estudio.best_params)
    with open(cache, "w") as f:
        json.dump({"modelo": model_name, "n_trials": n_trials,
                   "mejor_valor": estudio.best_value, "mejores_params": best,
                   "podados": sum(1 for t in estudio.trials
                                  if t.state.name == "PRUNED")},
                  f, indent=2, ensure_ascii=False)
    log.info("[%s] mejor PR-AUC %.4f | %s", model_name, estudio.best_value, best)
    return best


def aplicar_hiperparametros(cfg, best: dict) -> dict:
    """Traduce lo que devuelve Optuna a la forma que espera `cfg["gnn"]`."""
    g = cfg["gnn"]
    for k in ("lr", "dropout", "mlp_head_dim"):
        if k in best:
            g[k] = best[k]
    if "ancho" in best:
        # El default es 2, no 1: si `capas` faltara (un estudio antiguo, un
        # enqueue_trial incompleto) se construiría una GNN de una capa, que en
        # el grafo heterogéneo no llega a propagar nada y muere en build_model.
        g["hidden_dims"] = [best["ancho"]] * max(2, int(best.get("capas", 2)))
    return cfg


def weekly_val_aucs(model_name: str, seed: int, cfg) -> list[float]:
    """
    Walk-forward dentro del mes de validación: PR-AUC por semana (1..4).

    PR-AUC y no ROC-AUC, por coherencia con el resto del proyecto y porque con
    3,4% de fraude el ROC comprime las diferencias: la misma comparación que en
    PR-AUC da 0.0348 en ROC da 0.0073. Elegir arquitectura con una métrica que
    apenas distingue es pedir que la elección salga por azar.

    Por SEMANA y no sobre el mes entero porque el mes agregado se infla con la
    correlación temporal: medido en la ablación sin aristas, un modelo daba
    0.8524 sobre el mes completo y 0.6075 de media semanal — estaba separando
    por PERIODO, no por fraude. Dentro de una semana esa correlación no existe.
    """
    import torch
    from sklearn.metrics import average_precision_score

    from src.continual_learning.validate import score_nodes
    from src.gnn.models import build_model

    data = torch.load(resolve(cfg, "graph_dir") / "graph.pt", weights_only=False)
    ckpt = torch.load(resolve(cfg, "models_dir") / f"{model_name}_seed{seed}.pt",
                      weights_only=False)
    cfg["gnn"]["in_dim"] = ckpt["in_dim"]
    model = build_model(model_name, cfg, data.metadata())
    model.load_state_dict(ckpt["state_dict"])

    val_nodes = torch.where(data["transaction"].val_mask)[0].numpy()
    scores = score_nodes(model, data, val_nodes, cfg)
    y = data["transaction"].y.numpy()[val_nodes]
    weeks = data["transaction"].week_in_month.numpy()[val_nodes]

    aucs = []
    for w in sorted(np.unique(weeks)):
        m = weeks == w
        if y[m].sum() == 0 or y[m].sum() == m.sum():
            continue  # semana sin ambas clases: AUC indefinido, se omite
        aucs.append(float(average_precision_score(y[m], scores[m])))
    return aucs


def collect(cfg) -> dict:
    reports_dir = resolve(cfg, "reports_dir")
    results = {}
    for model in cfg["gnn"].get("arquitecturas", ["graphsage", "gatv2"]):
        runs = []
        for seed in cfg["gnn"]["seeds"]:
            f = reports_dir / f"{model}_seed{seed}_val.json"
            if not f.exists():
                log.warning("Falta %s — corre primero el entrenamiento.", f)
                continue
            with open(f) as fh:
                run = json.load(fh)
            # reportes viejos no traían estos campos
            run.setdefault("model", model)
            run.setdefault("seed", seed)
            # walk-forward: AUC por semana del mes de validación
            run["weekly_auc"] = weekly_val_aucs(model, seed, cfg)
            runs.append(run)
        if runs:
            all_weekly = [a for r in runs for a in r["weekly_auc"]]
            results[model] = {
                "runs": runs,
                # la selección usa el promedio walk-forward (semanas x seeds);
                # el AUC del mes completo queda como referencia
                "auc_mean": float(np.mean(all_weekly)),
                "auc_std": float(np.std(all_weekly)),
                "auc_month_mean": float(np.mean([r["auc_roc"] for r in runs])),
                "recall_mean": float(np.mean([r["recall"] for r in runs])),
                "pr_auc_mean": float(np.mean([r.get("pr_auc", 0) for r in runs])),
            }
    return results


def select(results: dict, cfg) -> dict:
    # Los nombres salen del CONFIG, no escritos a mano. Este bloque comparaba
    # results.get("gat") cuando la arquitectura pasó a llamarse "gatv2": la
    # búsqueda siempre devolvía None, se caía al else y GraphSAGE ganaba por
    # incomparecencia, con el mensaje "Única arquitectura disponible" aunque
    # las dos hubieran corrido. La comparación central del capstone estaba
    # desactivada en silencio.
    disponibles = [m for m in cfg["gnn"].get("arquitecturas", MODELS)
                   if results.get(m)]
    if not disponibles:
        raise SystemExit("Ninguna arquitectura tiene resultados: revisa el "
                         "paso `gnn`.")
    if len(disponibles) == 1:
        winner = disponibles[0]
        reason = (f"Única arquitectura con resultados: {winner}. Las demás "
                  f"({', '.join(m for m in cfg['gnn'].get('arquitecturas', MODELS) if m != winner)}) "
                  "no dejaron corridas.")
    else:
        orden = sorted(disponibles, key=lambda m: -results[m]["auc_mean"])
        primero, segundo = orden[0], orden[1]
        diff = results[primero]["auc_mean"] - results[segundo]["auc_mean"]
        if diff < TIE_MARGIN and "graphsage" in disponibles:
            # Empate técnico: decide producción, no el decimal de ruido.
            winner = "graphsage"
            reason = (f"Empate técnico (Δ={diff:.4f} < {TIE_MARGIN} entre "
                      f"{primero} y {segundo}). Gana GraphSAGE por costo de "
                      "inferencia fijo e inductividad (producción).")
        else:
            winner = primero
            reason = (f"Mayor PR-AUC walk-forward: {primero} {results[primero]['auc_mean']:.4f} "
                      f"contra {segundo} {results[segundo]['auc_mean']:.4f} (Δ={diff:+.4f}).")

    best_seed_runs = results[winner]["runs"]
    # la mejor seed también se elige por su promedio walk-forward.
    # OJO: la seed se lee del propio reporte — si faltara alguna corrida,
    # indexar cfg["gnn"]["seeds"] apuntaría a la seed equivocada.
    # PR-AUC semanal medio. El fallback a pr_auc mensual solo actúa si alguna
    # semana quedó sin ambas clases y no hubo curva que calcular.
    best_idx = int(np.argmax([np.mean(r["weekly_auc"]) if r.get("weekly_auc")
                              else r.get("pr_auc", 0.0) for r in best_seed_runs]))
    best_seed = best_seed_runs[best_idx]["seed"]

    # El KPI (0.93) es de AUC-ROC MENSUAL, así que se compara contra
    # `auc_month_mean`. Antes se comparaba contra `auc_mean`, que desde el
    # cambio a PR-AUC walk-forward vale ~0.29: el aviso saltaba SIEMPRE, aunque
    # el ROC mensual fuese 0.9435 y el KPI estuviera cumplido de sobra.
    kpi_ok = results[winner]["auc_month_mean"] > cfg["gnn"]["kpi_auc"]
    # La época del PICO de la corrida ganadora. Se anota aquí —y no solo dentro
    # del checkpoint— para que el paso `refit` pueda releerla sin cargar 2 MB de
    # pesos, y para que quede a la vista qué número se heredó.
    best_epoch = best_seed_runs[best_idx].get("best_epoch")
    return {
        "selected": winner,
        "seed": best_seed,
        "checkpoint": f"{winner}_seed{best_seed}.pt",
        "best_epoch": best_epoch,
        "reason": reason,
        "auc_mean": results[winner]["auc_mean"],
        "auc_std": results[winner]["auc_std"],
        "auc_month_mean": results[winner]["auc_month_mean"],
        "pr_auc_mean": results[winner].get("pr_auc_mean"),
        "kpi_auc_target": cfg["gnn"]["kpi_auc"],
        "kpi_auc_ok": bool(kpi_ok),
        "kpi_medido_sobre": "auc_roc mensual",
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--skip-train", action="store_true",
                   help="No entrenar; solo comparar reportes existentes")
    p.add_argument("--force", action="store_true",
                   help="Reentrenar las 6 corridas desde cero (ignora el estado)")
    args = p.parse_args()

    cfg = load_config()
    ensure_dirs(cfg)
    if not args.skip_train:
        run_all(cfg, force=args.force)

    results = collect(cfg)
    log.info("--- Resumen por arquitectura ---")
    for m, r in results.items():
        log.info("%-10s AUC walk-forward %.4f ± %.4f (mes: %.4f) | recall %.4f | PR-AUC %.4f",
                 m, r["auc_mean"], r["auc_std"], r["auc_month_mean"],
                 r["recall_mean"], r["pr_auc_mean"])

    selection = select(results, cfg)
    log.info("SELECCIONADA: %s (%s)", selection["selected"], selection["reason"])
    if not selection["kpi_auc_ok"]:
        log.warning("OJO: AUC-ROC mensual %.4f no supera el KPI %.2f — "
                    "revisar features/grafo. (El PR-AUC walk-forward, que es "
                    "lo que SELECCIONA, es %.4f: son métricas distintas.)",
                    selection["auc_month_mean"], selection["kpi_auc_target"],
                    selection["auc_mean"])

    out = resolve(cfg, "models_dir") / "selected_model.json"
    with open(out, "w") as f:
        json.dump({"selection": selection, "results": results}, f, indent=2)
    log.info("Selección registrada en %s", out)


if __name__ == "__main__":
    main()
