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
from src.utils.common import (ensure_dirs, get_logger, load_config,
                              load_state, resolve, state_path, update_state)

log = get_logger("compare_gnns")

TIE_MARGIN = 0.005  # si el AUC difiere menos que esto, decide producción
# GAT va PRIMERO a propósito: es la arquitectura cara (atención con 4 cabezas
# sobre ~1.8M aristas por batch, ~12 GB de activaciones). Si va a fallar por
# memoria o a ir demasiado lenta, mejor saberlo en la primera corrida que tras
# hora y media de GraphSAGE. El orden NO afecta los resultados: train() llama a
# set_seed(seed) al inicio de cada corrida, así que cada una es independiente.
MODELS = ("gat", "graphsage")


def plan(cfg) -> list[tuple[str, int]]:
    """Las 6 corridas del protocolo, en orden fijo."""
    return [(m, s) for m in MODELS for s in cfg["gnn"]["seeds"]]


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
    log.info("--- Corridas GNN (%d arquitecturas x %d seeds) ---",
             len(MODELS), len(cfg["gnn"]["seeds"]))
    for model in MODELS:
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

    for i, (model, seed) in enumerate(pending, 1):
        info = None if force else resume_info(model, seed, cfg)
        desde = (f"retoma en la época {info['epoch'] + 1} de {cfg['gnn']['epochs']}"
                 if info else f"desde cero (época 1 de {cfg['gnn']['epochs']})")
        log.info("=== [%d/%d] %s seed=%d — %s ===",
                 i, len(pending), model, seed, desde)
        train(model, seed, cfg, force=force)


def weekly_val_aucs(model_name: str, seed: int, cfg) -> list[float]:
    """Walk-forward dentro del mes de validación: AUC por semana (1..4).
    Scorea los nodos de validación una sola vez y corta por semana."""
    import torch
    from sklearn.metrics import roc_auc_score

    from src.continual_learning.validate import score_nodes
    from src.gnn.models import build_model

    data = torch.load(resolve(cfg, "graph_dir") / "graph.pt", weights_only=False)
    ckpt = torch.load(resolve(cfg, "models_dir") / f"{model_name}_seed{seed}.pt",
                      weights_only=False)
    cfg["gnn"]["in_dim"] = ckpt["in_dim"]
    model = build_model(model_name, cfg)
    model.load_state_dict(ckpt["state_dict"])

    val_nodes = torch.where(data.val_mask)[0].numpy()
    scores = score_nodes(model, data, torch.tensor(val_nodes), cfg)
    y = data.y.numpy()[val_nodes]
    weeks = data.week_in_month.numpy()[val_nodes]

    aucs = []
    for w in sorted(np.unique(weeks)):
        m = weeks == w
        if y[m].sum() == 0 or y[m].sum() == m.sum():
            continue  # semana sin ambas clases: AUC indefinido, se omite
        aucs.append(float(roc_auc_score(y[m], scores[m])))
    return aucs


def collect(cfg) -> dict:
    reports_dir = resolve(cfg, "reports_dir")
    results = {}
    for model in ("graphsage", "gat"):
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
    sage, gat = results.get("graphsage"), results.get("gat")
    if sage and gat:
        diff = sage["auc_mean"] - gat["auc_mean"]
        if abs(diff) < TIE_MARGIN:
            winner, reason = "graphsage", (
                f"Empate técnico en AUC (Δ={diff:+.4f} < {TIE_MARGIN}). Gana "
                "GraphSAGE por costo de inferencia fijo e inductividad (producción).")
        elif diff > 0:
            winner, reason = "graphsage", f"Mayor AUC promedio (Δ={diff:+.4f})."
        else:
            winner, reason = "gat", f"Mayor AUC promedio (Δ={-diff:+.4f})."
    else:
        winner = "graphsage" if sage else "gat"
        reason = "Única arquitectura con resultados disponibles."

    best_seed_runs = results[winner]["runs"]
    # la mejor seed también se elige por su promedio walk-forward.
    # OJO: la seed se lee del propio reporte — si faltara alguna corrida,
    # indexar cfg["gnn"]["seeds"] apuntaría a la seed equivocada.
    best_idx = int(np.argmax([np.mean(r["weekly_auc"]) if r.get("weekly_auc")
                              else r["auc_roc"] for r in best_seed_runs]))
    best_seed = best_seed_runs[best_idx]["seed"]

    kpi_ok = results[winner]["auc_mean"] > cfg["gnn"]["kpi_auc"]
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
        "kpi_auc_target": cfg["gnn"]["kpi_auc"],
        "kpi_auc_ok": bool(kpi_ok),
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
        log.warning("OJO: AUC %.4f no supera el KPI %.2f — revisar features/grafo.",
                    selection["auc_mean"], selection["kpi_auc_target"])

    out = resolve(cfg, "models_dir") / "selected_model.json"
    with open(out, "w") as f:
        json.dump({"selection": selection, "results": results}, f, indent=2)
    log.info("Selección registrada en %s", out)


if __name__ == "__main__":
    main()
