"""
REENTRENAMIENTO PROFUNDO — cuando el fine-tuning no basta.

Se programa cuando la doble validación falla en AMBOS frentes (el patrón
nuevo contradice los viejos): el orquestador deja el pendiente en
artifacts/pending_deep_retrain.json y este módulo lo ejecuta.

Diferencias con el fine-tuning:
- Entrena DESDE CERO (misma arquitectura seleccionada) sobre el train
  original COMPLETO + los casos de adaptación del patrón conflictivo,
  con pos_weight recalculado sobre esa unión. Así la red reconcilia los
  patrones contradictorios en vez de parchar la frontera de decisión.
- Es lento (horas, no minutos) — por eso es la última carta del dial y se
  ejecuta como proceso aparte, no dentro del ciclo semanal.

Validación antes de desplegar (misma vara que el ciclo normal):
- ¿Aprendió?  recall sobre la VERIFICACIÓN del patrón >= KPI (70%)
- ¿No olvidó? recall sobre el SET DE CONTROL sin caída vs el modelo vigente
Si pasa: production_model.pt + actualización de buffer/control (regla de
oro) + se borra el pendiente. Si no pasa: se documenta y el modelo vigente
sigue en producción.

Uso:
  python -m src.continual_learning.deep_retrain            # lee el pendiente
  python -m src.continual_learning.deep_retrain --epochs 15
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.continual_learning.control_set import ControlSet
from src.continual_learning.replay_buffer import ReplayBuffer
from src.continual_learning.validate import score_nodes
from src.gnn.models import build_model
from src.gnn.sampling import make_neighbor_loader
from src.utils.common import get_logger, load_config, resolve, set_seed
from src.utils.metrics import recall_at_threshold

log = get_logger("cl.deep_retrain")


def _load_production_or_selected(cfg, models_dir):
    """El modelo vigente: producción si existe, si no el seleccionado."""
    prod = models_dir / "production_model.pt"
    if prod.exists():
        ckpt = torch.load(prod, weights_only=False)
    else:
        with open(models_dir / "selected_model.json") as f:
            sel = json.load(f)["selection"]
        ckpt = torch.load(models_dir / sel["checkpoint"], weights_only=False)
    cfg["gnn"]["in_dim"] = ckpt["in_dim"]
    model = build_model(ckpt["model_name"], cfg)
    model.load_state_dict(ckpt["state_dict"])
    return model, ckpt["model_name"]


def deep_retrain(epochs: int | None = None):
    cfg = load_config()
    set_seed(42)
    models_dir = resolve(cfg, "models_dir")
    artifacts_dir = resolve(cfg, "artifacts_dir")
    reports_dir = resolve(cfg, "reports_dir")

    pending_f = artifacts_dir / "pending_deep_retrain.json"
    if not pending_f.exists():
        log.info("No hay reentrenamiento profundo pendiente (%s). Nada que hacer.",
                 pending_f)
        return
    with open(pending_f) as f:
        pending = json.load(f)
    adapt = np.array(pending["adapt_nodes"], dtype=np.int64)
    verif = np.array(pending["verif_nodes"], dtype=np.int64)
    log.info("Pendiente %s: %d adaptación / %d verificación",
             pending["pattern_id"], len(adapt), len(verif))

    data = torch.load(resolve(cfg, "graph_dir") / "graph.pt", weights_only=False)
    old_model, model_name = _load_production_or_selected(cfg, models_dir)

    # --- conjunto de entrenamiento: train original COMPLETO + adaptación ---
    seed_mask = data.train_mask.clone()
    seed_mask[torch.tensor(adapt, dtype=torch.long)] = True
    y = data.y.numpy().astype(int)
    seeds_idx = torch.where(seed_mask)[0].numpy()
    n_pos = int(y[seeds_idx].sum())
    pos_weight = (len(seeds_idx) - n_pos) / max(1, n_pos)  # regla de siempre
    log.info("Entrenando desde cero: %d nodos (pos_weight=%.1f)",
             len(seeds_idx), pos_weight)

    t0 = time.time()
    model = build_model(model_name, cfg)   # pesos NUEVOS: desde cero
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))
    opt = torch.optim.Adam(model.parameters(), lr=cfg["gnn"]["lr"])
    loader = make_neighbor_loader(data, cfg["gnn"]["fanouts"], seed_mask,
                                  cfg["gnn"]["batch_size"], shuffle=True)
    n_epochs = epochs or cfg["gnn"]["epochs"]
    model.train()
    for ep in range(1, n_epochs + 1):
        total, seen = 0.0, 0
        for batch in loader:
            opt.zero_grad()
            out = model(batch.x, batch.edge_index)[: batch.batch_size]
            yb = batch.y[: batch.batch_size].float()
            loss = criterion(out.squeeze(-1), yb)
            loss.backward()
            opt.step()
            total += loss.item() * batch.batch_size
            seen += batch.batch_size
        log.info("Época %02d | loss %.4f", ep, total / max(1, seen))

    hours = (time.time() - t0) / 3600
    # --- validación con la misma vara del ciclo normal ---
    control = ControlSet(cfg).load()   # estado persistido en artifacts/
    ctrl_nodes = control.node_indices()
    kpi = cfg["continual_learning"]["validation"]["recall_verification_min"]
    max_drop = cfg["continual_learning"]["validation"]["control_max_drop"]
    thr = cfg["gnn"]["threshold"]

    r_verif = recall_at_threshold(
        y[verif], score_nodes(model, data, torch.tensor(verif), cfg), thr)
    r_ctrl_new = recall_at_threshold(
        y[ctrl_nodes], score_nodes(model, data, torch.tensor(ctrl_nodes), cfg), thr)
    r_ctrl_old = recall_at_threshold(
        y[ctrl_nodes], score_nodes(old_model, data, torch.tensor(ctrl_nodes), cfg), thr)

    learned = r_verif >= kpi
    kept = (r_ctrl_old - r_ctrl_new) <= max_drop
    deploy = bool(learned and kept)
    report = {
        "pattern_id": pending["pattern_id"], "mode": "deep_retrain",
        "recall_verification": round(float(r_verif), 4), "kpi": kpi,
        "recall_control_new": round(float(r_ctrl_new), 4),
        "recall_control_old": round(float(r_ctrl_old), 4),
        "hours": round(hours, 2), "deploy": deploy,
    }
    log.info("Deep retrain -> aprendió: %s (%.2f) | mantiene control: %s "
             "(%.2f vs %.2f) | %.1f h | despliega: %s",
             learned, r_verif, kept, r_ctrl_new, r_ctrl_old, hours, deploy)

    if deploy:
        torch.save({"model_name": model_name, "in_dim": cfg["gnn"]["in_dim"],
                    "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                    "deployed_after": f"deep_retrain_{pending['pattern_id']}"},
                   models_dir / "production_model.pt")
        # regla de oro: adaptación -> buffer / verificación -> control
        buffer = ReplayBuffer(cfg).load()  # estado persistido en artifacts/
        s_adapt = score_nodes(model, data, torch.tensor(adapt), cfg)
        buffer.update_with_adaptation(
            [{"node_idx": int(n), "y": int(y[n]), "score": float(s)}
             for n, s in zip(adapt, s_adapt)], pending["pattern_id"])
        m_all = data.month.numpy()
        control.update_with_verification(
            [{"node_idx": int(n), "y": int(y[n]), "month": int(m_all[n])}
             for n in verif], pending["pattern_id"],
            buffer_nodes=set(int(i) for i in buffer.node_indices()))
        pending_f.unlink()
        log.info("DESPLIEGUE profundo OK: production_model.pt actualizado, "
                 "conjuntos al día, pendiente cerrado.")
    else:
        log.warning("El reentrenamiento profundo NO pasó la validación: el "
                    "modelo vigente sigue en producción. Documentado en el "
                    "reporte; siguiente paso sugerido: aristas nuevas.")

    with open(reports_dir / f"deep_retrain_{pending['pattern_id']}.json", "w") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=None)
    args = p.parse_args()
    deep_retrain(args.epochs)
