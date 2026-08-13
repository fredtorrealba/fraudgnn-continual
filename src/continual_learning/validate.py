"""
CL Paso 6 — VALIDACIÓN post fine-tuning (¿está mejor que el modelo anterior?).

Tres mediciones, siempre sobre distribución REAL sin balanceo:
  1. ¿APRENDIÓ?  recall sobre VERIFICACIÓN (30% apartado) >= 70% (KPI)
                 y ADEMÁS mejor que el recall del modelo ANTERIOR sobre
                 el patrón nuevo (si no mejora, el ciclo no aporta).
  2. ¿OLVIDÓ?    recall sobre el SET DE CONTROL histórico: el modelo nuevo
                 no puede caer más que `control_max_drop` respecto del
                 modelo anterior sobre los MISMOS datos viejos.
  3. ¿TIEMPO?    ciclo completo (disparo -> despliegue) < 48h (KPI).

El modelo nuevo SOLO se despliega si pasa 1 y 2 (y se reporta 3).

Manejo de fallas — dial estabilidad-plasticidad (un solo dial, dos direcciones):
  - Falla control (olvidó)      -> ESTABILIDAD: +buffer, -LR, congelar capa 3,
                                   menos épocas.
  - Falla verificación (no aprendió) -> PLASTICIDAD: +nuevos, +LR, descongelar,
                                   más épocas.
  - Fallan ambos -> el patrón contradice los viejos: reentrenamiento profundo
                    programado o documentar el trade-off.
Cada reintento cuesta minutos — por eso las 48h tienen holgura.
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.gnn.sampling import fanouts, make_neighbor_loader
from src.utils.common import get_device, get_logger, load_config
from src.utils.metrics import full_report, recall_at_threshold

log = get_logger("cl.validate")


@torch.no_grad()
def score_nodes(model, data, node_idx: np.ndarray, cfg) -> np.ndarray:
    """Scores del modelo sobre nodos específicos, vía subgrafos sampleados."""
    device = get_device()
    model = model.to(device).eval()
    mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    # as_tensor en vez de tensor(): si node_idx YA es un tensor, tensor() copia
    # y avisa con UserWarning. as_tensor no copia ni avisa.
    mask[torch.as_tensor(node_idx, dtype=torch.long)] = True
    # SIN num_workers a propósito: este loader se crea de nuevo en cada intento
    # de cada semana (4 semanas x 3 intentos x N validaciones) sobre conjuntos
    # de 2 a 5000 nodos. Levantar procesos persistentes para eso es más caro que
    # el muestreo, y al destruirlos tan seguido PyTorch escupe cientos de
    # "Bad file descriptor" y "semaphore released too many times" al cerrar.
    loader = make_neighbor_loader(data, num_neighbors=fanouts(cfg),
                                  input_nodes=mask, batch_size=512, shuffle=False)
    scores = np.zeros(data.num_nodes, dtype=np.float32)
    for batch in loader:
        batch = batch.to(device)
        s = torch.sigmoid(model(batch.x, batch.edge_index)[: batch.batch_size])
        scores[batch.n_id[: batch.batch_size].cpu().numpy()] = s.cpu().numpy()
    model.cpu()
    return scores[node_idx]


def validate_cycle(new_model, old_model, data,
                   verification_nodes: np.ndarray,
                   control_nodes: np.ndarray,
                   elapsed_hours: float,
                   cfg: dict | None = None) -> dict:
    """
    Compara modelo NUEVO vs ANTERIOR sobre:
      - datos del patrón NUEVO (verificación, que ninguno entrenó)
      - datos ANTIGUOS (set de control, que ninguno entrenó)
    Devuelve el veredicto + la dirección del dial si falla.
    """
    cfg = cfg or load_config()
    v = cfg["continual_learning"]["validation"]
    thr = cfg["gnn"]["threshold"]

    y_verif = data.y[torch.tensor(verification_nodes)].numpy()
    y_ctrl = data.y[torch.tensor(control_nodes)].numpy()

    # --- patrón nuevo (todos son fraudes confirmados -> recall directo) ---
    s_new_v = score_nodes(new_model, data, verification_nodes, cfg)
    s_old_v = score_nodes(old_model, data, verification_nodes, cfg)
    recall_new_on_pattern = recall_at_threshold(y_verif, s_new_v, thr)
    recall_old_on_pattern = recall_at_threshold(y_verif, s_old_v, thr)

    # --- datos antiguos (control, distribución real) ---
    s_new_c = score_nodes(new_model, data, control_nodes, cfg)
    s_old_c = score_nodes(old_model, data, control_nodes, cfg)
    recall_new_on_control = recall_at_threshold(y_ctrl, s_new_c, thr)
    recall_old_on_control = recall_at_threshold(y_ctrl, s_old_c, thr)

    learned = (recall_new_on_pattern >= v["recall_verification_min"] and
               recall_new_on_pattern > recall_old_on_pattern)
    not_forgot = recall_new_on_control >= recall_old_on_control - v["control_max_drop"]
    on_time = elapsed_hours < v["max_hours"]

    verdict = {
        "learned": bool(learned),
        "not_forgot": bool(not_forgot),
        "on_time": bool(on_time),
        "deploy": bool(learned and not_forgot),
        "recall_pattern_new_model": recall_new_on_pattern,
        "recall_pattern_old_model": recall_old_on_pattern,
        "recall_control_new_model": recall_new_on_control,
        "recall_control_old_model": recall_old_on_control,
        "control_report_new": full_report(y_ctrl, s_new_c, thr),
        "elapsed_hours": round(elapsed_hours, 2),
        "dial": None,
    }

    if not verdict["deploy"]:
        if not not_forgot and not learned:
            verdict["dial"] = "deep_retrain"   # fallan ambos: patrón contradictorio
        elif not not_forgot:
            verdict["dial"] = "stability"      # olvidó -> proteger lo viejo
        else:
            verdict["dial"] = "plasticity"     # no aprendió -> aprender lo nuevo

    log.info("Validación -> aprendió: %s (%.2f vs %.2f del anterior, KPI>=%.2f) | "
             "olvidó: %s (%.2f vs %.2f) | <48h: %s | despliega: %s | dial: %s",
             learned, recall_new_on_pattern, recall_old_on_pattern,
             v["recall_verification_min"], not not_forgot,
             recall_new_on_control, recall_old_on_control,
             on_time, verdict["deploy"], verdict["dial"])
    return verdict


def dial_overrides(direction: str, cfg: dict | None = None) -> dict | None:
    """Traduce la dirección del dial a overrides del fine-tuning."""
    cfg = cfg or load_config()
    d = cfg["continual_learning"]["dial"]
    if direction == "stability":
        s = d["stability"]
        return {"mix_new": s["mix_new"], "lr_scale": s["lr_scale"],
                "freeze_layer3": s["freeze_layer3"], "epochs": s["epochs"]}
    if direction == "plasticity":
        p = d["plasticity"]
        return {"mix_new": p["mix_new"], "lr_scale": p["lr_scale"],
                "unfreeze_layer2": p["unfreeze_layer2"], "epochs": p["epochs"]}
    return None  # deep_retrain: fuera del alcance del reintento automático
