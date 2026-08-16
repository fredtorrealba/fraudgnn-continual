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
from src.gnn.sampling import make_hetero_loader
from src.utils.common import get_device, get_logger, load_config
from src.utils.metrics import full_report, recall_at_threshold

log = get_logger("cl.validate")


@torch.no_grad()
def score_nodes(model, data, node_idx, cfg) -> np.ndarray:
    """P(fraude) de esos nodos. Envoltorio sobre `embed_and_score_nodes`."""
    return embed_and_score_nodes(model, data, node_idx, cfg)[2]


@torch.no_grad()
def embed_and_score_nodes(model, data, node_idx, cfg):
    """
    LOS DOS embeddings y el score, en UNA sola pasada de muestreo.

    Devuelve (completo, vecinos, scores) porque cada cabeza necesita uno
    distinto y calcularlos por separado recorrería el grafo dos veces:

        completo  "yo + mi vecindario"   -> para `solo_gnn`, que no recibe
                                            nada más: sin las features propias
                                            no sabría nada de la transacción
        vecinos   "solo mi vecindario"   -> para `gnn_mas_tabular`, que YA
                                            recibe las features propias en su
                                            bloque tabular: incluirlas otra vez
                                            sería gastar columnas en repetir

    El score ES `sigmoid(classifier(embedding))`, así que calcularlos por
    separado recorrería el grafo dos veces para el mismo resultado.
    Devuelve (embeddings [n, dim], scores [n]).

    El SCORE sale siempre del camino completo: es el del modelo real.

    Se reserva por nodos ÚNICOS pedidos, no por nodos del grafo — con un
    embedding de 256 dimensiones, dimensionarlo al grafo entero serían ~600 MB
    por llamada. Y se pasa por `np.unique` porque `node_idx` puede traer
    repeticiones (el balanceo de semillas repite fraudes): con un mapeo directo
    id->fila, las repeticiones quedarían en ceros SIN avisar.
    """
    device = get_device()
    model = model.to(device).eval()

    node_idx = np.asarray(node_idx, dtype=np.int64)
    uniq, inv = np.unique(node_idx, return_inverse=True)
    mask = torch.zeros(data["transaction"].num_nodes, dtype=torch.bool)
    mask[torch.as_tensor(uniq, dtype=torch.long)] = True

    # Los workers se deciden POR TAMAÑO, no por una constante. Los dos usos de
    # esta función son opuestos:
    #   CL         decenas de llamadas por ciclo sobre cientos de nodos
    #              -> levantar 12 procesos cuesta más que el muestreo entero
    #   oof.py     4 folds de ~102.000 nodos + ~171.000 fuera de ventana
    #              -> con 0 workers, la etapa muestrea en UN solo núcleo
    # Con la constante a 0 el segundo caso desperdiciaba 15 de los 16 vCPU.
    # Los workers los decide make_hetero_loader por tamaño; aquí solo el batch.
    # Inferencia: sin gradientes que retener, cabe un batch mucho mayor que en
    # entrenamiento. Solo afecta a la velocidad, nunca al resultado.
    bs = 2048 if len(uniq) >= 20_000 else 512
    loader = make_hetero_loader(data, cfg, mask, shuffle=False, batch_size=bs)

    pos = np.full(data["transaction"].num_nodes, -1, dtype=np.int64)
    pos[uniq] = np.arange(len(uniq))
    dim = model.dim_embedding
    emb = np.zeros((len(uniq), dim), dtype=np.float32)      # completo
    embv = np.zeros((len(uniq), dim), dtype=np.float32)     # solo vecinos
    sc = np.zeros(len(uniq), dtype=np.float32)

    for batch in loader:
        batch = batch.to(device)
        n = batch["transaction"].batch_size
        e_vec, e_full = model.embed(batch.x_dict, batch.edge_index_dict, batch,
                                    solo_vecinos=True)
        s = torch.sigmoid(model.classifier[3](
            model.classifier[2](e_full[:n]))).squeeze(-1)
        fila = pos[batch["transaction"].n_id[:n].cpu().numpy()]
        emb[fila] = e_full[:n].cpu().numpy()
        embv[fila] = e_vec[:n].cpu().numpy()
        sc[fila] = s.cpu().numpy()

    model.cpu()
    return emb[inv], embv[inv], sc[inv]


def _as_scorer(obj, data, cfg):
    """
    Normaliza a `callable(node_idx) -> np.float32[]`.

    Un `nn.Module` se envuelve en `score_nodes`, así que el modo GNN sola queda
    EXACTAMENTE igual que antes. Un sistema híbrido ya expone esa interfaz vía
    `HybridSystem.scorer()`. Gracias a esto `validate.py` no importa xgboost y
    la doble validación sirve para los dos sistemas sin ramas condicionales.
    """
    if callable(obj) and not hasattr(obj, "state_dict"):
        return obj
    return lambda node_idx: score_nodes(obj, data, node_idx, cfg)


def validate_cycle(new_model, old_model, data,
                   verification_nodes: np.ndarray,
                   control_nodes: np.ndarray,
                   elapsed_hours: float,
                   cfg: dict | None = None,
                   threshold: float | None = None) -> dict:
    """
    Compara sistema NUEVO vs ANTERIOR sobre:
      - datos del patrón NUEVO (verificación, que ninguno entrenó)
      - datos ANTIGUOS (set de control, que ninguno entrenó)
    Devuelve el veredicto + la dirección del dial si falla.

    `new_model`/`old_model` aceptan un modelo de PyTorch o un scorer ya
    preparado (ver `_as_scorer`), lo que permite validar la GNN sola o el
    sistema híbrido completo con el mismo código.

    `threshold` es explícito porque los dos sistemas NO comparten escala: la
    GNN entrena con pos_weight ~27 y sus scores están inflados, mientras la
    cabeza XGBoost devuelve probabilidades calibradas. Usar 0.5 para el híbrido
    dejaría casi cero alertas y ningún ciclo llegaría a desplegar.
    """
    cfg = cfg or load_config()
    v = cfg["continual_learning"]["validation"]
    thr = cfg["gnn"]["threshold"] if threshold is None else float(threshold)
    score_new = _as_scorer(new_model, data, cfg)
    score_old = _as_scorer(old_model, data, cfg)

    y_verif = data.y[torch.tensor(verification_nodes)].numpy()
    y_ctrl = data.y[torch.tensor(control_nodes)].numpy()

    # --- patrón nuevo (todos son fraudes confirmados -> recall directo) ---
    s_new_v = score_new(verification_nodes)
    s_old_v = score_old(verification_nodes)
    recall_new_on_pattern = recall_at_threshold(y_verif, s_new_v, thr)
    recall_old_on_pattern = recall_at_threshold(y_verif, s_old_v, thr)

    # --- datos antiguos (control, distribución real) ---
    s_new_c = score_new(control_nodes)
    s_old_c = score_old(control_nodes)
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
