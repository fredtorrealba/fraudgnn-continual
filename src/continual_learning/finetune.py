"""
CL Paso 5 — FINE-TUNING (el reentrenamiento con el grafo local).

Diseño:
- Cada batch mezcla 40% casos NUEVOS (adaptación) + 60% REPLAY BUFFER.
  Más nuevos = aprende rápido pero olvida; más viejos = no olvida pero
  aprende lento. La mezcla es el dial principal del trade-off.
- pos_weight RECALCULADO al batch mezclado (~1.2, no 27.6): el buffer viene
  50/50 así que el conjunto ya está casi balanceado por composición.
  SIN SMOTE (balanceo por selección de casos reales).
- LR DIFERENCIADO por grupo de capas: el concept drift mueve la FRONTERA de
  decisión (capas finales), no la comprensión estructural del grafo (capas
  tempranas). Congelar además acelera el ciclo -> KPI <48h.
    capa 1 -> LR = 0 (congelada)
    capa 2 -> 1e-5
    capa 3 -> 1e-4
    clasificador -> 1e-3
- Pocas épocas (5-10) sobre pocos cientos de muestras -> MINUTOS, no horas.
- El entrenamiento usa el GRAFO LOCAL: cada nodo semilla entrena sobre su
  subgrafo sampleado (NeighborLoader 15-10-5), nunca el grafo completo.
"""
import copy
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.gnn.sampling import make_neighbor_loader
from src.utils.common import get_device, get_logger, load_config

log = get_logger("cl.finetune")


def finetune(model, data, adapt_nodes: np.ndarray, buffer_nodes: np.ndarray,
             cfg: dict | None = None, overrides: dict | None = None):
    """
    Devuelve (modelo_ajustado, info). No muta el modelo de entrada: trabaja
    sobre una copia — el original solo se reemplaza si pasa la validación.

    overrides: ajustes del dial estabilidad-plasticidad, p.ej.
      {"mix_new": 0.30, "lr_scale": 0.5, "freeze_layer3": True, "epochs": 5}
    """
    cfg = cfg or load_config()
    ft = dict(cfg["continual_learning"]["finetune"])
    overrides = overrides or {}

    mix_new = overrides.get("mix_new", ft["mix_new"])
    epochs = overrides.get("epochs", ft["epochs"])
    lr_scale = overrides.get("lr_scale", 1.0)

    lrs = {"layer1": ft["lr_layer1"],
           "layer2": ft["lr_layer2"] * lr_scale,
           "layer3": ft["lr_layer3"] * lr_scale,
           "classifier": ft["lr_classifier"] * lr_scale}
    if overrides.get("freeze_layer3"):
        lrs["layer3"] = 0.0                       # más estabilidad
    if overrides.get("unfreeze_layer2"):
        lrs["layer2"] = ft["lr_layer3"] * lr_scale  # más plasticidad

    device = get_device()
    model = copy.deepcopy(model).to(device)
    model.train()
    # OJO: los BatchNorm quedan en eval(). Con lotes chicos y sesgados a
    # fraude (mezcla 40/60), actualizar sus estadísticas móviles corre la
    # normalización de TODA la red y colapsa el modelo (olvido catastrófico
    # artificial). Congelarlos es práctica estándar en fine-tuning.
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm1d):
            m.eval()

    # --- composición del set de fine-tuning: 40% nuevos / 60% buffer ---
    n_new = len(adapt_nodes)
    n_buf = int(round(n_new * (1 - mix_new) / max(mix_new, 1e-6)))
    rng = np.random.default_rng(42)
    buf_sample = rng.choice(buffer_nodes, size=min(n_buf, len(buffer_nodes)),
                            replace=False)
    seed_nodes = np.concatenate([adapt_nodes, buf_sample])

    # pos_weight recalculado AL CONJUNTO QUE ENTRENA AHORA (~1.2)
    y_mix = data.y[torch.tensor(seed_nodes, dtype=torch.long)].numpy()
    pos_weight = float((y_mix == 0).sum() / max(1, (y_mix == 1).sum()))
    log.info("Fine-tuning: %d nuevos + %d buffer | mezcla %.0f/%.0f | "
             "pos_weight=%.2f | LRs=%s | %d épocas",
             n_new, len(buf_sample), 100 * mix_new, 100 * (1 - mix_new),
             pos_weight, lrs, epochs)

    seed_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    seed_mask[torch.tensor(seed_nodes, dtype=torch.long)] = True
    loader = make_neighbor_loader(data, num_neighbors=cfg["gnn"]["fanouts"],
                                  input_nodes=seed_mask,
                                  batch_size=ft["batch_size"], shuffle=True)
    # Sin num_workers: el fine-tuning trabaja con decenas de muestras (mezcla
    # 40/60 de casos nuevos + buffer). Un pool de procesos para 1-2 batches es
    # puro overhead, y su destrucción repetida ensucia el log.

    optimizer = torch.optim.Adam(model.param_groups(lrs))
    criterion = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(pos_weight, device=device))

    t0 = time.time()
    for epoch in range(1, epochs + 1):
        total, n = 0.0, 0
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            logits = model(batch.x, batch.edge_index)[: batch.batch_size]
            loss = criterion(logits, batch.y[: batch.batch_size])
            loss.backward()
            optimizer.step()
            total += loss.item() * batch.batch_size
            n += batch.batch_size
        log.info("  época %d/%d | loss %.4f", epoch, epochs, total / max(n, 1))

    minutes = (time.time() - t0) / 60
    log.info("Fine-tuning completo en %.1f min.", minutes)
    return model.cpu(), {"pos_weight": pos_weight, "mix_new": mix_new,
                         "lrs": lrs, "epochs": epochs,
                         "minutes": round(minutes, 2)}
