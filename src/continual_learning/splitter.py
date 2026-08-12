"""
CL Paso 2 — SPLIT del patrón nuevo: 70% ADAPTACIÓN / 30% VERIFICACIÓN.

Regla de oro (invariante del sistema):
  - ADAPTACIÓN entrena -> después del ciclo va SOLO al replay buffer.
  - VERIFICACIÓN nunca entrena -> después del ciclo va SOLO al set de control.
  - Buffer y control JAMÁS se cruzan (si un caso entrenado cayera al control,
    el recall de "olvido" se mediría con datos memorizados: recall inflado,
    olvido invisible = data leakage).

El split es aleatorio (semilla fija por ciclo para reproducibilidad).
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.common import get_logger, load_config

log = get_logger("cl.splitter")


def split_new_pattern(tids: list[int], cfg: dict | None = None,
                      seed: int = 42) -> tuple[list[int], list[int]]:
    """Devuelve (tids_adaptacion, tids_verificacion)."""
    cfg = cfg or load_config()
    frac = cfg["continual_learning"]["split"]["adaptation"]
    rng = np.random.default_rng(seed)
    tids = np.array(tids)
    perm = rng.permutation(len(tids))
    n_adapt = max(1, int(round(len(tids) * frac)))
    adapt = tids[perm[:n_adapt]].tolist()
    verif = tids[perm[n_adapt:]].tolist()
    log.info("Split patrón nuevo: %d adaptación (%.0f%%) / %d verificación",
             len(adapt), 100 * frac, len(verif))
    assert not set(adapt) & set(verif), "Adaptación y verificación se cruzan"
    return adapt, verif
