"""
CL Paso 4 — SET DE CONTROL HISTÓRICO (~5.000 muestras, congeladas desde día 1).

Su único rol: VALIDAR el olvido. El modelo NUNCA entrena con esto.
Es disjunto del replay buffer por construcción (regla de oro).

A diferencia del buffer, el control debe ser REPRESENTATIVO, no difícil:
- Construcción inicial: muestreo aleatorio ESTRATIFICADO (clase x mes) del
  train, excluyendo explícitamente todo lo que cayó al buffer.
- Actualización tras cada ciclo: entran los casos de VERIFICACIÓN (que nunca
  entrenaron, máx ~200 por patrón) -> salen históricos por muestreo aleatorio
  estratificado. Desde el ciclo siguiente el control vigila el olvido de los
  históricos Y del patrón recién aprendido.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.common import get_logger, load_config, resolve

log = get_logger("cl.control")


class ControlSet:
    def __init__(self, cfg: dict | None = None):
        self.cfg = cfg or load_config()
        c = self.cfg["continual_learning"]["control_set"]
        self.size = c["size"]
        self.max_intake = c["max_intake_per_pattern"]
        self.entries: list[dict] = []   # {node_idx, y, month, origin}
        self.path = resolve(self.cfg, "artifacts_dir") / "control_set.json"

    def build_initial(self, node_idx: np.ndarray, y: np.ndarray,
                      months: np.ndarray, buffer_nodes: set[int], seed: int = 42):
        """Estratificado (clase x mes), DISJUNTO del buffer."""
        rng = np.random.default_rng(seed)
        eligible = np.array([i for i in range(len(node_idx))
                             if int(node_idx[i]) not in buffer_nodes])
        cells = [(cls, m) for cls in (0, 1) for m in np.unique(months)]
        # cuota proporcional a la presencia real de cada celda (representativo)
        sel: list[int] = []
        total_eligible = len(eligible)
        for cls, m in cells:
            cell = eligible[(y[eligible] == cls) & (months[eligible] == m)]
            quota = int(round(self.size * len(cell) / total_eligible))
            take = min(quota, len(cell))
            sel.extend(rng.permutation(cell)[:take].tolist())
        # ajuste fino al tamaño exacto
        if len(sel) > self.size:
            sel = list(rng.permutation(sel)[: self.size])
        elif len(sel) < self.size:
            pool = np.setdiff1d(eligible, sel)
            sel.extend(rng.permutation(pool)[: self.size - len(sel)].tolist())

        self.entries = [{"node_idx": int(node_idx[i]), "y": int(y[i]),
                         "month": int(months[i]), "origin": "historical"}
                        for i in sel]
        overlap = {e["node_idx"] for e in self.entries} & buffer_nodes
        assert not overlap, f"LEAKAGE: {len(overlap)} nodos en buffer Y control"
        log.info("Control inicial: %d entradas (%.2f%% fraude), disjunto del buffer.",
                 len(self.entries),
                 100 * np.mean([e["y"] for e in self.entries]))
        self.save()

    def update_with_verification(self, verif_entries: list[dict],
                                 pattern_id: str, buffer_nodes: set[int],
                                 seed: int = 42):
        """
        Entra la verificación (nunca entrenó) -> salen históricos al azar
        estratificado. REGLA DE ORO: nada que haya entrenado entra acá.
        """
        rng = np.random.default_rng(seed)
        # blindaje contra leakage: rechazar cualquier nodo que esté en el buffer
        clean = [e for e in verif_entries if e["node_idx"] not in buffer_nodes]
        dropped = len(verif_entries) - len(clean)
        if dropped:
            log.warning("Se rechazaron %d casos de verificación presentes en el "
                        "buffer (violarían la invariante).", dropped)
        if len(clean) > self.max_intake:
            keep = rng.permutation(len(clean))[: self.max_intake]
            clean = [clean[i] for i in keep]
        for e in clean:
            e["origin"] = pattern_id

        n_out = max(0, len(self.entries) + len(clean) - self.size)
        if n_out > 0:
            hist_idx = [i for i, e in enumerate(self.entries)
                        if e["origin"] == "historical"]
            # aleatorio estratificado por clase entre los históricos
            by_class = {0: [i for i in hist_idx if self.entries[i]["y"] == 0],
                        1: [i for i in hist_idx if self.entries[i]["y"] == 1]}
            frac1 = len(by_class[1]) / max(1, len(hist_idx))
            n1 = int(round(n_out * frac1))
            out = (list(rng.permutation(by_class[1])[:n1]) +
                   list(rng.permutation(by_class[0])[: n_out - n1]))
            out_set = set(out[:n_out])
            self.entries = [e for i, e in enumerate(self.entries)
                            if i not in out_set]
        self.entries.extend(clean)
        log.info("Control actualizado (+%d %s, -%d): %d entradas.",
                 len(clean), pattern_id, n_out, len(self.entries))
        self.save()

    def node_indices(self) -> np.ndarray:
        return np.array([e["node_idx"] for e in self.entries], dtype=np.int64)

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(self.entries, f)

    def load(self) -> "ControlSet":
        with open(self.path) as f:
            self.entries = json.load(f)
        return self
