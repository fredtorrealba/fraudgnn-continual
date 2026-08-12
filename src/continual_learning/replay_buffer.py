"""
CL Paso 3 — REPLAY BUFFER (~10.000 muestras).

Resuelve el catastrophic forgetting (Kirkpatrick 2017): fine-tuning solo con
casos nuevos sobreescribe los pesos viejos. El buffer es memoria de
ENTRENAMIENTO — NO valida nada (para eso está el set de control, disjunto).

Construcción inicial (comprime ~470K casos de train en 10K):
  - Por CLASE: 50/50 fraude-legítima, balanceado por SELECCIÓN de casos
    reales (no SMOTE — sintéticos no tienen aristas).
  - Por TIEMPO: cuota de muestras de cada mes de entrenamiento.
  - Por DIFICULTAD: prioriza casos FRONTERA (score 0.4-0.7 del modelo
    recién entrenado) — los más fáciles de olvidar. Los de score 0.98
    están "grabados profundo".

Actualización tras cada ciclo (tamaño fijo 10.000):
  - ENTRAN los casos de adaptación (máx ~800 si el patrón es grande).
  - SALEN igual cantidad: primero REDUNDANTES (duplicados de zona densa),
    luego FÁCILES (scores extremos). NUNCA salen los frontera.
  - Piso de base histórica: 60% del buffer siempre es histórico, para que
    los patrones aprendidos no desplacen la base original.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.common import get_logger, load_config, resolve

log = get_logger("cl.buffer")


class ReplayBuffer:
    """
    Guarda índices de NODO del grafo (no copias de features): el fine-tuning
    samplea los subgrafos frescos desde el grafo con NeighborLoader.
    Cada entrada: (node_idx, y, score, origin) donde origin es 'historical'
    o el id del patrón (p.ej. 'pattern_3').
    """

    def __init__(self, cfg: dict | None = None):
        self.cfg = cfg or load_config()
        b = self.cfg["continual_learning"]["replay_buffer"]
        self.size = b["size"]
        self.frontier_lo, self.frontier_hi = b["frontier_range"]
        self.historical_floor = b["historical_floor"]
        self.max_intake = b["max_intake_per_pattern"]
        self.class_balance = b["class_balance"]
        self.entries: list[dict] = []
        self.path = resolve(self.cfg, "artifacts_dir") / "replay_buffer.json"

    # ---------- construcción inicial ----------
    def build_initial(self, node_idx: np.ndarray, y: np.ndarray,
                      scores: np.ndarray, months: np.ndarray, seed: int = 42):
        """
        Estratifica train (meses 1-4) en 10K:
        cuota por (clase x mes), dentro de cada celda prioriza frontera.
        """
        rng = np.random.default_rng(seed)
        per_class = int(self.size * self.class_balance)      # 5.000 y 5.000
        sel: list[int] = []
        for cls in (1, 0):  # primero fraudes (escasos)
            cls_mask = y == cls
            # Resguardo: nunca tomar más del 70% de los casos disponibles de
            # una clase — el set de control necesita casos de ambas clases y
            # es DISJUNTO del buffer. Con el dataset real esto no restringe
            # (hay ~16K fraudes en train); protege corridas con pocos datos.
            cls_cap = int(0.7 * cls_mask.sum())
            per_class_eff = min(per_class, cls_cap)
            m_unique = np.unique(months[cls_mask])
            quota = per_class_eff // len(m_unique)
            for m in m_unique:
                cell = np.where(cls_mask & (months == m))[0]
                if len(cell) == 0:
                    continue
                take = min(quota, len(cell))
                sel.extend(self._pick_frontier_first(cell, scores, take, rng))
            # relleno si algún mes no alcanzó la cuota
            missing = per_class_eff - sum(1 for i in sel if y[i] == cls)
            if missing > 0:
                pool = np.setdiff1d(np.where(cls_mask)[0], sel)
                extra = min(missing, len(pool))
                sel.extend(self._pick_frontier_first(pool, scores, extra, rng))

        self.entries = [{
            "node_idx": int(node_idx[i]), "y": int(y[i]),
            "score": float(scores[i]), "origin": "historical",
        } for i in sel]
        n_frontier = sum(1 for e in self.entries if self._is_frontier(e["score"]))
        log.info("Buffer inicial: %d entradas (%d fraudes, %d frontera).",
                 len(self.entries), sum(e["y"] for e in self.entries), n_frontier)
        self.save()

    def _is_frontier(self, s: float) -> bool:
        return self.frontier_lo <= s <= self.frontier_hi

    def _pick_frontier_first(self, cell_idx, scores, take, rng):
        """Dentro de una celda: primero todos los frontera, luego aleatorio."""
        s = scores[cell_idx]
        frontier = cell_idx[(s >= self.frontier_lo) & (s <= self.frontier_hi)]
        rest = np.setdiff1d(cell_idx, frontier)
        picked = list(rng.permutation(frontier)[:take])
        if len(picked) < take:
            picked += list(rng.permutation(rest)[: take - len(picked)])
        return [int(i) for i in picked]

    # ---------- actualización post-ciclo ----------
    def update_with_adaptation(self, adapt_entries: list[dict],
                               pattern_id: str, seed: int = 42):
        """
        Entra la adaptación (sampleada si es grande) -> salen redundantes y
        fáciles, nunca frontera, respetando el piso histórico del 60%.
        REGLA DE ORO: solo datos que ENTRENARON pueden entrar acá.
        """
        rng = np.random.default_rng(seed)
        if len(adapt_entries) > self.max_intake:
            keep = rng.permutation(len(adapt_entries))[: self.max_intake]
            adapt_entries = [adapt_entries[i] for i in keep]
        for e in adapt_entries:
            e["origin"] = pattern_id

        n_out = max(0, len(self.entries) + len(adapt_entries) - self.size)
        removed = 0
        if n_out > 0:
            before = len(self.entries)
            self._evict(n_out, rng)
            removed = before - len(self.entries)
        # El buffer es de TAMAÑO FIJO. Si frontera + piso histórico impiden
        # expulsar lo suficiente, se recorta la entrada nueva (no se infla
        # el buffer): protege la memoria antigua por diseño.
        room = self.size - len(self.entries)
        if len(adapt_entries) > room:
            log.warning("Piso/frontera limitan la evicción: entran %d de %d "
                        "casos de %s.", room, len(adapt_entries), pattern_id)
            adapt_entries = adapt_entries[:room]
        self.entries.extend(adapt_entries)
        log.info("Buffer actualizado (+%d %s, -%d): %d entradas.",
                 len(adapt_entries), pattern_id, removed, len(self.entries))
        self.save()

    def _evict(self, n_out: int, rng):
        """Orden de salida (según diseño): 1° REDUNDANTES, 2° FÁCILES
        (scores extremos). Nunca frontera. Respeta el piso histórico.

        Redundante = misma clase + mismo origen + score casi idéntico
        (redondeado a 0.02): el modelo los trata igual, así que aportan lo
        mismo al repaso — se conserva un representante por grupo y los
        demás son los primeros candidatos a salir.
        """
        n_hist = sum(1 for e in self.entries if e["origin"] == "historical")
        min_hist = int(self.size * self.historical_floor)

        def evictable(e):
            return not self._is_frontier(e["score"])  # frontera intocable

        # --- 1) marcar redundantes (uno por grupo se salva, al azar) ---
        seen: set[tuple] = set()
        redundant: list[int] = []
        others: list[int] = []
        for i in rng.permutation(len(self.entries)):
            e = self.entries[int(i)]
            if not evictable(e):
                continue
            key = (e["y"], e["origin"], round(e["score"] / 0.02) * 0.02)
            if key in seen:
                redundant.append(int(i))      # copia informacional -> sale 1°
            else:
                seen.add(key)
                others.append(int(i))

        # --- 2) el resto ordenado por "facilidad" (score extremo sale antes) ---
        others.sort(key=lambda i: -abs(self.entries[i]["score"] - 0.55))
        candidates = redundant + others

        # --- 3) tomar n_out respetando el piso de base histórica ---
        allowed_hist = max(0, n_hist - min_hist)
        out_idx: set[int] = set()
        hist_removed = 0
        for i in candidates:
            if len(out_idx) >= n_out:
                break
            if self.entries[i]["origin"] == "historical":
                if hist_removed >= allowed_hist:
                    continue                  # piso histórico: no sale más base
                hist_removed += 1
            out_idx.add(i)
        self.entries = [e for i, e in enumerate(self.entries) if i not in out_idx]

    # ---------- acceso ----------
    def node_indices(self) -> np.ndarray:
        return np.array([e["node_idx"] for e in self.entries], dtype=np.int64)

    def refresh_scores(self, scores_by_node: dict[int, float]):
        """Tras desplegar un modelo nuevo, los scores del buffer se refrescan
        (la 'dificultad' es relativa al modelo vigente)."""
        for e in self.entries:
            if e["node_idx"] in scores_by_node:
                e["score"] = float(scores_by_node[e["node_idx"]])
        self.save()

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(self.entries, f)

    def load(self) -> "ReplayBuffer":
        with open(self.path) as f:
            self.entries = json.load(f)
        return self
