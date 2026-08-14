"""
El sistema híbrido en operación: GNN + columnas del grafo + cabeza XGBoost.

    transacción ──┬─► 431 features propias        (ya están en data.x)
                  ├─► 8 columnas estructurales    (graph_features.parquet)
                  └─► gnn_score                   (la red puntúa)
                              ↓
                        cabeza XGBoost  ──►  P(fraude)

Expone la MISMA interfaz que `score_nodes`: una función que recibe índices de
nodo y devuelve un vector alineado. Eso permite que el ciclo de continual
learning y la comparación final traten a la GNN sola y al híbrido de forma
intercambiable, sin ramas condicionales repartidas por el código.

Las 431 columnas se leen de `data.x`, que ya está en memoria y en el mismo
orden de fila: no hay que releer el parquet ni hacer joins dentro del bucle.

xgboost se importa PEREZOSAMENTE dentro de los métodos. Este módulo sí convive
con torch, y en macOS cargar ambos runtimes de OpenMP a la vez segfaultea; el
proceso que lo use debe haber llamado antes a `src.utils.omp.guard_omp()`.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.continual_learning.validate import score_nodes
from src.utils.common import get_logger

log = get_logger("hybrid.system")


class HybridSystem:
    """
    GNN + cabeza. Con `head=None` se comporta como la GNN sola, lo que permite
    comparar ambos sistemas con el mismo código.
    """

    def __init__(self, gnn, head, struct: np.ndarray | None, cfg,
                 umbral: float | None = None):
        self.gnn = gnn
        self.head = head
        self.struct = struct
        self.cfg = cfg
        self.umbral = umbral if umbral is not None else cfg["gnn"]["threshold"]

    @property
    def es_hibrido(self) -> bool:
        return self.head is not None

    def score(self, data, node_idx: np.ndarray) -> np.ndarray:
        """P(fraude) para esos nodos, alineado con `node_idx`."""
        g = score_nodes(self.gnn, data, node_idx, self.cfg)
        if not self.es_hibrido:
            return g
        idx = np.asarray(node_idx, dtype=np.int64)
        X = np.hstack([
            data.x[idx].numpy().astype(np.float32),
            self.struct[idx],
            g.reshape(-1, 1).astype(np.float32),
        ])
        return self.head.inplace_predict(X)

    def scorer(self, data):
        """La función que consume `validate_cycle`."""
        return lambda node_idx: self.score(data, node_idx)


def cargar_struct(cfg) -> np.ndarray | None:
    """Las 8 columnas estructurales, o None si el paso `graph` es antiguo."""
    from src.hybrid.features import load_struct, ruta
    if not ruta(cfg).exists():
        log.warning("Falta %s: el sistema opera en modo GNN sola",
                    ruta(cfg).name)
        return None
    struct, _ = load_struct(cfg)
    return struct


def cargar_cabeza(cfg, nombre: str = "hybrid_head_prod.json"):
    """La cabeza de producción, o None si aún no se ha entrenado."""
    from src.hybrid.head import cargar
    from src.utils.common import resolve
    if not (resolve(cfg, "models_dir") / nombre).exists():
        log.warning("Falta %s: el sistema opera en modo GNN sola", nombre)
        return None
    return cargar(cfg, nombre)


def cargar_umbral(cfg) -> float:
    """El umbral por presupuesto que fijó `hybrid_refit`, o el del config."""
    import json
    from src.utils.common import resolve
    ruta = resolve(cfg, "reports_dir") / "hybrid_thresholds.json"
    if ruta.exists():
        with open(ruta) as f:
            return float(json.load(f)["umbral"])
    return float(cfg["gnn"]["threshold"])
