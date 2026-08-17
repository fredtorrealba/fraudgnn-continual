"""
El sistema híbrido en operación: GNN + cabeza XGBoost. DORMIDO con el CL.

    transacción ──┬─► features propias            (ya están en data.x)
                  └─► gnn_score / embedding       (la red puntúa)
                              ↓
                        cabeza XGBoost  ──►  P(fraude)

OJO al reactivar: se escribió para el esquema de 4 variantes numeradas y las
8 columnas estructurales, que SE RETIRARON (2026-08-17, medidas en −0.0013).
`score()` todavía las espera en `self.struct`; hay que portarlo al esquema de
tres cabezas + `grados_entidad.parquet` — está en la deuda de
`continual_learning.md`.

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
from src.continual_learning.validate import embed_and_score_nodes
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
        emb, g = embed_and_score_nodes(self.gnn, data, node_idx, self.cfg)
        if not self.es_hibrido:
            return g
        idx = np.asarray(node_idx, dtype=np.int64)
        base = data.x[idx].numpy().astype(np.float32)
        est = self.struct[idx]
        # Qué espera la cabeza se deduce de ELLA, no de la config: así el mismo
        # código sirve para la variante del escalar (431+8+1) y la del
        # embedding (431+8+dim), y un modelo entrenado con una no se puede
        # servir por accidente con la otra.
        esperado = self.head.num_features()
        if esperado == base.shape[1] + est.shape[1] + 1:
            extra = g.reshape(-1, 1).astype(np.float32)
        else:
            extra = emb
        X = np.hstack([base, est, extra])
        assert X.shape[1] == esperado, (
            f"La cabeza espera {esperado} columnas y se le arman "
            f"{X.shape[1]} ({base.shape[1]} base + {est.shape[1]} "
            f"estructurales + {extra.shape[1]})")
        return self.head.inplace_predict(X)

    def scorer(self, data):
        """La función que consume `validate_cycle`."""
        return lambda node_idx: self.score(data, node_idx)


def cargar_struct(cfg) -> np.ndarray | None:
    """
    Siempre None: las 8 columnas estructurales SE RETIRARON (2026-08-17).

    Medidas en su momento: −0.0013 de PR-AUC — una versión pobre de lo que ya
    calcula la GNN, duplicando C1-C14 y D1-D15. `features.py` se eliminó (está
    en el historial de git si hiciera falta); esta función se conserva porque
    `HybridSystem` ya acepta `struct=None` y así el CL no necesita cambios al
    reactivarse. El equivalente vigente y medido son los `__grado_*` de
    `grados_entidad.parquet`, que las cabezas ya reciben vía `cargar_tabla`.
    """
    return None


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
