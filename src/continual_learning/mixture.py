"""
Composición del set de adaptación: 40% casos nuevos / 60% replay buffer.

Vive aparte porque la usan DOS modelos: el fine-tuning de la GNN
(`finetune.py`) y el warm start de la cabeza XGBoost del sistema híbrido
(`src/hybrid/head.py`). Con la misma función y la misma semilla, ambos
entrenan sobre las FILAS IDÉNTICAS — no sobre dos muestras parecidas —, que es
lo que hace comparable la adaptación de las dos piezas.

Es numpy puro sobre índices de nodo: no depende de torch ni de PyG.
"""
import numpy as np

SEMILLA = 42        # fija: los reintentos del dial deben ver el mismo buffer


def mezcla_40_60(adapt_nodes: np.ndarray, buffer_nodes: np.ndarray,
                 mix_new: float, seed: int = SEMILLA) -> np.ndarray:
    """
    Índices de nodo del conjunto de adaptación.

    `mix_new` es la fracción que deben representar los casos nuevos (0.40 por
    defecto en config). El tamaño del buffer se deriva de ahí:

        n_buf = n_new * (1 - mix_new) / mix_new

    Con 11 casos nuevos y mix_new=0.4 salen ~16 del buffer -> 27 filas totales.
    Si el buffer tiene menos entradas de las pedidas se usa entero.
    """
    n_new = len(adapt_nodes)
    n_buf = int(round(n_new * (1 - mix_new) / max(mix_new, 1e-6)))
    rng = np.random.default_rng(seed)
    buf_sample = rng.choice(buffer_nodes,
                            size=min(n_buf, len(buffer_nodes)),
                            replace=False)
    return np.concatenate([adapt_nodes, buf_sample])
