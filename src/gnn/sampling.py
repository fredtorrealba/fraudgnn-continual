"""
Neighbor sampling 15-10-5 con fallback propio.

PyG NeighborLoader requiere pyg-lib o torch-sparse (compilación nativa que
suele fallar según la versión de torch/CUDA). Este módulo expone
`make_neighbor_loader(...)`: usa NeighborLoader si las libs están, y si no,
cae a `SimpleNeighborLoader` — una implementación en numpy del mismo
protocolo (sampleo de N vecinos fijos por capa, subgrafo local, nunca el
grafo completo). Es más lenta pero 100% portable; para el dataset completo
conviene instalar torch-sparse.

La interfaz del batch imita a NeighborLoader en lo que usa el proyecto:
  batch.x, batch.edge_index, batch.y, batch.batch_size, batch.n_id
con los nodos semilla SIEMPRE en las primeras `batch_size` posiciones.
"""
import numpy as np
import torch


def fanouts(cfg: dict) -> list[int]:
    """
    Fanouts recortados al número de capas del modelo.

    Cada capa es un salto, así que muestrear 3 saltos para un modelo de 1 capa
    sería traer miles de nodos que nunca se usan. Se toman los PRIMEROS N
    valores: el fanout grande corresponde al salto más cercano, que es donde
    está la señal.
    """
    g = cfg.get("gnn", {})
    n = len(g.get("hidden_dims", [256, 128, 64]))
    return list(g["fanouts"])[:n]


def loader_opts(cfg: dict) -> dict:
    """Opciones de muestreo del config, listas para pasar al loader."""
    g = cfg.get("gnn", {})
    return {"num_workers": g.get("num_workers", 0),
            "pin_memory": g.get("pin_memory", False)}


def _has_pyg_sampler() -> bool:
    try:
        import pyg_lib  # noqa: F401
        return True
    except ImportError:
        pass
    try:
        import torch_sparse  # noqa: F401
        return True
    except ImportError:
        return False


class _MiniBatch:
    __slots__ = ("x", "edge_index", "y", "batch_size", "n_id")

    def __init__(self, x, edge_index, y, batch_size, n_id):
        self.x, self.edge_index, self.y = x, edge_index, y
        self.batch_size, self.n_id = batch_size, n_id

    def to(self, device):
        self.x = self.x.to(device)
        self.edge_index = self.edge_index.to(device)
        self.y = self.y.to(device)
        return self


class SimpleNeighborLoader:
    """Sampling k-hop con fanouts fijos, sobre listas de adyacencia numpy."""

    _adj_cache: dict[int, np.ndarray] = {}

    def __init__(self, data, num_neighbors, input_nodes, batch_size,
                 shuffle=True, seed=42):
        self.data = data
        self.fanouts = list(num_neighbors)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed)
        self.seeds = torch.where(input_nodes)[0].numpy() \
            if input_nodes.dtype == torch.bool else np.asarray(input_nodes)
        self._build_adjacency()

    def _build_adjacency(self):
        key = id(self.data)
        if key in SimpleNeighborLoader._adj_cache:
            self.indptr, self.indices = SimpleNeighborLoader._adj_cache[key]
            return
        ei = self.data.edge_index.numpy()
        n = self.data.num_nodes
        order = np.argsort(ei[0], kind="stable")
        src_sorted, dst_sorted = ei[0][order], ei[1][order]
        self.indptr = np.zeros(n + 1, dtype=np.int64)
        np.add.at(self.indptr, src_sorted + 1, 1)
        self.indptr = np.cumsum(self.indptr)
        self.indices = dst_sorted
        SimpleNeighborLoader._adj_cache[key] = (self.indptr, self.indices)

    def _neighbors(self, node: int, k: int) -> np.ndarray:
        lo, hi = self.indptr[node], self.indptr[node + 1]
        neigh = self.indices[lo:hi]
        if len(neigh) <= k:
            return neigh
        return self.rng.choice(neigh, size=k, replace=False)

    def __iter__(self):
        seeds = self.rng.permutation(self.seeds) if self.shuffle else self.seeds
        for i in range(0, len(seeds), self.batch_size):
            batch_seeds = seeds[i:i + self.batch_size]
            nodes = list(batch_seeds)
            seen = set(nodes)
            frontier = list(batch_seeds)
            edges = []
            for k in self.fanouts:                 # 15 -> 10 -> 5
                nxt = []
                for u in frontier:
                    for v in self._neighbors(int(u), k):
                        edges.append((int(v), int(u)))  # mensaje vecino->nodo
                        if v not in seen:
                            seen.add(int(v))
                            nodes.append(int(v))
                            nxt.append(int(v))
                frontier = nxt
            local = {n: j for j, n in enumerate(nodes)}
            if edges:
                ei = torch.tensor([[local[a] for a, _ in edges],
                                   [local[b] for _, b in edges]],
                                  dtype=torch.long)
                ei = torch.cat([ei, ei.flip(0)], dim=1)
            else:
                ei = torch.zeros((2, 0), dtype=torch.long)
            n_id = torch.tensor(nodes, dtype=torch.long)
            yield _MiniBatch(self.data.x[n_id], ei, self.data.y[n_id],
                             len(batch_seeds), n_id)

    def __len__(self):
        return int(np.ceil(len(self.seeds) / self.batch_size))


def make_neighbor_loader(data, num_neighbors, input_nodes, batch_size,
                         shuffle=True, num_workers=0, pin_memory=False):
    """
    NeighborLoader de PyG si hay backend nativo; si no, el fallback.

    `num_workers` y `pin_memory` vienen de config.yaml (gnn.num_workers y
    gnn.pin_memory) y SOLO aplican al NeighborLoader de PyG: el fallback es
    una clase propia, no un DataLoader, y los ignora.
    """
    if _has_pyg_sampler():
        from torch_geometric.loader import NeighborLoader
        # pyg-lib exige tensores contiguos. Los grafos generados antes de
        # arreglar build_graph traen edge_index como vista transpuesta
        # (stride (1,2)) y revientan en index_sort: se normalizan aquí para
        # no obligar a reconstruir el grafo.
        if not data.edge_index.is_contiguous():
            data.edge_index = data.edge_index.contiguous()
        extra = {}
        if num_workers and int(num_workers) > 0:
            # persistent_workers evita respawnear los procesos en CADA época
            # (30 épocas x 6 corridas = mucho arranque desperdiciado).
            extra = {"num_workers": int(num_workers), "persistent_workers": True}
        return NeighborLoader(data, num_neighbors=num_neighbors,
                              input_nodes=input_nodes,
                              batch_size=batch_size, shuffle=shuffle,
                              pin_memory=bool(pin_memory), **extra)
    return SimpleNeighborLoader(data, num_neighbors, input_nodes,
                                batch_size, shuffle)
