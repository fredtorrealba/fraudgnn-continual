"""
Paso 2 — Construcción del grafo homogéneo (PyTorch Geometric).

Diseño (según la definición del proyecto):
- NODOS = transacciones (cada una con sus 432 features + isFraud).
- ARISTAS = entidad compartida entre dos transacciones:
    * misma tarjeta   -> huella card1+card2+card3+card5+addr1
    * mismo email     -> P_emaildomain (se ignoran dominios masivos tipo gmail
                         combinándolo con card1 para no crear hubs falsos)
    * mismo dispositivo -> DeviceInfo + id_30 + id_31
- Reglas anti-explosión:
    * ventana temporal de 30 días entre transacciones conectadas
    * tope de 50 aristas por nodo (se conectan las más cercanas en el tiempo)

Salida: data/graph/graph.pt  (torch_geometric.data.Data con x, edge_index, y,
masks de split y metadatos de tiempo).
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.common import ensure_dirs, get_logger, load_config, resolve

log = get_logger("build_graph")


def entity_keys(df: pd.DataFrame, cfg: dict) -> dict:
    """Construye la clave de entidad por cada tipo de arista."""
    def joined(cols):
        return df[cols].fillna("nan").astype(str).agg("|".join, axis=1)

    keys = {}
    # Tarjeta: concatenación de la huella completa
    card_cols = [f"raw__{c}" for c in cfg["graph"]["edge_entities"]["card"]]
    keys["card"] = joined(card_cols)

    # Email: dominio + card1 (evita hub gigante en gmail.com)
    keys["email"] = joined(["raw__P_emaildomain", "raw__card1"]).where(
        ~df["raw__P_emaildomain"].isna(), np.nan)

    # Dispositivo: DeviceInfo + versiones de SO/navegador
    dev_cols = [f"raw__{c}" for c in cfg["graph"]["edge_entities"]["device"]]
    keys["device"] = joined(dev_cols).where(~df["raw__DeviceInfo"].isna(), np.nan)
    return keys


def build_edges(df: pd.DataFrame, cfg: dict) -> np.ndarray:
    """
    Para cada entidad compartida conecta transacciones dentro de la ventana
    temporal, respetando el tope de aristas por nodo.
    """
    window_s = cfg["graph"]["window_days"] * 86400
    max_deg = cfg["graph"]["max_edges_per_node"]
    dts = df["TransactionDT"].values

    degree = np.zeros(len(df), dtype=np.int32)
    edges = set()
    keys = entity_keys(df, cfg)

    for etype, key_series in keys.items():
        groups = defaultdict(list)
        for idx, k in enumerate(key_series.values):
            if isinstance(k, str) and "nan" not in k.split("|")[0:1]:
                groups[k].append(idx)

        n_edges_type = 0
        for _, idxs in groups.items():
            if len(idxs) < 2:
                continue
            idxs.sort(key=lambda i: dts[i])  # ordenados en el tiempo
            # Cada nodo se conecta hacia atrás con sus vecinos temporales más
            # cercanos de la misma entidad (dentro de la ventana y del tope).
            for pos in range(1, len(idxs)):
                i = idxs[pos]
                for prev_pos in range(pos - 1, -1, -1):
                    j = idxs[prev_pos]
                    if dts[i] - dts[j] > window_s:
                        break
                    if degree[i] >= max_deg or degree[j] >= max_deg:
                        continue
                    e = (j, i) if j < i else (i, j)
                    if e not in edges:
                        edges.add(e)
                        degree[i] += 1
                        degree[j] += 1
                        n_edges_type += 1
        log.info("Aristas tipo %-7s: %d", etype, n_edges_type)

    if not edges:
        return np.zeros((2, 0), dtype=np.int64)
    arr = np.array(list(edges), dtype=np.int64).T
    # Grafo no dirigido -> duplicar en ambos sentidos.
    # ascontiguousarray NO es cosmético: .T y [::-1] devuelven vistas en orden
    # Fortran, torch.tensor() PRESERVA esos strides, y el sampler nativo
    # (pyg_lib.ops.index_sort) falla con "Input should be contiguous".
    return np.ascontiguousarray(
        np.concatenate([arr, arr[::-1]], axis=1))


def main():
    cfg = load_config()
    ensure_dirs(cfg)
    proc_dir, graph_dir = resolve(cfg, "processed_dir"), resolve(cfg, "graph_dir")

    df = pd.read_parquet(proc_dir / "full.parquet")
    with open(proc_dir / "feature_cols.json") as f:
        meta = json.load(f)
    feature_cols = meta["feature_cols"]

    log.info("Construyendo aristas para %d nodos...", len(df))
    edge_index = build_edges(df, cfg)
    log.info("Total aristas (dirigidas x2): %d", edge_index.shape[1])

    from torch_geometric.data import Data
    data = Data(
        x=torch.tensor(df[feature_cols].values, dtype=torch.float32),
        edge_index=torch.tensor(edge_index, dtype=torch.long),
        y=torch.tensor(df["isFraud"].values, dtype=torch.float32),
    )
    data.transaction_id = torch.tensor(df["TransactionID"].values, dtype=torch.long)
    data.transaction_dt = torch.tensor(df["TransactionDT"].values, dtype=torch.long)
    data.month = torch.tensor(df["month"].values, dtype=torch.long)
    data.week_in_month = torch.tensor(df["week_in_month"].values, dtype=torch.long)
    # Cada nodo lleva: features (x) + etiqueta (y) + fraud_score.
    # El score parte en NaN y lo va llenando el modelo al operar.
    data.fraud_score = torch.full((data.num_nodes,), float("nan"))
    for split in ["train", "val", "test"]:
        setattr(data, f"{split}_mask",
                torch.tensor((df["split"] == split).values, dtype=torch.bool))

    torch.save(data, graph_dir / "graph.pt")
    log.info("Grafo: %d nodos, %d aristas, %d features",
             data.num_nodes, data.edge_index.shape[1], data.x.shape[1])
    if data.x.shape[1] != cfg["gnn"]["in_dim"]:
        log.warning("in_dim real (%d) != config (%d). Actualiza gnn.in_dim en "
                    "config.yaml antes de entrenar.", data.x.shape[1], cfg["gnn"]["in_dim"])


if __name__ == "__main__":
    main()
