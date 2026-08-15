"""
Muestreo de vecindarios sobre el GRAFO HETEROGÉNEO.

DOS DECISIONES QUE NO SON COSMÉTICAS

1. VECINOS POR RECENCIA, NO AL AZAR
   `temporal_strategy="last"` + `time_attr` resuelve dos cosas de un tiro:
     · solo se bajan vecinos con tiempo ANTERIOR al del nodo raíz  -> causalidad
       estricta. El diseño anterior construía las aristas hacia atrás pero luego
       hacía el grafo NO DIRIGIDO, así que un nodo podía ver el futuro.
     · y de esos, los N MÁS RECIENTES, que es lo que importa en fraude: lo que
       hizo esa tarjeta ayer pesa más que lo que hizo hace tres semanas.
   Exige que las aristas estén ordenadas por tiempo (lo hace build_graph).

2. SEMILLAS BALANCEADAS
   Un lote al azar de 1.024 transacciones trae ~36 fraudes y ~988 normales. Con
   `balanceo_semillas` se repiten fraudes REALES hasta llenar la mitad del lote.
   No se inventa ningún nodo ni arista — un nodo sintético no tendría vecinos y
   habría que inventarle el grafo. Cada repetición ve un vecindario distinto
   (el muestreo cambia), así que funciona como aumento de datos.

   Arregla algo que `pos_weight` NO toca: el BatchNorm calcula media y varianza
   CON EL LOTE, y con 988 normales contra 36 fraudes esas estadísticas son las
   de una compra normal. El fraude se normalizaba contra una referencia ajena.

   OJO: al balancear hay que BAJAR `pos_weight` (ver gnn.pos_weight_con_balanceo)
   o el desbalance se corrige dos veces y la red sobrepredice fraude.

REQUISITO: el grafo heterogéneo necesita el sampler nativo (pyg-lib o
torch-sparse). El fallback casero en numpy solo servía para grafos homogéneos y
se ha retirado; si falta el sampler, se aborta con instrucciones.
"""
import numpy as np
import torch

TXN = "transaction"


def _tiene_sampler_nativo() -> bool:
    for m in ("pyg_lib", "torch_sparse"):
        try:
            __import__(m)
            return True
        except ImportError:
            continue
    return False


def loader_opts(cfg: dict) -> dict:
    """Opciones de muestreo del config, listas para el loader."""
    g = cfg.get("gnn", {})
    return {"num_workers": g.get("num_workers", 0),
            "pin_memory": g.get("pin_memory", False),
            "sin_aristas": g.get("sin_aristas", False)}


def fanouts_hetero(data, cfg: dict) -> dict:
    """
    Fanouts POR TIPO DE ARISTA, con tantos saltos como capas tenga el modelo.

    En un grafo bipartito llegar de una transacción a otra cuesta DOS saltos:

        transacción -> [entidad] -> otra transacción de la misma entidad

    Por eso los dos sentidos llevan valores distintos:
      · transaction -> entidad : 1. Una transacción pertenece como mucho a UNA
        entidad de cada tipo, así que pedir más no trae nada.
      · entidad -> transaction : `graph.vecinos_por_entidad` (10 por defecto).

    Con 5 tipos de entidad y 10 por entidad, el vecindario de una transacción es
    de hasta 50 transacciones, todas anteriores a ella.
    """
    # Default de 2 capas: el bipartito necesita dos saltos para que una
    # transacción alcance a otra. El [256] de antes venía del grafo homogéneo.
    n_capas = len(cfg["gnn"].get("hidden_dims", [64, 64]))
    por_entidad = int(cfg["graph"].get("vecinos_por_entidad", 10))
    out = {}
    for et in data.edge_types:
        cuantos = 1 if et[0] == TXN else por_entidad
        out[et] = [cuantos] * n_capas
    return out


def semillas_balanceadas(data, mask, cfg, generador=None) -> torch.Tensor:
    """
    Índices semilla con ~50% de fraude, repitiendo fraudes REALES.

    Devuelve un tensor de índices (con repeticiones) apto para `input_nodes`.
    Si no hay fraudes o el balanceo está desactivado, devuelve la máscara tal cual.
    """
    if not cfg["gnn"].get("balanceo_semillas", False):
        return mask
    idx = torch.where(mask)[0]
    y = data[TXN].y[idx]
    fraude, normal = idx[y == 1], idx[y == 0]
    if len(fraude) == 0 or len(normal) == 0:
        return mask
    rng = generador or np.random.default_rng(42)
    # tantas normales como el conjunto original, y fraudes repetidos hasta igualar
    n = len(normal)
    reps = rng.choice(len(fraude), size=n, replace=True)
    return torch.cat([normal, fraude[torch.tensor(reps, dtype=torch.long)]])


def make_hetero_loader(data, cfg, mask, shuffle=True, batch_size=None,
                       balancear=False, seed=42):
    """
    NeighborLoader heterogéneo con muestreo temporal.

    `mask` puede ser una máscara booleana o un tensor de índices (lo que
    devuelve `semillas_balanceadas`).
    """
    if not _tiene_sampler_nativo():
        raise SystemExit(
            "El grafo heterogéneo necesita el sampler nativo de PyG y no está "
            "instalado.\n"
            "  pip install pyg-lib torch-sparse -f "
            "https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html\n"
            "Ver scripts/setup_runpod.sh, que lo resuelve automáticamente.")

    from torch_geometric.loader import NeighborLoader

    opts = loader_opts(cfg)
    if opts["sin_aristas"]:
        # ABLACIÓN: se vacían todas las aristas. Cada transacción queda aislada
        # y el modelo se reduce a una MLP con la misma capacidad. Responde una
        # sola pregunta: ¿el AUC depende del grafo o nunca lo estaba usando?
        # NO toca graph.pt: el cambio es en memoria, por corrida.
        data = data.clone()
        for et in data.edge_types:
            data[et].edge_index = torch.empty((2, 0), dtype=torch.long)

    semillas = semillas_balanceadas(data, mask, cfg,
                                    np.random.default_rng(seed)) if balancear else mask

    extra = {}
    if opts["num_workers"]:
        # persistent_workers evita respawnear procesos en CADA época
        extra = {"num_workers": int(opts["num_workers"]), "persistent_workers": True}

    return NeighborLoader(
        data,
        num_neighbors=fanouts_hetero(data, cfg),
        input_nodes=(TXN, semillas),
        batch_size=batch_size or cfg["gnn"]["batch_size"],
        shuffle=shuffle,
        time_attr="time",              # causalidad: solo vecinos anteriores
        temporal_strategy="last",      # y de esos, los más recientes
        pin_memory=bool(opts["pin_memory"]),
        **extra,
    )


def proporcion_fraude(batch) -> float:
    """Fracción de fraude entre los nodos SEMILLA del lote (para el log)."""
    n = batch[TXN].batch_size
    return float(batch[TXN].y[:n].mean()) if n else 0.0
