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
            "prefetch_factor": g.get("prefetch_factor", 2),
            "sin_aristas": g.get("sin_aristas", False)}


def fanouts_hetero(data, cfg: dict) -> dict:
    """
    Fanouts POR TIPO DE ARISTA, con tantos saltos como capas tenga el modelo.

    En un grafo bipartito llegar de una transacción a otra cuesta DOS saltos:

        transacción -> [entidad] -> otra transacción de la misma entidad

    OJO CON LA DIRECCIÓN. `NeighborLoader` muestrea HACIA ATRÁS: para calcular
    un nodo baja sus vecinos de ENTRADA. Así que el fanout de un tipo de arista
    (origen, rel, destino) dice cuántos ORÍGENES se traen al expandir un
    DESTINO — al revés de lo que sugiere leer el nombre.

        ('transaction','en',entidad)      destino = entidad
            se usa al expandir una ENTIDAD y trae TRANSACCIONES
            -> aquí van los `vecinos_por_entidad`

        (entidad,'tiene','transaction')   destino = transacción
            se usa al expandir una TRANSACCIÓN y trae ENTIDADES
            -> aquí va 1: una transacción pertenece a UNA entidad de cada tipo

    ESTUVO AL REVÉS y anulaba el grafo entero: se pedían 10 entidades por
    transacción (y solo hay 1) y UNA transacción por entidad, así que el
    vecindario era de un nodo. En el log real, con 2048 semillas salían 2049
    nodos de transacción — un vecino en total. Y con `time_attr` activo, cero.
    La GNN llevaba TODAS las corridas heterogéneas entrenando como una MLP.

    Verificado con NeighborLoader aislado, semilla t=900 sobre 10 transacciones:
        invertido  -> [600, 900]                    1 vecino
        correcto   -> [0, 100, ..., 900]            los 9, y solo pasado

    Con 5 tipos de entidad y 10 por entidad, el vecindario de una transacción es
    de hasta 50 transacciones, todas anteriores a ella.
    """
    # Default de 2 capas: el bipartito necesita dos saltos para que una
    # transacción alcance a otra. El [256] de antes venía del grafo homogéneo.
    n_capas = len(cfg["gnn"].get("hidden_dims", [64, 64]))
    por_entidad = int(cfg["graph"].get("vecinos_por_entidad", 10))
    out = {}
    for et in data.edge_types:
        # et[0] == TXN  ->  ('transaction','en',entidad)  ->  expande ENTIDAD
        cuantos = por_entidad if et[0] == TXN else 1
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


def cerrar_loader(loader) -> None:
    """
    Mata los workers de un loader EN CUANTO deja de usarse.

    Con `persistent_workers=True` los procesos siguen vivos hasta que el
    intérprete termina. Nadie los cerraba, así que se acumulaban: cada trial de
    Optuna dejaba 12 procesos colgados y el proceso tardaba minutos en morir
    joinéandolos todos, DESPUÉS de haber escrito su último log.

    Medido en el smoke (2 trials x 2 arquitecturas = 4 loaders):
        último log de `gnn` 05:11:29  ->  la etapa acabó 05:23:31   12 min
        último log de `oof` 05:23:42  ->  la etapa acabó 05:25:43    2 min

    En la corrida real son 30 trials x 2 arquitecturas = 60 loaders, o sea 720
    procesos acumulados. Por eso esto no es cosmético.

    Va con try/except porque `_shutdown_workers` es API privada de PyTorch: si
    cambia de nombre, se pierde la limpieza pero no se rompe el entrenamiento.
    """
    try:
        it = getattr(loader, "_iterator", None)
        if it is not None:
            it._shutdown_workers()
            loader._iterator = None
    except Exception:
        pass


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

    # Los workers se escalan con el TAMAÑO, no se copian del config a ciegas.
    # Levantar 12 procesos cuesta ~60 s cuando el padre ya tiene contexto CUDA y
    # el grafo en memoria: cada fork lo hereda. Eso se paga por CADA loader, y
    # `oof` crea uno por fold y Optuna uno por trial. Medido en el smoke:
    # entrenar un fold de 4.000 nodos tardaba 2 s el primero y 61 s el segundo,
    # todo en el arranque de los workers, no en muestrear.
    # Con 410.000 nodos (la corrida real) el reparto sí compensa y se usan los 12.
    # Dos casos con costes opuestos, y `shuffle` los distingue sin más pistas:
    #   shuffle=True   ENTRENAR. El loader se crea una vez y se recorre en cada
    #                  época (persistent_workers), así que el arranque se
    #                  amortiza y los workers compensan siempre.
    #   shuffle=False  PUNTUAR. Se crea y se usa UNA vez: arrancar 12 procesos
    #                  ES el coste. Medido: un fold del OOF pasó de 61 s a 2 s
    #                  al quitarlos.
    # El umbral de tamaño sigue para conjuntos grandes de inferencia, donde
    # muestrear sí domina sobre el arranque.
    n_semillas = int(semillas.sum()) if semillas.dtype == torch.bool else len(semillas)
    n_w = int(opts["num_workers"]) if (shuffle or n_semillas >= 20_000) else 0
    extra = {}
    if n_w:
        # persistent_workers evita respawnear procesos en CADA época.
        # prefetch_factor = lotes que cada worker deja LISTOS por adelantado.
        # Con el defecto (2) los workers terminan y se quedan esperando: medido
        # en la corrida real, 12 workers al 2,5% de CPU y la GPU al 5% — nadie
        # saturado porque todos esperaban. Subirlo hace que sigan muestreando
        # mientras la GPU consume, que es lo que llena la tubería.
        # Coste: RAM = workers x prefetch x tamaño del lote.
        extra = {"num_workers": n_w, "persistent_workers": True,
                 "prefetch_factor": int(opts["prefetch_factor"])}

    # Generador EXPLÍCITO para el shuffle. Sin él, DataLoader usa el generador
    # global de torch, así que el orden de los lotes depende de cuánto azar se
    # haya consumido antes de crear el loader. Funciona por accidente mientras
    # nadie meta una llamada aleatoria en medio; con uno propio, el orden es el
    # mismo siempre.
    gen = torch.Generator()
    gen.manual_seed(int(seed))

    return NeighborLoader(
        data,
        num_neighbors=fanouts_hetero(data, cfg),
        input_nodes=(TXN, semillas),
        batch_size=batch_size or cfg["gnn"]["batch_size"],
        shuffle=shuffle,
        generator=gen,
        time_attr="time",              # causalidad: solo vecinos anteriores
        temporal_strategy="last",      # y de esos, los más recientes
        pin_memory=bool(opts["pin_memory"]),
        **extra,
    )


def proporcion_fraude(batch) -> float:
    """Fracción de fraude entre los nodos SEMILLA del lote (para el log)."""
    n = batch[TXN].batch_size
    return float(batch[TXN].y[:n].mean()) if n else 0.0
