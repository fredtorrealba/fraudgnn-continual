"""
Paso 3 — GRAFO HETEROGÉNEO de transacciones y entidades.

QUÉ CAMBIÓ Y POR QUÉ
Antes las transacciones se conectaban ENTRE SÍ cuando compartían una entidad
(10,99 M aristas). Ahora la entidad es un NODO y las aristas son bipartitas:

    transacción  <--->  [uid] [card] [email] [device] [net]

Tres consecuencias:

1. El nodo de entidad tiene su PROPIO vector aprendido. Antes el vecindario se
   resumía con una media fija; ahora "esta tarjeta" es una representación que la
   red entrena y comparte entre todas sus transacciones.
2. Las aristas bajan a ~2,5 M: cada transacción tiene como mucho una arista por
   tipo de entidad, en vez de hasta 50 vecinos directos.
3. Es INDUCTIVO: los nodos de entidad no llevan features propias, su contenido
   sale de agregar sus transacciones. Una entidad que aparece por primera vez en
   el mes 6 funciona igual que una vista en el mes 1.

LA REGLA DE NULOS
Si CUALQUIER columna de la clave es nula, no hay nodo ni arista. Antes se
rellenaba con el texto "nan" y se concatenaba, así que un cliente con `addr1` y
el MISMO cliente sin `addr1` (~11% de las filas) caían en grupos distintos: el
patrón de ausencia se había vuelto parte de la identidad.

`device` y `net` solo cubren ~24% de las filas porque la tabla `identity` se une
con LEFT JOIN. No es un fallo: esas transacciones simplemente no tienen esa
arista, y la cobertura real se reporta en graph_meta.json.

CAUSALIDAD
Aquí NO se filtra por tiempo. La causalidad la impone el muestreo con
`temporal_strategy="last"` (ver gnn/sampling.py), que solo baja vecinos con
`transaction_dt` ANTERIOR al del nodo raíz. Eso arregla de raíz la fuga del
diseño anterior, donde el grafo se hacía no dirigido y un nodo podía ver el
futuro.

Salidas: data/graph/graph.pt (HeteroData) + data/graph/graph_meta.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.common import ensure_dirs, get_logger, load_config, resolve

log = get_logger("build_graph")


def clave_entidad(df: pd.DataFrame, spec: dict) -> pd.Series:
    """
    Clave de una entidad, o NaN si le falta algún componente.

    `usa_d1` añade (día - D1) a la clave. D1 es "días desde el primer uso de la
    tarjeta", así que restarlo del día de la transacción da una CONSTANTE por
    cliente: el día en que esa tarjeta empezó a usarse. Es lo que distingue a
    clientes que colisionan en el mismo `card1`, y es el truco que encontraron
    los ganadores de la competición de Kaggle sobre este dataset.
    """
    cols = [f"raw__{c}" for c in spec["cols"]]
    faltan = [c for c in cols if c not in df.columns]
    if faltan:
        log.warning("Columnas ausentes, la entidad se omite: %s", faltan)
        return pd.Series(np.nan, index=df.index, dtype=object)

    # nulo en CUALQUIER componente -> sin clave. Nunca se rellena con "nan".
    valido = df[cols].notna().all(axis=1)
    partes = [df[c].astype(str) for c in cols]

    if spec.get("usa_d1"):
        if "D1" not in df.columns:
            log.warning("Falta D1: la entidad de cliente cae a la clave sin él")
        else:
            dia = (df["TransactionDT"] // 86400).astype("float64")
            inicio = dia - df["D1"].astype("float64")
            valido &= df["D1"].notna()
            partes.append(inicio.astype("Int64").astype(str))

    clave = partes[0].str.cat(partes[1:], sep="|") if len(partes) > 1 else partes[0]
    return clave.where(valido, np.nan)


def main():
    cfg = load_config()
    ensure_dirs(cfg)
    proc, graph_dir = resolve(cfg, "processed_dir"), resolve(cfg, "graph_dir")

    df = pd.read_parquet(proc / "full.parquet")
    with open(proc / "feature_cols.json") as f:
        feature_cols = json.load(f)["feature_cols"]

    # MISMA ABLACIÓN QUE LAS CABEZAS XGBOOST. Si se excluyen V/C/D del modelo
    # tabular pero la GNN las sigue viendo, el embedding las promedia y se las
    # devuelve a XGBoost por la puerta de atrás: la variante con grafo tendría
    # esas columnas —disfrazadas— y el control no. Ganaría por copiarlas, no
    # por descubrir nada, y la ablación no mediría lo que dice medir.
    from src.hybrid.head import filtrar_prefijos
    prefijos = (cfg.get("xgboost") or {}).get("excluir_prefijos") or []
    if prefijos:
        antes = len(feature_cols)
        feature_cols = filtrar_prefijos(feature_cols, prefijos)
        log.info("ABLACIÓN %s: la GNN también pierde esas columnas -> %d "
                 "features por nodo en vez de %d", prefijos, len(feature_cols), antes)

    # --- dos features que la GNN necesita y no tenía -----------------------
    # Sin ellas no puede aprender lo que daban las columnas C y D, por mucho que
    # se las quitemos al tabular:
    #   · el TIEMPO no estaba entre las features (TransactionDT se excluye por
    #     ser identificador). La red elegía vecinos por fecha pero no sabía
    #     CUÁNDO ocurrió nada, así que no podía calcular deltas.
    #   · el TAMAÑO de la entidad tampoco: el muestreo capa a 10 vecinos, así
    #     que una tarjeta con 3 compras y otra con 300 le llegaban iguales.
    # Se añaden normalizadas para no descuadrar la escala del resto.
    # Las temporales ya vienen en feature_cols desde `preprocess`.


    from torch_geometric.data import HeteroData
    data = HeteroData()

    # --- nodos transacción: los ÚNICOS con features propias ------------------
    data["transaction"].x = torch.tensor(df[feature_cols].values, dtype=torch.float32)
    data["transaction"].y = torch.tensor(df["isFraud"].values, dtype=torch.float32)
    # time_attr del muestreo temporal: debe ser int64 y estar en el nodo
    data["transaction"].time = torch.tensor(df["TransactionDT"].values, dtype=torch.long)
    data["transaction"].transaction_id = torch.tensor(df["TransactionID"].values,
                                                      dtype=torch.long)
    data["transaction"].month = torch.tensor(df["month"].values, dtype=torch.long)
    data["transaction"].week_in_month = torch.tensor(df["week_in_month"].values,
                                                     dtype=torch.long)
    for split in ("train", "val", "test"):
        data["transaction"][f"{split}_mask"] = torch.tensor(
            (df["split"] == split).values, dtype=torch.bool)

    n_txn = len(df)
    tope = int(cfg["graph"].get("max_entity_degree", 500))
    meta = {"n_transacciones": n_txn, "max_entity_degree": tope, "entidades": {}}

    log.info("Grafo heterogéneo sobre %d transacciones | poda de entidad > %d",
             n_txn, tope)
    grados_extra: dict[str, np.ndarray] = {}

    for nombre, spec in cfg["graph"]["entidades"].items():
        claves = clave_entidad(df, spec)
        presentes = claves.notna()
        if not presentes.any():
            log.warning("Entidad '%s': ninguna fila la tiene, se omite", nombre)
            continue

        codigos, unicas = pd.factorize(claves[presentes], sort=False)
        tam = np.bincount(codigos, minlength=len(unicas))

        # PODA: una entidad con miles de transacciones tendría como vector la
        # media del dataset entero, y ese vector se repartiría a todos sus
        # vecinos borrando las diferencias en vez de crearlas (over-smoothing
        # por hub). Además explota la memoria al muestrear.
        gigantes = tam > tope
        n_podadas = int(gigantes.sum())
        vivas = ~gigantes[codigos]

        filas = np.where(presentes.values)[0][vivas]
        ent_idx = codigos[vivas]
        # renumerar tras la poda para no dejar huecos
        remap, ent_idx = np.unique(ent_idx, return_inverse=True)

        et = ("transaction", f"en_{nombre}", nombre)
        rev = (nombre, f"tiene_{nombre}", "transaction")
        src = torch.tensor(filas, dtype=torch.long)
        dst = torch.tensor(ent_idx, dtype=torch.long)
        data[et].edge_index = torch.stack([src, dst])
        data[rev].edge_index = torch.stack([dst, src])
        data[nombre].num_nodes = int(len(remap))

        # GRADO de la entidad como feature del nodo: cuántas transacciones
        # tiene la entidad a la que pertenece esta transacción, ANTES de capar
        # a 10 en el muestreo. Es lo que le permite a la red aprender lo que
        # daban las columnas C (conteos por entidad).
        if cfg["graph"].get("features_derivadas", True):
            grado = np.bincount(ent_idx, minlength=len(remap))[ent_idx]
            col = np.zeros(n_txn, dtype=np.float32)
            col[filas] = np.log1p(grado)          # log: los hubs no dominan
            grados_extra[f"__grado_{nombre}"] = col

        cob = len(filas) / n_txn
        meta["entidades"][nombre] = {
            "columnas": spec["cols"], "usa_d1": bool(spec.get("usa_d1")),
            "n_nodos": int(len(remap)), "n_aristas": int(len(filas)),
            "cobertura": round(cob, 4),
            "podadas_por_grado": n_podadas,
            "txn_por_entidad_media": round(len(filas) / max(len(remap), 1), 1),
            "degenerada": bool(len(filas) / max(len(remap), 1) < 2),
        }
        media = len(filas) / max(len(remap), 1)
        log.info("  %-7s %7d nodos | %7d aristas | cobertura %5.1f%% | "
                 "%4.1f txn/entidad | %d podadas",
                 nombre, len(remap), len(filas), 100 * cob, media, n_podadas)
        # Una entidad con ~1 transacción por grupo NO CONECTA NADA: cada
        # transacción cuelga de su propio nodo y el paso de mensajes no
        # transporta información de nadie. Suele significar que la clave es
        # demasiado específica (p.ej. añadir card1 a algo que ya era casi único).
        if media < 2:
            log.warning("    '%s' es DEGENERADA (%.1f txn/entidad): la clave es "
                        "demasiado específica y no conecta transacciones entre "
                        "sí. Revisa sus columnas o quítala de config.", nombre, media)
        # El extremo contrario: pocos grupos enormes conectan medio grafo con el
        # otro medio, y su vector acaba siendo la media del dataset.
        elif media > tope / 2:
            log.warning("    '%s' es muy GRUESA (%.1f txn/entidad): puede que la "
                        "poda por grado esté haciendo casi todo el trabajo.",
                        nombre, media)

    if grados_extra:
        extra = np.stack([grados_extra[k] for k in sorted(grados_extra)], axis=1)
        data["transaction"].x = torch.cat(
            [data["transaction"].x, torch.tensor(extra, dtype=torch.float32)], dim=1)
        feature_cols = feature_cols + sorted(grados_extra)
        log.info("Grados de entidad añadidos como features: %s",
                 sorted(grados_extra))

    meta["feature_cols_gnn"] = feature_cols
    meta["n_features_gnn"] = len(feature_cols)

    # --- las aristas deben ir ORDENADAS por tiempo del origen ---------------
    # `temporal_strategy="last"` (gnn/sampling.py) exige que, para cada nodo
    # destino, sus aristas entrantes estén ordenadas por el tiempo del origen.
    # Si no lo están, "los N más recientes" devuelve cualquier cosa.
    for nombre in list(meta["entidades"]):
        rev = (nombre, f"tiene_{nombre}", "transaction")
        ei = data[rev].edge_index
        t = data["transaction"].time[ei[1]]
        orden = torch.argsort(ei[0] * (int(t.max()) + 1) + t)
        data[rev].edge_index = ei[:, orden]

    torch.save(data, graph_dir / "graph.pt")
    meta["n_aristas_total"] = int(sum(v["n_aristas"] for v in meta["entidades"].values()))
    with open(graph_dir / "graph_meta.json", "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    log.info("Guardado: %d tipos de entidad, %d aristas (x2 con las inversas)",
             len(meta["entidades"]), meta["n_aristas_total"])
    # INFO, no WARNING: `train_gnn` y `compare_gnns` sobrescriben in_dim con el
    # ancho REAL del grafo antes de construir el modelo, así que el valor del
    # config no se usa nunca. Como el ancho depende de la ablación y de las
    # features derivadas, el config quedaría desfasado a cada cambio y el aviso
    # saltaría siempre sin que hubiera nada que arreglar.
    real = data["transaction"].x.shape[1]
    if real != cfg["gnn"].get("in_dim"):
        log.info("in_dim: %d features por transacción (el config dice %s; manda "
                 "el real, se toma del grafo en cada entrenamiento)",
                 real, cfg["gnn"].get("in_dim"))


if __name__ == "__main__":
    main()
