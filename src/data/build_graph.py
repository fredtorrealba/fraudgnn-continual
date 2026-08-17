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


def _previas_por_entidad(ent_idx: np.ndarray, tiempo: np.ndarray) -> np.ndarray:
    """
    Cuántas transacciones ANTERIORES tiene cada fila dentro de su entidad.

    El conteo total (`bincount`) mira al futuro: una transacción de enero
    llevaba el número de compras que su tarjeta acumularía hasta junio. Es una
    fuga que no rompe nada y falsea el resultado, justo el tipo que hay que
    cazar a mano.

    Se ordena por (entidad, tiempo) y la posición dentro del grupo ES el número
    de anteriores. Vectorizado: con 500.000 filas un bucle en Python tardaría
    más que el resto de la etapa.
    """
    orden = np.lexsort((tiempo, ent_idx))       # primero por entidad, luego por tiempo
    e = ent_idx[orden]
    # inicio de cada grupo dentro del array ordenado
    nuevo_grupo = np.r_[True, e[1:] != e[:-1]]
    inicio = np.repeat(np.flatnonzero(nuevo_grupo), np.diff(
        np.r_[np.flatnonzero(nuevo_grupo), len(e)]))
    previas = np.empty(len(e), dtype=np.int64)
    previas[orden] = np.arange(len(e)) - inicio
    return previas


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
    # Se guardan CRUDAS de momento: la normalización va al final, cuando ya
    # estén también las `__grado_*` que se calculan más abajo. Normalizar aquí
    # dejaría esas cinco columnas sin tipificar — lo cazó el test la primera vez.
    data["transaction"].x = torch.tensor(df[feature_cols].values,
                                         dtype=torch.float32)
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

    por_entidad = int(cfg["graph"].get("vecinos_por_entidad", 10))
    log.info("Grafo heterogéneo sobre %d transacciones | %s | el muestreo baja "
             "%d vecinas por entidad",
             n_txn,
             f"poda de entidad > {tope}" if tope > 0 else "SIN poda por grado",
             por_entidad)
    grados_extra: dict[str, np.ndarray] = {}

    # MESES DEL GRAFO. Por defecto, solo los que alguna ventana usa. Construir
    # entidades y aristas sobre meses que nadie mira cuesta tiempo y memoria,
    # pero sobre todo METE FUTURO: el `__grado_*` contaba las transacciones de
    # la entidad en los SEIS meses, así que una fila del mes 1 llevaba un grado
    # que incluía su actividad del mes 6. Los nodos de transacción se conservan
    # todos (node_idx es el índice de fila de full.parquet, contrato implícito);
    # lo que se limita son las ARISTAS y los grados.
    meses_grafo = cfg["graph"].get("meses")
    if not meses_grafo:
        from src.utils.ventanas import verificar
        v = verificar(cfg, df["month"].values, df["week_in_month"].values)
        import numpy as _np
        meses_grafo = sorted(_np.unique(
            df["month"].values[_np.logical_or.reduce(list(v.values()))]).tolist())
    en_juego = df["month"].isin(meses_grafo).values
    log.info("Grafo sobre los meses %s: %d de %d transacciones "
             "(el resto queda sin aristas)",
             meses_grafo, int(en_juego.sum()), len(df))

    for nombre, spec in cfg["graph"]["entidades"].items():
        claves = clave_entidad(df, spec)
        presentes = claves.notna() & en_juego
        if not presentes.any():
            log.warning("Entidad '%s': ninguna fila la tiene, se omite", nombre)
            continue

        codigos, unicas = pd.factorize(claves[presentes], sort=False)
        filas_pres = np.where(presentes.values)[0]

        # E2 — PODA POR GRADO MÁXIMO. Con `max_entity_degree: 0` no se poda y
        # el grafo queda con TODAS sus conexiones, que es lo recomendado.
        #
        # La justificación original era el over-smoothing: una entidad con miles
        # de transacciones tendría como vector la media del dataset y se la
        # repartiría a todos sus vecinos. Pero eso ya lo impide el MUESTREO:
        # `vecinos_por_entidad: 10` hace que el vector de la entidad se calcule
        # SIEMPRE con 10 transacciones, sean 30 o 4.887 las que tenga. Una
        # tarjeta enorme y una pequeña producen vectores igual de específicos.
        #
        # Y podar cuesta caro. Medido con `max_entity_degree: 500`:
        #   · 72 entidades de las 10.106 de `card` (el 0,7%) dejaban sin arista
        #     al 24,6% del dataset — 54.274 transacciones
        #   · esas transacciones tenían MÁS fraude que la media (3,39% vs 3,11%)
        #   · y `__grado_card` se les quedaba en 0, el MISMO valor que una
        #     transacción sin card1: la red no distinguía "no tengo tarjeta" de
        #     "mi tarjeta lleva 3.000 compras". La feature quedaba invertida
        #     justo en las entidades con más historial.
        #
        # Si al quitarla el resultado empeora, la causa serán las entidades que
        # NO son una identidad real (prepago compartida, valores por defecto):
        # ahí los 10 vecinos son desconocidos. Pero eso pide un filtro de
        # calidad de entidad, no un tope por número — el tope mantiene las 501
        # primeras conexiones a esa misma entidad ruidosa y corta las demás.
        #
        # Cuando se poda, el corte es POR TRANSACCIÓN y en su momento, no de la
        # entidad entera. Antes se sumaban todas sus filas del periodo y, si el
        # total pasaba del tope, se borraba la entidad COMPLETA: una tarjeta con
        # 12 compras en enero y 600 en febrero perdía también las de enero, que
        # eran perfectamente normales. Futuro decidiendo el pasado.
        #
        # `previas` es el número de transacciones ANTERIORES de esa entidad, así
        # que sirve para las tres cosas: este corte, el grado mínimo de más
        # abajo y el `__grado_*`. Se calcula una sola vez.
        previas = _previas_por_entidad(
            codigos, df["TransactionDT"].values[filas_pres])
        vivas = (previas <= tope) if tope > 0 else np.ones(len(previas), bool)
        n_podadas = int((~vivas).sum())
        n_ent_afectadas = int(len(np.unique(codigos[~vivas]))) if n_podadas else 0

        filas = filas_pres[vivas]
        ent_idx = codigos[vivas]
        previas_vivas = previas[vivas]
        # renumerar tras la poda para no dejar huecos
        remap, ent_idx = np.unique(ent_idx, return_inverse=True)

        et = ("transaction", f"en_{nombre}", nombre)
        rev = (nombre, f"tiene_{nombre}", "transaction")
        src = torch.tensor(filas, dtype=torch.long)
        dst = torch.tensor(ent_idx, dtype=torch.long)

        # GRADO MÍNIMO, y ASIMÉTRICO. Una entidad con una sola transacción no
        # conecta a nadie: su vector es esa misma transacción, y al bajar se la
        # devuelve. La transacción se recibe a sí misma. En `uid` le pasaba a
        # 59.189 de 94.901 nodos (62%), el 26,8% del dataset.
        #
        # Las dos direcciones NO se podan igual, y esto es lo importante:
        #
        #   SUBIDA  transaction -> entidad    se conservan TODAS
        #           si se quitara la primera compra de un cliente, las
        #           siguientes nunca sabrían de ella
        #
        #   BAJADA  entidad -> transaction    solo si `previas >= minimo`
        #           una transacción sin ninguna anterior en su entidad no
        #           tiene de quién enterarse: lo único que recibiría es su
        #           propio eco
        #
        # CAUSAL por la misma razón que la poda de arriba: se mira cuántas
        # había ANTES, no cuántas acabará teniendo la entidad. Contar el total
        # dejaría que la actividad de junio decidiera si una compra de enero
        # tiene vecinos. `previas == 0` es exactamente "soy la primera".
        minimo = int(cfg["graph"].get("min_previas_entidad", 1))
        recibe = previas_vivas >= minimo
        n_sin_vecinos = int((~recibe).sum())

        data[et].edge_index = torch.stack([src, dst])
        data[rev].edge_index = torch.stack([dst[recibe], src[recibe]])
        data[nombre].num_nodes = int(len(remap))

        # GRADO de la entidad como feature del nodo: cuántas transacciones
        # tiene la entidad a la que pertenece esta transacción, ANTES de capar
        # a 10 en el muestreo. Es lo que le permite a la red aprender lo que
        # daban las columnas C (conteos por entidad).
        if cfg["graph"].get("features_derivadas", True):
            # CAUSAL: cuántas transacciones ANTERIORES tiene la entidad, no
            # cuántas tiene en total. El conteo total mira al futuro —una fila
            # de enero llevaba el grado que su tarjeta alcanzaría en junio— y
            # eso es una fuga que no rompe nada y falsea los resultados.
            grado = previas_vivas          # ya calculado para la poda
            col = np.zeros(n_txn, dtype=np.float32)
            col[filas] = np.log1p(grado)          # log: los hubs no dominan
            grados_extra[f"__grado_{nombre}"] = col

        # Contra las transacciones EN JUEGO, no contra las 582.429 del dataset:
        # los meses fuera de las ventanas están excluidos por diseño y dividir
        # por ellos hacía parecer que uid cubría un 34% cuando cubre el 89%.
        cob = len(filas) / max(int(en_juego.sum()), 1)
        meta["entidades"][nombre] = {
            "columnas": spec["cols"], "usa_d1": bool(spec.get("usa_d1")),
            "n_nodos": int(len(remap)), "n_aristas": int(len(filas)),
            "cobertura": round(cob, 4),
            # Ahora se podan TRANSACCIONES (las que llegan pasado el tope),
            # no entidades enteras. `entidades_afectadas` dice a cuántas.
            "txn_podadas_por_grado": n_podadas,
            "entidades_afectadas": n_ent_afectadas,
            "txn_por_entidad_media": round(len(filas) / max(len(remap), 1), 1),
            "degenerada": bool(len(filas) / max(len(remap), 1) < 2),
            # Aristas de BAJADA que se quitaron por no tener ninguna
            # transacción anterior en su entidad (ver `min_previas_entidad`).
            "aristas_subida": int(len(filas)),
            "aristas_bajada": int(recibe.sum()),
            "sin_anteriores": n_sin_vecinos,
        }
        media = len(filas) / max(len(remap), 1)
        log.info("  %-7s %7d nodos | subida %7d / bajada %7d aristas | "
                 "cobertura %5.1f%% | %4.1f txn/entidad | %d podadas por grado "
                 "en %d entidades | %d sin anteriores",
                 nombre, len(remap), len(filas), int(recibe.sum()), 100 * cob,
                 media, n_podadas, n_ent_afectadas, n_sin_vecinos)
        # Una entidad con ~1 transacción por grupo NO CONECTA NADA: cada
        # transacción cuelga de su propio nodo y el paso de mensajes no
        # transporta información de nadie. Suele significar que la clave es
        # demasiado específica (p.ej. añadir card1 a algo que ya era casi único).
        if media < 2:
            log.warning("    '%s' es DEGENERADA (%.1f txn/entidad): la clave es "
                        "demasiado específica y no conecta transacciones entre "
                        "sí. Revisa sus columnas o quítala de config.", nombre, media)
        # El extremo contrario: pocos grupos enormes conectan medio grafo con el
        # otro medio. Solo tiene sentido avisar SI HAY TOPE — sin él, `tope / 2`
        # vale 0 y la condición se cumplía siempre: el aviso saltaba en las cinco
        # entidades a la vez, incluida `uid` con 2,1 txn/entidad, que es
        # justamente lo contrario de gruesa. Un aviso que salta siempre enseña a
        # ignorarlo.
        elif tope > 0 and media > tope / 2:
            log.warning("    '%s' es muy GRUESA (%.1f txn/entidad): puede que la "
                        "poda por grado esté haciendo casi todo el trabajo.",
                        nombre, media)

        # Sin tope, el riesgo cambia de forma. Ya no es el over-smoothing —el
        # muestreo baja 10 vecinas y punto— sino QUÉ 10: con miles de candidatas,
        # "las 10 más recientes anteriores" pueden caer todas en la misma hora y
        # describir un pico de actividad en vez del comportamiento del cliente.
        # Es informativo, no un fallo: se mide comparando corridas.
        gigantes = int((np.bincount(ent_idx) > 50 * por_entidad).sum())
        if gigantes:
            log.info("    '%s': %d entidades con más de %d transacciones. El "
                     "muestreo bajará solo %d, muy juntas en el tiempo.",
                     nombre, gigantes, 50 * por_entidad, por_entidad)

    if grados_extra:
        extra = np.stack([grados_extra[k] for k in sorted(grados_extra)], axis=1)
        data["transaction"].x = torch.cat(
            [data["transaction"].x, torch.tensor(extra, dtype=torch.float32)], dim=1)
        feature_cols = feature_cols + sorted(grados_extra)
        log.info("Grados de entidad añadidos como features: %s",
                 sorted(grados_extra))

    # ── NORMALIZACIÓN, con TODAS las columnas ya presentes ──────────────────
    # Va aquí y no al cargar el grafo a propósito: si se hiciera al entrar a la
    # red habría que acordarse en cada punto de entrada (`train_gnn`, `embed`, y
    # el CL cuando vuelva), y uno que se olvide entrenaría con datos crudos sin
    # que nadie se entere. Es el mismo patrón que ya falló con `sin_aristas`.
    #
    # Y va DESPUÉS de añadir los `__grado_*`: normalizarlo antes dejaba esas
    # cinco columnas sin tipificar, y `__grado_card` se quedaba con el 6,5% de
    # la varianza. Lo cazó `tests/test_normalizacion.py` en su primera pasada.
    #
    # Sin esto, `id_02` se llevaba el 99,55% de la varianza y la red no veía
    # nada más. Ver src/data/normalizacion.py para el diagnóstico completo.
    from src.data.normalizacion import normalizar
    from src.utils.ventanas import mascara
    _crudo = data["transaction"].x.numpy()
    _entrena = mascara(cfg, "gnn_entrena", df["month"].values,
                       df["week_in_month"].values)
    _norm, _par_norm = normalizar(_crudo, feature_cols, _entrena, log=log)
    data["transaction"].x = torch.tensor(_norm, dtype=torch.float32)
    # El crudo se conserva para AUDITAR qué se transformó y para probar otra
    # normalización sin reconstruir el grafo. Un grafo normalizado sin su
    # original es una caja negra.
    data["transaction"].x_crudo = torch.tensor(_crudo, dtype=torch.float32)

    meta["feature_cols_gnn"] = feature_cols
    meta["n_features_gnn"] = len(feature_cols)
    # Los parámetros de normalización viajan con el grafo: sin ellos no se
    # puede saber qué se le hizo a cada columna ni deshacerlo.
    meta["normalizacion"] = _par_norm

    # --- las aristas deben ir ORDENADAS por tiempo del origen ---------------
    # `temporal_strategy="last"` (gnn/sampling.py) exige que, para cada nodo
    # destino, sus aristas entrantes estén ordenadas por el tiempo del origen.
    # Si no lo están, "los N más recientes" devuelve cualquier cosa.
    for nombre in list(meta["entidades"]):
        rev = (nombre, f"tiene_{nombre}", "transaction")
        ei = data[rev].edge_index
        t = data["transaction"].time[ei[1]]
        # stable=True: dos transacciones de la MISMA entidad pueden compartir
        # TransactionDT (son segundos). Sin estabilidad, el desempate varía entre
        # corridas y con él qué vecinos baja `temporal_strategy="last"`.
        orden = torch.argsort(ei[0] * (int(t.max()) + 1) + t, stable=True)
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
