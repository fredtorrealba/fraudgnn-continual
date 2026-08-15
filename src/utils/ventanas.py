"""
Ventanas temporales del experimento: qué filas hace cada trabajo.

POR QUÉ EXISTE ESTE ARCHIVO
El diseño original repartía por MESES y usaba los meses 1-4 para entrenar la GNN
*y* las cabezas a la vez. Eso obligaba al OOF: como la red memorizaba esos meses,
había que entrenar K redes y que cada una describiera el trozo que no vio.

El OOF resolvía la memorización pero creaba otro problema peor: cada una de las K
redes aprende sus PROPIOS pesos, así que la dimensión 7 de su embedding mide algo
distinto en cada una. XGBoost aprendía cortes sobre unos ejes y los aplicaba
sobre otros. Medido: `gnn_mas_tabular` cortó por early stopping en 2 árboles
contra los 517 de `control`, y su PR-AUC cayó de 0.4803 a 0.1159.

La solución es separar las ventanas: la GNN entrena en un bloque y las cabezas en
OTRO. Así una sola red describe todo lo que las cabezas ven, nunca lo memorizó, y
las 32 columnas significan lo mismo en toda la tabla. El OOF sobra.

LOS CINCO BLOQUES, cada uno con UN solo trabajo:

    gnn_entrena       entrena las 6 redes
    gnn_valida        elige cuál gana
    cabezas_entrenan  entrena las 3 cabezas XGBoost
    cabezas_validan   nº de árboles y umbral
    examen            el número final; no se toca hasta el informe

Ninguno se solapa: `verificar()` lo comprueba y aborta si se cruzan.
"""
import numpy as np


def _como_lista(spec):
    """Un bloque puede ser un dict o una lista de dicts (para cruzar meses)."""
    if spec is None:
        return []
    return spec if isinstance(spec, list) else [spec]


def mascara(cfg: dict, bloque: str, meses, semanas) -> np.ndarray:
    """
    Máscara booleana de las filas que pertenecen a `bloque`.

    `meses` y `semanas` son los arrays de la tabla (o del grafo): mes 1-6 y
    semana 1-4 dentro del mes.
    """
    v = (cfg.get("ventanas") or {}).get(bloque)
    if v is None:
        raise KeyError(f"Falta ventanas.{bloque} en config.yaml")
    meses = np.asarray(meses)
    semanas = np.asarray(semanas)
    out = np.zeros(len(meses), dtype=bool)
    for parte in _como_lista(v):
        m = meses == int(parte["mes"])
        s = parte.get("semanas")
        if s:
            m &= np.isin(semanas, [int(x) for x in s])
        out |= m
    return out


def todas(cfg: dict) -> list[str]:
    return list((cfg.get("ventanas") or {}).keys())


def verificar(cfg: dict, meses, semanas, log=None) -> dict:
    """
    Comprueba que los bloques NO se solapan y devuelve su tamaño.

    Un solapamiento significa que unas filas hacen dos trabajos: entrenar y
    examinar, por ejemplo. Es el fallo que invalida un experimento sin que nada
    se rompa, así que se aborta.
    """
    nombres = [n for n in todas(cfg) if n != "activo"]
    ms = {n: mascara(cfg, n, meses, semanas) for n in nombres}
    for i, a in enumerate(nombres):
        for b in nombres[i + 1:]:
            cruce = int((ms[a] & ms[b]).sum())
            if cruce:
                raise SystemExit(
                    f"Las ventanas '{a}' y '{b}' comparten {cruce:,} filas. "
                    f"Cada bloque debe hacer UN solo trabajo: revisa "
                    f"`ventanas` en config.yaml.")
    if log:
        for n in nombres:
            log.info("  ventana %-18s %7d filas", n, int(ms[n].sum()))
        fuera = int((~np.logical_or.reduce(list(ms.values()))).sum())
        log.info("  %d filas fuera de todas las ventanas (meses reservados)", fuera)
    return {n: ms[n] for n in nombres}


def mascaras_grafo(cfg: dict, data, log=None) -> dict:
    """
    Las máscaras de todos los bloques, como tensores booleanos del grafo.

    El grafo guarda `month` y `week_in_month` por transacción, así que no hace
    falta volver a leer el parquet.
    """
    import torch

    m = data["transaction"].month.numpy()
    w = data["transaction"].week_in_month.numpy()
    return {k: torch.from_numpy(v)
            for k, v in verificar(cfg, m, w, log).items()}
