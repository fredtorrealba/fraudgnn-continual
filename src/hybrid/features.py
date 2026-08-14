"""
Columnas estructurales del grafo para el modelo tabular.

Resumen numérico del vecindario de cada transacción, calculado con aritmética
simple sobre las relaciones de entidad. Es la mitad "sin red neuronal" del
sistema híbrido: lo que la GNN obtiene promediando vecinos, aquí se cuenta
explícitamente y se le entrega a XGBoost como columnas más.

DOS REGLAS QUE DEFINEN SU VALIDEZ
1. NUNCA se usa `isFraud`, ni propio ni de los vecinos. Cero riesgo de fuga de
   etiqueta, así que estas columnas valen para los 6 meses por igual.
2. Solo cuentan vecinos ANTERIORES en el tiempo (`dt[j] < dt[i]`). Un sistema
   en producción no puede ver transacciones que aún no han ocurrido; incluirlas
   inflaría los resultados con información del futuro.

DIFERENCIA CON LAS ARISTAS DEL GRAFO
Se calculan sobre la relación de entidad COMPLETA, sin el tope de
`graph.max_edges_per_node` (50). Dos motivos:
  - `n_vecinos` censurado a 50 destruiría justo la señal antifraude que
    interesa ("esta tarjeta aparece 500 veces este mes").
  - Así las columnas no dependen del tope: cambiarlo no invalida la cabeza
    XGBoost ya entrenada.

Salida: data/processed/graph_features.parquet, una fila por nodo en el MISMO
orden que full.parquet (node_idx == índice de fila).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.common import get_logger, resolve

log = get_logger("hybrid.features")

COLUMNAS = ["n_vecinos", "n_vecinos_card", "n_vecinos_email", "n_vecinos_device",
            "monto_medio_vecinos", "monto_max_vecinos",
            "horas_desde_vecino_previo", "n_entidades_distintas"]

TIPOS = ("card", "email", "device")


def _acumular(claves: pd.Series, dts: np.ndarray, montos: np.ndarray,
              ventana_s: int, n: int):
    """
    Recorre cada grupo de entidad y acumula, para cada nodo, los estadísticos
    de sus vecinos ANTERIORES dentro de la ventana temporal.

    Devuelve (conteo, suma_montos, max_montos, dt_del_vecino_mas_reciente).
    Ventana deslizante de dos punteros sobre el grupo ordenado por tiempo: cada
    nodo mira hacia atrás hasta salirse de la ventana, así que el coste es
    lineal en el número de pares dentro de ella.
    """
    cnt = np.zeros(n, dtype=np.int32)
    suma = np.zeros(n, dtype=np.float64)
    mx = np.full(n, np.nan, dtype=np.float64)
    ult = np.full(n, np.nan, dtype=np.float64)

    grupos = {}
    for idx, k in enumerate(claves.values):
        if not isinstance(k, str) or k.startswith("nan|") or k == "nan":
            continue
        grupos.setdefault(k, []).append(idx)

    for idxs in grupos.values():
        if len(idxs) < 2:
            continue
        idxs.sort(key=lambda i: dts[i])
        ini = 0
        for pos in range(1, len(idxs)):
            i = idxs[pos]
            while dts[i] - dts[idxs[ini]] > ventana_s:
                ini += 1
            if ini >= pos:
                continue
            previos = idxs[ini:pos]                 # todos anteriores a i
            m = montos[previos]
            cnt[i] += len(previos)
            suma[i] += m.sum()
            mx[i] = np.nanmax([mx[i], m.max()])
            ult[i] = np.nanmax([ult[i], dts[idxs[pos - 1]]])
    return cnt, suma, mx, ult


def construir(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Las 8 columnas estructurales, una fila por nodo en el orden de df."""
    from src.data.build_graph import entity_keys

    n = len(df)
    ventana_s = cfg["graph"]["window_days"] * 86400
    dts = df["TransactionDT"].values.astype(np.int64)
    montos = df["TransactionAmt"].values.astype(np.float64)
    claves = entity_keys(df, cfg)

    total_cnt = np.zeros(n, dtype=np.int32)
    total_suma = np.zeros(n, dtype=np.float64)
    total_max = np.full(n, np.nan)
    ult_global = np.full(n, np.nan)
    n_entidades = np.zeros(n, dtype=np.int8)
    por_tipo = {}

    for etype in TIPOS:
        cnt, suma, mx, ult = _acumular(claves[etype], dts, montos, ventana_s, n)
        por_tipo[etype] = cnt
        total_cnt += cnt
        total_suma += suma
        total_max = np.fmax(total_max, mx)
        ult_global = np.fmax(ult_global, ult)
        n_entidades += (cnt > 0).astype(np.int8)
        log.info("  vecinos previos tipo %-7s: %d nodos con al menos uno",
                 etype, int((cnt > 0).sum()))

    with np.errstate(invalid="ignore", divide="ignore"):
        medio = np.where(total_cnt > 0, total_suma / np.maximum(total_cnt, 1), np.nan)
        horas = (dts - ult_global) / 3600.0

    # SIN VECINOS PREVIOS -> centinela -1, no NaN.
    # XGBoost trataría los NaN nativamente como rama de faltantes, que sería lo
    # semánticamente correcto, pero SMOTE (imblearn) los rechaza y el baseline
    # lo usa. Imputar en un consumidor y no en otro haría que el modelo
    # entrenara sobre -1 y predijera sobre NaN, así que se resuelve aquí, de una
    # vez, para que TODOS vean lo mismo.
    # -1 no pierde información: `n_vecinos == 0` ya codifica "no tiene vecinos",
    # así que el árbol puede aprender a ignorar el resto en ese caso. Y es
    # imposible como monto o como horas, así que no se confunde con un valor
    # real.
    SIN_VECINOS = -1.0
    medio = np.where(total_cnt > 0, medio, SIN_VECINOS)
    total_max = np.where(total_cnt > 0, total_max, SIN_VECINOS)
    horas = np.where(np.isfinite(ult_global), horas, SIN_VECINOS)

    out = pd.DataFrame({
        "n_vecinos": total_cnt,
        "n_vecinos_card": por_tipo["card"],
        "n_vecinos_email": por_tipo["email"],
        "n_vecinos_device": por_tipo["device"],
        "monto_medio_vecinos": medio.astype(np.float32),
        "monto_max_vecinos": total_max.astype(np.float32),
        "horas_desde_vecino_previo": horas.astype(np.float32),
        "n_entidades_distintas": n_entidades,
    })
    aislados = int((total_cnt == 0).sum())
    log.info("Columnas estructurales: %d nodos sin vecino previo (%.1f%%)",
             aislados, 100 * aislados / n)
    return out


def ruta(cfg) -> Path:
    return resolve(cfg, "processed_dir") / "graph_features.parquet"


def load_struct(cfg) -> tuple[np.ndarray, list[str]]:
    """
    Matriz (N, 8) float32 alineada por índice de nodo, y los nombres.

    Los nodos sin vecino previo llevan -1 como centinela (ver `construir`): no
    hay NaN, para que SMOTE y XGBoost vean exactamente los mismos valores.
    """
    df = pd.read_parquet(ruta(cfg))
    return df[COLUMNAS].values.astype(np.float32), list(COLUMNAS)
