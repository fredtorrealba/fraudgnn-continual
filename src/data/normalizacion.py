"""
Normalización de las features que entran a la GNN.

POR QUÉ EXISTE ESTE ARCHIVO. Hasta el 16/08 las features iban CRUDAS al grafo, y
una sola columna se llevaba el 99,55% de la varianza:

    id_02          [ 1.058 , 999.595 ]   std 73.420
    TransactionAmt [   0,3 ,   5.279 ]   std    211
    la mediana                            rango 19,5

A un ÁRBOL eso le da igual: parte por umbrales y solo usa el ORDEN, así que es
invariante a la escala (verificado: XGBoost da predicciones bit-idénticas con y
sin z-score). A una RED no: la primera capa calcula `W·x`, y un componente
13.000 veces mayor que el resto domina la suma y acapara el gradiente.

Síntomas que dejó, y que se leyeron durante semanas sin entenderlos:

    bns.1.running_var  hasta 579.055     BatchNorm domando la explosión
    best_epoch: 2 de 50                  tocaba techo enseguida
    gnn_sola ROC 0,4540 en `habitual`    por debajo del azar justo donde
                                         más vecinos tiene

DOS PROBLEMAS DISTINTOS, DOS TRATAMIENTOS

  1. ESCALA — las columnas que son cantidades reales (importes, distancias,
     tiempos) tienen colas muy pesadas. Se les aplica log con signo, z-score y
     un CLIP en ±10 desviaciones: el log baja el peor valor de 96 a 44 sigmas,
     pero 44 sigmas siguen bastando para acaparar un gradiente.

  2. ORDEN INVENTADO — las categóricas que `preprocessing` codificó como
     enteros por orden ALFABÉTICO:

         anonymous.com  2      gmail.com  17      outlook.com  36
         aol.com        3      hotmail.com 20     yahoo.com    54

     La red lee 17 y 20 como "parecidos" y 17 y 54 como "lejanos". Es un orden
     que no existe. Se sustituyen por FRECUENCIA CAUSAL RELATIVA: qué fracción
     de las transacciones ANTERIORES llevaba ese mismo valor. Eso sí es una
     cantidad con sentido ("tarjeta muy usada" contra "tarjeta nueva").

     Es la idea del 2º lugar de Kaggle sobre este mismo dataset («descarté
     card1 y me quedé con su codificación de frecuencia»), con dos cambios que
     esta ventana exige:

       CAUSAL     la tabla estática contada en `gnn_entrena` (días 1-15)
                  CADUCA: una tarjeta que aparece 300 veces entre los días
                  16 y 52 llegaba al examen con frecuencia 0, indistinguible
                  de una tarjeta nueva. Contar solo lo anterior a cada fila
                  no caduca nunca y no puede mirar el futuro. Es el mismo
                  principio que `previas_por_grupo` aplica a los `__grado_*`.

       RELATIVA   el conteo acumulado crece de forma monótona con el tiempo:
                  "ProductCD=W lleva 40.000 apariciones" en el examen queda
                  fuera de todo el rango visto al ajustar el z-score — la
                  misma no-estacionariedad que mató a `__pos_temporal`. La
                  FRACCIÓN es estacionaria: si W es el 70% del tráfico, vale
                  ~0,7 en enero y en junio.

TODO SE AJUSTA CON `gnn_entrena` Y SE APLICA A TODO. Medias y desviaciones
salen de los días 1-15 y se aplican a las 220.806 filas. La frecuencia causal
no se "ajusta": es un hecho por fila, como los `__grado_*`.

NO TOCA A XGBOOST. Las cabezas leen `full.parquet`; el grafo lleva su propia
copia transformada. El `control` no se mueve un decimal y la comparación sigue
siendo válida.
"""
import numpy as np

# Tras log + z-score el peor valor real queda en 44 desviaciones: suficiente
# para acaparar gradiente. ±10 no toca al 99,99% de los valores y le pone techo
# al resto. Va en los parámetros de graph_meta.json para que sea auditable.
CLIP = 10.0

# `id_02` NO va por frecuencia aunque sea de alta cardinalidad: el 88% de sus
# valores aparecen una sola vez, así que la frecuencia la dejaría casi binaria.
# Va por el camino de las cantidades.
NO_FRECUENCIA = {"id_02"}

# Las que `preprocessing` codificó desde texto o son identificadores: su número
# es un NOMBRE, no una cantidad.
CATEGORICAS = {
    "ProductCD", "card1", "card2", "card3", "card4", "card5", "card6",
    "addr1", "addr2", "P_emaildomain", "R_emaildomain",
    "M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9",
    "DeviceType", "DeviceInfo",
    "id_12", "id_15", "id_16", "id_23", "id_27", "id_28", "id_29",
    "id_30", "id_31", "id_33", "id_34", "id_35", "id_36", "id_37", "id_38",
}


def previas_por_grupo(grupo_idx: np.ndarray, tiempo: np.ndarray) -> np.ndarray:
    """
    Cuántas filas ANTERIORES tiene cada fila dentro de su grupo.

    El conteo total (`bincount`) mira al futuro: una transacción de enero
    llevaba el número de compras que su tarjeta acumularía hasta junio. Es una
    fuga que no rompe nada y falsea el resultado, justo el tipo que hay que
    cazar a mano.

    Se ordena por (grupo, tiempo) y la posición dentro del grupo ES el número
    de anteriores. Vectorizado: con 500.000 filas un bucle en Python tardaría
    más que el resto de la etapa. `lexsort` es estable, así que los empates de
    TransactionDT se desempatan por orden de fila — el mismo criterio que
    `_delta_tarjeta` y el orden de aristas.

    Vive aquí (y no en build_graph, donde nació como `_previas_por_entidad`)
    porque lo usan tres consumidores: los `__grado_*` del grafo, la frecuencia
    causal de esta normalización y el reparto conocidos/nuevos del informe.
    build_graph lo importa; al revés sería un import circular.
    """
    orden = np.lexsort((tiempo, grupo_idx))     # primero por grupo, luego por tiempo
    e = grupo_idx[orden]
    # inicio de cada grupo dentro del array ordenado
    nuevo_grupo = np.r_[True, e[1:] != e[:-1]]
    inicio = np.repeat(np.flatnonzero(nuevo_grupo), np.diff(
        np.r_[np.flatnonzero(nuevo_grupo), len(e)]))
    previas = np.empty(len(e), dtype=np.int64)
    previas[orden] = np.arange(len(e)) - inicio
    return previas


def _log_con_signo(x: np.ndarray) -> np.ndarray:
    """
    `sign(x) · log1p(|x|)` — comprime las colas SIN romperse con los negativos.

    `log1p(x)` a secas no vale aquí: 16 de las 70 columnas tienen valores
    negativos (`id_14` llega a −660, y `-1` es el centinela de categoría no
    vista), así que daría NaN en las 220.806 filas.

    Medido sobre estos datos: baja el peor valor tipificado de 96,2 a 44,0
    desviaciones y quita 1.824 filas con extremos. El reparto de varianza queda
    igual de bueno.
    """
    return np.sign(x) * np.log1p(np.abs(x))


def frecuencia_causal(x: np.ndarray, tiempo: np.ndarray) -> np.ndarray:
    """
    Fracción de las transacciones ANTERIORES que llevaba este mismo valor.

    En [0,1]. La primera aparición de un valor da 0 ("nunca visto hasta ahora",
    que es informativo); las primeras filas del dataset tienen denominadores
    pequeños y son ruidosas, pero son pocas y solo caen en `gnn_entrena`.

    No necesita log: una proporción ya está acotada.
    """
    codigos = np.unique(x, return_inverse=True)[1]
    previas = previas_por_grupo(codigos, tiempo)
    # posición de cada fila en el orden temporal = cuántas filas hay antes
    orden = np.argsort(tiempo, kind="stable")
    pos = np.empty(len(x), dtype=np.int64)
    pos[orden] = np.arange(len(x))
    return previas / np.maximum(pos, 1)


def transformar_base(X: np.ndarray, cols: list[str],
                     tiempo: np.ndarray) -> tuple[np.ndarray, dict[str, str]]:
    """
    La transformación por columna, ANTES del z-score.

    Separada de `normalizar` para que los tests puedan reconstruirla desde
    `x_crudo` + `graph_meta.json` y comparar: un grafo normalizado que no se
    puede recomputar es una caja negra.
    """
    B = np.empty(X.shape, dtype=np.float64)
    tipos: dict[str, str] = {}
    for i, c in enumerate(cols):
        x = X[:, i].astype(np.float64)
        if c in CATEGORICAS and c not in NO_FRECUENCIA:
            B[:, i] = frecuencia_causal(x, tiempo)
            tipos[c] = "frecuencia_causal"
        else:
            B[:, i] = _log_con_signo(x)
            tipos[c] = "cantidad"
    return B, tipos


def normalizar(X: np.ndarray, cols: list[str], entrena: np.ndarray,
               tiempo: np.ndarray, log=None) -> tuple[np.ndarray, dict]:
    """
    Ajusta el z-score con `entrena` y devuelve (X normalizado, parámetros).

    Es el único punto de entrada: hacerlo en `build_graph` y no al cargar el
    grafo significa que NINGÚN camino de código puede saltárselo. `train_gnn`,
    `embed` y —cuando vuelva— el CL leen el mismo `graph.pt` ya transformado.

    Los parámetros van a `graph_meta.json` para que la transformación sea
    auditable y reversible.
    """
    B, tipos = transformar_base(X, cols, tiempo)
    par = {"metodo": "frecuencia-causal-relativa | log-con-signo + z-score",
           "ajustado_con": "gnn_entrena", "n_filas_ajuste": int(entrena.sum()),
           "clip": CLIP, "columnas": {}}
    Z = np.empty(X.shape, dtype=np.float32)
    for i, c in enumerate(cols):
        mu = float(B[entrena, i].mean())
        sd = float(B[entrena, i].std())
        sd = sd if sd > 1e-12 else 1.0
        Z[:, i] = np.clip((B[:, i] - mu) / sd, -CLIP, CLIP).astype(np.float32)
        par["columnas"][c] = {"tipo": tipos[c], "media": mu, "std": sd}

    if log:
        n_f = sum(1 for t in tipos.values() if t == "frecuencia_causal")
        var = Z.var(0); o = np.argsort(-var)
        log.info("Normalización: %d columnas por frecuencia causal relativa, "
                 "%d por log-con-signo; z-score ajustado con %d filas de "
                 "gnn_entrena, clip en ±%.0f",
                 n_f, len(cols) - n_f, int(entrena.sum()), CLIP)
        log.info("  varianza: la mayor columna era el %.2f%% del total y ahora "
                 "es el %.2f%% | |x| max %.1f",
                 100 * X.var(0).max() / X.var(0).sum(),
                 100 * var[o[0]] / var.sum(), float(np.abs(Z).max()))
    return Z, par
