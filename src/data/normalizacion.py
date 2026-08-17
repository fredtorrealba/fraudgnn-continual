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

  1. ESCALA — 33 columnas son cantidades reales (importes, distancias, tiempos)
     con colas muy pesadas. Se les aplica log con signo y luego z-score.

  2. ORDEN INVENTADO — 37 columnas son categóricas que `preprocessing` codificó
     como enteros por orden ALFABÉTICO:

         anonymous.com  2      gmail.com  17      outlook.com  36
         aol.com        3      hotmail.com 20     yahoo.com    54

     La red lee 17 y 20 como "parecidos" y 17 y 54 como "lejanos". Es un orden
     que no existe. Se sustituyen por CODIFICACIÓN DE FRECUENCIA: cuántas veces
     aparece ese valor. Eso sí es una cantidad con sentido ("tarjeta muy usada"
     contra "tarjeta nueva") y generaliza a valores nunca vistos, que reciben 0.

     Es lo que hizo el 2º lugar de Kaggle sobre este mismo dataset: «muchos
     valores de card1 solo aparecen en test; descarté la original y me quedé con
     la codificación de frecuencia».

TODO SE AJUSTA CON `gnn_entrena` Y SE APLICA A TODO. Medias, desviaciones y
tablas de frecuencia salen de los días 1-15 y se aplican a las 220.806 filas.
Calcularlas sobre el examen sería la misma fuga que evitan las ventanas.

NO TOCA A XGBOOST. Las cabezas leen `full.parquet`; el grafo lleva su propia
copia transformada. El `control` no se mueve un decimal y la comparación sigue
siendo válida.
"""
import numpy as np

# `id_02` NO va por frecuencia aunque sea de alta cardinalidad: el 88% de sus
# valores aparecen una sola vez, así que la frecuencia la dejaría casi binaria
# (un valor dominante con 148.734 filas y una cola de únicos). Va por el camino
# de las cantidades.
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


def ajustar(X: np.ndarray, cols: list[str], entrena: np.ndarray) -> dict:
    """
    Calcula los parámetros con las filas de `entrena` y NADA MÁS.

    Devuelve un dict serializable a JSON: va a `graph_meta.json` para que la
    transformación sea auditable y reversible. Sin eso, un grafo normalizado es
    una caja negra — no se puede comprobar qué se le hizo ni deshacerlo.
    """
    par = {"metodo": "frecuencia | log-con-signo + z-score",
           "ajustado_con": "gnn_entrena", "n_filas_ajuste": int(entrena.sum()),
           "columnas": {}}
    for i, c in enumerate(cols):
        x = X[:, i].astype(np.float64)
        if c in CATEGORICAS and c not in NO_FRECUENCIA:
            # Tabla de frecuencias contada SOLO en entrena. Un valor que no
            # aparezca ahí recibirá 0: "nunca visto", que es informativo.
            vals, cnt = np.unique(x[entrena], return_counts=True)
            f = np.zeros(len(x))
            idx = np.searchsorted(vals, x)
            dentro = (idx < len(vals)) & (vals[np.clip(idx, 0, len(vals)-1)] == x)
            f[dentro] = cnt[idx[dentro]]
            base = _log_con_signo(f)          # las frecuencias también tienen cola
            tipo = "frecuencia"
            extra = {"n_valores": int(len(vals))}
        else:
            base = _log_con_signo(x)
            tipo = "cantidad"
            extra = {}
        mu, sd = float(base[entrena].mean()), float(base[entrena].std())
        par["columnas"][c] = {"tipo": tipo, "media": mu,
                              "std": sd if sd > 1e-12 else 1.0, **extra}
    return par


def aplicar(X: np.ndarray, cols: list[str], par: dict,
            tablas: dict | None = None) -> np.ndarray:
    """Aplica los parámetros de `ajustar` a TODAS las filas."""
    Z = np.empty_like(X, dtype=np.float32)
    for i, c in enumerate(cols):
        p = par["columnas"][c]
        x = X[:, i].astype(np.float64)
        base = _log_con_signo(tablas[c] if p["tipo"] == "frecuencia" else x)
        Z[:, i] = ((base - p["media"]) / p["std"]).astype(np.float32)
    return Z


def normalizar(X: np.ndarray, cols: list[str], entrena: np.ndarray,
               log=None) -> tuple[np.ndarray, dict]:
    """
    Ajusta con `entrena` y devuelve (X normalizado, parámetros).

    Es el único punto de entrada: hacerlo en `build_graph` y no al cargar el
    grafo significa que NINGÚN camino de código puede saltárselo. `train_gnn`,
    `embed` y —cuando vuelva— el CL leen el mismo `graph.pt` ya transformado.
    """
    par = ajustar(X, cols, entrena)
    tablas = {}
    for i, c in enumerate(cols):
        if par["columnas"][c]["tipo"] != "frecuencia":
            continue
        x = X[:, i].astype(np.float64)
        vals, cnt = np.unique(x[entrena], return_counts=True)
        f = np.zeros(len(x))
        idx = np.searchsorted(vals, x)
        dentro = (idx < len(vals)) & (vals[np.clip(idx, 0, len(vals)-1)] == x)
        f[dentro] = cnt[idx[dentro]]
        tablas[c] = f
    Z = aplicar(X, cols, par, tablas)

    if log:
        n_f = sum(1 for c in cols if par["columnas"][c]["tipo"] == "frecuencia")
        var = Z.var(0); o = np.argsort(-var)
        log.info("Normalización: %d columnas por frecuencia, %d por "
                 "log-con-signo + z-score (ajustado con %d filas de gnn_entrena)",
                 n_f, len(cols) - n_f, int(entrena.sum()))
        log.info("  varianza: la mayor columna era el %.2f%% del total y ahora "
                 "es el %.2f%% | |x| max %.1f",
                 100 * X.var(0).max() / X.var(0).sum(),
                 100 * var[o[0]] / var.sum(), float(np.abs(Z).max()))
    return Z, par
