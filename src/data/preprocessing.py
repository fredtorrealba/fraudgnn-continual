"""
Paso 1 — Preprocesamiento del IEEE-CIS.

Hace tres cosas:
1. Merge de train_transaction + train_identity (left join por TransactionID).
2. Limpieza: encoding de categóricas, imputación de NAs, matriz final de
   432 features numéricas por transacción.
3. División temporal ESTRICTA (walk-forward, sin data leakage):
   meses 1-4 = TRAIN | mes 5 = VAL | mes 6 = TEST (dividido en semanas).

El mes se deriva de TransactionDT (segundos desde t0): mes = DT // (86400*30).
Nada del futuro se usa para transformar el pasado: los encoders y la
imputación se AJUSTAN solo con TRAIN y se aplican a VAL/TEST.

Salidas (data/processed/):
  full.parquet          dataset completo procesado + columnas month/week
  feature_cols.json     lista de columnas de features (orden fijo)
  split_masks.parquet   TransactionID -> split (train/val/test) y semana
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.common import ensure_dirs, get_logger, load_config, resolve

log = get_logger("preprocessing")

# Umbral para fabricar un flag de ausencia `<col>__na`. Por debajo, el flag es
# casi constante y no aporta; por encima, la imputación por mediana está
# borrando una señal real (la tabla identity falta en ~76% de las filas y ESO
# es información: el patrón de NaN de las V agrupa las columnas por origen).
UMBRAL_FLAG_NA = 0.20

# Columnas que se usan para construir ARISTAS del grafo: se conservan crudas
# (además de su versión codificada) porque el builder del grafo las necesita.
def edge_raw_cols(cfg: dict) -> list[str]:
    """
    Columnas que hay que preservar SIN codificar, para construir las entidades.

    Se derivan de `graph.entidades` en vez de escribirse a mano. La lista fija
    que había aquí se quedó atrás cuando se añadieron las entidades `device` y
    `net`: faltaban id_33 e id_13/17/19/20, y `build_graph` las omitía enteras
    con un WARNING fácil de pasar por alto. Dos de las cinco entidades no
    existían y el grafo se construía igualmente.

    Derivarlas del config significa que añadir una entidad nueva no obliga a
    tocar este archivo: si está en `graph.entidades`, su columna se preserva.
    """
    cols = []
    for spec in (cfg.get("graph", {}).get("entidades") or {}).values():
        for c in spec.get("cols", []):
            if c not in cols:
                cols.append(c)
    return cols


def load_raw(raw_dir: Path) -> pd.DataFrame:
    tx = pd.read_csv(raw_dir / "train_transaction.csv")
    ident = pd.read_csv(raw_dir / "train_identity.csv")
    df = tx.merge(ident, on="TransactionID", how="left")
    log.info("Merge: %d filas, %d columnas", *df.shape)
    # El merge deja el DataFrame repartido en muchos bloques internos. Con
    # 400+ columnas, CUALQUIER inserción posterior dispara PerformanceWarning.
    # Una copia aquí lo consolida de una vez y evita el aviso aguas abajo.
    return df.copy()


def _delta_tarjeta(df: pd.DataFrame) -> pd.Series:
    """
    Segundos desde la transacción anterior de la MISMA card1, en log1p.

    CAUSAL: `diff()` sobre el orden temporal mira solo hacia atrás, nunca al
    futuro. La primera compra de cada tarjeta no tiene anterior y queda en NaN:
    el flag `__tiene_anterior` guarda esa información y la imputación general
    la rellena con la mediana de train. Antes se marcaba con -1, pero para la
    GNN ese centinela es un punto más de la recta —tras normalizar queda pegado
    a "hace muy poco"— y mezclaba dos significados en una columna. Un árbol lo
    rodeaba partiendo en -0.5; la red no puede.

    NO REORDENA el DataFrame. El índice de fila de full.parquet es el índice de
    nodo del grafo (contrato implícito que protegen los asserts de hybrid/), así
    que se ordena una copia, se calcula, y se devuelve al orden original con
    reindex.
    """
    orden = df["TransactionDT"].astype("float64").argsort(kind="stable")
    tmp = df.iloc[orden]
    delta = tmp.groupby("card1")["TransactionDT"].diff()
    delta = delta.reindex(df.index)
    return np.log1p(delta)


def add_temporal_columns(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    spm = cfg["data"]["seconds_per_month"]
    # assign() en vez de dos asignaciones sueltas: insertar columnas una a una
    # en un DataFrame de 400+ columnas lo fragmenta y dispara PerformanceWarning.
    dt = df["TransactionDT"].astype("float64")
    hora = (dt % 86400) / 86400.0
    delta = _delta_tarjeta(df)
    return df.assign(
        # El reloj. Va AQUÍ y no en build_graph porque no tiene nada de grafo, y
        # porque calculado allí no llegaba al parquet: la GNN sabía la hora y las
        # cabezas tabulares no. Esa información entraba en `gnn_mas_tabular`
        # dentro del embedding, así que parte del "aporte del grafo" era el reloj.
        # Solo la CÍCLICA. `__pos_temporal` (posición en los 6 meses) se quitó:
        # train ocupa [0, 0.667] y validación [0.667, 0.834] — 0% de
        # solapamiento. Los árboles no extrapolan, así que fuera de la ventana
        # de entrenamiento se comporta como una constante; dentro, en cambio,
        # da un valor único por fila y permite memorizar. Medido: los meses
        # in-sample subieron de 0.53 a 0.86 de PR-AUC y el mes 5 no se movió.
        # La hora del día sí generaliza: se solapa al 99.9% entre splits.
        __hora_dia=hora,
        # La MISMA hora en seno/coseno, porque [0,1] rompe el ciclo justo en la
        # medianoche: 23:59 y 00:01 quedan en los extremos opuestos de la recta.
        # A un árbol le basta con partir dos veces; para `W·x` de la GNN esa
        # discontinuidad es artificial y cae de madrugada, donde vive el fraude.
        # La lineal se conserva: a los árboles les da cortes más simples.
        __hora_sin=np.sin(2 * np.pi * hora),
        __hora_cos=np.cos(2 * np.pi * hora),
        # Segundos desde la compra ANTERIOR de la misma tarjeta, en log.
        # Es lo que daba D1 y que la ablación quitó, y a diferencia de la
        # posición absoluta SÍ generaliza: "hace 2 horas" significa lo mismo en
        # enero que en junio, así que el valor cae dentro del rango aprendido.
        # NaN si no hay anterior; lo rellena la imputación general con la
        # mediana de train, y el flag de abajo conserva la distinción.
        __delta_anterior=delta,
        # "¿Existe una compra anterior de esta tarjeta?" — separado del delta
        # para que ni la red ni SMOTE confundan "no hay anterior" con un valor
        # real (ver _delta_tarjeta).
        __tiene_anterior=delta.notna().astype(np.float32),
        month=(df["TransactionDT"] // spm).astype(int) + 1,
        # Semana relativa dentro del mes (simula el "futuro que llega" en test)
        week_in_month=(((df["TransactionDT"] % spm) // (spm // 4)) + 1)
                      .clip(1, 4).astype(int),
    )


def encode_and_impute(df: pd.DataFrame, train_mask: pd.Series, cfg: dict):
    """Label encoding de categóricas + imputación, ajustados SOLO con train."""
    id_like = {"TransactionID", "isFraud", "TransactionDT", "month", "week_in_month"}
    feature_cols = [c for c in df.columns if c not in id_like]

    cat_cols = [c for c in feature_cols
                if not pd.api.types.is_numeric_dtype(df[c])]
    num_cols = [c for c in feature_cols if c not in cat_cols]

    # Guardar crudas para las aristas ANTES de codificar.
    # Se añaden TODAS de una vez con concat: en bucle serían ~10 inserciones
    # sucesivas sobre un DataFrame ancho, que es justo lo que pandas penaliza
    # con PerformanceWarning ("DataFrame is highly fragmented").
    pedidas = edge_raw_cols(cfg)
    crudas = {f"raw__{c}": df[c].astype(str).replace("nan", np.nan)
              for c in pedidas if c in df.columns}
    if crudas:
        df = pd.concat([df, pd.DataFrame(crudas, index=df.index)], axis=1)
    # Avisar AQUÍ, donde se puede arreglar, y no dos etapas más tarde cuando
    # build_graph descarta la entidad entera sin decir de dónde venía.
    ausentes = [c for c in pedidas if c not in df.columns]
    if ausentes:
        log.warning("Estas columnas las piden las entidades de config y NO "
                    "están en el dataset: %s. Las entidades que dependan de "
                    "ellas se omitirán en `graph`.", ausentes)

    # Categóricas: mapa de categorías aprendido en train; lo no visto -> -1
    for c in cat_cols:
        cats = df.loc[train_mask, c].astype(str).fillna("__NA__").unique()
        mapping = {v: i for i, v in enumerate(sorted(cats))}
        df[c] = df[c].astype(str).fillna("__NA__").map(mapping).fillna(-1).astype(np.int32)

    # FLAGS DE AUSENCIA, antes de imputar — la imputación por mediana coloca el
    # faltante en el centro de la distribución, que es donde más se confunde
    # con un valor real. Un árbol puede rodearlo; la red no distingue "no había
    # dato" de "el dato era la mediana". Las categóricas no lo necesitan: su
    # "__NA__" ya es una categoría propia.
    #
    # DEDUPLICADOS por patrón: las V comparten ~15 patrones de NaN en bloque
    # (vienen de las mismas tablas de origen — el 1er lugar de Kaggle las
    # agrupó exactamente así), y 300 flags idénticos no aportan nada y diluyen
    # los splits. Se emite UN flag por patrón, con el nombre de su primera
    # columna: `V12__na` representa a todo su bloque. El nombre hereda el
    # prefijo a propósito: así la ablación ["V","C","D"] se lleva también sus
    # flags y no devuelve por la ventana lo que quita la puerta.
    #
    # La TASA se mide en train (elegir columnas mirando el examen sería fuga);
    # el VALOR del flag es un hecho por fila y no se ajusta con nada.
    tr = train_mask.values
    vistos: dict[bytes, str] = {}
    flags: dict[str, np.ndarray] = {}
    for c in num_cols:
        if c.startswith("__"):          # derivadas propias: su ausencia ya
            continue                    # tiene flag dedicado (__tiene_anterior)
        na = df[c].isna().values
        if na[tr].mean() <= UMBRAL_FLAG_NA:
            continue
        patron = np.packbits(na).tobytes()
        if patron in vistos:
            continue
        vistos[patron] = c
        flags[f"{c}__na"] = na.astype(np.float32)
    if flags:
        df = pd.concat([df, pd.DataFrame(flags, index=df.index)], axis=1)
        feature_cols = feature_cols + list(flags)
        log.info("Flags de ausencia: %d patrones distintos entre las columnas "
                 "con más de %.0f%% de NaN en train -> %s%s",
                 len(flags), 100 * UMBRAL_FLAG_NA, sorted(flags)[:6],
                 " ..." if len(flags) > 6 else "")

    # Numéricas: mediana de train (el patrón de ausencia ya quedó en los flags)
    for c in num_cols:
        med = df.loc[train_mask, c].median()
        df[c] = df[c].fillna(0.0 if np.isnan(med) else med)

    df[num_cols] = df[num_cols].astype(np.float32)
    # Los dos bucles de arriba reescriben cientos de columnas una a una y
    # vuelven a fragmentar el frame; se consolida antes de devolverlo.
    return df.copy(), feature_cols


def main():
    cfg = load_config()
    ensure_dirs(cfg)
    raw_dir, out_dir = resolve(cfg, "raw_dir"), resolve(cfg, "processed_dir")

    df = load_raw(raw_dir)
    df = add_temporal_columns(df, cfg)

    # MESES EN JUEGO. Por defecto, los que usan las `ventanas`. Los demás se
    # descartan aquí y no llegan a existir: así ninguna estadística, mapa de
    # categorías ni mediana puede calcularse con ellos por descuido.
    from src.utils.ventanas import verificar
    v_all = verificar(cfg, df["month"].values, df["week_in_month"].values)
    usados = np.logical_or.reduce(list(v_all.values()))
    meses = cfg["data"].get("meses") or sorted(
        np.unique(df["month"].values[usados]).tolist())
    antes = len(df)
    df = df[df["month"].isin(meses)].reset_index(drop=True)
    log.info("Meses en juego %s: %d de %d transacciones (%d descartadas)",
             meses, len(df), antes, antes - len(df))

    # EL AJUSTE VA SOLO CON DATOS DE ENTRENAMIENTO. El codificador de
    # categóricas y las medianas de imputación son PARTE DEL MODELO: ajustarlos
    # con datos de validación o de examen es fuga.
    #
    # Antes se usaba `train_months` = meses 1-4. Con las ventanas, el bloque de
    # examen es el mes 2 semana 4, así que su codificación se calculaba con
    # estadísticas de los meses 3 y 4 — POSTERIORES a él. Fuga temporal que no
    # rompe nada y falsea los resultados.
    v = verificar(cfg, df["month"].values, df["week_in_month"].values)
    train_mask = pd.Series(v["gnn_entrena"] | v["cabezas_entrenan"],
                           index=df.index)
    log.info("Codificadores e imputación ajustados con %d filas "
             "(gnn_entrena + cabezas_entrenan), no con el dataset entero",
             int(train_mask.sum()))
    df, feature_cols = encode_and_impute(df, train_mask, cfg)

    # `split` se conserva por compatibilidad con informes antiguos, pero quien
    # manda son las `ventanas`: ningún módulo del pipeline lo usa ya.
    df["split"] = np.select(
        [df["month"].isin(set(cfg["data"]["train_months"])),
         df["month"] == cfg["data"]["val_month"],
         df["month"] == cfg["data"]["test_month"]],
        ["train", "val", "test"], default="otro")

    y_tr = df.loc[train_mask, "isFraud"]
    ratio = (y_tr == 0).sum() / max(1, (y_tr == 1).sum())
    log.info("Desbalance en las ventanas de entrenamiento: %.1f legítimas "
             "por fraude (pos_weight)", ratio)

    df.to_parquet(out_dir / "full.parquet", index=False)
    with open(out_dir / "feature_cols.json", "w") as f:
        json.dump({"feature_cols": feature_cols, "pos_weight_train": float(ratio)}, f, indent=2)
    df[["TransactionID", "split", "month", "week_in_month", "isFraud"]].to_parquet(
        out_dir / "split_masks.parquet", index=False)

    log.info("Split -> train: %d | val: %d | test: %d",
             (df.split == "train").sum(), (df.split == "val").sum(), (df.split == "test").sum())


if __name__ == "__main__":
    main()
