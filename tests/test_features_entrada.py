"""
INVARIANTE — las features derivadas de la entrada dicen lo que prometen.

Guarda los fallos de la familia "la columna existe pero no significa eso":

  CICLO       __hora_dia en [0,1] rompe el ciclo en la medianoche: 23:59 y
              00:01 quedaban en extremos opuestos de la recta. __hora_sin/cos
              tienen que dejarlas VECINAS.

  CENTINELA   __delta_anterior marcaba "no hay anterior" con -1. Tras
              normalizar, ese centinela era un punto más de la recta, pegado a
              "hace muy poco": la red no podía separar los dos significados.
              Ahora va en __tiene_anterior y el delta se imputa con la mediana.

  AUSENCIA    la imputación por mediana coloca el faltante en el CENTRO de la
              distribución — indistinguible de un valor real. Los flags
              `<col>__na` conservan el patrón, deduplicados por bloque (las V
              comparten ~15 patrones) y con nombre que la ablación captura.

  FRECUENCIA  la tabla estática contada en gnn_entrena CADUCABA: un valor que
              explota después del día 15 llegaba al examen con frecuencia 0,
              igual que uno nunca visto. La frecuencia causal relativa cuenta
              lo ANTERIOR a cada fila: no caduca y no puede mirar el futuro.

Todo sintético: no necesita ningún artefacto. Sin GPU.

    python tests/test_features_entrada.py
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.normalizacion import (CLIP, frecuencia_causal,      # noqa: E402
                                    normalizar, previas_por_grupo)
from src.data.preprocessing import (UMBRAL_FLAG_NA,               # noqa: E402
                                    add_temporal_columns, encode_and_impute)
from src.hybrid.head import filtrar_prefijos                      # noqa: E402


def _check(fallos, ok, nombre, detalle=""):
    print(f"  [{'OK ' if ok else 'MAL'}] {nombre}{' · ' + detalle if detalle else ''}")
    if not ok:
        fallos.append(nombre)


def _df_sintetico(n=400, seed=7):
    rng = np.random.default_rng(seed)
    dt = np.sort(rng.integers(0, 86400 * 55, n))
    df = pd.DataFrame({
        "TransactionID": np.arange(n),
        "isFraud": rng.integers(0, 2, n),
        "TransactionDT": dt,
        "TransactionAmt": rng.lognormal(3, 1, n).astype(np.float64),
        "card1": rng.choice([111, 222, 333, 444], n).astype(np.float64),
    })
    # dos columnas con el MISMO patrón de NaN (como los bloques V), una con
    # otro patrón, y una casi completa que NO debe generar flag
    patron_a = rng.random(n) < 0.5
    patron_b = rng.random(n) < 0.6
    df["id_09"] = np.where(patron_a, np.nan, rng.random(n))
    df["id_10"] = np.where(patron_a, np.nan, rng.random(n))
    df["V300"] = np.where(patron_b, np.nan, rng.random(n))
    df["dist9"] = np.where(rng.random(n) < 0.05, np.nan, rng.random(n))
    return df


def main() -> int:
    fallos = []
    cfg = {"data": {"seconds_per_month": 86400 * 30}, "graph": {"entidades": {}}}
    df = _df_sintetico()
    n = len(df)

    # ── hora cíclica ──────────────────────────────────────────────────────
    df = add_temporal_columns(df, cfg)
    ident = np.abs(df["__hora_sin"] ** 2 + df["__hora_cos"] ** 2 - 1).max()
    _check(fallos, ident < 1e-9, "sin²+cos² = 1", f"desvío máx {ident:.1e}")

    # 23:59 y 00:01 tienen que ser VECINAS en (sin, cos) y lejanas en la recta
    h = pd.DataFrame({"TransactionDT": [86400 - 60, 60],
                      "card1": [1.0, 1.0]})
    h = add_temporal_columns(h, cfg)
    d_ciclo = float(np.hypot(h["__hora_sin"].diff().iloc[1],
                             h["__hora_cos"].diff().iloc[1]))
    d_recta = float(abs(h["__hora_dia"].diff().iloc[1]))
    _check(fallos, d_ciclo < 0.02 and d_recta > 0.9,
           "medianoche continua en (sin,cos)",
           f"ciclo {d_ciclo:.4f} vs recta {d_recta:.4f}")

    # ── __delta_anterior + __tiene_anterior ───────────────────────────────
    # la primera compra de cada tarjeta: sin anterior, flag en 0
    orden = df["TransactionDT"].argsort(kind="stable")
    primera = ~df.iloc[orden].duplicated("card1").reindex(df.index[orden]).values
    es_primera = np.zeros(n, bool)
    es_primera[orden[primera]] = True
    coincide = ((df["__tiene_anterior"].values == 0) == es_primera).all()
    _check(fallos, coincide, "__tiene_anterior = 0 exactamente en la primera "
           "compra de cada tarjeta")
    sin_ant_nan = df.loc[~df["__tiene_anterior"].astype(bool),
                         "__delta_anterior"].isna().all()
    _check(fallos, bool(sin_ant_nan), "sin anterior -> delta NaN (ya no -1)")

    # una a mano: la fila k de la card 222 contra su anterior real
    filas_222 = df.index[df["card1"] == 222].tolist()
    if len(filas_222) >= 2:
        a, b = filas_222[0], filas_222[1]
        esperado = np.log1p(df.loc[b, "TransactionDT"] - df.loc[a, "TransactionDT"])
        _check(fallos, abs(df.loc[b, "__delta_anterior"] - esperado) < 1e-9,
               "delta = log1p(segundos hasta la anterior de la MISMA tarjeta)")

    # ── flags de ausencia ─────────────────────────────────────────────────
    train_mask = pd.Series(np.arange(n) < n // 2, index=df.index)
    med_esperada = df.loc[train_mask, "__delta_anterior"].median()
    na_antes = df["id_09"].isna().values.copy()
    df2, feature_cols = encode_and_impute(df.copy(), train_mask, cfg)

    _check(fallos, "id_09__na" in feature_cols and "V300__na" in feature_cols,
           "hay flag para cada patrón con más NaN que el umbral "
           f"({UMBRAL_FLAG_NA:.0%})")
    _check(fallos, "id_10__na" not in feature_cols,
           "patrones idénticos -> UN solo flag (id_10 comparte el de id_09)")
    _check(fallos, "dist9__na" not in feature_cols,
           "sin flag por debajo del umbral (dist9, 5% NaN)")
    _check(fallos, (df2["id_09__na"].values == na_antes.astype(np.float32)).all(),
           "el flag reproduce exactamente el patrón de NaN")
    _check(fallos, "V300__na" not in filtrar_prefijos(feature_cols, ["V"]),
           "la ablación [V] se lleva también V300__na (hereda el prefijo)")

    _check(fallos, not df2[[c for c in feature_cols
                            if pd.api.types.is_numeric_dtype(df2[c])]]
           .isna().any().any(), "tras imputar no queda ningún NaN")
    _check(fallos, (df2["__delta_anterior"] != -1.0).all(),
           "el centinela -1 desapareció del delta")
    imputadas = ~df2["__tiene_anterior"].astype(bool)
    _check(fallos, np.allclose(df2.loc[imputadas, "__delta_anterior"],
                               med_esperada),
           "el delta sin anterior se imputa con la MEDIANA de train",
           f"mediana {med_esperada:.4f}")

    # ── frecuencia causal relativa ────────────────────────────────────────
    rng = np.random.default_rng(11)
    m = 300
    tiempo = rng.permutation(m).astype(np.int64)      # únicos: sin empates
    x = rng.choice([5.0, 6.0, 7.0, 8.0], m)
    f = frecuencia_causal(x, tiempo)
    # fuerza bruta: SOLO las filas estrictamente anteriores
    esperado = np.array([
        (x[tiempo < tiempo[i]] == x[i]).sum() / max((tiempo < tiempo[i]).sum(), 1)
        for i in range(m)])
    _check(fallos, np.allclose(f, esperado),
           "frecuencia causal = apariciones ANTERIORES / total anterior "
           "(fuerza bruta, 300 filas)")
    _check(fallos, float(f.max()) <= 1.0 and float(f.min()) >= 0.0,
           "la frecuencia relativa vive en [0,1]")

    # LA PROPIEDAD QUE MOTIVÓ EL CAMBIO: un valor que solo aparece DESPUÉS de
    # la ventana de ajuste no puede quedarse en "nunca visto" para siempre.
    tiempo2 = np.arange(m, dtype=np.int64)
    x2 = np.full(m, 1.0)
    x2[m // 2:] = 99.0                    # el 99 solo existe en la 2ª mitad
    f2 = frecuencia_causal(x2, tiempo2)
    _check(fallos, f2[m // 2] == 0.0 and f2[-1] > 0.4,
           "un valor nuevo empieza en 0 y su frecuencia CRECE con su uso "
           "(la tabla estática lo dejaba en 0 para siempre)",
           f"primera aparición {f2[m//2]:.2f} -> última {f2[-1]:.2f}")

    # y es estacionaria: la fracción de un valor estable no depende de en qué
    # mes se mida (el conteo acumulado sí dependía: crecía monótono)
    x3 = rng.choice([1.0, 2.0], m, p=[0.7, 0.3])
    f3 = frecuencia_causal(x3, np.arange(m, dtype=np.int64))
    temprano = f3[40:60][x3[40:60] == 1.0].mean()
    tarde = f3[-60:][x3[-60:] == 1.0].mean()
    _check(fallos, abs(temprano - tarde) < 0.15,
           "estacionaria: la fracción de un valor estable no deriva",
           f"temprano {temprano:.2f} vs tarde {tarde:.2f}")

    # ── normalizar: ajuste, clip y tipos ──────────────────────────────────
    X = np.stack([rng.lognormal(4, 2, m),               # cola pesada
                  rng.normal(0, 1, m),
                  rng.choice([2.0, 17.0, 54.0], m)], axis=1)
    X[0, 0] = 1e9                                       # un extremo absurdo
    entrena = np.arange(m) < m // 2
    Z, par = normalizar(X, ["monto", "z", "card1"], entrena,
                        np.arange(m, dtype=np.int64))
    _check(fallos, float(np.abs(Z).max()) <= CLIP,
           f"clip: ningún |z| supera {CLIP:.0f}",
           f"máx {np.abs(Z).max():.2f}")
    _check(fallos, par["columnas"]["card1"]["tipo"] == "frecuencia_causal"
           and par["columnas"]["monto"]["tipo"] == "cantidad",
           "cada columna declara su tipo en los parámetros")
    _check(fallos, par.get("clip") == CLIP and par.get("ajustado_con") == "gnn_entrena",
           "los parámetros llevan clip y con qué se ajustó (auditable)")
    mu_ent = np.abs(Z[entrena, 1].mean())
    _check(fallos, mu_ent < 0.05, "tipificado: media ~0 en las filas de ajuste",
           f"|media| {mu_ent:.3f}")

    # previas_por_grupo sigue siendo el de siempre (lo comparten grafo,
    # frecuencia causal e informe): posición dentro del grupo ordenado
    g = np.array([0, 1, 0, 0, 1])
    t = np.array([10, 20, 30, 40, 50])
    _check(fallos, (previas_por_grupo(g, t) == np.array([0, 0, 1, 2, 1])).all(),
           "previas_por_grupo cuenta lo anterior dentro del grupo")

    if fallos:
        print(f"\n  FALLAN {len(fallos)} COMPROBACIÓN(ES) DE ENTRADA:")
        for f_ in fallos:
            print(f"   · {f_}")
        return 1
    print("\n  OK — hora cíclica, delta sin centinela, flags de ausencia y "
          "frecuencia causal hacen lo que dicen.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
