"""
DIAGNÓSTICO — consistencia temporal por columna del embedding (MEJORAS punto 6).

La técnica del 1er lugar de Kaggle: entrenar con UNA sola feature en un periodo
y medirla en el siguiente. Ellos encontraron que el 5% de sus columnas tenían
AUC 0.60 en entrenamiento y 0.40 en validación — peor que el azar: un patrón
que existe en el presente y SE INVIERTE en el futuro no es ruido, es daño
activo, y XGBoost lo aprende con gusto.

Aquí se aplica a las columnas emb_/embv_ del parquet del embedding: AUC de
cada columna sola en `cabezas_entrenan` contra `cabezas_validan`, orientada
por la primera (la dirección de una dimensión latente es arbitraria).

MEDIDO LA PRIMERA VEZ (2026-08-17, embedding 64d de graphsage 256x3):

    embv_  mediana 0.6515 -> 0.6169   1 inversión débil (embv_16), 0 fuertes
    emb_   mediana 0.7349 -> 0.6884   2 inversiones débiles, 0 fuertes

La degradación es PAREJA, no de unas pocas columnas: la poda por consistencia
no aplica y el aporte negativo del grafo (−0.0207) no se explica por
dimensiones invertidas. Lo que sí muestra: embv_ a 0.65 de AUC mediana por
dimensión está en el régimen «redundante con lo tabular» que gnn.md documenta
(el embv_ de 0.526/dim aportaba +0.0053; el emb_ de 0.675/dim restaba −0.0325).

FALLA solo si hay inversiones FUERTES (>0.55 en entrenan y <0.5 en validan):
eso sí sería una poda pendiente y barata. Las débiles solo se informan.

Necesita `gnn_embed.parquet` (etapa embed). Sin GPU.

    python tests/test_consistencia_embedding.py
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.common import load_config, resolve      # noqa: E402
from src.utils.ventanas import mascara                 # noqa: E402


def main() -> int:
    from sklearn.metrics import roc_auc_score

    cfg = load_config()
    proc = resolve(cfg, "processed_dir")
    ruta = proc / "gnn_embed.parquet"
    if not ruta.exists():
        print(f"  SALTADO: falta {ruta.name}. Corre la etapa `embed`.")
        return 0

    df = pd.read_parquet(proc / "full.parquet",
                         columns=["isFraud", "month", "week_in_month"])
    oof = pd.read_parquet(ruta)
    ni = oof["node_idx"].values
    y = df["isFraud"].values[ni]

    m_e = mascara(cfg, "cabezas_entrenan", df["month"].values,
                  df["week_in_month"].values)
    m_v = mascara(cfg, "cabezas_validan", df["month"].values,
                  df["week_in_month"].values)
    en_e = np.isin(ni, np.where(m_e)[0])
    en_v = np.isin(ni, np.where(m_v)[0])
    if not (en_e.any() and en_v.any()):
        print("  SALTADO: el parquet no cubre cabezas_entrenan/validan.")
        return 0

    fallos = []
    for pref in ("embv_", "emb_"):
        cols = sorted((c for c in oof.columns
                       if c.startswith(pref) and c[len(pref):].isdigit()),
                      key=lambda c: int(c[len(pref):]))
        if not cols:
            continue
        debiles, fuertes, aes, avs = [], [], [], []
        for c in cols:
            a_e = roc_auc_score(y[en_e], oof[c].values[en_e])
            a_v = roc_auc_score(y[en_v], oof[c].values[en_v])
            if a_e < 0.5:                      # orientar por entrenan
                a_e, a_v = 1 - a_e, 1 - a_v
            aes.append(a_e); avs.append(a_v)
            if a_v < 0.5:
                (fuertes if a_e > 0.55 else debiles).append((c, a_e, a_v))
        print(f"  {pref}* ({len(cols)} col) · AUC mediana "
              f"{np.median(aes):.4f} -> {np.median(avs):.4f} · "
              f"{len(debiles)} inversión(es) débil(es), {len(fuertes)} fuerte(s)")
        for c, e, v in debiles:
            print(f"    ~ {c:<10} {e:.4f} -> {v:.4f}  (débil: se informa)")
        for c, e, v in fuertes:
            print(f"    ! {c:<10} {e:.4f} -> {v:.4f}  (FUERTE)")
            fallos.append(
                f"{c} tiene AUC {e:.3f} en entrenan y {v:.3f} en validan: "
                f"un patrón que se invierte en el tiempo. Candidata a poda "
                f"antes de entrenar las cabezas.")

    if fallos:
        print("\n  HAY COLUMNAS DEL EMBEDDING QUE SE INVIERTEN EN EL TIEMPO:")
        for f in fallos:
            print(f"   · {f}")
        return 1
    print("\n  OK — la degradación del embedding es pareja: ninguna columna "
          "se invierte con fuerza entre entrenan y validan.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
