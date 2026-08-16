"""
INVARIANTE — SMOTE hace lo que dice, en las tres cabezas.

No entrena nada: arma las matrices igual que `heads`, aplica SMOTE y comprueba.

Lo que guarda:

  SOLO SINTETIZA FRAUDE   las legítimas tienen que quedar EXACTAMENTE igual.
                          Si SMOTE tocara la clase mayoritaria estaría
                          cambiando el problema, no equilibrándolo.

  EL RATIO PEDIDO         `sampling_strategy` es un contrato: 0.5 significa que
                          la minoría llega al 50% de la mayoría. Se comprueba
                          contra el config, no contra un número escrito a mano.

  SIN NaN                 SMOTE interpola con kNN y un NaN envenena la
                          distancia. La matriz de entrada tiene que venir
                          limpia de `cargar_tabla`.

  POCOS DUPLICADOS        si los "sintéticos" son copias, no es SMOTE sino
                          sobremuestreo por repetición, y eso sobreajusta. Pero
                          algunos son inevitables: `solo_gnn` trabaja con 64
                          columnas de embedding y dos transacciones con el
                          mismo vecindario producen el MISMO vector, así que
                          interpolar entre ellas devuelve ese vector otra vez.
                          Medido: 462 de 38.574 (1,2%). Se avisa por encima del
                          5%, que ya sería el embedding colapsando.

  ESCALA COHERENTE        la norma de los sintéticos tiene que caer dentro del
                          rango de los reales. Si se dispara, SMOTE está
                          interpolando entre puntos lejanísimos y fabricando
                          fraudes que no se parecen a ninguno.

Necesita la etapa `embed` hecha (usa `gnn_embed.parquet`). Sin GPU.

    python tests/test_smote.py
"""
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.baseline_xgboost.smote_pipeline import apply_smote      # noqa: E402
from src.hybrid.head import cargar_tabla, cols_embedding, columnas  # noqa: E402
from src.utils.common import load_config, resolve                # noqa: E402
from src.utils.ventanas import verificar                         # noqa: E402


def main() -> int:
    cfg = load_config()
    if not (resolve(cfg, "processed_dir") / "gnn_embed.parquet").exists():
        print("  SALTADO: falta gnn_embed.parquet. Corre la etapa `embed`.")
        return 0

    ratio = float((cfg["xgboost"].get("smote") or {}).get("sampling_strategy", 0.5))
    df, base = cargar_tabla(cfg, "train")
    emb, embv = cols_embedding(df, "completo"), cols_embedding(df, "vecinos")
    v = verificar(cfg, df["month"].values, df["week_in_month"].values)
    tr = np.where(v["cabezas_entrenan"])[0]
    y = df["isFraud"].values.astype(int)[tr]
    fallos = []

    print(f"  cabezas_entrenan: {len(tr)} filas · {int(y.sum())} fraudes "
          f"({100*y.mean():.2f}%) · sampling_strategy {ratio}")

    for var in [str(x) for x in (cfg.get("hybrid") or {}).get("variantes", ())]:
        cols = columnas(var, base, emb, embv)
        X = df.iloc[tr][cols].values.astype(np.float32)
        mal = []

        nan = int(np.isnan(X).sum())
        if nan:
            mal.append(f"{nan} NaN en la matriz de entrada; SMOTE interpola con "
                       f"kNN y un NaN envenena la distancia")

        Xr, yr = apply_smote(X, y, cfg)

        # 1. las legítimas, intactas
        if int((1 - yr).sum()) != int((1 - y).sum()):
            mal.append(f"cambió el número de legítimas: {int((1-y).sum())} -> "
                       f"{int((1-yr).sum())}. SMOTE solo puede tocar la minoría")

        # 2. el ratio que pide el config
        mayoria = int((yr == 0).sum())
        esperado = int(round(mayoria * ratio))
        obtenido = int(yr.sum())
        if abs(obtenido - esperado) > max(2, 0.01 * esperado):
            mal.append(f"ratio incumplido: {obtenido} fraudes contra "
                       f"{esperado} = {mayoria} x {ratio}")

        # 3. los sintéticos no pueden ser copias
        sint = Xr[len(X):]
        if len(sint):
            dup = len(sint) - len(np.unique(sint, axis=0))
            pct_dup = 100 * dup / len(sint)
            if pct_dup > 5.0:
                mal.append(f"{dup} sintéticos duplicados ({pct_dup:.1f}%): el "
                           f"espacio de features está colapsando y SMOTE repite "
                           f"en vez de interpolar")

            # 4. escala coherente con los fraudes reales
            nr = np.linalg.norm(np.nan_to_num(X[y == 1]), axis=1)
            ns = np.linalg.norm(np.nan_to_num(sint), axis=1)
            if float(np.percentile(ns, 95)) > 3 * float(np.percentile(nr, 95)):
                mal.append(f"la norma de los sintéticos (p95 {np.percentile(ns,95):.1f}) "
                           f"triplica la de los reales (p95 {np.percentile(nr,95):.1f})")

        dupt = (f" · {pct_dup:.1f}% duplicados" if len(sint) and pct_dup else "")
        print(f"  [{'MAL' if mal else 'OK '}] {var:<16}{len(cols):>4} col · "
              f"{len(X)} -> {len(Xr)} filas · {100*yr.mean():.1f}% fraude · "
              f"{len(sint)} sintéticos{dupt}")
        fallos += [f"{var}: {m}" for m in mal]
        del X, Xr, sint

    if fallos:
        print("\n  FALLA EL INVARIANTE DE SMOTE:")
        for f in fallos:
            print(f"   · {f}")
        return 1
    print("\n  SMOTE OK — solo sintetiza fraude, respeta el ratio y no duplica.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
