"""
Diagnóstico del SMOTE para las TRES cabezas. No entrena nada: solo arma las
matrices como lo haría `heads`, aplica SMOTE y mide.

    python3 scripts/revisar_smote.py

Hay que correrlo DESPUÉS de la etapa `embed`: necesita gnn_embed.parquet.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.baseline_xgboost.smote_pipeline import apply_smote
from src.hybrid.head import cargar_tabla, cols_embedding, columnas
from src.utils.common import load_config
from src.utils.ventanas import verificar


def main():
    cfg = load_config()
    df, base = cargar_tabla(cfg, "train")
    emb, embv = cols_embedding(df, "completo"), cols_embedding(df, "vecinos")
    v = verificar(cfg, df["month"].values, df["week_in_month"].values)
    tr = np.where(v["cabezas_entrenan"])[0]
    y = df["isFraud"].values.astype(int)[tr]

    out = {"filas": int(len(tr)), "fraude_real": int(y.sum()),
           "pct_real": round(100 * float(y.mean()), 2), "cabezas": {}}
    print(f"\nVENTANA cabezas_entrenan: {len(tr):,} filas, {y.sum():,} fraudes "
          f"({100*y.mean():.2f}%)\n")

    for var in [str(x) for x in (cfg.get("hybrid") or {}).get("variantes", ())]:
        cols = columnas(var, base, emb, embv)
        X = df.iloc[tr][cols].values.astype(np.float32)
        nan = int(np.isnan(X).sum())
        Xr, yr = apply_smote(X, y, cfg)

        # ¿los sintéticos se parecen a los fraudes reales? Se compara la norma:
        # si SMOTE interpola bien, la distribución debe solaparse.
        reales = Xr[:len(X)][y == 1]
        sint = Xr[len(X):]
        nr = np.linalg.norm(np.nan_to_num(reales), axis=1)
        ns = np.linalg.norm(np.nan_to_num(sint), axis=1)

        d = {"n_columnas": len(cols),
             "nan_en_matriz": nan,
             "filas_antes": int(len(X)), "filas_despues": int(len(Xr)),
             "fraude_despues": int(yr.sum()),
             "pct_fraude_despues": round(100 * float(yr.mean()), 2),
             "legitimas_intactas": bool((1 - yr).sum() == (1 - y).sum()),
             "sinteticos": int(yr.sum() - y.sum()),
             "norma_reales": [round(float(np.percentile(nr, p)), 3) for p in (5, 50, 95)],
             "norma_sinteticos": [round(float(np.percentile(ns, p)), 3) for p in (5, 50, 95)],
             "duplicados_exactos": int(len(Xr) - len(np.unique(Xr, axis=0)))}
        out["cabezas"][var] = d
        print(f"  {var}")
        print(f"    columnas {d['n_columnas']:>3} | NaN {d['nan_en_matriz']} | "
              f"{d['filas_antes']:,} -> {d['filas_despues']:,} filas "
              f"({d['pct_fraude_despues']}% fraude)")
        print(f"    sintéticos {d['sinteticos']:,} | legítimas intactas: "
              f"{d['legitimas_intactas']} | duplicados {d['duplicados_exactos']}")
        print(f"    norma p5/p50/p95  reales {d['norma_reales']}  "
              f"sintéticos {d['norma_sinteticos']}")
        del X, Xr

    print("\n" + "=" * 60)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
