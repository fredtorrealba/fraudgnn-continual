"""
INVARIANTE — las features que entran a la GNN están normalizadas.

El fallo que guarda estuvo activo TODA la primera fase del proyecto y no produjo
ningún síntoma legible: las features iban crudas al grafo y una sola columna,
`id_02`, se llevaba el **99,55%** de la varianza. La red no veía 70 features,
veía esa columna y ruido.

A un árbol le da igual (usa el orden, no la magnitud) y por eso XGBoost rendía
bien con los mismos datos. A una red no: `W·x` con un componente 13.000 veces
mayor que el resto reparte casi todo el gradiente a un solo peso.

Las huellas que dejó, leídas durante semanas sin conectarlas:

    bns.1.running_var  579.055
    best_epoch: 2 de 50
    gnn_sola ROC 0,4540 en `habitual` — por debajo del azar justo donde
                                        más vecinos tiene

Lo que se comprueba:

  REPARTO      ninguna columna puede llevarse más del 15% de la varianza.
               El umbral es generoso a propósito: alarma, no termómetro. Hoy
               la mayor está en 2,28%; si alguna llega a 15 es que algo se
               quedó crudo.

  ESCALA       media ~0 y desviación ~1 en las filas de ajuste, que es la
               definición de estar tipificado.

  SIN EXTREMOS ningún |x| por encima de 200. Con log-con-signo el máximo real
               es 44; sin normalizar era 999.595.

  AUDITABLE    `graph_meta.json` guarda el método, con qué se ajustó y los
               parámetros por columna. Un grafo normalizado sin eso es una
               caja negra.

  CATEGÓRICAS  las que son identificadores tienen que ir por frecuencia. Su
               número es un NOMBRE (alfabético: gmail=17, hotmail=20) y
               tipificarlo deja el orden inventado intacto.

  SIN FUGA     los parámetros salen de `gnn_entrena` y de nada más.

Necesita `data/graph/graph.pt`. Sin GPU.

    python tests/test_normalizacion.py
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import torch

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.normalizacion import CATEGORICAS, NO_FRECUENCIA   # noqa: E402
from src.utils.common import load_config, resolve                # noqa: E402
from src.utils.ventanas import mascara                           # noqa: E402

MAX_VAR_PCT = 15.0      # ninguna columna puede dominar
MAX_ABS = 200.0         # ningún valor absurdo


def main() -> int:
    cfg = load_config()
    g = resolve(cfg, "graph_dir") / "graph.pt"
    m = resolve(cfg, "graph_dir") / "graph_meta.json"
    if not (g.exists() and m.exists()):
        print(f"  SALTADO: falta {g if not g.exists() else m}.")
        return 0

    data = torch.load(g, weights_only=False)
    meta = json.load(open(m))
    cols = meta["feature_cols_gnn"]
    X = data["transaction"].x.numpy().astype(np.float64)
    fallos = []

    # ── el reparto de varianza ────────────────────────────────────────────
    var = X.var(0)
    peor = int(np.argmax(var))
    pct = 100 * var[peor] / var.sum()
    ok = pct <= MAX_VAR_PCT
    print(f"  [{'OK ' if ok else 'MAL'}] reparto de varianza · la mayor es "
          f"{cols[peor]} con {pct:.2f}% (tope {MAX_VAR_PCT}%)")
    if not ok:
        fallos.append(
            f"'{cols[peor]}' se lleva el {pct:.1f}% de la varianza. Esa columna "
            f"entra sin normalizar y la red no va a mirar ninguna otra.")

    # cuántas columnas hacen falta para el 90%: si es 1 o 2, algo domina
    n90 = int(np.searchsorted(np.cumsum(np.sort(var)[::-1]) / var.sum(), 0.90) + 1)
    print(f"  [{'OK ' if n90 >= 10 else 'MAL'}] {n90} columnas para el 90% de la "
          f"varianza (de {len(cols)})")
    if n90 < 10:
        fallos.append(f"solo {n90} columnas explican el 90%: la entrada está "
                      f"dominada por unas pocas.")

    # ── escala ────────────────────────────────────────────────────────────
    ent = mascara(cfg, "gnn_entrena", data["transaction"].month.numpy(),
                  data["transaction"].week_in_month.numpy())
    mu, sd = X[ent].mean(0), X[ent].std(0)
    vivas = sd > 1e-9
    ok_mu = float(np.abs(mu[vivas]).max()) < 0.01
    ok_sd = float(np.abs(sd[vivas] - 1).max()) < 0.01
    print(f"  [{'OK ' if ok_mu and ok_sd else 'MAL'}] tipificado en gnn_entrena · "
          f"|media| max {np.abs(mu[vivas]).max():.2e} · "
          f"|std-1| max {np.abs(sd[vivas]-1).max():.2e}")
    if not (ok_mu and ok_sd):
        fallos.append("las columnas no tienen media 0 y desviación 1 en las "
                      "filas de ajuste: los parámetros no se aplicaron o se "
                      "calcularon con otro bloque.")

    # ── extremos ──────────────────────────────────────────────────────────
    amax = float(np.abs(X).max())
    print(f"  [{'OK ' if amax <= MAX_ABS else 'MAL'}] sin valores absurdos · "
          f"|x| max {amax:.1f} (tope {MAX_ABS})")
    if amax > MAX_ABS:
        fallos.append(f"hay valores de hasta {amax:.0f}: con eso los gradientes "
                      f"explotan en las primeras épocas.")

    if not np.isfinite(X).all():
        fallos.append("hay NaN o Inf tras normalizar. Sospechoso número uno: "
                      "log1p() sin signo sobre las 16 columnas con negativos.")

    # ── auditable ─────────────────────────────────────────────────────────
    par = meta.get("normalizacion")
    ok_meta = bool(par and par.get("columnas") and
                   len(par["columnas"]) == len(cols))
    print(f"  [{'OK ' if ok_meta else 'MAL'}] parámetros en graph_meta.json · "
          f"{len(par.get('columnas', {})) if par else 0} de {len(cols)} columnas")
    if not ok_meta:
        fallos.append("graph_meta.json no guarda los parámetros por columna: "
                      "la transformación no es auditable ni reversible.")
    elif par.get("ajustado_con") != "gnn_entrena":
        fallos.append(f"se ajustó con '{par.get('ajustado_con')}' en vez de "
                      f"gnn_entrena: eso es fuga.")

    # ── las categóricas van por frecuencia ────────────────────────────────
    if ok_meta:
        esperadas = [c for c in cols if c in CATEGORICAS and c not in NO_FRECUENCIA]
        mal = [c for c in esperadas if par["columnas"][c]["tipo"] != "frecuencia"]
        print(f"  [{'OK ' if not mal else 'MAL'}] {len(esperadas)} categóricas "
              f"por frecuencia · {len(cols)-len(esperadas)} por log+z")
        if mal:
            fallos.append(f"{len(mal)} categóricas no van por frecuencia "
                          f"({mal[:4]}): su número es alfabético y tipificarlo "
                          f"deja el orden inventado intacto.")

    # ── el crudo se conserva ──────────────────────────────────────────────
    tiene_crudo = "x_crudo" in data["transaction"]
    print(f"  [{'OK ' if tiene_crudo else '~~~'}] x_crudo conservado para auditar"
          f"{'' if tiene_crudo else ' — no está, no se puede verificar la transformación'}")

    if fallos:
        print("\n  FALLA EL INVARIANTE DE NORMALIZACIÓN:")
        for f in fallos:
            print(f"   · {f}")
        return 1
    print(f"\n  OK — la entrada de la GNN está normalizada y ninguna columna "
          f"domina ({pct:.2f}% la mayor).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
