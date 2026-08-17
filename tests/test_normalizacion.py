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
               El umbral es generoso a propósito: alarma, no termómetro.

  ESCALA       media ~0 y desviación ~1 en las filas de ajuste, que es la
               definición de estar tipificado. Tolerancia 0.05 porque el clip
               recorta unas pocas colas después del ajuste.

  CLIP         ningún |x| por encima del clip declarado en los parámetros.
               Sin normalizar el máximo era 999.595; con log+z quedaba en 44
               desviaciones — todavía bastantes para acaparar un gradiente.

  RECOMPUTABLE la comprobación fuerte: desde `x_crudo` + los parámetros de
               graph_meta.json se RECONSTRUYE la transformación entera y tiene
               que dar lo que hay en el grafo. Un grafo normalizado que no se
               puede recomputar es una caja negra.

  CATEGÓRICAS  las que son identificadores van por FRECUENCIA CAUSAL RELATIVA:
               su número es un NOMBRE (alfabético: gmail=17, hotmail=20) y
               tipificarlo deja el orden inventado intacto. Causal y relativa
               porque la tabla estática contada en gnn_entrena caducaba: un
               valor que explota después del día 15 llegaba al examen con
               frecuencia 0, igual que uno nunca visto.

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
from src.data.normalizacion import (CATEGORICAS, CLIP, NO_FRECUENCIA,  # noqa: E402
                                    transformar_base)
from src.utils.common import load_config, resolve                # noqa: E402
from src.utils.ventanas import mascara                           # noqa: E402

MAX_VAR_PCT = 15.0      # ninguna columna puede dominar


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

    par = meta.get("normalizacion")
    if not par:
        print("  MAL: graph_meta.json no trae parámetros de normalización.")
        return 1
    # Un grafo de la versión con tabla estática no es un fallo del código de
    # hoy, pero SÍ es un artefacto desactualizado: usarlo mediría otra cosa.
    if "causal" not in str(par.get("metodo", "")):
        print(f"  MAL: el grafo se construyó con la normalización anterior "
              f"({par.get('metodo')!r}). Reconstruye desde preprocess.\n"
              f"       OJO: `--from preprocess --force` relanza también la "
              f"búsqueda de Optuna; para conservarla, borra los artefactos de "
              f"datos/modelos a mano y corre SIN --force (receta en la memoria "
              f"de pipeline).")
        return 1
    clip = float(par.get("clip", CLIP))

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
    # El clip recorta DESPUÉS del ajuste, así que hay que separar dos casos:
    #   · columnas SIN valores recortados en entrena: media 0 y std 1 exactas.
    #   · columnas CON recorte (casi constantes con un puñado de extremos, o
    #     colas que pasan de ±clip): su std ENCOGE — eso es el clip trabajando,
    #     no un fallo. Lo que jamás puede pasar es que la std CREZCA por encima
    #     de 1: eso sí significaría parámetros de otro bloque o sin aplicar.
    recortada = (np.abs(X[ent]) >= clip - 1e-6).any(0)
    limpia = vivas & ~recortada
    ok_mu = float(np.abs(mu[limpia]).max()) < 0.01 if limpia.any() else True
    ok_sd = float(np.abs(sd[limpia] - 1).max()) < 0.01 if limpia.any() else True
    ok_crece = float(sd[vivas].max()) < 1.05
    print(f"  [{'OK ' if ok_mu and ok_sd and ok_crece else 'MAL'}] tipificado en "
          f"gnn_entrena · {int(limpia.sum())} columnas exactas "
          f"(|media| max {np.abs(mu[limpia]).max():.2e}, "
          f"|std-1| max {np.abs(sd[limpia]-1).max():.2e}) · "
          f"{int((vivas & recortada).sum())} con std encogida por el clip "
          f"(mín {sd[vivas].min():.2f})")
    if not (ok_mu and ok_sd):
        fallos.append("las columnas sin recorte no tienen media 0 y desviación "
                      "1 en las filas de ajuste: los parámetros no se aplicaron "
                      "o se calcularon con otro bloque.")
    if not ok_crece:
        fallos.append(f"hay columnas con std {sd[vivas].max():.2f} > 1 en las "
                      f"filas de ajuste: el clip solo puede encogerla, así que "
                      f"esos parámetros no salieron de este bloque.")

    # ── clip ──────────────────────────────────────────────────────────────
    amax = float(np.abs(X).max())
    print(f"  [{'OK ' if amax <= clip + 1e-6 else 'MAL'}] clip respetado · "
          f"|x| max {amax:.2f} (tope ±{clip:.0f})")
    if amax > clip + 1e-6:
        fallos.append(f"hay valores de hasta {amax:.1f} con clip declarado en "
                      f"{clip:.0f}: el clip no se aplicó.")

    if not np.isfinite(X).all():
        fallos.append("hay NaN o Inf tras normalizar. Sospechoso número uno: "
                      "log1p() sin signo sobre las columnas con negativos.")

    # ── auditable ─────────────────────────────────────────────────────────
    ok_meta = bool(par.get("columnas") and len(par["columnas"]) == len(cols))
    print(f"  [{'OK ' if ok_meta else 'MAL'}] parámetros en graph_meta.json · "
          f"{len(par.get('columnas', {}))} de {len(cols)} columnas")
    if not ok_meta:
        fallos.append("graph_meta.json no guarda los parámetros por columna: "
                      "la transformación no es auditable ni reversible.")
    elif par.get("ajustado_con") != "gnn_entrena":
        fallos.append(f"se ajustó con '{par.get('ajustado_con')}' en vez de "
                      f"gnn_entrena: eso es fuga.")

    # ── las categóricas van por frecuencia causal ─────────────────────────
    if ok_meta:
        esperadas = [c for c in cols if c in CATEGORICAS and c not in NO_FRECUENCIA]
        mal = [c for c in esperadas
               if par["columnas"][c]["tipo"] != "frecuencia_causal"]
        print(f"  [{'OK ' if not mal else 'MAL'}] {len(esperadas)} categóricas "
              f"por frecuencia causal · {len(cols)-len(esperadas)} por log+z")
        if mal:
            fallos.append(f"{len(mal)} categóricas no van por frecuencia causal "
                          f"({mal[:4]}): su número es alfabético y tipificarlo "
                          f"deja el orden inventado intacto.")

    # ── LA FUERTE: recomputar desde el crudo y comparar ───────────────────
    if "x_crudo" in data["transaction"] and ok_meta:
        crudo = data["transaction"].x_crudo.numpy().astype(np.float64)
        tiempo = data["transaction"].time.numpy()
        B, tipos = transformar_base(crudo, cols, tiempo)
        Z = np.empty_like(B, dtype=np.float32)
        for i, c in enumerate(cols):
            p = par["columnas"][c]
            Z[:, i] = np.clip((B[:, i] - p["media"]) / p["std"],
                              -clip, clip).astype(np.float32)
        diff = float(np.abs(Z - X).max())
        ok_rec = diff < 1e-4
        print(f"  [{'OK ' if ok_rec else 'MAL'}] recomputable desde x_crudo + "
              f"parámetros · desvío máx {diff:.2e}")
        if not ok_rec:
            fallos.append(
                f"reconstruir la transformación desde x_crudo difiere hasta en "
                f"{diff:.4f} de lo que hay en el grafo: o los parámetros de "
                f"graph_meta.json no son los que se aplicaron, o la "
                f"transformación cambió sin reconstruir el grafo.")
        tipos_meta = {c: par["columnas"][c]["tipo"] for c in cols}
        if tipos != tipos_meta:
            fallos.append("los tipos por columna del meta no coinciden con los "
                          "que produce transformar_base: versiones distintas.")
    else:
        print("  [~~~] x_crudo no está: la recomputación no se puede verificar")

    if fallos:
        print("\n  FALLA EL INVARIANTE DE NORMALIZACIÓN:")
        for f in fallos:
            print(f"   · {f}")
        return 1
    print(f"\n  OK — la entrada de la GNN está normalizada, recomputable y "
          f"ninguna columna domina ({pct:.2f}% la mayor).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
