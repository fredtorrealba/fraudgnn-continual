"""
Radiografía del grafo construido. Para decidir con números, no con intuición.

    python scripts/inspeccionar_grafo.py
    python scripts/inspeccionar_grafo.py --guardar antes.json
    python scripts/inspeccionar_grafo.py --contra antes.json     # el diff

Flujo típico para comparar dos configuraciones:

    # con la poda actual
    python scripts/inspeccionar_grafo.py --guardar /tmp/con_poda.json
    # cambias max_entity_degree en el config y reconstruyes
    python -m src.data.build_graph
    python scripts/inspeccionar_grafo.py --contra /tmp/con_poda.json

Qué mira, y por qué cada cosa:

  CONECTIVIDAD   cuántas transacciones se quedan sin recibir de NADIE. Son las
                 que el grafo no puede ayudar: para ellas la GNN es una MLP.

  VECINDARIO     a cuántas transacciones distintas llega cada una por sus 5
                 entidades. Es el material del que sale el embedding.

  GRADO          la feature `__grado_*`. Con poda, la transacción nº 502 de una
                 tarjeta recibía 0 — el MISMO valor que una sin card1. La
                 columna quedaba invertida justo donde más historial hay.

  FRAUDE         si lo que se pierde está sesgado. Cortar conexiones donde hay
                 más fraude que la media es peor que cortarlas al azar.

  ORDEN          `temporal_strategy="last"` exige las aristas de bajada
                 ordenadas por tiempo. Si no lo están, "los 10 más recientes"
                 devuelve cualquier cosa y nadie avisa.
"""
import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import torch

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.common import load_config, resolve      # noqa: E402

TXN = "transaction"


def radiografia(cfg) -> dict:
    data = torch.load(resolve(cfg, "graph_dir") / "graph.pt", weights_only=False)
    meta = json.load(open(resolve(cfg, "graph_dir") / "graph_meta.json"))
    N = data[TXN].num_nodes
    y = data[TXN].y.numpy()
    t = data[TXN].time.numpy()
    cols = meta["feature_cols_gnn"]

    r = {"config": {"max_entity_degree": cfg["graph"].get("max_entity_degree"),
                    "min_previas_entidad": cfg["graph"].get("min_previas_entidad"),
                    "vecinos_por_entidad": cfg["graph"].get("vecinos_por_entidad")},
         "n_transacciones": N, "fraude_global": round(100 * float(y.mean()), 3),
         "entidades": {}}

    recibe_alguna = np.zeros(N, bool)
    alcanzables = np.zeros(N, dtype=np.int64)

    for nt in [n for n in data.node_types if n != TXN]:
        sub = data[(TXN, f"en_{nt}", nt)].edge_index
        baj = data[(nt, f"tiene_{nt}", TXN)].edge_index
        n_ent = data[nt].num_nodes
        e_sub, txn_sub = sub[1].numpy(), sub[0].numpy()
        grado = np.bincount(e_sub, minlength=n_ent)
        vivos = grado[grado > 0]

        recibe = np.zeros(N, bool)
        if baj.numel():
            recibe[baj[1].numpy()] = True
            recibe_alguna |= recibe
        # a cuántas OTRAS transacciones llega cada una por esta entidad
        alcanzables[txn_sub] += np.maximum(grado[e_sub] - 1, 0)

        # el orden temporal que exige temporal_strategy="last"
        ordenado = True
        if baj.numel():
            clave = (baj[0].to(torch.int64) * (int(t.max()) + 1)
                     + torch.as_tensor(t[baj[1].numpy()]))
            ordenado = bool((clave[1:] >= clave[:-1]).all())

        col = cols.index(f"__grado_{nt}") if f"__grado_{nt}" in cols else None
        g = data[TXN].x[:, col].numpy() if col is not None else None
        sin_arista = np.zeros(N, bool)
        sin_arista[:] = True
        sin_arista[txn_sub] = False

        r["entidades"][nt] = {
            "nodos": int(n_ent),
            "aristas_subida": int(sub.shape[1]),
            "aristas_bajada": int(baj.shape[1]),
            "cobertura_pct": round(100 * len(np.unique(txn_sub)) / N, 2),
            "grado_medio": round(float(vivos.mean()), 2) if len(vivos) else 0.0,
            "grado_p50": int(np.median(vivos)) if len(vivos) else 0,
            "grado_p99": int(np.percentile(vivos, 99)) if len(vivos) else 0,
            "grado_max": int(vivos.max()) if len(vivos) else 0,
            "entidades_grado_1": int((grado == 1).sum()),
            "entidades_mayores_500": int((grado > 500).sum()),
            "txn_sin_recibir": int((~recibe).sum()),
            "orden_temporal_ok": ordenado,
            "grado_feature_cero_pct": (round(100 * float((g == 0).mean()), 2)
                                       if g is not None else None),
            "grado_feature_max": round(float(g.max()), 3) if g is not None else None,
        }

    sin_nadie = ~recibe_alguna
    r["conectividad"] = {
        "txn_sin_recibir_de_nadie": int(sin_nadie.sum()),
        "pct": round(100 * float(sin_nadie.mean()), 2),
        "fraude_en_las_aisladas": round(100 * float(y[sin_nadie].mean()), 3)
        if sin_nadie.any() else None,
        "fraude_en_las_conectadas": round(100 * float(y[~sin_nadie].mean()), 3),
    }
    q = np.percentile(alcanzables, [10, 50, 90, 99])
    r["vecindario"] = {"p10": int(q[0]), "mediana": int(q[1]),
                       "p90": int(q[2]), "p99": int(q[3]),
                       "media": round(float(alcanzables.mean()), 1)}
    return r


def _pinta(r, prev=None):
    def d(a, b):
        if prev is None or b is None:
            return ""
        dif = a - b
        return f"  ({dif:+,})".replace(",", ".") if dif else "  (=)"

    c = r["config"]
    print(f"\n  max_entity_degree {c['max_entity_degree']} · "
          f"min_previas_entidad {c['min_previas_entidad']} · "
          f"vecinos_por_entidad {c['vecinos_por_entidad']}")
    print(f"  {r['n_transacciones']:,} transacciones · fraude {r['fraude_global']}%"
          .replace(",", "."))

    print(f"\n  {'entidad':<8}{'nodos':>8}{'subida':>9}{'bajada':>9}{'cob%':>7}"
          f"{'gr.med':>8}{'gr.p99':>8}{'gr.max':>8}{'>500':>6}{'ord':>5}")
    print("  " + "─" * 76)
    for nt, e in r["entidades"].items():
        p = (prev or {}).get("entidades", {}).get(nt, {})
        print(f"  {nt:<8}{e['nodos']:>8}{e['aristas_subida']:>9}"
              f"{e['aristas_bajada']:>9}{e['cobertura_pct']:>7.1f}"
              f"{e['grado_medio']:>8.1f}{e['grado_p99']:>8}{e['grado_max']:>8}"
              f"{e['entidades_mayores_500']:>6}"
              f"{'ok' if e['orden_temporal_ok'] else 'MAL':>5}"
              f"{d(e['aristas_subida'], p.get('aristas_subida'))}")

    print(f"\n  __grado_* — la feature que sustituye a las columnas C")
    print(f"  {'entidad':<8}{'% en cero':>12}{'máximo':>10}")
    for nt, e in r["entidades"].items():
        p = (prev or {}).get("entidades", {}).get(nt, {})
        marca = ""
        if p.get("grado_feature_cero_pct") is not None:
            dif = e["grado_feature_cero_pct"] - p["grado_feature_cero_pct"]
            marca = f"   ({dif:+.2f} pts)" if abs(dif) > 0.001 else ""
        print(f"  {nt:<8}{e['grado_feature_cero_pct']:>11.2f}%"
              f"{e['grado_feature_max']:>10.2f}{marca}")

    co = r["conectividad"]
    pc = (prev or {}).get("conectividad", {})
    print(f"\n  CONECTIVIDAD")
    print(f"    sin recibir de ninguna entidad   {co['txn_sin_recibir_de_nadie']:>8}"
          f"  ({co['pct']}%)"
          f"{d(co['txn_sin_recibir_de_nadie'], pc.get('txn_sin_recibir_de_nadie'))}")
    print(f"    fraude en esas                   {co['fraude_en_las_aisladas']}%")
    print(f"    fraude en las conectadas         {co['fraude_en_las_conectadas']}%")

    v, pv = r["vecindario"], (prev or {}).get("vecindario", {})
    print(f"\n  VECINDARIO alcanzable por las 5 entidades (antes del tope de 10)")
    print(f"    p10 {v['p10']} · mediana {v['mediana']} · p90 {v['p90']} · "
          f"p99 {v['p99']} · media {v['media']}"
          f"{d(v['mediana'], pv.get('mediana'))}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Radiografía del grafo")
    ap.add_argument("--guardar", metavar="RUTA")
    ap.add_argument("--contra", metavar="RUTA", help="compara con un guardado")
    a = ap.parse_args()

    r = radiografia(load_config())
    prev = json.load(open(a.contra)) if a.contra else None
    if prev:
        print(f"\n  comparando contra {a.contra}")
    _pinta(r, prev)
    if a.guardar:
        Path(a.guardar).write_text(json.dumps(r, indent=2, ensure_ascii=False))
        print(f"\n  guardado en {a.guardar}")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
