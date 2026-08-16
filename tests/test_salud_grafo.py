"""
INVARIANTE — el grafo está sano: ninguna entidad se cayó en silencio.

No comprueba que los números sean BUENOS, sino que no son CATASTRÓFICOS. Los
umbrales son deliberadamente generosos: esto tiene que saltar cuando algo se
rompe (una entidad sin aristas, la mitad del dataset desconectado, una feature
en cero), no cuando una cifra se mueve unas décimas. Un test que salta con cada
ajuste enseña a ignorarlo.

Lo que guarda, y por qué cada cosa ya pasó o pudo pasar:

  ENTIDAD VACÍA      `EDGE_RAW_COLS` estaba escrito a mano y `device` y `net`
                     acabaron con cero nodos sin que nada avisara.

  ASIMETRÍA          la bajada nunca puede superar a la subida: E1 poda solo
                     una dirección.

  ORDEN TEMPORAL     `temporal_strategy="last"` coge un sufijo de las aristas
                     de cada entidad. Sin orden exacto, "las 10 más recientes"
                     devuelve cualquier cosa y nadie avisa.

  __grado_*          si queda en cero para TODAS las filas con arista, la
                     feature que sustituye a las columnas C no existe.

  DESCONEXIÓN        si más de la cuarta parte de las transacciones no recibe
                     de ninguna entidad, el grafo dejó de ser un grafo.

Además del test, trae el INFORME con los números finos y el diff entre
configuraciones — que es lo que sirve para DECIDIR, no solo para vigilar. La
decisión de E2 salió de comparar dos radiografías:

    python tests/test_salud_grafo.py                          # el test
    python tests/test_salud_grafo.py --informe                # los números
    python tests/test_salud_grafo.py --informe --guardar a.json
    python tests/test_salud_grafo.py --informe --contra a.json   # el diff
"""
import json
import sys
import warnings
from pathlib import Path

import argparse  # noqa: F401

import numpy as np
import torch

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.common import load_config, resolve      # noqa: E402

TXN = "transaction"
MAX_AISLADAS_PCT = 25.0        # generoso a propósito: alarma, no termómetro


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
    import argparse
    ap = argparse.ArgumentParser(description="Salud del grafo (test + informe)")
    ap.add_argument("--informe", action="store_true", help="números finos")
    ap.add_argument("--guardar", metavar="RUTA")
    ap.add_argument("--contra", metavar="RUTA", help="compara con un guardado")
    a = ap.parse_args()

    cfg = load_config()
    if a.informe or a.guardar or a.contra:
        import json as _json
        r = radiografia(cfg)
        _pinta(r, _json.load(open(a.contra)) if a.contra else None)
        if a.guardar:
            Path(a.guardar).write_text(_json.dumps(r, indent=2, ensure_ascii=False))
            print(f"\n  guardado en {a.guardar}")
        print()
        return 0

    g = resolve(cfg, "graph_dir") / "graph.pt"
    if not g.exists():
        print(f"  SALTADO: no existe {g}. Corre la etapa `graph`.")
        return 0

    data = torch.load(g, weights_only=False)
    meta = json.load(open(resolve(cfg, "graph_dir") / "graph_meta.json"))
    cols = meta["feature_cols_gnn"]
    N = data[TXN].num_nodes
    t = data[TXN].time
    fallos = []
    recibe_alguna = np.zeros(N, bool)

    esperadas = list(cfg["graph"]["entidades"])
    faltan = [e for e in esperadas if e not in data.node_types]
    if faltan:
        fallos.append(f"entidades del config que NO están en el grafo: {faltan}")

    for nt in [n for n in data.node_types if n != TXN]:
        sub = data[(TXN, f"en_{nt}", nt)].edge_index
        baj = data[(nt, f"tiene_{nt}", TXN)].edge_index
        mal = []

        if data[nt].num_nodes == 0 or sub.shape[1] == 0:
            mal.append("no tiene nodos o no tiene aristas")
        if baj.shape[1] > sub.shape[1]:
            mal.append(f"la bajada ({baj.shape[1]}) supera a la subida "
                       f"({sub.shape[1]}); E1 poda una sola dirección")

        if baj.numel():
            recibe_alguna[baj[1].numpy()] = True
            clave = baj[0].to(torch.int64) * (int(t.max()) + 1) + t[baj[1]].to(torch.int64)
            fuera = int((clave[1:] < clave[:-1]).sum())
            if fuera:
                mal.append(f"{fuera} aristas de bajada fuera de orden temporal")

        c = f"__grado_{nt}"
        if c in cols:
            v = data[TXN].x[:, cols.index(c)].numpy()
            con_arista = np.zeros(N, bool)
            con_arista[sub[0].numpy()] = True
            if con_arista.any() and float(v[con_arista].max()) == 0.0:
                mal.append(f"{c} vale 0 en TODAS las filas con arista")

        print(f"  [{'MAL' if mal else 'OK '}] {nt:<7} {data[nt].num_nodes:>7} nodos · "
              f"subida {sub.shape[1]:>7} · bajada {baj.shape[1]:>7}")
        fallos += [f"{nt}: {m}" for m in mal]

    aisladas = 100 * float((~recibe_alguna).mean())
    ok = aisladas <= MAX_AISLADAS_PCT
    print(f"  [{'OK ' if ok else 'MAL'}] transacciones sin recibir de nadie: "
          f"{int((~recibe_alguna).sum())} ({aisladas:.1f}%, tope {MAX_AISLADAS_PCT}%)")
    if not ok:
        fallos.append(f"el {aisladas:.1f}% de las transacciones no recibe de "
                      f"ninguna entidad; el grafo dejó de conectar")

    if not torch.isfinite(data[TXN].x).all():
        fallos.append("hay NaN o Inf en las features de transaction")

    if fallos:
        print("\n  GRAFO ENFERMO:")
        for f in fallos:
            print(f"   · {f}")
        return 1
    print("\n  Grafo sano — todas las entidades conectan y las features son finitas.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
