"""
TODAS las métricas de la corrida, en un solo sitio.

Lo llama `final_comparison` al terminar, así que sale con cada corrida del
pipeline. También se puede lanzar suelto:

    python3 -m src.comparison.resumen            solo imprime
    python3 -m src.comparison.resumen --json     además escribe resumen.json

Recoge lo que cada etapa dejó suelto y lo junta en cuatro bloques, con LAS
MISMAS SEIS MÉTRICAS en los tres momentos que importan: las redes en
`gnn_valida`, las cabezas en `cabezas_validan` y todas en `examen`. Más el
recall al 2% aparte, que es la métrica de negocio.

Existe porque los números acababan repartidos en ocho ficheros y había que
abrirlos todos para contar una historia.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.common import load_config, resolve


def _leer(p: Path):
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:
        return None


M6 = ("pr_auc", "auc_roc", "recall", "precision", "f1", "accuracy")
CAB = ("pr_auc", "roc", "recall", "prec", "f1", "acc")


def _seis(d: dict) -> dict:
    """Las SEIS métricas, siempre las mismas y en el mismo orden."""
    return {k: round(float(d.get(k, 0)), 4) for k in M6}


def _tabla(titulo: str, filas: dict, extra: str = ""):
    print(f"\n  {titulo}")
    print("  " + "-" * 72)
    print(f"   {'':<20}" + "".join(f"{c:>9}" for c in CAB) + extra)
    for nom, d in filas.items():
        print(f"   {nom:<20}" + "".join(f"{d[k]:>9.4f}" for k in M6))


def main(escribir_json: bool):
    cfg = load_config()
    rep, mod = resolve(cfg, "reports_dir"), resolve(cfg, "models_dir")
    out = {}

    # ── 1. las redes, medidas en gnn_valida ──────────────────────────────
    redes = {}
    for f in sorted(rep.glob("*_seed*_val.json")):
        d = _leer(f)
        if d:
            redes[f.stem.replace("_val", "")] = _seis(d)
    if redes:
        out["redes"] = redes
        _tabla("REDES GNN — medidas en gnn_valida", redes)
    sel = (_leer(mod / "selected_model.json") or {}).get("selection")
    if sel:
        out["gana"] = f"{sel['selected']} seed={sel['seed']}"
        print(f"\n   GANA: {out['gana']}  ·  {sel['reason']}")

    # ── 2. las cabezas, medidas en cabezas_validan ───────────────────────
    h = _leer(rep / "heads_variantes.json")
    if h:
        cab = {v: _seis(r) for v, r in h.get("variantes", {}).items()}
        out["cabezas"] = cab
        out["arboles"] = {v: r.get("n_estimators")
                          for v, r in h.get("variantes", {}).items()}
        _tabla("CABEZAS XGBoost — medidas en cabezas_validan", cab)
        print("   árboles: " + " · ".join(f"{v} {n}" for v, n in out["arboles"].items()))

    # ── 3. el examen ─────────────────────────────────────────────────────
    fc = _leer(rep / "final_comparison.json")
    if not fc:
        print("\n  Falta reports/final_comparison.json (¿corriste `final`?)\n")
        return
    ex = fc.get("examen", {})
    fin = {v: _seis(r) for v, r in ex.get("modelos", {}).items()}
    out["examen"] = fin
    _tabla(f"EXAMEN — {ex.get('n', 0):,} txn · {ex.get('n_fraud', 0):,} fraudes", fin)

    # ── 4. el 2%, aparte ─────────────────────────────────────────────────
    dos = next((f for f in ex.get("presupuesto", []) if f.get("pct") == 2.0), None)
    if dos:
        out["recall_2pct"] = {k: v for k, v in dos.items()
                              if k not in ("pct", "n_alertas")}
        out["alertas_2pct"] = dos["n_alertas"]
        print(f"\n  RECALL AL 2% DE ALERTAS  ({dos['n_alertas']:,} revisiones "
              f"de {ex.get('n', 0):,})")
        print("  " + "-" * 72)
        for k, v in out["recall_2pct"].items():
            n_f = ex.get("n_fraud", 0)
            print(f"   {k:<20}{v:>9.4f}   {int(v * n_f):>4} de {n_f} fraudes")

    # ── 4b. por historial del cliente ────────────────────────────────────
    # El grafo SOLO puede aportar donde el cliente tiene transacciones
    # anteriores. Un promedio sobre toda la población lo diluye con las filas
    # donde el grafo no tiene nada que decir.
    hist = ex.get("por_historial") or []
    if hist:
        out["por_historial"] = hist
        print(f"\n  POR HISTORIAL DEL CLIENTE (uid)")
        print("  " + "-" * 72)
        modelos = list(hist[0]["modelos"])
        print(f"   {'grupo':<11}{'txn':>7}{'fraudes':>9}{'%fr':>7}"
              + "".join(f"{m[:14]:>15}" for m in modelos))
        for f in hist:
            fila = "".join(
                f"{(f['modelos'][m]['pr_auc'] if f['modelos'][m]['pr_auc'] is not None else 0):>15.4f}"
                for m in modelos)
            print(f"   {f['grupo']:<11}{f['n']:>7}{f['n_fraud']:>9}"
                  f"{f['pct_fraude']:>6.1f}%{fila}")
        if "control" in modelos and "gnn_mas_tabular" in modelos:
            print(f"\n   aporte del grafo por grupo:")
            for f in hist:
                c = f["modelos"]["control"]["pr_auc"]
                g = f["modelos"]["gnn_mas_tabular"]["pr_auc"]
                if c is not None and g is not None:
                    print(f"     {f['grupo']:<11}{g - c:+.4f}"
                          f"   ({f['modelos']['gnn_mas_tabular']['capturados']} "
                          f"contra {f['modelos']['control']['capturados']} "
                          f"fraudes capturados de {f['n_fraud']})")

    # ── 5. el veredicto ──────────────────────────────────────────────────
    a = fc.get("atribucion")
    if a:
        b = a.get("bootstrap", {})
        imp = (a.get("importancia") or {}).get("gnn_mas_tabular", {})
        out["veredicto"] = {
            "aporte_del_grafo": a.get("aporte_del_grafo"),
            "ic95": b.get("ic95"),
            "significativo": b.get("significativo"),
            "embedding_pct_ganancia": (imp.get("ganancia_pct") or {}).get("embedding"),
        }
        print("\n  VEREDICTO")
        print("  " + "-" * 72)
        print(f"   aporte del grafo   {a.get('aporte_del_grafo', 0):+.4f}")
        if b:
            print(f"   IC95               [{b['ic95'][0]:+.4f}, {b['ic95'][1]:+.4f}]"
                  f"   {'SIGNIFICATIVO' if b.get('significativo') else 'no significativo'}")
        if imp:
            print(f"   embedding aporta   {(imp.get('ganancia_pct') or {}).get('embedding', 0):.1f}%"
                  f" de la ganancia ({imp.get('embedding_usadas', 0)}/"
                  f"{imp.get('embedding_totales', 0)} columnas)")

    if escribir_json:
        p = rep / "resumen.json"
        with open(p, "w") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"\n   -> {p}")
    print()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true", help="además escribe reports/resumen.json")
    main(ap.parse_args().json)
