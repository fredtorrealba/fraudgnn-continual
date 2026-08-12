"""
Explorador INTERACTIVO del grafo: genera un HTML navegable.

Carga la adyacencia completa en el navegador y parte mostrando un solo nodo
semilla; el resto del grafo se va descubriendo a medida que expandes:

  - doble clic en un nodo  -> expande sus vecinos
  - clic simple            -> ficha del nodo (tid, mes, split, etiqueta, score)
  - arrastrar / rueda      -> mover y hacer zoom
  - buscador por TransactionID
  - boton para expandir todo lo visible de una pasada

Uso:
  python scripts/graph_explorer.py                      # semilla: fraude del mes 6
  python scripts/graph_explorer.py --tid 3003456
  python scripts/graph_explorer.py --month 2 --hops 1   # arranca con 1 salto abierto
  python scripts/graph_explorer.py --graph data/graph/graph_scored.pt

Genera reports/grafo_interactivo.html y lo abre en el navegador.
Requiere conexion a internet la primera vez (carga vis-network por CDN).
"""
import argparse
import json
import sys
import webbrowser
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def build_payload(d, seed_idx: int, hops: int) -> dict:
    """Serializa el grafo completo + el conjunto inicial que se muestra."""
    n = d.num_nodes
    src, dst = d.edge_index.tolist()

    adj: list[list[int]] = [[] for _ in range(n)]
    for a, b in zip(src, dst):
        adj[a].append(b)          # edge_index ya viene en ambos sentidos

    has_score = hasattr(d, "fraud_score") and bool((~torch.isnan(d.fraud_score)).any())
    split = ["-"] * n
    for s in ("train", "val", "test"):
        m = getattr(d, f"{s}_mask", None)
        if m is not None:
            for i in m.nonzero(as_tuple=True)[0].tolist():
                split[i] = s

    # conjunto inicial: la semilla y su vecindario de `hops` saltos
    seen = {seed_idx}
    frontier = [seed_idx]
    for _ in range(max(0, hops)):
        nxt = []
        for i in frontier:
            for j in adj[i]:
                if j not in seen:
                    seen.add(j)
                    nxt.append(j)
        frontier = nxt

    return {
        "tid": d.transaction_id.tolist(),
        "y": d.y.int().tolist(),
        "month": d.month.tolist() if hasattr(d, "month") else [0] * n,
        "week": d.week_in_month.tolist() if hasattr(d, "week_in_month") else [0] * n,
        "split": split,
        "score": (torch.nan_to_num(d.fraud_score, nan=-1.0).tolist()
                  if has_score else None),
        "adj": adj,
        "seed": seed_idx,
        "initial": sorted(seen),
        "nNodes": n,
        "nEdges": len(src) // 2,
    }


HTML = r"""
<meta charset="utf-8">
<title>FraudGNN — explorador del grafo</title>
__VIS__
<style>
  :root { --bg:#12151c; --panel:#1b2029; --line:#2c3444; --txt:#e6e9ef; --dim:#94a0b4;
          --fraud:#e5484d; --legit:#4c88d8; --accent:#f0b429; }
  * { box-sizing:border-box; }
  body { margin:0; height:100vh; display:flex; background:var(--bg); color:var(--txt);
         font:13px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }
  #net { flex:1; height:100vh; }
  #side { width:310px; background:var(--panel); border-left:1px solid var(--line);
          padding:16px; overflow-y:auto; display:flex; flex-direction:column; gap:14px; }
  h1 { font-size:15px; margin:0; letter-spacing:.2px; }
  .sub { color:var(--dim); font-size:11px; margin-top:2px; }
  .box { background:#151a22; border:1px solid var(--line); border-radius:8px; padding:10px; }
  .row { display:flex; justify-content:space-between; padding:3px 0; font-size:12px; }
  .row span:first-child { color:var(--dim); }
  button { width:100%; background:#232b38; color:var(--txt); border:1px solid var(--line);
           border-radius:6px; padding:8px; cursor:pointer; font-size:12px; margin-bottom:6px; }
  button:hover { background:#2c3644; border-color:#3d4a5e; }
  input { width:100%; background:#151a22; color:var(--txt); border:1px solid var(--line);
          border-radius:6px; padding:7px; font-size:12px; }
  label { color:var(--dim); font-size:11px; display:block; margin-bottom:4px; }
  .legend { display:flex; align-items:center; gap:7px; font-size:12px; padding:2px 0; }
  .dot { width:11px; height:11px; border-radius:50%; }
  .hint { color:var(--dim); font-size:11px; border-top:1px solid var(--line); padding-top:10px; }
  kbd { background:#232b38; border:1px solid var(--line); border-radius:3px;
        padding:1px 4px; font-size:10px; }
</style>
<div id="net"></div>
<div id="side">
  <div>
    <h1>Explorador del grafo</h1>
    <div class="sub" id="meta"></div>
  </div>

  <div class="box">
    <div class="legend"><div class="dot" style="background:var(--fraud)"></div> fraude (y=1)</div>
    <div class="legend"><div class="dot" style="background:var(--legit)"></div> legítima (y=0)</div>
    <div class="legend"><div class="dot" style="background:none;border:2px solid var(--accent)"></div> semilla</div>
  </div>

  <div>
    <label>Buscar TransactionID</label>
    <input id="q" placeholder="p. ej. 3003456" />
  </div>

  <div>
    <label>Máx. vecinos por expansión</label>
    <input id="cap" type="number" value="25" min="1" max="200" />
  </div>

  <div>
    <button id="expandAll">Expandir todo lo visible</button>
    <button id="labels">Ocultar etiquetas</button>
    <button id="physics">Pausar physics</button>
    <button id="reset">Volver a la semilla</button>
  </div>

  <div class="box" id="info"><div class="sub">Haz clic en un nodo para ver su ficha.</div></div>

  <div class="hint">
    <kbd>doble clic</kbd> expande vecinos<br>
    <kbd>clic</kbd> ficha del nodo<br>
    <kbd>arrastrar</kbd> mover · <kbd>rueda</kbd> zoom<br><br>
    Los nodos con borde punteado tienen vecinos sin expandir.
  </div>
</div>
<script>
const D = __DATA__;
const FRAUD = "#e5484d", LEGIT = "#4c88d8", ACCENT = "#f0b429";

const tid2idx = new Map();
D.tid.forEach((t, i) => tid2idx.set(t, i));

const nodes = new vis.DataSet();
const edges = new vis.DataSet();
const shown = new Set();
const expanded = new Set();
const edgeKey = (a, b) => a < b ? a + "_" + b : b + "_" + a;

function styleOf(i) {
  const isFraud = D.y[i] === 1;
  const pend = D.adj[i].some(j => !shown.has(j));
  return {
    color: {
      background: isFraud ? FRAUD : LEGIT,
      border: i === D.seed ? ACCENT : (isFraud ? "#a32a2e" : "#31578f"),
    },
    borderWidth: i === D.seed ? 4 : (pend ? 3 : 1),
    shapeProperties: { borderDashes: pend && i !== D.seed ? [3, 3] : false },
    size: 8 + Math.min(14, D.adj[i].length * 0.5),
  };
}

function tooltip(i) {
  const s = D.score && D.score[i] >= 0 ? " · score " + D.score[i].toFixed(3) : "";
  return `tid ${D.tid[i]} · mes ${D.month[i]} · ${D.split[i]} · ` +
         `${D.y[i] === 1 ? "FRAUDE" : "legítima"} · grado ${D.adj[i].length}${s}`;
}

function addNode(i) {
  if (shown.has(i)) return false;
  shown.add(i);
  nodes.add(Object.assign({ id: i, label: String(D.tid[i]), title: tooltip(i) }, styleOf(i)));
  for (const j of D.adj[i]) {
    if (shown.has(j) && j !== i) {
      const k = edgeKey(i, j);
      if (!edges.get(k)) edges.add({ id: k, from: i, to: j });
    }
  }
  return true;
}

function refresh(list) {
  nodes.update([...list].filter(i => shown.has(i)).map(i => Object.assign({ id: i }, styleOf(i))));
}

function expand(i) {
  const cap = Math.max(1, parseInt(document.getElementById("cap").value) || 25);
  const pend = D.adj[i].filter(j => !shown.has(j));
  pend.slice(0, cap).forEach(addNode);
  expanded.add(i);
  refresh(new Set([i, ...D.adj[i]]));
  stats();
}

function stats() {
  document.getElementById("meta").textContent =
    `${shown.size} de ${D.nNodes.toLocaleString()} nodos · ` +
    `${edges.length} de ${D.nEdges.toLocaleString()} aristas`;
}

function showInfo(i) {
  const r = (k, v) => `<div class="row"><span>${k}</span><span>${v}</span></div>`;
  const pend = D.adj[i].filter(j => !shown.has(j)).length;
  document.getElementById("info").innerHTML =
    r("TransactionID", D.tid[i]) + r("mes", D.month[i] + " · sem " + D.week[i]) +
    r("split", D.split[i]) +
    r("etiqueta", D.y[i] === 1 ? "<b style='color:#e5484d'>FRAUDE</b>" : "legítima") +
    (D.score && D.score[i] >= 0 ? r("fraud_score", D.score[i].toFixed(4)) : "") +
    r("grado total", D.adj[i].length) + r("vecinos ocultos", pend);
}

const network = new vis.Network(document.getElementById("net"), { nodes, edges }, {
  nodes: { shape: "dot", font: { color: "#8994a6", size: 9, face: "monospace" } },
  edges: { color: { color: "#3d4757", highlight: ACCENT }, width: 1, smooth: false },
  physics: { solver: "forceAtlas2Based",
             forceAtlas2Based: { gravitationalConstant: -55, springLength: 90 },
             stabilization: { iterations: 220 } },
  interaction: { hover: true, tooltipDelay: 120, multiselect: true },
});

network.on("doubleClick", p => { if (p.nodes.length) expand(p.nodes[0]); });
network.on("click", p => { if (p.nodes.length) showInfo(p.nodes[0]); });

document.getElementById("expandAll").onclick = () => {
  [...shown].filter(i => !expanded.has(i)).forEach(expand);
};
document.getElementById("reset").onclick = () => {
  nodes.clear(); edges.clear(); shown.clear(); expanded.clear();
  D.initial.forEach(addNode); stats(); network.fit();
};
let labelsOn = true;
document.getElementById("labels").onclick = e => {
  labelsOn = !labelsOn;
  network.setOptions({ nodes: { font: { size: labelsOn ? 9 : 0 } } });
  e.target.textContent = labelsOn ? "Ocultar etiquetas" : "Mostrar etiquetas";
};
let phy = true;
document.getElementById("physics").onclick = e => {
  phy = !phy; network.setOptions({ physics: { enabled: phy } });
  e.target.textContent = phy ? "Pausar physics" : "Reanudar physics";
};
document.getElementById("q").onkeydown = e => {
  if (e.key !== "Enter") return;
  const i = tid2idx.get(parseInt(e.target.value));
  if (i === undefined) { e.target.value = "no encontrado"; return; }
  addNode(i); expand(i); showInfo(i);
  network.selectNodes([i]); network.focus(i, { scale: 1.1, animation: true });
};

D.initial.forEach(addNode);
stats();
showInfo(D.seed);
</script>
"""


VIS_URL = "https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"
VIS_CACHE = ROOT / "scripts" / ".vis-network.min.js"


def vis_script(offline: bool) -> str:
    """CDN por defecto; con --offline embebe la libreria dentro del HTML."""
    if not offline:
        return f'<script src="{VIS_URL}"></script>'

    if not VIS_CACHE.exists():
        import urllib.request
        print(f"Descargando vis-network una sola vez -> {VIS_CACHE.name} ...")
        try:
            urllib.request.urlretrieve(VIS_URL, VIS_CACHE)
        except Exception as e:                                   # noqa: BLE001
            sys.exit(f"No pude descargar vis-network ({e}).\n"
                     f"Con internet corre una vez sin --offline, o baja el archivo "
                     f"manualmente desde {VIS_URL} y guardalo en {VIS_CACHE}")
    js = VIS_CACHE.read_text(encoding="utf-8")
    # un </script> dentro del JS cerraria la etiqueta antes de tiempo
    return "<script>" + js.replace("</script>", "<\\/script>") + "</script>"


def main():
    p = argparse.ArgumentParser(description="Explorador interactivo del grafo")
    p.add_argument("--graph", default="data/graph/graph.pt")
    p.add_argument("--tid", type=int, help="TransactionID de la semilla")
    p.add_argument("--month", type=int, default=6,
                   help="mes del que sacar la semilla si no das --tid (default 6)")
    p.add_argument("--no-fraud", dest="fraud", action="store_false", default=True,
                   help="semilla legitima en vez de fraude")
    p.add_argument("--hops", type=int, default=1,
                   help="saltos abiertos al arrancar (default 1)")
    p.add_argument("--out", default="reports/grafo_interactivo.html")
    p.add_argument("--offline", action="store_true",
                   help="embeber vis-network en el HTML (funciona sin internet)")
    p.add_argument("--no-open", dest="open_browser", action="store_false", default=True)
    args = p.parse_args()

    gpath = Path(args.graph)
    gpath = gpath if gpath.is_absolute() else ROOT / gpath
    if not gpath.exists():
        sys.exit(f"No existe {gpath}. Corre antes: python -m src.data.build_graph")
    d = torch.load(gpath, weights_only=False)

    deg = torch.bincount(d.edge_index[0], minlength=d.num_nodes).float()
    if args.tid is not None:
        hit = (d.transaction_id == args.tid).nonzero(as_tuple=True)[0]
        if len(hit) == 0:
            sys.exit(f"No encontre la transaccion {args.tid}.")
        seed = int(hit[0])
    else:
        mask = (d.y == 1) if args.fraud else (d.y == 0)
        if hasattr(d, "month"):
            mask = mask & (d.month == args.month)
        cand = mask.nonzero(as_tuple=True)[0]
        if len(cand) == 0:
            sys.exit("Ningun nodo cumple el filtro; prueba otro --month o --no-fraud.")
        seed = int(cand[deg[cand].argmax()])   # el mas conectado: hay algo que ver

    payload = build_payload(d, seed, args.hops)
    out = Path(args.out)
    out = out if out.is_absolute() else ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    html = (HTML.replace("__DATA__", json.dumps(payload, separators=(",", ":")))
                .replace("__VIS__", vis_script(args.offline)))
    out.write_text(html, encoding="utf-8")

    mb = out.stat().st_size / 1e6
    print(f"Generado {out}  ({mb:.1f} MB"
          + (", autocontenido: no necesita internet)" if args.offline else ")"))
    print(f"Semilla: txn {int(d.transaction_id[seed])} | "
          f"{'FRAUDE' if d.y[seed] == 1 else 'legitima'} | "
          f"mes {int(d.month[seed]) if hasattr(d, 'month') else '?'} | "
          f"grado {int(deg[seed])} | arranca con {len(payload['initial'])} nodos visibles")
    if args.open_browser:
        webbrowser.open(out.as_uri())


if __name__ == "__main__":
    main()
