"""
Visualizaciones del modelo y del grafo. Dos modos:

  --mode tsne   (por defecto) t-SNE del espacio latente aprendido por la GNN
  --mode graph  el grafo dibujado: nodos, aristas y los fraudes en ROJO

------------------------------------------------------------------------------
MODO GRAPH
------------------------------------------------------------------------------
Dibuja la topología para ver DÓNDE caen los fraudes. Como el grafo son cientos
de islas inconexas (cada tarjeta/dispositivo forma su grupo), un force-directed
global daría una nube sin estructura: se calcula el layout de CADA componente
por separado y se empacan ordenadas por tamaño. Los fraudes se pintan encima y
más grandes para que no queden tapados por las legítimas.

  python scripts/plot_embeddings.py --mode graph
  python scripts/plot_embeddings.py --mode graph --month 6      # solo el test
  python scripts/plot_embeddings.py --mode graph --hide-isolated
  python scripts/plot_embeddings.py --mode graph --out reports/g.svg   # zoom infinito

No necesita modelo entrenado: solo data/graph/graph.pt (paso 2).

------------------------------------------------------------------------------
MODO TSNE
------------------------------------------------------------------------------
t-SNE del espacio latente aprendido por la GNN.

Proyecta en 2D los embeddings de 64 dims (la salida de conv3, justo antes del
clasificador) y los colorea por clase. Sirve para mostrar QUE aprendio la red,
no solo cuanto acierta.

El modo estrella es la comparacion ANTES vs DESPUES del continual learning:

    python scripts/plot_embeddings.py --auto

Dibuja dos paneles con los mismos nodos del mes 6: a la izquierda el modelo
pre-CL, a la derecha el desplegado tras el ciclo. Los fraudes EMERGENTES (los
que el modelo pre-CL dejaba pasar con score < threshold) van marcados con una
X amarilla en ambos paneles: se ve como pasan de estar diluidos entre las
legitimas a agruparse. Esa es la evidencia visual del OE3.

Los embeddings se extraen con un forward hook sobre classifier[0]: no se
modifica nada de src/gnn/models.py.

Uso:
  python scripts/plot_embeddings.py --auto              # antes vs despues del CL
  python scripts/plot_embeddings.py                     # solo el modelo seleccionado
  python scripts/plot_embeddings.py --ckpt models/graphsage_seed42.pt
  python scripts/plot_embeddings.py --split val --max-points 4000
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.gnn.models import build_model                       # noqa: E402
from src.utils.common import load_config, resolve            # noqa: E402

FRAUD_C = "#e5484d"
LEGIT_C = "#4c88d8"
EMERG_C = "#f0b429"


def resolve_checkpoints(cfg, args) -> list[tuple[str, Path]]:
    """Devuelve [(titulo, ruta)] segun el modo pedido."""
    models_dir = resolve(cfg, "models_dir")

    if args.ckpt:
        return [("modelo", Path(args.ckpt) if Path(args.ckpt).is_absolute()
                 else ROOT / args.ckpt)]

    sel_file = models_dir / "selected_model.json"
    if not sel_file.exists():
        sys.exit("Falta models/selected_model.json. Corre antes:\n"
                 "  python -m src.gnn.compare_gnns")
    with open(sel_file) as f:
        sel = json.load(f)["selection"]      # el JSON anida todo bajo "selection"
    pre = models_dir / sel["checkpoint"]

    if not args.auto:
        return [(f"modelo seleccionado ({sel['checkpoint']})", pre)]

    post = models_dir / "production_model.pt"
    if not post.exists():
        sys.exit("Falta models/production_model.pt (no hay modelo post-CL).\n"
                 "Corre antes:  python -m src.continual_learning.cl_orchestrator\n"
                 "O usa el script sin --auto para ver solo el modelo actual.")
    return [("ANTES del CL", pre), ("DESPUES del CL", post)]


def embed(ckpt_path: Path, data, cfg):
    """Carga el checkpoint y devuelve (embeddings 64d, scores) de todo el grafo."""
    if not ckpt_path.exists():
        sys.exit(f"No existe el checkpoint {ckpt_path}")
    ckpt = torch.load(ckpt_path, weights_only=False)
    cfg["gnn"]["in_dim"] = ckpt["in_dim"]
    model = build_model(ckpt["model_name"], cfg)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    grabbed = {}
    hook = model.classifier[0].register_forward_hook(
        lambda m, inp, out: grabbed.__setitem__("z", inp[0].detach()))
    with torch.no_grad():
        logits = model(data.x, data.edge_index)   # forward completo (no sampleado)
    hook.remove()

    return grabbed["z"].numpy(), torch.sigmoid(logits).numpy()


def pick_nodes(data, args, cfg) -> np.ndarray:
    """Nodos a proyectar: por defecto el mes de test, con todos los fraudes."""
    mask = torch.ones(data.num_nodes, dtype=torch.bool)
    if args.split and args.split != "all":
        mask &= getattr(data, f"{args.split}_mask")
    if args.month is not None and hasattr(data, "month"):
        mask &= data.month == args.month
    idx = mask.nonzero(as_tuple=True)[0].numpy()
    if len(idx) == 0:
        sys.exit("Ningun nodo cumple el filtro (--split / --month).")

    if len(idx) <= args.max_points:
        return idx
    # muestreo estratificado: los fraudes son escasos, se conservan todos
    y = data.y.numpy()[idx]
    fraud, legit = idx[y == 1], idx[y == 0]
    rng = np.random.default_rng(args.seed)
    keep = max(0, args.max_points - len(fraud))
    legit = rng.choice(legit, size=min(keep, len(legit)), replace=False)
    print(f"  muestreo: {len(fraud)} fraudes (todos) + {len(legit)} legitimas")
    return np.concatenate([fraud, legit])


# =============================================================================
# MODO GRAPH — la topología, con los fraudes en rojo
# =============================================================================
def node_filter(data, args):
    """Máscara de nodos según --split / --month (None = todos)."""
    mask = torch.ones(data.num_nodes, dtype=torch.bool)
    if args.split and args.mode == "graph" and args.split != "all":
        mask &= getattr(data, f"{args.split}_mask")
    if args.month is not None and hasattr(data, "month"):
        mask &= data.month == args.month
    return mask


def build_nx(data, keep_mask, max_nodes: int, seed: int):
    """Grafo networkx de los nodos filtrados, con muestreo si son demasiados."""
    import networkx as nx

    idx = keep_mask.nonzero(as_tuple=True)[0]
    if len(idx) > max_nodes:
        # se conservan TODOS los fraudes: con 3% de positivos, un muestreo
        # uniforme dejaría la imagen sin nada rojo que mirar
        y = data.y[idx].numpy()
        rng = np.random.default_rng(seed)
        fraud = idx[y == 1]
        legit = idx[y == 0].numpy()
        legit = rng.choice(legit, size=max(0, max_nodes - len(fraud)), replace=False)
        idx = torch.tensor(np.concatenate([fraud.numpy(), legit]))
        print(f"  [!] {int(keep_mask.sum()):,} nodos superan el tope de {max_nodes:,}: "
              f"se muestrearon {len(idx):,} (todos los fraudes + legítimas al azar)")

    keep = torch.zeros(data.num_nodes, dtype=torch.bool)
    keep[idx] = True
    src, dst = data.edge_index
    sel = keep[src] & keep[dst]
    remap = torch.full((data.num_nodes,), -1, dtype=torch.long)
    order = idx.sort().values
    remap[order] = torch.arange(len(order))

    G = nx.Graph()
    G.add_nodes_from(range(len(order)))
    G.add_edges_from(zip(remap[src[sel]].tolist(), remap[dst[sel]].tolist()))
    return G, order


def layout_components(G, seed: int, hide_isolated: bool):
    """Layout por componente + empaquetado en filas, ordenado por tamaño."""
    import networkx as nx

    comps = sorted(nx.connected_components(G), key=len, reverse=True)
    multi = [c for c in comps if len(c) > 1]
    singles = [next(iter(c)) for c in comps if len(c) == 1]

    pos = {}
    radii = [math.sqrt(len(c)) for c in multi]
    row_w = max(1.0, math.sqrt(sum(2 * r + 1 for r in radii) * max(radii, default=1) * 2.2))

    x = y = row_h = 0.0
    for comp, r in zip(multi, radii):
        w = 2 * r + 1.0
        if x + w > row_w and x > 0:
            x, y, row_h = 0.0, y - row_h, 0.0
        sub = G.subgraph(comp)
        p = nx.spring_layout(sub, seed=seed, iterations=50,
                             k=1.1 / math.sqrt(len(comp)))
        arr = np.array([p[v] for v in sub.nodes()])
        arr = arr / (np.abs(arr).max() or 1.0) * r
        for v, (px, py) in zip(sub.nodes(), arr):
            pos[v] = (x + r + px, y - r + py)
        x += w
        row_h = max(row_h, w)

    if singles and not hide_isolated:
        cols = max(1, int(math.sqrt(len(singles) * 2.5)))
        y0 = y - row_h - 3.0
        for i, v in enumerate(singles):
            pos[v] = ((i % cols) * 0.55, y0 - (i // cols) * 0.55)
    return pos, len(multi), len(singles)


def draw_graph(data, args):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.collections import LineCollection
        from matplotlib.lines import Line2D
    except ImportError:
        sys.exit("Falta matplotlib:  pip install matplotlib")

    keep = node_filter(data, args)
    print(f"Construyendo el grafo ({int(keep.sum()):,} nodos filtrados)...")
    G, order = build_nx(data, keep, args.max_nodes, args.seed)
    y = data.y[order].numpy()
    print(f"  {G.number_of_nodes():,} nodos | {G.number_of_edges():,} aristas | "
          f"{int(y.sum()):,} fraudes")

    print("Calculando layout por componentes...")
    pos, n_comp, n_iso = layout_components(G, args.seed, args.hide_isolated)
    print(f"  {n_comp:,} componentes con aristas | {n_iso:,} aislados"
          + (" (ocultos)" if args.hide_isolated else ""))

    drawn = sorted(pos.keys())
    P = np.array([pos[v] for v in drawn])
    yv = y[drawn]

    bg, fg = ("#0e1117", "#e6e9ef") if args.dark else ("white", "#222222")
    print("Dibujando...")
    fig, ax = plt.subplots(figsize=(args.figsize, args.figsize * 0.72), facecolor=bg)
    ax.set_facecolor(bg)

    seg = [(pos[a], pos[b]) for a, b in G.edges() if a in pos and b in pos]
    ax.add_collection(LineCollection(seg, colors="#3a4658" if args.dark else "#9aa7b8",
                                     linewidths=0.25, alpha=args.edge_alpha, zorder=1))
    leg = yv == 0
    ax.scatter(P[leg, 0], P[leg, 1], s=args.node_size, c=LEGIT_C, linewidths=0, zorder=2)
    ax.scatter(P[~leg, 0], P[~leg, 1], s=args.node_size * 2.6, c=FRAUD_C,
               linewidths=0, zorder=3)          # el fraude encima: si no, queda tapado

    scope = []
    if args.month is not None:
        scope.append(f"mes {args.month}")
    if args.split and args.split != "all":
        scope.append(args.split)
    ax.set_title(
        f"Grafo{' — ' + ', '.join(scope) if scope else ''}\n"
        f"{len(drawn):,} nodos · {len(seg):,} aristas · {int(yv.sum()):,} fraudes "
        f"({yv.mean() * 100:.2f}%) · {n_comp:,} componentes · {n_iso:,} aislados",
        fontsize=13, color=fg, pad=14)
    ax.legend(handles=[
        Line2D([], [], marker="o", ls="", color=FRAUD_C, label="fraude (y=1)"),
        Line2D([], [], marker="o", ls="", color=LEGIT_C, label="legítima (y=0)"),
    ], loc="lower right", frameon=False, labelcolor=fg)
    ax.autoscale_view()
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout()

    out = Path(args.out)
    out = out if out.is_absolute() else ROOT / out
    if out.name == "tsne_embeddings.png":            # default del otro modo
        out = out.with_name("grafo.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=args.dpi, facecolor=bg, bbox_inches="tight")
    print(f"Guardado en {out}  ({out.stat().st_size / 1e6:.1f} MB)")
    if out.suffix.lower() == ".svg":
        print("  Es vectorial: ábrelo en el navegador y haz zoom sin perder nitidez.")


def main():
    p = argparse.ArgumentParser(description="t-SNE del espacio latente de la GNN")
    p.add_argument("--mode", choices=["tsne", "graph"], default="tsne",
                   help="tsne: espacio latente | graph: la topología con fraudes en rojo")
    p.add_argument("--graph", default="data/graph/graph.pt")
    p.add_argument("--ckpt", help="checkpoint concreto (.pt)")
    p.add_argument("--auto", action="store_true",
                   help="comparar pre-CL vs production_model.pt")
    p.add_argument("--split", choices=["train", "val", "test", "all"], default="test")
    p.add_argument("--month", type=int, help="filtrar por mes (opcional)")
    p.add_argument("--out", default="reports/tsne_embeddings.png")
    p.add_argument("--seed", type=int, default=42)
    # --- solo modo tsne ---
    p.add_argument("--max-points", type=int, default=3000)
    p.add_argument("--perplexity", type=float, default=30.0)
    # --- solo modo graph ---
    p.add_argument("--max-nodes", type=int, default=20000,
                   help="tope de nodos dibujados; sobre eso se muestrea (modo graph)")
    p.add_argument("--hide-isolated", action="store_true",
                   help="ocultar los nodos sin aristas (modo graph)")
    p.add_argument("--node-size", type=float, default=2.5)
    p.add_argument("--edge-alpha", type=float, default=0.22)
    p.add_argument("--figsize", type=float, default=22.0)
    p.add_argument("--dpi", type=int, default=200)
    p.add_argument("--dark", action="store_true", help="fondo oscuro (modo graph)")
    args = p.parse_args()

    cfg = load_config()
    gpath = Path(args.graph)
    gpath = gpath if gpath.is_absolute() else ROOT / gpath
    if not gpath.exists():
        sys.exit(f"No existe {gpath}. Corre antes: python -m src.data.build_graph")
    data = torch.load(gpath, weights_only=False)

    if args.mode == "graph":                 # no necesita modelo entrenado
        draw_graph(data, args)
        return

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        from sklearn.manifold import TSNE
    except ImportError:
        sys.exit("Falta matplotlib:  pip install matplotlib")

    ckpts = resolve_checkpoints(cfg, args)
    nodes = pick_nodes(data, args, cfg)
    y = data.y.numpy()[nodes]
    print(f"Proyectando {len(nodes):,} nodos ({int(y.sum())} fraudes)")

    # Los fraudes emergentes se definen SIEMPRE con el primer modelo (el "antes"):
    # son los que ese modelo dejaba pasar por debajo del umbral de alerta.
    thr = cfg["gnn"]["threshold"]
    panels, emergent = [], None
    for title, path in ckpts:
        print(f"[{title}] {path.name}")
        z_all, s_all = embed(path, data, cfg)
        if emergent is None:
            emergent = (y == 1) & (s_all[nodes] < thr)
            print(f"  fraudes emergentes (score < {thr} con el 1er modelo): "
                  f"{int(emergent.sum())}")
        perp = min(args.perplexity, max(5.0, (len(nodes) - 1) / 3))
        z = TSNE(n_components=2, random_state=args.seed, perplexity=perp,
                 init="pca").fit_transform(z_all[nodes])
        panels.append((title, z, s_all[nodes]))

    fig, axes = plt.subplots(1, len(panels), figsize=(8.2 * len(panels), 7.2),
                             squeeze=False)
    for ax, (title, z, scores) in zip(axes[0], panels):
        ax.scatter(z[y == 0, 0], z[y == 0, 1], s=9, c=LEGIT_C, alpha=0.45,
                   linewidths=0, label="legítima")
        normal_fraud = (y == 1) & ~emergent
        ax.scatter(z[normal_fraud, 0], z[normal_fraud, 1], s=26, c=FRAUD_C,
                   alpha=0.9, linewidths=0, label="fraude")
        if emergent.any():
            ax.scatter(z[emergent, 0], z[emergent, 1], s=64, c=EMERG_C,
                       marker="X", linewidths=0.4, edgecolors="#5a4300",
                       label="fraude emergente")
        rec = (scores[y == 1] >= thr).mean() if (y == 1).any() else 0.0
        rec_e = scores[emergent] >= thr
        extra = f" · recall emergentes {rec_e.mean():.0%}" if emergent.any() else ""
        ax.set_title(f"{title}\nrecall {rec:.0%}{extra}", fontsize=12)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color("#d0d5dd")

    axes[0][0].legend(handles=[
        Line2D([], [], marker="o", ls="", color=LEGIT_C, label="legítima (y=0)"),
        Line2D([], [], marker="o", ls="", color=FRAUD_C, label="fraude detectado"),
        Line2D([], [], marker="X", ls="", color=EMERG_C, label="fraude emergente"),
    ], loc="best", frameon=False, fontsize=9)

    scope = args.split + (f", mes {args.month}" if args.month else "")
    fig.suptitle(f"Espacio latente de la GNN (t-SNE de los embeddings 64d) — {scope}",
                 fontsize=13)
    if len(panels) > 1:
        fig.text(0.5, 0.015, "Cada panel tiene su propio t-SNE: los ejes NO son "
                 "comparables entre paneles, sí lo es la ESTRUCTURA (qué se "
                 "agrupa con qué).", ha="center", fontsize=9, color="#667085")
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))

    out = Path(args.out)
    out = out if out.is_absolute() else ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Guardado en {out}")


if __name__ == "__main__":
    main()
