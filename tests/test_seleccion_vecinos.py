"""
¿El muestreo baja de verdad las 10 transacciones MÁS RECIENTES ANTERIORES?

    python scripts/probar_seleccion_vecinos.py
    python scripts/probar_seleccion_vecinos.py --semillas 200

Es la pieza de la que depende todo el diseño temporal, y se pide con dos
opciones de `NeighborLoader` (`gnn/sampling.py`):

    time_attr="time"             solo vecinos ANTERIORES a la raíz
    temporal_strategy="last"     y de esos, los N más recientes

Que se lo pidamos no significa que lo haga. Esto lo comprueba contra la verdad
calculada a mano desde el grafo.

DOS MODOS, según lo que haya instalado:

  COMPLETO   con `pyg-lib`. Muestrea de verdad y compara nodo a nodo lo que
             bajó contra lo que debería haber bajado.

  PRECONDICIÓN   sin `pyg-lib` (macOS). No puede llamar al sampler, pero sí
             verificar lo que el sampler necesita: que las aristas de bajada
             estén ordenadas por (entidad, tiempo). `temporal_strategy="last"`
             coge un sufijo de esa lista, así que si el orden no es exacto,
             "los 10 más recientes" devuelve cualquier cosa y nadie avisa.
             Además simula la selección e imprime qué debería salir.
"""
import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import torch

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.gnn.sampling import TXN, _tiene_sampler_nativo     # noqa: E402
from src.utils.common import load_config, resolve           # noqa: E402


def _verdad(data, raiz: int, nt: str, k: int):
    """Las k transacciones que DEBERÍAN bajar por esa entidad, a mano."""
    t = data[TXN].time.numpy()
    baj = data[(nt, f"tiene_{nt}", TXN)].edge_index.numpy()
    sub = data[(TXN, f"en_{nt}", nt)].edge_index.numpy()
    # ¿a qué entidad pertenece la raíz?
    ent = sub[1][sub[0] == raiz]
    if not len(ent):
        return None, None
    e = int(ent[0])
    # ¿esa entidad entrega a la raíz? (poda de grado mínimo, E1)
    if not (baj[1][baj[0] == e] == raiz).any():
        return e, None
    hermanas = sub[0][sub[1] == e]                    # todas las de la entidad
    anteriores = hermanas[t[hermanas] <= t[raiz]]     # solo pasado (<= incluye la raíz)
    orden = np.argsort(t[anteriores], kind="stable")
    return e, anteriores[orden][-k:]                  # las k más recientes


def precondicion(data) -> int:
    """Sin sampler: comprobar el orden que `temporal_strategy` da por hecho."""
    t = data[TXN].time
    fallos = 0
    print("  ORDEN de las aristas de bajada (lo que exige temporal_strategy)")
    for nt in [n for n in data.node_types if n != TXN]:
        ei = data[(nt, f"tiene_{nt}", TXN)].edge_index
        if not ei.numel():
            continue
        clave = ei[0].to(torch.int64) * (int(t.max()) + 1) + t[ei[1]].to(torch.int64)
        desordenadas = int((clave[1:] < clave[:-1]).sum())
        print(f"    [{'OK ' if not desordenadas else 'MAL'}] {nt:<7} "
              f"{ei.shape[1]:>7} aristas · {desordenadas} fuera de orden")
        fallos += bool(desordenadas)
    return fallos


def main() -> int:
    ap = argparse.ArgumentParser(description="Selección de vecinos por recencia")
    ap.add_argument("--semillas", type=int, default=50)
    a = ap.parse_args()

    cfg = load_config()
    k = int(cfg["graph"].get("vecinos_por_entidad", 10))
    data = torch.load(resolve(cfg, "graph_dir") / "graph.pt", weights_only=False)
    t = data[TXN].time.numpy()
    print(f"\n  vecinos_por_entidad = {k} · {data[TXN].num_nodes} transacciones\n")

    fallos = precondicion(data)

    # Qué DEBERÍA bajar, calculado a mano. Se imprime siempre: sirve de
    # referencia para comparar con lo que el pod devuelva.
    rng = np.random.default_rng(42)
    entidades = [n for n in data.node_types if n != TXN]
    muestras, ventanas = [], []
    for raiz in rng.choice(data[TXN].num_nodes, size=a.semillas, replace=False):
        for nt in entidades:
            e, esperadas = _verdad(data, int(raiz), nt, k)
            if esperadas is None or len(esperadas) < 2:
                continue
            muestras.append(len(esperadas))
            # ¿en cuánto tiempo caben esas k? (el riesgo de las entidades enormes)
            ventanas.append((t[esperadas].max() - t[esperadas].min()) / 3600)
    print(f"\n  SIMULACIÓN sobre {a.semillas} semillas · {len(muestras)} pares "
          f"(transacción, entidad) con vecindario")
    print(f"    vecinas que bajarían: mediana {int(np.median(muestras))} de {k}")
    q = np.percentile(ventanas, [10, 50, 90])
    print(f"    ventana temporal que cubren: p10 {q[0]:.1f}h · "
          f"mediana {q[1]:.1f}h · p90 {q[2]:.1f}h")
    apretadas = 100 * float(np.mean(np.asarray(ventanas) < 1))
    print(f"    casos donde las {k} caben en menos de 1 hora: {apretadas:.1f}%")

    if not _tiene_sampler_nativo():
        print("\n  MODO PRECONDICIÓN: falta `pyg-lib`, no se puede llamar al "
              "sampler.\n  Corre esto en el pod para la comprobación completa.")
        return 1 if fallos else 0

    # ---- modo completo: se muestrea de verdad y se compara ----
    from src.gnn.sampling import cerrar_loader, make_hetero_loader
    cfg["gnn"]["num_workers"] = 0
    mask = torch.zeros(data[TXN].num_nodes, dtype=torch.bool)
    idx = rng.choice(data[TXN].num_nodes, size=a.semillas, replace=False)
    mask[torch.as_tensor(idx)] = True
    loader = make_hetero_loader(data, cfg, mask, shuffle=False, batch_size=1)

    revisados = malos = futuro = 0
    try:
        for b in loader:
            raiz = int(b[TXN].n_id[0])
            traidas = set(b[TXN].n_id.numpy().tolist()) - {raiz}
            if not traidas:
                continue
            # 1. ninguna puede ser POSTERIOR a la raíz
            futuro += sum(1 for n in traidas if t[n] > t[raiz])
            # 2. todas tienen que estar entre las k más recientes de ALGUNA
            #    de las entidades de la raíz
            permitidas = set()
            for nt in entidades:
                _, esp = _verdad(data, raiz, nt, k)
                if esp is not None:
                    permitidas |= set(int(x) for x in esp)
            malos += len(traidas - permitidas - {raiz})
            revisados += len(traidas)
    finally:
        cerrar_loader(loader)

    print(f"\n  MODO COMPLETO: {revisados} vecinas muestreadas")
    print(f"    [{'OK ' if not futuro else 'MAL'}] posteriores a su raíz: {futuro}")
    print(f"    [{'OK ' if not malos else 'MAL'}] fuera de las {k} más recientes: "
          f"{malos}")
    if futuro or malos:
        print("\n    El muestreo NO respeta 'las k más recientes anteriores'. "
              "Revisa time_attr y temporal_strategy en gnn/sampling.py.")
        return 1
    print(f"\n  OK — el muestreo baja exactamente las {k} más recientes anteriores.")
    return 1 if fallos else 0


if __name__ == "__main__":
    raise SystemExit(main())
