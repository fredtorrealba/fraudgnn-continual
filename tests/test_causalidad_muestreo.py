"""
A2 — ¿el muestreo mira solo hacia atrás?

Es EL invariante del diseño temporal: si un vecino muestreado es posterior a
su transacción raíz, la GNN está viendo el futuro y todas las métricas del
proyecto son optimistas por construcción, sin que nada falle ni avise.

`make_hetero_loader` lo pide con `time_attr="time"` y `temporal_strategy="last"`
(gnn/sampling.py:257). Esto comprueba que PyG lo esté cumpliendo de verdad, que
es distinto de que se lo hayamos pedido.

SOLO CORRE EN EL POD: necesita el sampler nativo (`pyg-lib`), que no está en
macOS. Por eso la auditoría del 15/08 no pudo verificar este punto.

    python scripts/verificar_causalidad.py                # 20 lotes
    python scripts/verificar_causalidad.py --lotes 100    # más exigente

Sale con código 1 si encuentra una violación, para poder encadenarlo en CI.

La comprobación es POR SEMILLA, no por lote. `time_attr` activa `disjoint`
automáticamente en PyG: cada semilla arrastra su propio subgrafo y el tensor
`batch` dice a qué semilla pertenece cada nodo. Comparar contra el máximo del
lote entero sería una prueba mucho más débil —la pasaría un muestreo que le
diera a la transacción más antigua los vecinos de la más reciente—.
"""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.gnn.sampling import TXN, cerrar_loader, make_hetero_loader
from src.utils.common import get_logger, load_config, resolve, set_seed

log = get_logger("verificar_causalidad")


def main() -> int:
    ap = argparse.ArgumentParser(description="A2: causalidad del muestreo")
    ap.add_argument("--lotes", type=int, default=20)
    ap.add_argument("--semillas", type=int, default=8192)
    args = ap.parse_args()

    cfg = load_config()
    set_seed(42)
    # Sin workers: son 20 lotes, arrancar procesos cuesta más que muestrear.
    cfg["gnn"]["num_workers"] = 0

    ruta = resolve(cfg, "graph_dir") / "graph.pt"
    data = torch.load(ruta, weights_only=False)
    log.info("Grafo: %s | %d transacciones | %d tipos de arista",
             ruta, data[TXN].num_nodes, len(data.edge_types))

    mask = torch.zeros(data[TXN].num_nodes, dtype=torch.bool)
    gen = torch.Generator().manual_seed(42)
    mask[torch.randperm(data[TXN].num_nodes, generator=gen)[:args.semillas]] = True

    loader = make_hetero_loader(data, cfg, mask, shuffle=False, batch_size=512)
    viol = comparados = n_sem = n_nod = 0
    peor = 0.0
    sin_disjoint = False

    try:
        for i, b in enumerate(loader):
            if i >= args.lotes:
                break
            n = b[TXN].batch_size
            t = b[TXN].time
            n_sem += n
            n_nod += b[TXN].num_nodes

            asignacion = getattr(b[TXN], "batch", None)
            if asignacion is None:
                # Sin `disjoint` no hay forma de saber de qué semilla vino cada
                # nodo. Se degrada al máximo del lote y se avisa: la prueba
                # sigue detectando fugas groseras, pero no las finas.
                sin_disjoint = True
                mal = t[n:] > t[:n].max()
                viol += int(mal.sum())
                comparados += len(t) - n
                if mal.any():
                    peor = max(peor, float((t[n:][mal] - t[:n].max()).max()))
                continue

            # Cada nodo se compara contra SU raíz.
            raiz = t[:n][asignacion]
            mal = t > raiz
            mal[:n] = False                      # la semilla es su propia raíz
            viol += int(mal.sum())
            comparados += len(t) - n
            if mal.any():
                peor = max(peor, float((t[mal] - raiz[mal]).max()))
    finally:
        cerrar_loader(loader)

    if sin_disjoint:
        log.warning("El lote no trae `batch`: `disjoint` no está activo. La "
                    "comprobación se hizo contra el máximo del lote, que es "
                    "MÁS DÉBIL. Revisa que `time_attr` esté puesto.")

    log.info("%d semillas -> %d nodos transaction (x%.1f de expansión)",
             n_sem, n_nod, n_nod / max(1, n_sem))
    log.info("Vecinos comparados: %d", comparados)

    if viol:
        log.error("FUGA TEMPORAL: %d vecinos (%.4f%%) son POSTERIORES a su "
                  "raíz. El peor se adelanta %.1f días.",
                  viol, 100 * viol / max(1, comparados), peor / 86400)
        log.error("Las métricas del proyecto NO son válidas mientras esto pase.")
        return 1

    log.info("OK: ningún vecino es posterior a su raíz. El muestreo solo mira "
             "hacia atrás.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
