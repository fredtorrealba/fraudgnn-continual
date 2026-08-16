"""
INVARIANTE E1 — el grafo en disco coincide con `graph.min_previas_entidad`.

Una entidad con una sola transacción no conecta a nadie: su vector ES esa
transacción, y al bajar se la devuelve. La transacción se recibe a sí misma. En
`uid` pasaba en 59.189 de sus 94.901 nodos (62%), el 26,8% del dataset.

Lo que se prueba, y es lo fácil de romper:

  1. La poda es ASIMÉTRICA
       SUBIDA  transaction -> entidad   TODAS. Si se podara, las compras
                                        siguientes de un cliente nunca sabrían
                                        de la primera.
       BAJADA  entidad -> transaction   solo con `previas >= minimo`.

  2. La cuenta es EXACTA. Para cada entidad se quitan `min(minimo, grado)`
     aristas de bajada — las de sus primeras transacciones. Sale de que
     `previas` es la posición dentro del grupo ordenado por tiempo, así que
     recorre 0..grado-1 sin huecos.

  3. Se comprueba el valor QUE HAYA en el config, incluido `0`. Una versión
     anterior de este test se saltaba el 0 en vez de verificar que desactiva
     la poda, así que no probaba lo que el comentario del config promete.

  4. CAUSAL: ninguna entidad entrega a una transacción anterior a su primera.
     Se recalcula desde el grafo, sin reutilizar el código que lo construyó —
     si usara la misma función, aprobaría el mismo error dos veces.

Necesita `data/graph/graph.pt` construido. No necesita GPU ni `pyg-lib`.

    python tests/test_grado_minimo_entidad.py
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import torch

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.common import load_config, resolve      # noqa: E402

TXN = "transaction"


def main() -> int:
    cfg = load_config()
    minimo = int(cfg["graph"].get("min_previas_entidad", 1))
    ruta = resolve(cfg, "graph_dir") / "graph.pt"
    if not ruta.exists():
        print(f"  SALTADO: no existe {ruta}. Corre la etapa `graph`.")
        return 0

    data = torch.load(ruta, weights_only=False)
    t = data[TXN].time.numpy()
    fallos = []
    print(f"  config: min_previas_entidad = {minimo}"
          f"{'   (poda DESACTIVADA)' if minimo <= 0 else ''}")

    for nt in [n for n in data.node_types if n != TXN]:
        sub = data[(TXN, f"en_{nt}", nt)].edge_index         # txn -> entidad
        baj = data[(nt, f"tiene_{nt}", TXN)].edge_index      # entidad -> txn
        n_ent = data[nt].num_nodes
        e_sub, txn_sub = sub[1].numpy(), sub[0].numpy()
        grado = np.bincount(e_sub, minlength=n_ent)
        mal = []

        # 1+2. La cuenta exacta, para el valor que sea.
        esperado = int(np.minimum(max(minimo, 0), grado).sum())
        quitadas = sub.shape[1] - baj.shape[1]
        if quitadas != esperado:
            mal.append(f"se quitaron {quitadas} aristas de bajada y tocaban "
                       f"{esperado} = suma de min({minimo}, grado) por entidad")

        # 3. Con minimo <= 0 la poda no debe existir: las dos direcciones iguales.
        if minimo <= 0 and baj.shape[1] != sub.shape[1]:
            mal.append(f"con minimo={minimo} la poda debería estar desactivada "
                       f"y la bajada ({baj.shape[1]}) no iguala a la subida "
                       f"({sub.shape[1]})")

        # 4. Causal: nadie recibe siendo anterior a la primera de su entidad.
        e_baj, txn_baj = baj[0].numpy(), baj[1].numpy()
        primero = np.full(n_ent, np.iinfo(np.int64).max, dtype=np.int64)
        np.minimum.at(primero, e_sub, t[txn_sub])
        antes = int((t[txn_baj] < primero[e_baj]).sum()) if len(txn_baj) else 0
        if antes:
            mal.append(f"{antes} aristas de bajada van a transacciones "
                       f"ANTERIORES a la primera de su entidad (imposible)")

        # 5. Con la poda activa, las entidades de grado <= minimo quedan mudas.
        if minimo > 0:
            entrega = np.zeros(n_ent, dtype=bool)
            entrega[e_baj] = True
            cortas = np.flatnonzero((grado > 0) & (grado <= minimo))
            vivas = int(entrega[cortas].sum())
            if vivas:
                mal.append(f"{vivas} entidades con grado <= {minimo} siguen "
                           f"entregando; deberían quedar mudas")
            mudas = f" · {len(cortas)} mudas"
        else:
            mudas = ""

        print(f"  [{'MAL' if mal else 'OK '}] {nt:<7} subida {sub.shape[1]:>7} · "
              f"bajada {baj.shape[1]:>7} · -{quitadas}{mudas}")
        fallos += [f"{nt}: {m}" for m in mal]

    if fallos:
        print("\n  FALLA EL INVARIANTE E1:")
        for f in fallos:
            print(f"   · {f}")
        return 1
    print(f"\n  E1 OK — el grafo coincide con min_previas_entidad = {minimo}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
