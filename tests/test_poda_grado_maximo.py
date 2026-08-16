"""
INVARIANTE E2 — el grafo tiene las aristas que dicen los datos, ni una menos.

Se comprueba contra `full.parquet`, NO contra el propio grafo: se recalculan las
claves de entidad desde las columnas originales y se cuenta cuántas
transacciones deberían tener arista. Verificar el grafo consigo mismo aprobaría
el mismo error dos veces.

Con `max_entity_degree: 0` (recomendado) toda transacción con clave no nula
tiene su arista de subida. La poda por grado era redundante con el muestreo
—`vecinos_por_entidad: 10` ya limita lo que baja cada entidad— y cara: con 500,
72 entidades de las 10.106 de `card` dejaban sin arista al 24,6% del dataset,
justo el tramo con más fraude, y ponían `__grado_card` a 0 en las tarjetas con
MÁS historial.

Con `max_entity_degree > 0` se comprueba la cuenta de la poda, que es
`Σ max(0, grado - (tope+1))` por entidad: sobrevive hasta la transacción con
`previas == tope`, o sea las primeras `tope+1`.

Necesita `data/graph/graph.pt` y `data/processed/full.parquet`. Sin GPU.

    python tests/test_poda_grado_maximo.py
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.build_graph import _previas_por_entidad, clave_entidad   # noqa: E402
from src.utils.common import load_config, resolve                      # noqa: E402

TXN = "transaction"


def main() -> int:
    cfg = load_config()
    tope = int(cfg["graph"].get("max_entity_degree", 500))
    g = resolve(cfg, "graph_dir") / "graph.pt"
    p = resolve(cfg, "processed_dir") / "full.parquet"
    if not (g.exists() and p.exists()):
        print(f"  SALTADO: falta {g if not g.exists() else p}.")
        return 0

    data = torch.load(g, weights_only=False)
    df = pd.read_parquet(p)
    fallos = []
    print(f"  config: max_entity_degree = {tope}"
          f"{'   (SIN poda)' if tope <= 0 else ''}")

    for nombre, spec in cfg["graph"]["entidades"].items():
        if nombre not in data.node_types:
            continue
        # Lo que DEBERÍA haber, recalculado desde las columnas originales.
        claves = clave_entidad(df, spec)
        presentes = claves.notna()
        # `graph.meses` limita las aristas (no los nodos de transacción)
        meses = cfg["graph"].get("meses")
        if meses:
            presentes &= df["month"].isin(meses)
        cod, _ = pd.factorize(claves[presentes], sort=False)
        filas = np.where(presentes.values)[0]
        previas = _previas_por_entidad(cod, df["TransactionDT"].values[filas])
        esperadas = int((previas <= tope).sum()) if tope > 0 else int(len(previas))

        reales = int(data[(TXN, f"en_{nombre}", nombre)].edge_index.shape[1])
        ok = reales == esperadas
        if not ok:
            fallos.append(
                f"{nombre}: el grafo tiene {reales} aristas de subida y los datos "
                f"dicen {esperadas}"
                + ("" if tope > 0 else
                   ". Con max_entity_degree=0 no debe perderse ninguna."))

        # Sin poda, además: ninguna entidad puede estar truncada.
        extra = ""
        if tope <= 0:
            e_sub = data[(TXN, f"en_{nombre}", nombre)].edge_index[1].numpy()
            grado_grafo = np.bincount(e_sub, minlength=data[nombre].num_nodes)
            grado_datos = np.bincount(cod, minlength=data[nombre].num_nodes)
            n = min(len(grado_grafo), len(grado_datos))
            truncadas = int((grado_grafo[:n] < grado_datos[:n]).sum())
            if truncadas:
                fallos.append(f"{nombre}: {truncadas} entidades truncadas pese a "
                              f"max_entity_degree=0")
            extra = f" · grado máx {int(grado_datos.max())}"

        print(f"  [{'OK ' if ok else 'MAL'}] {nombre:<7} subida {reales:>7} "
              f"(datos: {esperadas}){extra}")

    # La feature de grado no puede quedar topada por la poda.
    import json
    cols = json.load(open(resolve(cfg, "graph_dir") / "graph_meta.json"))["feature_cols_gnn"]
    if tope <= 0 and "__grado_card" in cols:
        v = data[TXN].x[:, cols.index("__grado_card")].numpy()
        techo = float(np.log1p(500))
        if v.max() <= techo + 1e-6:
            fallos.append(
                f"__grado_card llega como mucho a {v.max():.3f} = log1p(500). "
                f"Sigue topada por la poda vieja.")
        else:
            print(f"  [OK ] __grado_card máx {v.max():.2f} "
                  f"(el techo de la poda era {techo:.2f})")

    if fallos:
        print("\n  FALLA EL INVARIANTE E2:")
        for f in fallos:
            print(f"   · {f}")
        return 1
    print(f"\n  E2 OK — el grafo coincide con los datos y con max_entity_degree "
          f"= {tope}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
