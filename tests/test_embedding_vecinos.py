"""
INVARIANTE E0 — el embedding "solo vecinos" tiene que contener vecinos.

Por qué existe: este fallo estuvo activo toda la primera fase del proyecto SIN
producir un solo síntoma. Nada falló, no hubo warning, y las métricas salían
plausibles. `embv_*` era `constante + 4 proyecciones de las features propias` y
CERO información del vecindario, porque se capturaba al final de la capa 1 —
cuando los nodos de entidad todavía están en ceros y no han repartido nada.

El resultado del capstone (`aporte del grafo`) se calcula con esas columnas, así
que el fallo no daba un error: daba una respuesta equivocada a la pregunta de la
tesis. Por eso se prueba con un grafo de juguete y no esperando a una corrida.

No necesita GPU, ni `pyg-lib`, ni datos. Dos segundos.

    python tests/test_embedding_vecinos.py
"""
import sys
import warnings
from pathlib import Path

import torch

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from torch_geometric.data import HeteroData          # noqa: E402
from src.gnn.models import FraudGraphSAGE            # noqa: E402

D = 4
ENT = ["uid", "card", "email", "device", "net"]
UMBRAL = 1e-4


def _grafo(feat_vecinas: torch.Tensor) -> HeteroData:
    """txn_0 es la raíz; las demás cuelgan del mismo `uid`, así que son vecinas."""
    d = HeteroData()
    d["transaction"].x = torch.cat([torch.ones(1, D), feat_vecinas])
    for e in ENT:
        d[e].num_nodes = 1
    n = feat_vecinas.shape[0] + 1
    todas = list(range(n))
    d["transaction", "en_uid", "uid"].edge_index = torch.tensor([todas, [0] * n])
    d["uid", "tiene_uid", "transaction"].edge_index = torch.tensor([[0] * n, todas])
    for e in ENT[1:]:                       # las demás entidades: solo la raíz
        d["transaction", f"en_{e}", e].edge_index = torch.tensor([[0], [0]])
        d[e, f"tiene_{e}", "transaction"].edge_index = torch.tensor([[0], [0]])
    return d


def _modelo():
    meta = (["transaction"] + ENT,
            [("transaction", f"en_{e}", e) for e in ENT] +
            [(e, f"tiene_{e}", "transaction") for e in ENT])
    return FraudGraphSAGE(metadata=meta, in_dim=D, hidden_dims=[8, 8],
                          mlp_dim=4, dropout=0.0,
                          aggr=["mean", "max", "std"]).eval()


@torch.no_grad()
def _vecinos(m, g):
    return m.encode(g.x_dict, g.edge_index_dict, g, solo_vecinos=True)[1][0]


def main() -> int:
    torch.manual_seed(0)
    m = _modelo()
    fallos = []

    # 1. MISMA raíz, vecinas opuestas -> el embedding TIENE que moverse.
    a = _grafo(torch.zeros(2, D))
    b = _grafo(torch.full((2, D), 9.0))
    dif = float((_vecinos(m, a) - _vecinos(m, b)).abs().max())
    ok = dif > UMBRAL
    print(f"  [{'OK ' if ok else 'MAL'}] reacciona a los vecinos      "
          f"cambio = {dif:.6f}  (umbral {UMBRAL})")
    if not ok:
        fallos.append(
            "El embedding 'solo vecinos' NO cambia al cambiar los vecinos.\n"
            "     Se está capturando antes de que los nodos de entidad repartan\n"
            "     su resumen. Revisa `encode()`: debe recogerse en la ÚLTIMA capa\n"
            "     (`i == len(self.convs) - 1`), no en la primera.")

    # 2. Con más vecinas el resumen cambia -> también debe moverse.
    c = _grafo(torch.full((5, D), 9.0))
    dif2 = float((_vecinos(m, b) - _vecinos(m, c)).abs().max())
    ok2 = dif2 > UMBRAL
    print(f"  [{'OK ' if ok2 else 'MAL'}] reacciona a CUÁNTOS vecinos  "
          f"cambio = {dif2:.6f}")
    if not ok2:
        fallos.append("El embedding no distingue 2 vecinas de 5.")

    # 3. Se restan los CINCO términos de raíz, no uno.
    #    `_termino_vecinos` hacía `return` dentro del bucle y dejaba 4 copias de
    #    las features propias dentro del embedding que existe para quitarlas.
    #    OJO con el ModuleDict de PyG: iterarlo directo (`for et in convs`)
    #    devuelve las claves SERIALIZADAS ('<transaction___en_uid___uid>') y la
    #    comparación `et[2] == "transaction"` da 0 siempre. Solo `.items()` y
    #    `.keys()` devuelven la tupla. Esta prueba llegó a reportar "0 aristas
    #    entrantes" por eso mismo.
    conv = m.convs[-1]
    entrantes = [(et, sub) for et, sub in conv.convs.items()
                 if et[2] == "transaction"]
    restados = sum(1 for _, sub in entrantes
                   if getattr(sub, "lin_r", None) is not None)
    ok3 = len(entrantes) == 5 and restados == len(entrantes)
    print(f"  [{'OK ' if ok3 else 'MAL'}] resta las {len(entrantes)} aristas "
          f"entrantes    (lin_r encontrados: {restados})")
    if not ok3:
        fallos.append(
            f"Se esperaban 5 aristas entrantes a 'transaction' con lin_r y hay "
            f"{len(entrantes)} con {restados} lin_r.\n"
            "     `_termino_vecinos` tiene que restarlas TODAS: HeteroConv suma\n"
            "     una convolución por tipo de arista, así que la salida lleva un\n"
            "     término lin_r(x_i) por cada una. Con `return` dentro del bucle\n"
            "     solo se quitaba la primera.")

    if fallos:
        print("\n  FALLA EL INVARIANTE E0:")
        for f in fallos:
            print(f"   · {f}")
        return 1
    print("\n  E0 OK — el embedding 'solo vecinos' contiene vecinos.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
