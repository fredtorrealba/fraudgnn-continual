"""
Las dos redes GNN del proyecto. Misma arquitectura base
(432 -> 256 -> 128 -> 64 + cabeza MLP), difieren SOLO en la agregación:

- GraphSAGE (principal): agregación MEAN de vecinos sampleados. Costo por
  nodo fijo e independiente del tamaño del grafo, inductivo (infiere sobre
  nodos nunca vistos) -> la elección para producción.
- GAT (alternativa): pesos de atención APRENDIDOS por vecino. Más expresiva,
  más costosa.

3 capas = el nodo "ve" hasta 3 saltos. Más capas -> over-smoothing.
Output: 1 logit -> sigmoid = P(fraude). El threshold vive fuera del modelo.

Nota de continual learning: las capas se exponen como conv1/conv2/conv3/
classifier para que el fine-tuning pueda asignar LR diferenciado por grupo
(capa 1 congelada, capa 2 casi congelada, capa 3 LR bajo, clasificador LR
normal).
"""
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GATConv, SAGEConv


class _BaseGNN(nn.Module):
    """
    Esqueleto compartido: N convoluciones + MLP head.

    N = len(hidden_dims), así que la PROFUNDIDAD es configurable desde
    config.yaml sin tocar código:
        hidden_dims: [256, 128, 64]  -> 3 capas = 3 saltos
        hidden_dims: [256, 128]      -> 2 capas
        hidden_dims: [256]           -> 1 capa

    Importa porque cada capa es un SALTO en el grafo, y la señal no se reparte
    por igual: medida sobre este dataset, la separación fraude/legítima es de
    3.9x a 1 salto, 0.7x a 2 y 0.3x a 3 — a partir del segundo salto se agrega
    ruido (over-smoothing). Los `fanouts` deben tener tantos elementos como
    capas; `src.gnn.sampling.fanouts()` se encarga de recortarlos.
    """

    def __init__(self, in_dim: int, hidden_dims: list[int], mlp_dim: int,
                 dropout: float):
        super().__init__()
        if not hidden_dims:
            raise ValueError("hidden_dims no puede estar vacío")
        self.dropout = dropout
        dims = [in_dim, *hidden_dims]
        self.convs = nn.ModuleList(
            self._make_conv(dims[i], dims[i + 1]) for i in range(len(hidden_dims)))
        self.bns = nn.ModuleList(nn.BatchNorm1d(h) for h in hidden_dims)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dims[-1], mlp_dim), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(mlp_dim, 1),
        )

    @property
    def n_capas(self) -> int:
        return len(self.convs)

    def _make_conv(self, in_c, out_c):  # pragma: no cover - abstracta
        raise NotImplementedError

    def encode(self, x, edge_index):
        """
        Representación del nodo TRAS las convoluciones y ANTES del clasificador.

        Es el vector que resume "yo + mi vecindario": `hidden_dims[-1]`
        dimensiones (256 con una capa). El sistema híbrido puede consumirlo
        entero en vez del escalar que devuelve `forward`, que lo colapsa a un
        único número y descarta 255 de esas 256 dimensiones.
        """
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    def forward(self, x, edge_index):
        return self.classifier(self.encode(x, edge_index)).squeeze(-1)  # logits

    @property
    def dim_embedding(self) -> int:
        return self.classifier[0].in_features

    @torch.no_grad()
    def predict_proba(self, x, edge_index):
        self.eval()
        return torch.sigmoid(self.forward(x, edge_index))

    def param_groups(self, lrs: dict) -> list[dict]:
        """
        Grupos de parámetros con LR diferenciado para el fine-tuning:
        lrs = {"layer1": 0.0, "layer2": 1e-5, "layer3": 1e-4, "classifier": 1e-3}

        Con menos de 3 capas se usan los ÚLTIMOS N valores de la lista, para
        conservar el gradiente de plasticidad que persigue el diseño: la capa
        más profunda es la más libre de moverse y la primera la más congelada.
        Con 1 capa, esa capa recibe `layer3` (la más plástica) — congelarla
        dejaría solo el clasificador entrenando.
        BatchNorm de cada capa acompaña a su convolución.
        """
        escala = [lrs["layer1"], lrs["layer2"], lrs["layer3"]][-self.n_capas:]
        grupos = [
            {"params": list(c.parameters()) + list(b.parameters()), "lr": lr}
            for c, b, lr in zip(self.convs, self.bns, escala)
        ]
        grupos.append({"params": self.classifier.parameters(),
                       "lr": lrs["classifier"]})
        return grupos


class FraudGraphSAGE(_BaseGNN):
    """RED 1 (principal): agregación MEAN, inductiva, costo fijo."""

    def _make_conv(self, in_c, out_c):
        return SAGEConv(in_c, out_c, aggr="mean")


class FraudGAT(_BaseGNN):
    """RED 2 (alternativa): atención por vecino aprendida."""

    def __init__(self, in_dim, hidden_dims, mlp_dim, dropout, heads: int = 4):
        self.heads = heads
        super().__init__(in_dim, hidden_dims, mlp_dim, dropout)

    def _make_conv(self, in_c, out_c):
        # concat=False -> promedio de cabezas, respeta las dims de hidden_dims
        return GATConv(in_c, out_c, heads=self.heads, concat=False)


def build_model(name: str, cfg: dict):
    g = cfg["gnn"]
    common = dict(in_dim=g["in_dim"], hidden_dims=g["hidden_dims"],
                  mlp_dim=g["mlp_head_dim"], dropout=g["dropout"])
    if name.lower() in ("graphsage", "sage"):
        return FraudGraphSAGE(**common)
    if name.lower() == "gat":
        return FraudGAT(**common, heads=g["gat_heads"])
    raise ValueError(f"Modelo desconocido: {name}")
