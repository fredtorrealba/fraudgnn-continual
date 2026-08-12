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
    """Esqueleto compartido: 3 convoluciones + MLP head."""

    def __init__(self, in_dim: int, hidden_dims: list[int], mlp_dim: int,
                 dropout: float):
        super().__init__()
        self.dropout = dropout
        h1, h2, h3 = hidden_dims
        self.conv1 = self._make_conv(in_dim, h1)
        self.conv2 = self._make_conv(h1, h2)
        self.conv3 = self._make_conv(h2, h3)
        self.bn1, self.bn2, self.bn3 = (nn.BatchNorm1d(h) for h in (h1, h2, h3))
        self.classifier = nn.Sequential(
            nn.Linear(h3, mlp_dim), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(mlp_dim, 1),
        )

    def _make_conv(self, in_c, out_c):  # pragma: no cover - abstracta
        raise NotImplementedError

    def forward(self, x, edge_index):
        for conv, bn in ((self.conv1, self.bn1), (self.conv2, self.bn2),
                         (self.conv3, self.bn3)):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.classifier(x).squeeze(-1)  # logits

    @torch.no_grad()
    def predict_proba(self, x, edge_index):
        self.eval()
        return torch.sigmoid(self.forward(x, edge_index))

    def param_groups(self, lrs: dict) -> list[dict]:
        """
        Grupos de parámetros con LR diferenciado para el fine-tuning:
        lrs = {"layer1": 0.0, "layer2": 1e-5, "layer3": 1e-4, "classifier": 1e-3}
        (BatchNorm de cada capa acompaña a su convolución.)
        """
        return [
            {"params": list(self.conv1.parameters()) + list(self.bn1.parameters()),
             "lr": lrs["layer1"]},
            {"params": list(self.conv2.parameters()) + list(self.bn2.parameters()),
             "lr": lrs["layer2"]},
            {"params": list(self.conv3.parameters()) + list(self.bn3.parameters()),
             "lr": lrs["layer3"]},
            {"params": self.classifier.parameters(), "lr": lrs["classifier"]},
        ]


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
        # concat=False -> promedio de cabezas, mantiene las dimensiones 256/128/64
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
