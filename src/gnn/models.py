"""
Las dos GNN del proyecto, sobre GRAFO HETEROGÉNEO.

    transacción  <--->  [uid] [card] [email] [device] [net]

Cada capa `HeteroConv` aplica una convolución POR TIPO DE ARISTA y suma los
resultados en el nodo destino. Con dos capas:

    capa 1   las transacciones vecinas  ->  el nodo de entidad
             (la entidad aprende un vector que resume "este cliente")
    capa 2   los nodos de entidad       ->  la transacción raíz

Los nodos de entidad NO tienen features propias: entran como ceros y su
contenido sale enteramente de agregar sus transacciones. Eso hace el modelo
INDUCTIVO — una entidad vista por primera vez en el mes 6 funciona igual.

ARQUITECTURAS
- GraphSAGE: agregación explícita (`mean`/`max`/`std`), coste fijo por nodo.
- GATv2: pesos de atención APRENDIDOS por vecino. Se usa GATv2Conv y no GATConv
  porque la atención de GAT es estática —el ranking de vecinos no depende del
  nodo que pregunta— y GATv2 lo corrige (Brody et al., ICLR 2022).

LA AGREGACIÓN IMPORTA
`aggr=["mean","max","std"]` en vez de solo la media. La media destruye la
dispersión del vecindario, y la dispersión es lo ÚNICO que no está ya en el
dataset: C1-C14 son conteos, D1-D15 deltas y V1-V339 agregados de historial —
ninguna trae varianza. La salida sigue teniendo `hidden_dims[-1]` dimensiones;
solo crecen los parámetros de la capa.

LA SALIDA PARA XGBOOST
`encode()` devuelve la representación de 256; `embed()` la de `mlp_head_dim`
(32), que es la capa oculta del clasificador y la que consume el sistema
híbrido. Con `solo_vecinos=True` se descuenta el término de raíz para que
XGBoost no reciba dos veces las features propias de la transacción.
"""
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GATv2Conv, HeteroConv, SAGEConv

TXN = "transaction"


class _BaseHeteroGNN(nn.Module):
    """
    Esqueleto compartido: N capas HeteroConv + cabeza MLP.

    `metadata` es la tupla (node_types, edge_types) del HeteroData, así que el
    modelo se adapta solo a las entidades que el grafo traiga. Si una entidad se
    omite por falta de columnas (ver build_graph), aquí no hay que tocar nada.
    """

    def __init__(self, metadata, in_dim: int, hidden_dims: list[int],
                 mlp_dim: int, dropout: float, aggr, **kw):
        super().__init__()
        if not hidden_dims:
            raise ValueError("hidden_dims no puede estar vacío")
        if len(hidden_dims) < 2:
            # En el grafo BIPARTITO llegar de una transacción a otra cuesta DOS
            # saltos: transacción -> entidad -> transacción. Los nodos de
            # entidad entran como CEROS, así que con una sola capa la
            # transacción agrega ceros y el vecindario NO influye en absoluto —
            # verificado: la diferencia entre correr con y sin aristas es
            # exactamente 0.000000. Sería una MLP disfrazada de GNN.
            raise ValueError(
                f"hidden_dims tiene {len(hidden_dims)} capa y el grafo "
                "heterogéneo necesita AL MENOS 2: transacción -> entidad -> "
                "transacción. Con una sola, los nodos de entidad siguen siendo "
                "ceros cuando llegan a la transacción y el grafo no aporta "
                "nada. Usa p. ej. hidden_dims: [64, 64].")
        self.node_types, self.edge_types = metadata
        self.dropout = dropout
        self.aggr = list(aggr) if isinstance(aggr, (list, tuple)) else aggr
        self.in_dim = in_dim
        self.extra = kw

        dims = [in_dim, *hidden_dims]
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for i in range(len(hidden_dims)):
            # (-1, -1) deja que PyG infiera las dimensiones de origen y destino
            # por tipo de arista: los nodos de entidad entran sin features y su
            # tamaño lo fija la capa anterior.
            conv = HeteroConv({et: self._make_conv(dims[i], dims[i + 1])
                               for et in self.edge_types}, aggr="sum")
            self.convs.append(conv)
            # BatchNorm solo en las transacciones: son los únicos nodos con
            # semántica estable entre lotes. Los de entidad cambian de
            # composición en cada muestreo.
            self.bns.append(nn.BatchNorm1d(hidden_dims[i]))

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dims[-1], mlp_dim), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(mlp_dim, 1),
        )

    @property
    def n_capas(self) -> int:
        return len(self.convs)

    @property
    def dim_embedding(self) -> int:
        """El vector que consume XGBoost: la capa oculta del clasificador."""
        return self.classifier[0].out_features

    def _make_conv(self, in_c, out_c):  # pragma: no cover - abstracta
        raise NotImplementedError

    def _dict_inicial(self, x_dict, edge_index_dict, batch) -> dict:
        """
        Los nodos de entidad no traen features: entran como ceros.

        No es un relleno perezoso, es la decisión que hace el modelo INDUCTIVO.
        Si cada entidad tuviera su propio embedding aprendido, el modelo sería
        transductivo y no sabría qué hacer con una tarjeta que aparece por
        primera vez en el mes 6. Así su contenido sale ENTERAMENTE de agregar
        sus transacciones, y una entidad nueva funciona igual que una conocida.

        El número de nodos de cada tipo se toma del batch si lo trae; si no, se
        deduce del mayor índice que aparece en las aristas.
        """
        ref = x_dict[TXN]
        out = dict(x_dict)
        for nt in self.node_types:
            if nt in out and out[nt] is not None and out[nt].numel():
                continue
            n = 0
            if batch is not None and nt in getattr(batch, "node_types", []):
                n = int(batch[nt].num_nodes or 0)
            if not n:                      # sin batch: deducir de las aristas
                for et, ei in edge_index_dict.items():
                    if ei.numel() == 0:
                        continue
                    if et[0] == nt:
                        n = max(n, int(ei[0].max()) + 1)
                    if et[2] == nt:
                        n = max(n, int(ei[1].max()) + 1)
            out[nt] = torch.zeros(n, self.in_dim, device=ref.device, dtype=ref.dtype)
        return out

    def encode(self, x_dict, edge_index_dict, batch=None, solo_vecinos: bool = False):
        """
        Representación de las transacciones tras las convoluciones.

        Con `solo_vecinos=True` devuelve además el término que NO viene del
        propio nodo. En SAGEConv la salida es `lin_l(agregado) + lin_r(x_i)`;
        como XGBoost ya tiene `x_i` entre sus columnas tabulares, entregarle
        también `lin_r(x_i)` sería repetirle lo que ya sabe. Se despeja exacto
        restando, sin reimplementar la convolución.
        """
        x = self._dict_inicial(x_dict, edge_index_dict, batch)
        vecinos = None
        # ÚLTIMA capa, no la primera. Los nodos de entidad entran en CEROS
        # (`_dict_inicial`, es lo que los hace inductivos) y su vector se calcula
        # en la capa 0: si se captura ahí, la transacción está leyendo ceros y el
        # embedding "solo vecinos" no contiene ni un vecino.
        #
        # Medido antes del arreglo: las 64 columnas `embv_` tenían AUC mediana
        # 0.5263 por dimensión —azar— y eran, en cristiano,
        # `constante + 4 proyecciones lineales de las features propias`.
        # En la última capa las entidades ya llevan agregadas sus transacciones,
        # así que `lin_l(entidades)` sí es el camino del vecindario.
        ultima = len(self.convs) - 1
        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            h = conv(x, edge_index_dict)
            if solo_vecinos and i == ultima and TXN in h:
                # Se captura ANTES de bn/relu/dropout a propósito: `bn` es
                # BatchNorm y pasarle un segundo tensor le corrompería las
                # estadísticas móviles durante el entrenamiento.
                vecinos = self._termino_vecinos(conv, h[TXN], x[TXN])
            nuevo = {}
            for nt, v in h.items():
                if nt == TXN:
                    v = bn(v)
                nuevo[nt] = F.dropout(F.relu(v), p=self.dropout,
                                      training=self.training)
            # los tipos sin aristas entrantes en esta capa conservan su valor
            for nt, v in x.items():
                nuevo.setdefault(nt, v)
            x = nuevo
        return (x[TXN], vecinos) if solo_vecinos else x[TXN]

    def _termino_vecinos(self, conv, h_txn, x_txn):
        """
        La parte de la representación que viene SOLO del vecindario.

        `conv` es la capa donde se captura: de ahí salen los `lin_r` a restar.

        XGBoost ya tiene las 431 features propias entre sus columnas tabulares:
        entregarle también la proyección de esas mismas features sería repetirle
        lo que ya sabe y gastar capacidad del embedding en ello. Cada
        arquitectura lo despeja a su manera (ver las subclases).
        """
        raise NotImplementedError

    def embed(self, x_dict, edge_index_dict, batch=None, solo_vecinos: bool = False):
        """El vector de `mlp_head_dim` dimensiones que consume XGBoost."""
        r = self.encode(x_dict, edge_index_dict, batch, solo_vecinos)
        h, vec = r if solo_vecinos else (r, None)
        proyecta = nn.Sequential(self.classifier[0], self.classifier[1])
        return (proyecta(vec if vec is not None else h), proyecta(h)) \
            if solo_vecinos else proyecta(h)

    def forward(self, x_dict, edge_index_dict, batch=None):
        return self.classifier(self.encode(x_dict, edge_index_dict, batch)).squeeze(-1)

    def param_groups(self, lrs: dict) -> list[dict]:
        """
        LR diferenciado para el fine-tuning del continual learning: el drift
        mueve la FRONTERA, no la estructura. Con menos de 3 capas se usan los
        ÚLTIMOS N valores, para que la capa más profunda siga siendo la más
        libre y la primera la más congelada.
        """
        escala = [lrs["layer1"], lrs["layer2"], lrs["layer3"]][-self.n_capas:]
        grupos = [{"params": list(c.parameters()) + list(b.parameters()), "lr": lr}
                  for c, b, lr in zip(self.convs, self.bns, escala)]
        grupos.append({"params": self.classifier.parameters(), "lr": lrs["classifier"]})
        return grupos


class FraudGraphSAGE(_BaseHeteroGNN):
    """Agregación explícita: mean / max / std, o la combinación de las tres."""

    def _make_conv(self, in_c, out_c):
        return SAGEConv((-1, -1), out_c, aggr=self.aggr)

    def _termino_vecinos(self, conv, h_txn, x_txn):
        """
        SAGEConv calcula `lin_l(agregado) + lin_r(x_i)`, así que el término de
        vecinos se despeja restando — exacto, sin reimplementar la convolución.

        SE RESTAN LAS CINCO, no una. A `transaction` entran cinco tipos de arista
        (uid, card, email, device, net) y `HeteroConv` los suma con `aggr="sum"`,
        así que la salida lleva CINCO términos `lin_r_k(x_i)`, uno por tipo. La
        versión anterior hacía `return` dentro del bucle en la primera y dejaba
        cuatro copias de las features propias dentro del embedding "solo
        vecinos" — justo lo que la función existe para quitar.

        `conv` se recibe como argumento en vez de usar `self.convs[0]`: hay que
        restar los `lin_r` de la MISMA capa donde se captura.
        """
        resto = h_txn
        for et, sub in conv.convs.items():
            if et[2] == TXN:                       # aristas que ENTRAN a transacción
                lin_r = getattr(sub, "lin_r", None)
                if lin_r is not None:
                    resto = resto - lin_r(x_txn)
        return resto


class FraudGATv2(_BaseHeteroGNN):
    """Atención aprendida por vecino. `aggr` no aplica: la atención la sustituye."""

    def _make_conv(self, in_c, out_c):
        return GATv2Conv((-1, -1), out_c, heads=self.extra.get("heads", 4),
                         concat=False, add_self_loops=False)

    def _termino_vecinos(self, conv, h_txn, x_txn):
        """
        Con `add_self_loops=False` el nodo NO se atiende a sí mismo: la salida
        de GATv2 es ya una suma ponderada de vecinos, sin término de raíz que
        descontar. (`lin_r` existe en GATv2Conv pero interviene en el cálculo
        de los coeficientes de atención, no se suma a la salida.)
        """
        return h_txn


def cfg_arquitectura(model_name: str, cfg: dict, ckpt: dict | None = None) -> dict:
    """
    El cfg con la arquitectura REAL de esa red, no la del config global.

    Desde que cada arquitectura usa SUS hiperparámetros de Optuna, `cfg["gnn"]`
    ya no describe a ninguna red concreta: graphsage puede haber salido con
    [128,128,128] y gatv2 con [128,128], mientras el config sigue diciendo
    [64,64]. Reconstruir desde el config daba:

        RuntimeError: size mismatch for convs.0... copying a param with shape
        torch.Size([128, 280]) ... current model is torch.Size([64, 280])

    Se resuelve en tres saltos, de más fiable a menos:
      1. el propio checkpoint, si trae `hidden_dims` (lo guarda train_gnn)
      2. reports/optuna_<arq>.json, el cache de la búsqueda
      3. el config tal cual (sin Optuna, o red entrenada antes de este cambio)
    """
    import json
    from src.utils.common import resolve

    c = json.loads(json.dumps(cfg))
    g = c["gnn"]
    if ckpt and ckpt.get("hidden_dims"):
        g["hidden_dims"] = list(ckpt["hidden_dims"])
        if ckpt.get("mlp_head_dim"):
            g["mlp_head_dim"] = int(ckpt["mlp_head_dim"])
        return c

    cache = resolve(cfg, "reports_dir") / f"optuna_{model_name}.json"
    if cache.exists():
        with open(cache) as f:
            best = (json.load(f) or {}).get("mejores_params") or {}
        if "ancho" in best:
            g["hidden_dims"] = [best["ancho"]] * max(2, int(best.get("capas", 2)))
        for k in ("dropout", "mlp_head_dim", "lr"):
            if k in best:
                g[k] = best[k]
    return c


def build_model(name: str, cfg: dict, metadata):
    g = cfg["gnn"]
    common = dict(metadata=metadata, in_dim=g["in_dim"],
                  hidden_dims=g["hidden_dims"], mlp_dim=g["mlp_head_dim"],
                  dropout=g["dropout"], aggr=g.get("aggr", "mean"))
    n = name.lower()
    if n in ("graphsage", "sage"):
        return FraudGraphSAGE(**common)
    if n in ("gatv2", "gat"):
        return FraudGATv2(**common, heads=g.get("gat_heads", 4))
    raise ValueError(f"Modelo desconocido: {name}. Usa graphsage o gatv2.")
