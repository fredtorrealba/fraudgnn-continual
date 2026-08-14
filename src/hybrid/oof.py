"""
Paso 5 / 8 — `gnn_score` honesto por validación cruzada (out-of-fold).

EL PROBLEMA QUE RESUELVE
La GNN ganadora se entrenó con los meses 1-4, así que MEMORIZÓ parte de esas
etiquetas. Si se usara su score sobre esas mismas transacciones como columna de
entrada de XGBoost, la cabeza vería una columna casi perfecta y aprendería
"copia el gnn_score". En el mes 6 —donde la red no ha memorizado nada— esa
confianza aprendida se rompe. Es el modo de fallo clásico del stacking.

LA SOLUCIÓN
Partir la ventana de entrenamiento en K trozos y entrenar K redes, cada una
dejando un trozo fuera. Cada transacción recibe el score de una red que NUNCA
la vio, así que la columna refleja lo que la GNN acierta de verdad y no lo que
recuerda. Las K redes se descartan: solo sobrevive la columna.

FOLDS ALEATORIOS ESTRATIFICADOS POR (month, isFraud), NO TEMPORALES
Con folds temporales cada red vería una cantidad y una posición distinta de
datos, y `gnn_score` quedaría correlacionado con el mes POR CONSTRUCCIÓN.
XGBoost es especialmente bueno encontrando eso: usaría la columna como proxy
del calendario. Es exactamente el artefacto que este proyecto ya midió (AUC
0.85 mensual contra 0.61 semanal en la ablación sin aristas). Estratificando
por (mes, clase), las K redes ven la misma composición temporal y el mismo
balance, y la columna no inyecta señal de calendario.

Los meses posteriores a la ventana NO llevan OOF: su score sale del modelo real
—que nunca los vio— y ese es el protocolo estándar.

    --window train      meses 1-4  (tras la etapa `gnn`)
    --window trainval   meses 1-5  (tras la etapa `refit`)

Salidas: data/processed/gnn_oof_{train,trainval}.parquet + su informe.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.gnn.models import build_model
from src.gnn.train_gnn import make_loader
from src.continual_learning.validate import score_nodes
from src.utils.common import (ensure_dirs, get_device, get_logger, load_config,
                              resolve, set_seed)

log = get_logger("hybrid.oof")

VENTANAS = {"train": "train_mask", "trainval": ("train_mask", "val_mask")}


def ruta_parquet(cfg, window: str) -> Path:
    return resolve(cfg, "processed_dir") / f"gnn_oof_{window}.parquet"


def ruta_informe(cfg, window: str) -> Path:
    return resolve(cfg, "reports_dir") / f"oof_{window}.json"


def _mascara(data, window: str) -> torch.Tensor:
    if window == "train":
        return data.train_mask
    return data.train_mask | data.val_mask


def _receta(cfg, models_dir: Path, window: str) -> tuple[str, int, int]:
    """
    Arquitectura, semilla y épocas que replican las K redes.

    Se heredan del modelo que corresponde a la ventana: el ganador de `gnn`
    para `train`, el refit para `trainval`. Sin early stopping — dentro de un
    fold no queda conjunto de validación.
    """
    with open(models_dir / "selected_model.json") as f:
        sel = json.load(f)["selection"]
    epocas = sel.get("best_epoch")
    if window == "trainval":
        rep = resolve(cfg, "reports_dir") / "refit.json"
        if rep.exists():
            with open(rep) as f:
                epocas = json.load(f).get("epocas", epocas)
    if not epocas:
        raise SystemExit(
            "No sé cuántas épocas usar. Falta 'best_epoch' en "
            "selected_model.json: vuelve a ejecutar el paso gnn (--only gnn --force).")
    return sel["selected"], int(sel["seed"]), int(epocas)


def _folds(data, nodos: np.ndarray, k: int, seed: int) -> np.ndarray:
    """Asignación de fold estratificada por (mes, clase)."""
    meses = data.month.numpy()[nodos]
    clases = data.y.numpy()[nodos].astype(int)
    fold = np.empty(len(nodos), dtype=np.int8)
    rng = np.random.default_rng(seed)
    for m in np.unique(meses):
        for c in (0, 1):
            sel = np.where((meses == m) & (clases == c))[0]
            rng.shuffle(sel)
            fold[sel] = np.arange(len(sel)) % k
    return fold


def _entrenar(nombre, seed, epocas, data, mask_train, cfg, device):
    """Una red del OOF: desde cero, épocas fijas, sin early stopping."""
    set_seed(seed)
    model = build_model(nombre, cfg).to(device)
    y_tr = data.y[mask_train]
    pos_weight = float((y_tr == 0).sum() / max(1, (y_tr == 1).sum()))
    opt = torch.optim.Adam(model.parameters(), lr=cfg["gnn"]["lr"])
    crit = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(pos_weight, device=device))
    loader = make_loader(data, mask_train, cfg, shuffle=True)
    for ep in range(1, epocas + 1):
        model.train()
        total, n = 0.0, 0
        for batch in loader:
            batch = batch.to(device)
            opt.zero_grad()
            logits = model(batch.x, batch.edge_index)[: batch.batch_size]
            loss = crit(logits, batch.y[: batch.batch_size])
            loss.backward()
            opt.step()
            total += loss.item() * batch.batch_size
            n += batch.batch_size
        if ep == 1 or ep == epocas:
            log.info("    época %02d/%02d | loss %.4f", ep, epocas, total / max(n, 1))
    return model


def _diagnostico(scores: np.ndarray, fold: np.ndarray, meses: np.ndarray) -> dict:
    """
    ¿Se coló señal de fold o de calendario en la columna?

    Si un modelo trivial puede predecir el FOLD a partir de `gnn_score`, es que
    las K redes no son intercambiables y la columna lleva un artefacto del
    procedimiento. El AUC debe salir ~0.50.
    """
    from sklearn.metrics import roc_auc_score
    out = {"por_fold": {}, "por_mes": {}}
    for f in np.unique(fold):
        s = scores[fold == f]
        out["por_fold"][int(f)] = {"n": int(len(s)), "media": float(s.mean()),
                                   "std": float(s.std())}
    for m in np.unique(meses):
        s = scores[meses == m]
        out["por_mes"][int(m)] = {"n": int(len(s)), "media": float(s.mean())}
    aucs = []
    for f in np.unique(fold):
        try:
            aucs.append(roc_auc_score((fold == f).astype(int), scores))
        except ValueError:
            pass
    out["auc_predecir_fold"] = float(np.mean(aucs)) if aucs else None
    out["nota"] = ("auc_predecir_fold debe salir ~0.50. Si se aleja, las K redes "
                   "no son intercambiables y gnn_score lleva un artefacto del "
                   "procedimiento, no señal de fraude.")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--window", choices=list(VENTANAS), default="train")
    args = ap.parse_args()

    cfg = load_config()
    ensure_dirs(cfg)
    models_dir = resolve(cfg, "models_dir")
    k = int((cfg.get("hybrid") or {}).get("oof_folds", 4))

    data = torch.load(resolve(cfg, "graph_dir") / "graph.pt", weights_only=False)
    cfg["gnn"]["in_dim"] = data.x.shape[1]
    device = get_device()

    # La suposición de la que cuelga todo el sistema híbrido: el índice de nodo
    # es el índice de fila del parquet. Si se rompiera, las columnas se unirían
    # a las transacciones equivocadas en silencio.
    df = pd.read_parquet(resolve(cfg, "processed_dir") / "full.parquet",
                         columns=["TransactionID"])
    assert np.array_equal(data.transaction_id.numpy(), df["TransactionID"].values), \
        "node_idx no coincide con la fila de full.parquet: reconstruye el grafo"

    nombre, seed, epocas = _receta(cfg, models_dir, args.window)
    mask = _mascara(data, args.window)
    nodos = torch.where(mask)[0].numpy()
    log.info("OOF %s | %s seed=%d, %d épocas | %d nodos en %d folds",
             args.window, nombre, seed, epocas, len(nodos), k)

    fold = _folds(data, nodos, k, seed)
    scores = np.zeros(len(nodos), dtype=np.float32)
    t0 = time.time()
    for f in range(k):
        fuera = fold == f
        log.info("  fold %d/%d: entrena con %d, puntúa %d",
                 f + 1, k, int((~fuera).sum()), int(fuera.sum()))
        mask_tr = torch.zeros(data.num_nodes, dtype=torch.bool)
        mask_tr[torch.tensor(nodos[~fuera], dtype=torch.long)] = True
        model = _entrenar(nombre, seed, epocas, data, mask_tr, cfg, device)
        scores[fuera] = score_nodes(model, data, nodos[fuera], cfg)
        del model
    minutos = round((time.time() - t0) / 60, 1)

    pd.DataFrame({"node_idx": nodos, "gnn_score": scores}).to_parquet(
        ruta_parquet(cfg, args.window), index=False)

    diag = _diagnostico(scores, fold, data.month.numpy()[nodos])
    informe = {"window": args.window, "folds": k, "modelo": nombre, "seed": seed,
               "epocas": epocas, "n_nodos": int(len(nodos)), "minutos": minutos,
               "diagnostico": diag}
    with open(ruta_informe(cfg, args.window), "w") as fh:
        json.dump(informe, fh, indent=2, ensure_ascii=False)

    log.info("OOF listo en %.1f min | auc_predecir_fold %.4f (debe ser ~0.50) -> %s",
             minutos, diag["auc_predecir_fold"] or float("nan"),
             ruta_parquet(cfg, args.window).name)
    return informe


if __name__ == "__main__":
    main()
