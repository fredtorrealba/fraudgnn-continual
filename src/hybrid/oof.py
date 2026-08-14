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
El parquet cubre TODAS las filas del grafo: dentro de la ventana con el score
out-of-fold, y fuera con el modelo real que nunca las vio (el ganador de `gnn`
para la ventana `train`, el refit para `trainval`).
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
from src.continual_learning.validate import embed_and_score_nodes
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
    # Además del escalar se guarda el EMBEDDING: el vector de la última capa,
    # antes de que el clasificador de la GNN lo colapse a un solo número. El
    # escalar es todo lo que la red puede decirle al modelo tabular hoy, y
    # compite contra 431 columnas; el embedding ensancha ese canal x256.
    # Ambos salen de las MISMAS K redes out-of-fold, así que el embedding es
    # tan honesto como el score: ninguna red vio la transacción que describe.
    emb = None
    t0 = time.time()
    for f in range(k):
        fuera = fold == f
        log.info("  fold %d/%d: entrena con %d, puntúa %d",
                 f + 1, k, int((~fuera).sum()), int(fuera.sum()))
        mask_tr = torch.zeros(data.num_nodes, dtype=torch.bool)
        mask_tr[torch.tensor(nodos[~fuera], dtype=torch.long)] = True
        model = _entrenar(nombre, seed, epocas, data, mask_tr, cfg, device)
        h, s_fold = embed_and_score_nodes(model, data, nodos[fuera], cfg)
        scores[fuera] = s_fold
        if emb is None:
            emb = np.zeros((len(nodos), h.shape[1]), dtype=np.float32)
        emb[fuera] = h
        del model, h
    minutos = round((time.time() - t0) / 60, 1)

    # El diagnóstico se calcula AQUÍ, antes de añadir los nodos de fuera de la
    # ventana: mide si `gnn_score` delata el fold, y esos nodos no tienen fold.
    diag = _diagnostico(scores, fold, data.month.numpy()[nodos])

    # --- meses POSTERIORES a la ventana: el modelo REAL, no OOF -------------
    # Sin esto, el paso `hybrid` valida sobre el mes 5 con gnn_score y embedding
    # en NaN: entrena las variantes con esas columnas y las evalúa donde no
    # existen. El efecto medido fue brutal — la variante del embedding cortaba
    # por early stopping en 6 árboles porque sus 256 columnas eran NaN en
    # validación, y la del escalar quedaba artificialmente por debajo de la
    # de 431. La comparación entre variantes era inválida.
    #
    # Es HONESTO usar aquí el modelo real: para `train` es el ganador del paso
    # `gnn`, entrenado solo con meses 1-4, que nunca vio el 5 ni el 6; para
    # `trainval` es el refit, que nunca vio el 6. Ninguno memorizó lo que
    # puntúa, que es la condición que persigue todo el esquema out-of-fold.
    fuera_ventana = torch.where(~mask)[0].numpy()
    if len(fuera_ventana):
        # El modelo se elige POR VENTANA, no con ruta_modelo_operativo(): esa
        # devuelve el refit si existe, y el refit entrenó con meses 1-5. Usarlo
        # para puntuar el mes 5 en la ventana `train` sería fuga: puntuaría
        # datos que memorizó. Cada ventana usa el modelo que NO vio lo que va
        # a puntuar.
        if args.window == "train":
            ruta_real = models_dir / f"{nombre}_seed{seed}.pt"
            etiqueta = f"{nombre} seed={seed} (meses 1-4)"
        else:
            ruta_real = models_dir / "refit_model.pt"
            etiqueta = "refit (meses 1-5)"
        if not ruta_real.exists():
            raise SystemExit(
                f"Falta {ruta_real.name}, necesario para puntuar los meses "
                f"fuera de la ventana '{args.window}'.")
        ck = torch.load(ruta_real, weights_only=False)
        cfg_real = dict(cfg)
        cfg_real["gnn"] = {**cfg["gnn"], "in_dim": ck["in_dim"]}
        real = build_model(ck["model_name"], cfg_real)
        real.load_state_dict(ck["state_dict"])
        log.info("Meses fuera de la ventana (%d nodos): los puntúa %s, que "
                 "nunca los vio", len(fuera_ventana), etiqueta)
        h_out, s_out = embed_and_score_nodes(real, data, fuera_ventana, cfg)
        del real
        nodos = np.concatenate([nodos, fuera_ventana])
        scores = np.concatenate([scores, s_out])
        emb = np.vstack([emb, h_out])
        orden = np.argsort(nodos)
        nodos, scores, emb = nodos[orden], scores[orden], emb[orden]

    # Escala de los embeddings dentro y fuera de la ventana. Los de dentro los
    # producen las K redes out-of-fold y los de fuera el modelo real: son redes
    # DISTINTAS, y su BatchNorm arrastra estadísticas distintas. Si las normas
    # difieren mucho, XGBoost aprendería cortes sobre una escala y los aplicaría
    # sobre otra. Medido en el smoke test (2 épocas) la mediana apenas se movía
    # y el desajuste por columna era 0.09 sobre 1, pero conviene vigilarlo:
    # normas muy dispares invalidarían la variante del embedding.
    if len(fuera_ventana):
        dentro = ~np.isin(nodos, fuera_ventana)
        n_d = np.linalg.norm(emb[dentro], axis=1)
        n_f = np.linalg.norm(emb[~dentro], axis=1)
        log.info("Norma del embedding — dentro de la ventana: mediana %.2f "
                 "(p95 %.2f) | fuera: mediana %.2f (p95 %.2f)",
                 np.median(n_d), np.percentile(n_d, 95),
                 np.median(n_f), np.percentile(n_f, 95))
        r = np.median(n_f) / max(np.median(n_d), 1e-8)
        if not 0.5 < r < 2.0:
            log.warning("Las escalas difieren x%.1f. La cabeza entrenaría con "
                        "una y serviría con otra: interpreta con cuidado los "
                        "resultados de la variante del embedding.", r)

    salida = {"node_idx": nodos, "gnn_score": scores}
    salida.update({f"emb_{i}": emb[:, i] for i in range(emb.shape[1])})
    pd.DataFrame(salida).to_parquet(ruta_parquet(cfg, args.window), index=False)
    log.info("Guardado: %d filas | gnn_score + embedding de %d dimensiones",
             len(nodos), emb.shape[1])

    informe = {"window": args.window, "folds": k, "modelo": nombre, "seed": seed,
               "epocas": epocas, "n_nodos": int(int(mask.sum())),
               "n_filas_parquet": int(len(nodos)), "minutos": minutos,
               "dim_embedding": int(emb.shape[1]),
               "diagnostico": diag}
    with open(ruta_informe(cfg, args.window), "w") as fh:
        json.dump(informe, fh, indent=2, ensure_ascii=False)

    log.info("OOF listo en %.1f min | auc_predecir_fold %.4f (debe ser ~0.50) -> %s",
             minutos, diag["auc_predecir_fold"] or float("nan"),
             ruta_parquet(cfg, args.window).name)
    return informe


if __name__ == "__main__":
    main()
