"""
Paso 4 — Entrenamiento ORIGINAL de una GNN (GraphSAGE o GAT).

Claves del diseño:
- Weighted loss: BCEWithLogitsLoss(pos_weight = N_legítimas/N_fraudes ≈ 27.6).
  La etiqueta actúa como interruptor en la fórmula — un fraude mal clasificado
  genera un gradiente 27.6x más fuerte. Sin SMOTE en la GNN (los sintéticos
  no tienen aristas).
- Neighbor sampling 15-10-5 (NeighborLoader): la red NUNCA ve el grafo
  completo; cada nodo entrena/infiere sobre su subgrafo de ~750 nodos.
- División temporal estricta: entrena meses 1-4, valida mes 5 (early stopping
  sobre AUC de validación).

REANUDACIÓN AUTOMÁTICA (no hay que pasar ningún flag):
- Al terminar cada época se guarda models/{model}_seed{seed}_resume.pt con
  pesos + optimizador + época + mejor estado + semillas de los RNG, y se
  actualiza artifacts/pipeline_state.json (se CREA solo en la primera corrida).
- Si el proceso muere y lo vuelves a lanzar, detecta ese archivo y sigue
  desde la época siguiente. Si la corrida ya estaba terminada, no reentrena:
  devuelve el reporte existente.
- Al terminar bien, el _resume.pt se borra y queda solo el checkpoint final.
- Para reentrenar desde cero a propósito: --force

Uso:
  python -m src.gnn.train_gnn --model graphsage --seed 42
  python -m src.gnn.train_gnn --model gat --seed 42
  python -m src.gnn.train_gnn --model gat --seed 42 --force   # ignora lo hecho

Salida: models/{model}_seed{seed}.pt + reports/{model}_seed{seed}_val.json
"""
import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.gnn.models import build_model
from src.gnn.sampling import fanouts, loader_opts, make_neighbor_loader
from src.utils.common import (ensure_dirs, get_device, get_logger,
                              get_run_state, load_config, resolve, set_seed,
                              update_state)
from src.utils.metrics import full_report

log = get_logger("train_gnn")


def run_key(model_name: str, seed: int) -> str:
    return f"{model_name}_seed{seed}"


def run_paths(model_name: str, seed: int, cfg: dict) -> tuple[Path, Path, Path]:
    """(checkpoint final, reporte de validación, checkpoint de reanudación)."""
    key = run_key(model_name, seed)
    return (resolve(cfg, "models_dir") / f"{key}.pt",
            resolve(cfg, "reports_dir") / f"{key}_val.json",
            resolve(cfg, "models_dir") / f"{key}_resume.pt")


def is_done(model_name: str, seed: int, cfg: dict) -> bool:
    """Corrida terminada = existen SUS DOS salidas finales en disco.
    El disco manda sobre el archivo de estado: si borras un checkpoint,
    la corrida vuelve a estar pendiente."""
    ckpt, rep, _ = run_paths(model_name, seed, cfg)
    return ckpt.exists() and rep.exists()


def resume_info(model_name: str, seed: int, cfg: dict) -> dict | None:
    """Si quedó un checkpoint de reanudación, dice en qué época quedó.
    None = la corrida no está a medias (o nunca empezó)."""
    _, _, resume_path = run_paths(model_name, seed, cfg)
    if not resume_path.exists():
        return None
    try:
        ck = torch.load(resume_path, map_location="cpu", weights_only=False)
    except Exception:                              # checkpoint ilegible
        return None
    return {"epoch": int(ck.get("epoch", 0)),
            "best_auc": float(ck.get("best_auc", 0.0))}


def _atomic_torch_save(obj, path: Path):
    """Guardado a prueba de cortes: si el proceso muere a mitad de escritura,
    el archivo bueno anterior sigue intacto."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _rng_state(**loaders) -> dict:
    """Estado de TODOS los generadores: torch, numpy, random y el de cada
    neighbor sampler (SimpleNeighborLoader lleva su propio np.Generator, que
    avanza época a época; sin esto la reanudación cambiaría el muestreo)."""
    st = {"torch": torch.get_rng_state(),
          "numpy": np.random.get_state(),
          "python": random.getstate()}
    for name, loader in loaders.items():
        rng = getattr(loader, "rng", None)        # None con NeighborLoader de PyG
        st[name] = rng.bit_generator.state if rng is not None else None
    return st


def _restore_rng(st: dict, **loaders):
    if not st:
        return
    torch.set_rng_state(st["torch"])
    np.random.set_state(st["numpy"])
    random.setstate(st["python"])
    for name, loader in loaders.items():
        rng = getattr(loader, "rng", None)
        if rng is not None and st.get(name) is not None:
            rng.bit_generator.state = st[name]


def ruta_modelo_operativo(cfg) -> tuple[Path, str]:
    """
    Qué checkpoint entra en operación: el del refit si existe, si no el
    ganador del paso GNN. Devuelve (ruta, etiqueta) para poder decirlo en el
    log — importa saber cuál de los dos produjo un resultado.
    """
    models_dir = resolve(cfg, "models_dir")
    refit = models_dir / "refit_model.pt"
    if refit.exists():
        return refit, "refit (reentrenado con train+val)"
    with open(models_dir / "selected_model.json") as f:
        sel = json.load(f)["selection"]
    return models_dir / sel["checkpoint"], f"{sel['selected']} seed {sel['seed']}"


def make_loader(data, mask, cfg, shuffle=True):
    return make_neighbor_loader(
        data,
        num_neighbors=fanouts(cfg),            # recortados a las capas del modelo
        input_nodes=mask,
        batch_size=cfg["gnn"]["batch_size"],
        shuffle=shuffle,
        **loader_opts(cfg),
    )


@torch.no_grad()
def evaluate(model, loader, device):
    """Scores sobre los nodos semilla de cada batch (distribución real)."""
    model.eval()
    ys, ss = [], []
    for batch in loader:
        batch = batch.to(device)
        logits = model(batch.x, batch.edge_index)[: batch.batch_size]
        ys.append(batch.y[: batch.batch_size].cpu().numpy())
        ss.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(ys), np.concatenate(ss)


def train(model_name: str, seed: int, cfg: dict | None = None,
          force: bool = False):
    cfg = cfg or load_config()
    ensure_dirs(cfg)
    key = run_key(model_name, seed)
    ckpt_path, rep_path, resume_path = run_paths(model_name, seed, cfg)

    # ¿Ya está lista? No se reentrena: se devuelve el reporte guardado.
    if is_done(model_name, seed, cfg) and not force:
        with open(rep_path) as fh:
            done_rep = json.load(fh)
        log.info("[%s] ya estaba terminada (AUC %.4f) — se salta.",
                 key, done_rep.get("auc_roc", float("nan")))
        update_state(cfg, key, status="done", model=model_name, seed=seed,
                     auc_roc=done_rep.get("auc_roc"))
        return done_rep

    if force and resume_path.exists():
        resume_path.unlink()                       # --force = arrancar limpio

    set_seed(seed)
    device = get_device()

    data = torch.load(resolve(cfg, "graph_dir") / "graph.pt", weights_only=False)
    # in_dim real puede diferir del nominal — el modelo se adapta al grafo
    cfg["gnn"]["in_dim"] = data.x.shape[1]

    # pos_weight = N_negativos / N_positivos DEL SET QUE ENTRENA AHORA
    y_tr = data.y[data.train_mask]
    pos_weight = float((y_tr == 0).sum() / max(1, (y_tr == 1).sum()))
    hd = cfg["gnn"]["hidden_dims"]
    log.info("[%s seed=%d] %d capa(s) %d->%s->%d->1 | fanouts %s | batch %d | "
             "pos_weight %.2f%s", model_name, seed, len(hd), cfg["gnn"]["in_dim"],
             "->".join(map(str, hd)), cfg["gnn"]["mlp_head_dim"],
             fanouts(cfg), cfg["gnn"]["batch_size"], pos_weight,
             "  [SIN ARISTAS]" if cfg["gnn"].get("sin_aristas") else "")


    model = build_model(model_name, cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["gnn"]["lr"])
    criterion = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(pos_weight, device=device))

    train_loader = make_loader(data, data.train_mask, cfg, shuffle=True)
    # El tamaño real del primer batch es la prueba de que la ablación se aplicó:
    # sin aristas son exactamente batch_size nodos; con aristas, muchos más.
    _b = next(iter(train_loader))
    log.info("[%s seed=%d] batch real: %d nodos, %d aristas", key.split("_")[0],
             seed, _b.x.shape[0], _b.edge_index.shape[1])
    val_loader = make_loader(data, data.val_mask, cfg, shuffle=False)

    best_auc, best_state, bad_epochs, best_epoch = 0.0, None, 0, 0
    start_epoch, resumed, minutes_before = 1, False, 0.0

    if resume_path.exists():
        ck = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["state_dict"])
        optimizer.load_state_dict(ck["optimizer"])
        best_auc, best_state = ck["best_auc"], ck["best_state"]
        bad_epochs = ck["bad_epochs"]
        best_epoch = ck.get("best_epoch", 0)
        minutes_before = ck.get("minutes", 0.0)    # tiempo de la corrida previa
        start_epoch = ck["epoch"] + 1
        _restore_rng(ck.get("rng"), train=train_loader, val=val_loader)
        resumed = True
        log.info("[%s] Retoma en la época %d de %d (quedó en la %d, mejor AUC "
                 "%.4f, %d sin mejorar).", key, start_epoch,
                 cfg["gnn"]["epochs"], ck["epoch"], best_auc, bad_epochs)
    else:
        log.info("[%s] Empieza desde cero (época 1 de %d).",
                 key, cfg["gnn"]["epochs"])

    update_state(cfg, key, status="running", model=model_name, seed=seed,
                 epoch=start_epoch - 1, best_auc=best_auc, resumed=resumed,
                 total_epochs=cfg["gnn"]["epochs"])

    # una época son cientos de batches y varios minutos: sin esto el log se
    # queda mudo y no se sabe si avanza o se colgó
    log_every = cfg["gnn"].get("log_every", 50)
    n_batches = len(train_loader)

    t0 = time.time()
    for epoch in range(start_epoch, cfg["gnn"]["epochs"] + 1):
        model.train()
        total_loss = 0.0
        t_epoch = time.time()
        for i, batch in enumerate(train_loader, 1):
            batch = batch.to(device)
            optimizer.zero_grad()
            logits = model(batch.x, batch.edge_index)[: batch.batch_size]
            loss = criterion(logits, batch.y[: batch.batch_size])
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.batch_size

            if log_every and (i % log_every == 0 or i == n_batches):
                transcurrido = (time.time() - t_epoch) / 60
                falta = transcurrido * (n_batches - i) / i   # ritmo actual
                log.info("  época %02d | batch %d/%d (%d%%) | loss %.4f | "
                         "%.1f min, faltan ~%.1f",
                         epoch, i, n_batches, 100 * i // n_batches,
                         loss.item(), transcurrido, falta)

        if log_every:
            log.info("  época %02d | entrenamiento listo, validando...", epoch)
        y_val, s_val = evaluate(model, val_loader, device)
        rep = full_report(y_val, s_val, cfg["gnn"]["threshold"])
        auc = rep.get("auc_roc", 0.0)
        log.info("Época %02d | loss %.4f | val AUC %.4f | val recall %.4f "
                 "| %.1f min", epoch, total_loss / int(data.train_mask.sum()),
                 auc, rep["recall"], (time.time() - t_epoch) / 60)

        if auc > best_auc:
            best_auc, bad_epochs, best_epoch = auc, 0, epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1

        # --- punto de guardado: cuesta ~1 s y ahorra hasta una época entera --
        minutes = minutes_before + (time.time() - t0) / 60
        _atomic_torch_save(
            {"model_name": model_name, "seed": seed, "epoch": epoch,
             "in_dim": cfg["gnn"]["in_dim"],
             "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
             "optimizer": optimizer.state_dict(),
             "best_auc": best_auc, "best_state": best_state,
             "bad_epochs": bad_epochs, "best_epoch": best_epoch,
             "minutes": minutes,
             "rng": _rng_state(train=train_loader, val=val_loader)},
            resume_path)
        update_state(cfg, key, status="running", model=model_name, seed=seed,
                     epoch=epoch, best_auc=best_auc, last_auc=auc,
                     resumed=resumed, minutes=round(minutes, 1),
                     total_epochs=cfg["gnn"]["epochs"])

        if bad_epochs >= cfg["gnn"]["patience"]:
            log.info("Early stopping (paciencia %d).", cfg["gnn"]["patience"])
            break

    model.load_state_dict(best_state)
    y_val, s_val = evaluate(model, val_loader, device)
    final_rep = full_report(y_val, s_val, cfg["gnn"]["threshold"])
    final_rep["train_minutes"] = round(minutes_before + (time.time() - t0) / 60, 1)
    final_rep["pos_weight"] = pos_weight
    # model/seed en el reporte: el comparador ya no depende del orden de la
    # lista de seeds para saber de quién es cada resultado
    final_rep["model"] = model_name
    final_rep["seed"] = seed
    final_rep["resumed"] = resumed
    # época del PICO, no la última: con patience=5 la corrida sigue 5 épocas
    # más allá de su mejor momento. Es el número que necesita el refit para
    # entrenar sin early stopping si algún día se usa el modo de épocas fijas.
    final_rep["best_epoch"] = best_epoch

    # los pesos se guardan en CPU: así el checkpoint se puede cargar en
    # cualquier máquina (Linux/CUDA, Mac/MPS o CPU pelado)
    _atomic_torch_save({"model_name": model_name, "seed": seed,
                        "in_dim": cfg["gnn"]["in_dim"],
                        "best_epoch": best_epoch,
                        "state_dict": {k: v.cpu()
                                       for k, v in model.state_dict().items()}},
                       ckpt_path)
    with open(rep_path, "w") as f:
        json.dump(final_rep, f, indent=2)
    resume_path.unlink(missing_ok=True)            # corrida cerrada: ya no hace falta
    update_state(cfg, key, status="done", model=model_name, seed=seed,
                 auc_roc=final_rep["auc_roc"], best_auc=best_auc,
                 minutes=final_rep["train_minutes"], resumed=resumed)
    log.info("[%s] LISTA — AUC %.4f | recall %.4f | %.1f min -> %s",
             key, final_rep["auc_roc"], final_rep["recall"],
             final_rep["train_minutes"], ckpt_path.name)
    return final_rep


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["graphsage", "gat"], required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--force", action="store_true",
                   help="Reentrenar desde cero aunque ya exista el resultado")
    args = p.parse_args()
    train(args.model, args.seed, force=args.force)
