"""
Paso 6 — REFIT del modelo ganador sobre train + validación.

POR QUÉ
El mes 5 se gasta en DECIDIR: qué arquitectura gana y en qué época deja de
mejorar. Tomadas esas decisiones, sus 85,303 transacciones quedan sin usar.
Este paso las incorpora, antes de que el mes 6 se toque por primera vez (que
ocurre en `cl`).

CÓMO
Se hereda del ganador la ARQUITECTURA y el NÚMERO DE ÉPOCAS que el mes 5
determinó, y se entrena DESDE CERO con todo lo disponible hasta el mes 5:

    entrena  ->  meses 1-5 completos   (495,904 txn, +21%)
    épocas   ->  las del pico de la corrida original (best_epoch)
    test     ->  mes 6, intacto

Es el procedimiento estándar de refit (el mismo que aplica sklearn con
`GridSearchCV(refit=True)`): la validación elige los hiperparámetros y luego
se reentrena con todos los datos disponibles.

Desde cero, no fine-tuning del checkpoint anterior: continuar desde esos pesos
sobrepesaría el mes 5 —serían los últimos gradientes— y ataría el resultado al
mínimo concreto en que cayó la corrida original.

LIMITACIÓN, que conviene declarar en la memoria
Sin mes 5 no hay early stopping: las épocas se heredan, medidas con un 21%
menos de datos. Es una aproximación asumida por el procedimiento — con más
datos cada época son más actualizaciones, así que el óptimo real podría estar
una o dos épocas antes.

Salidas: models/refit_model.pt, reports/refit.json
"""
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.gnn.models import TXN, build_model, cfg_arquitectura
from src.gnn.train_gnn import make_loader
from src.utils.common import (ensure_dirs, get_device, get_logger, load_config,
                              resolve, set_seed)

log = get_logger("refit")


def epocas_del_ganador(cfg, models_dir: Path, reports_dir: Path, sel: dict) -> tuple[int, str]:
    """
    Épocas a reproducir: la del PICO de la corrida ganadora, no la última.
    Con patience=5 el entrenamiento sigue 5 épocas más allá de su mejor
    momento; reproducir esas de más sería sobreajustar a propósito.

    Se busca en tres sitios, de más explícito a más enterrado, para que
    relanzar SOLO este paso nunca dependa de tener los 6 checkpoints:
      1. gnn.refit_epochs del config   (override manual)
      2. selected_model.json           (lo escribe compare_gnns)
      3. el checkpoint / el *_val.json (retrocompatibilidad)
    Devuelve (épocas, de dónde salió).
    """
    manual = (cfg.get("gnn") or {}).get("refit_epochs")
    if manual:
        return int(manual), "gnn.refit_epochs del config"

    if sel.get("best_epoch"):
        return int(sel["best_epoch"]), "selected_model.json"

    ck_path = models_dir / sel["checkpoint"]
    if ck_path.exists():
        ck = torch.load(ck_path, weights_only=False)
        if ck.get("best_epoch"):
            return int(ck["best_epoch"]), sel["checkpoint"]

    rep = reports_dir / f"{sel['selected']}_seed{sel['seed']}_val.json"
    if rep.exists():
        with open(rep) as f:
            n = json.load(f).get("best_epoch")
        if n:
            return int(n), rep.name

    raise SystemExit(
        "No encuentro en qué época paró la corrida ganadora. Opciones:\n"
        "  - ponla a mano en config.yaml -> gnn.refit_epochs\n"
        "  - o vuelve a ejecutar el paso gnn (--only gnn --force), que ahora "
        "guarda best_epoch en selected_model.json")


def main():
    cfg = load_config()
    ensure_dirs(cfg)
    models_dir, reports_dir = resolve(cfg, "models_dir"), resolve(cfg, "reports_dir")

    with open(models_dir / "selected_model.json") as f:
        sel = json.load(f)["selection"]
    nombre, seed = sel["selected"], sel["seed"]
    n_epocas, origen = epocas_del_ganador(cfg, models_dir, reports_dir, sel)
    log.info("Ganadora: %s (seed %d) | %d épocas (según %s)",
             nombre, seed, n_epocas, origen)

    set_seed(seed)
    device = get_device()
    data = torch.load(resolve(cfg, "graph_dir") / "graph.pt", weights_only=False)
    cfg["gnn"]["in_dim"] = data[TXN].x.shape[1]

    entrena = data[TXN].train_mask | data[TXN].val_mask   # meses 1-5 completos
    n_tr, antes = int(entrena.sum()), int(data[TXN].train_mask.sum())
    log.info("Entrena %d txn (+%.0f%% sobre las %d de la corrida original) | "
             "mes 6 intacto (%d)", n_tr, 100 * (n_tr - antes) / antes, antes,
             int(data.test_mask.sum()))

    y_tr = data[TXN].y[entrena]
    pos_weight = float((y_tr == 0).sum() / max(1, (y_tr == 1).sum()))
    log.info("pos_weight recalculado: %.2f", pos_weight)

    # Arquitectura de la GANADORA (checkpoint o cache de Optuna), no la del
    # config: los pesos son nuevos, pero la forma tiene que ser la misma.
    cfg = cfg_arquitectura(nombre, cfg)
    model = build_model(nombre, cfg, data.metadata()).to(device)  # pesos NUEVOS
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg["gnn"]["lr"],
        weight_decay=float(cfg["gnn"].get("weight_decay", 0.0)))
    criterion = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(pos_weight, device=device))
    balancear = bool(cfg["gnn"].get("balanceo_semillas", False))
    train_loader = make_loader(data, entrena, cfg, shuffle=True,
                               balancear=balancear)

    # Sin early stopping: no queda conjunto de validación. La única señal
    # disponible es la loss de entrenamiento, que se registra para poder ver
    # que la optimización avanzó como en la corrida original.
    perdidas, t0 = [], time.time()
    for epoca in range(1, n_epocas + 1):
        model.train()
        total = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            n = batch[TXN].batch_size
            logits = model(batch.x_dict, batch.edge_index_dict, batch)[:n]
            loss = criterion(logits, batch[TXN].y[:n])
            loss.backward()
            optimizer.step()
            total += loss.item() * batch[TXN].batch_size
        perdidas.append(round(total / n_tr, 4))
        log.info("Época %02d/%02d | loss %.4f", epoca, n_epocas, perdidas[-1])

    minutos = round((time.time() - t0) / 60, 1)
    torch.save({"model_name": nombre, "seed": seed,
                "in_dim": cfg["gnn"]["in_dim"], "best_epoch": n_epocas,
                "refit": True,
                "state_dict": {k: v.cpu() for k, v in model.state_dict().items()}},
               models_dir / "refit_model.pt")

    informe = {
        "modelo": nombre, "seed": seed,
        "entrenamiento": {"n": n_tr, "n_original": antes,
                          "incremento_pct": round(100 * (n_tr - antes) / antes, 1),
                          "meses": "1-5", "pos_weight": pos_weight},
        "epocas": n_epocas,
        "origen_epocas": origen,
        "loss_por_epoca": perdidas,
        "minutos": minutos,
        "nota": ("Refit estándar: la validación eligió arquitectura y épocas, "
                 "y se reentrena desde cero con todos los datos hasta el mes 5. "
                 "Sin early stopping (no queda conjunto de validación): las "
                 "épocas se heredan, medidas con un 21% menos de datos."),
    }
    with open(reports_dir / "refit.json", "w") as f:
        json.dump(informe, f, indent=2, ensure_ascii=False)

    log.info("Refit listo — %d épocas | loss %.4f -> %.4f | %.1f min "
             "-> refit_model.pt", n_epocas, perdidas[0], perdidas[-1], minutos)
    return informe


if __name__ == "__main__":
    main()
