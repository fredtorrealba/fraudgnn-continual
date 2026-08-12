"""
Utilidades comunes: carga de configuración, logging, semillas, dispositivo
y el archivo de estado del entrenamiento (para reanudar tras una caída).
"""
import json
import logging
import os
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]

_DEVICE_LOGGED = False   # el dispositivo se anuncia una sola vez por proceso


def load_config(path: str | None = None) -> dict:
    """Carga config/config.yaml (o una ruta alternativa)."""
    cfg_path = Path(path) if path else ROOT / "config" / "config.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_logger(name: str) -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )
    return logging.getLogger(name)


def set_seed(seed: int = 42):
    """Reproducibilidad: numpy, random y torch (si está disponible)."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            torch.mps.manual_seed(seed)
    except ImportError:
        pass


def get_device():
    """
    Dispositivo de cómputo, portable entre Linux y macOS.

    Por defecto: CUDA (NVIDIA) si existe, si no CPU. La GPU de Apple Silicon
    (MPS) NO se usa automáticamente: en esta carga el cuello de botella es el
    neighbor sampling (CPU) y no las multiplicaciones, así que MPS aporta poco
    y las operaciones de atención de GAT tienen soporte irregular. Queda
    disponible como opt-in explícito.

    Se puede forzar con la variable de entorno FRAUDGNN_DEVICE:
        FRAUDGNN_DEVICE=mps  python -m src.gnn.train_gnn --model graphsage
        FRAUDGNN_DEVICE=cpu  python -m src.gnn.compare_gnns

    IMPORTANTE: una corrida completa debe usar SIEMPRE el mismo dispositivo.
    CPU y GPU dan resultados numéricamente distintos, y la comparación del
    paso 5 (GraphSAGE vs GAT) exige que ambos se midan en igualdad de
    condiciones.
    """
    import torch

    global _DEVICE_LOGGED
    forced = os.environ.get("FRAUDGNN_DEVICE", "").strip().lower()
    if forced:
        if forced == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("FRAUDGNN_DEVICE=cuda pero no hay CUDA disponible.")
        if forced == "mps" and not (getattr(torch.backends, "mps", None)
                                    and torch.backends.mps.is_available()):
            raise RuntimeError("FRAUDGNN_DEVICE=mps pero MPS no está disponible "
                               "(¿macOS con Apple Silicon y torch >= 2.0?).")
        dev, how = torch.device(forced), "forzado por FRAUDGNN_DEVICE"
    elif torch.cuda.is_available():
        dev, how = torch.device("cuda"), f"auto · {torch.cuda.get_device_name(0)}"
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        # MPS existe pero no se activa solo: hay que pedirlo con FRAUDGNN_DEVICE=mps
        dev, how = torch.device("cpu"), "auto · MPS disponible (usa FRAUDGNN_DEVICE=mps)"
    else:
        dev, how = torch.device("cpu"), "auto · sin GPU disponible"

    if not _DEVICE_LOGGED:                       # una sola vez por proceso
        _DEVICE_LOGGED = True
        get_logger("device").info("Dispositivo de cómputo: %s (%s)", dev.type, how)
    return dev


def ensure_dirs(cfg: dict):
    """Crea los directorios declarados en config.paths si no existen."""
    for key, rel in cfg["paths"].items():
        (ROOT / rel).mkdir(parents=True, exist_ok=True)


def resolve(cfg: dict, path_key: str) -> Path:
    """Ruta absoluta a partir de una clave de config.paths."""
    return ROOT / cfg["paths"][path_key]


# ---------------------------------------------------------------------------
# Estado del pipeline — artifacts/pipeline_state.json
#
# UN solo archivo JSON con dos niveles, para saber siempre qué falta:
#   "steps": los pasos del pipeline (download, preprocess, graph, xgboost,
#            gnn, cl, final) con status pending|running|done|failed.
#   "runs":  las 6 corridas GNN "{modelo}_seed{seed}" con status, la última
#            época COMPLETADA y el mejor AUC de validación.
# Se crea solo en la primera corrida. Los scripts lo leen al arrancar: lo que
# está "done" se salta, lo que quedó a medias se retoma desde su checkpoint.
# ---------------------------------------------------------------------------

def _atomic_write(path: Path, text: str):
    """Escribe por archivo temporal + rename: un corte no deja JSON corrupto."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def state_path(cfg: dict) -> Path:
    return resolve(cfg, "artifacts_dir") / "pipeline_state.json"


def load_state(cfg: dict) -> dict:
    p = state_path(cfg)
    if not p.exists():
        return {"steps": {}, "runs": {}}
    try:
        st = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:                   # archivo a medio escribir
        return {"steps": {}, "runs": {}}
    st.setdefault("steps", {})
    st.setdefault("runs", {})
    return st


def _update_section(cfg: dict, section: str, key: str, **fields) -> dict:
    st = load_state(cfg)
    entry = st.setdefault(section, {}).setdefault(key, {})
    entry.update(fields)
    now = datetime.now().isoformat(timespec="seconds")
    entry["updated"] = now
    st["updated"] = now
    p = state_path(cfg)
    p.parent.mkdir(parents=True, exist_ok=True)    # primera corrida: lo crea
    _atomic_write(p, json.dumps(st, indent=2))
    return st


def update_state(cfg: dict, run_key: str, **fields) -> dict:
    """Actualiza (merge) una corrida GNN y persiste el archivo."""
    return _update_section(cfg, "runs", run_key, **fields)


def update_step(cfg: dict, step_name: str, **fields) -> dict:
    """Actualiza (merge) un paso del pipeline y persiste el archivo."""
    return _update_section(cfg, "steps", step_name, **fields)


def get_run_state(cfg: dict, run_key: str) -> dict:
    return load_state(cfg)["runs"].get(run_key, {})
