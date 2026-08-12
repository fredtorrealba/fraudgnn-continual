"""
Runner del pipeline completo, con memoria de lo que YA está hecho.

Reemplaza la lista de comandos a ciegas de scripts/run_pipeline.sh: antes de
cada paso mira si sus archivos de salida existen y, si están, lo salta. Así
un corte (Ctrl-C, batería, kernel panic) no obliga a rehacer la descarga, el
preprocesamiento ni los modelos ya entrenados: se relanza el mismo comando y
sigue donde quedó.

Dos niveles de memoria:
  PASO    los archivos de salida en data/, models/ y reports/ — el disco es
          la verdad. Si borras una salida, ese paso vuelve a estar pendiente.
  ÉPOCA   dentro del paso 4-5, cada corrida GNN (2 modelos x 3 seeds) tiene
          su propio checkpoint de reanudación (ver src/gnn/train_gnn.py).

El avance queda registrado en artifacts/pipeline_state.json, que se CREA solo
en la primera corrida.

Uso:
  python -m src.pipeline                 # corre lo que falte, en orden
  python -m src.pipeline --status        # solo mostrar en qué va (no ejecuta)
  python -m src.pipeline --from gnn      # desde ese paso en adelante
  python -m src.pipeline --only cl       # un solo paso
  python -m src.pipeline --force xgboost # rehacer ese paso aunque esté listo
  python -m src.pipeline --force         # rehacer TODO desde cero
"""
import argparse
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.common import (ensure_dirs, get_logger, load_config, load_state,
                              resolve, state_path, update_step)

log = get_logger("pipeline")


@dataclass
class Step:
    name: str                       # identificador corto para --only/--from
    title: str                      # lo que se ve en el log
    module: str                     # se ejecuta como python -m <module>
    outputs: list[tuple[str, str]]  # (clave de config.paths, archivo)
    args: list[str] = field(default_factory=list)

    def output_paths(self, cfg) -> list[Path]:
        return [resolve(cfg, key) / name for key, name in self.outputs]

    def missing(self, cfg) -> list[Path]:
        return [p for p in self.output_paths(cfg) if not p.exists()]

    def is_done(self, cfg) -> bool:
        return not self.missing(cfg)


# El pipeline del capstone, en orden. Las salidas son las que cada módulo
# escribe al terminar bien — por eso sirven como marca de "paso completado".
STEPS = [
    Step("download", "[0] Descarga del dataset IEEE-CIS",
         "src.data.download_ieee_cis",
         [("raw_dir", "train_transaction.csv"), ("raw_dir", "train_identity.csv")]),
    Step("preprocess", "[1] Preprocesamiento + split temporal",
         "src.data.preprocessing",
         [("processed_dir", "full.parquet"), ("processed_dir", "split_masks.parquet"),
          ("processed_dir", "feature_cols.json")]),
    Step("graph", "[2] Construcción del grafo (PyG)",
         "src.data.build_graph",
         [("graph_dir", "graph.pt")]),
    Step("xgboost", "[3] Baseline XGBoost (SMOTE + Optuna) — queda CONGELADO",
         "src.baseline_xgboost.train_xgboost",
         [("models_dir", "xgboost_baseline.json"),
          ("reports_dir", "xgboost_val_metrics.json")]),
    # Este paso tiene reanudación propia por seed y por época: aunque se corte
    # a la mitad, al relanzarlo sigue desde la última época guardada.
    Step("gnn", "[4-5] GraphSAGE vs GAT (3 seeds c/u) + selección",
         "src.gnn.compare_gnns",
         [("models_dir", "selected_model.json")]),
    Step("cl", "[7] Ciclo de Continual Learning (mes 6 por semanas)",
         "src.continual_learning.cl_orchestrator",
         [("reports_dir", "cl_cycles.json"),
          ("reports_dir", "gnn_cl_test_scores.npz"),
          ("graph_dir", "graph_scored.pt")]),
    Step("final", "[8] Comparación final GNN+CL vs XGBoost (OE4)",
         "src.comparison.final_comparison",
         [("reports_dir", "final_comparison.json")]),
]

BY_NAME = {s.name: s for s in STEPS}
ROOT = Path(__file__).resolve().parents[1]


def _rel(p: Path) -> str:
    """Ruta corta para el log; absoluta si cae fuera del proyecto."""
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)


def gnn_progress(cfg) -> str:
    """Sub-avance del paso 4-5: cuántas de las 6 corridas están listas."""
    runs = load_state(cfg).get("runs", {})
    if not runs:
        return ""
    done = sum(1 for r in runs.values() if r.get("status") == "done")
    medias = [f"{k} quedó en la época {r['epoch']}" if r.get("epoch")
              else f"{k} empezada, aún sin épocas completas"
              for k, r in runs.items() if r.get("status") == "running"]
    extra = f" | {medias[0]}" if medias else ""
    return f"  ({done}/{len(runs)} corridas listas{extra})"


def show_status(cfg):
    st = load_state(cfg)
    log.info("Estado del pipeline (%s)", state_path(cfg))
    for i, step in enumerate(STEPS, 1):
        guardado = st.get("steps", {}).get(step.name, {})
        if step.is_done(cfg):
            marca, detalle = "LISTO   ", ""
            if guardado.get("minutes"):
                detalle = f"  ({guardado['minutes']} min)"
        else:
            marca = "PENDIENTE"
            faltan = step.missing(cfg)
            detalle = f"  falta: {_rel(faltan[0])}"
            if guardado.get("status") in ("failed", "interrupted"):
                detalle += f"  [{guardado['status']} el {guardado.get('updated')}]"
        sub = gnn_progress(cfg) if step.name == "gnn" else ""
        log.info("  %d. %-10s %s %s%s%s", i, step.name, marca, step.title,
                 detalle, sub)


def run_step(step: Step, cfg) -> bool:
    """Ejecuta un paso como subproceso. True si terminó bien."""
    cmd = [sys.executable, "-m", step.module, *step.args]
    log.info("== %s ==", step.title)
    log.info("   $ %s", " ".join(cmd[1:]))
    update_step(cfg, step.name, status="running", title=step.title)
    t0 = time.time()
    try:
        subprocess.run(cmd, cwd=ROOT, check=True)
    except KeyboardInterrupt:
        update_step(cfg, step.name, status="interrupted",
                    minutes=round((time.time() - t0) / 60, 1))
        log.warning("Interrumpido en '%s'. Relanza el mismo comando para "
                    "seguir desde acá.", step.name)
        raise
    except subprocess.CalledProcessError as e:
        update_step(cfg, step.name, status="failed", returncode=e.returncode,
                    minutes=round((time.time() - t0) / 60, 1))
        log.error("El paso '%s' falló (código %d). Se detiene el pipeline; al "
                  "relanzar se retoma desde acá.", step.name, e.returncode)
        return False

    minutes = round((time.time() - t0) / 60, 1)
    faltan = step.missing(cfg)
    if faltan:
        # terminó sin error pero no dejó sus salidas: no se marca como hecho
        update_step(cfg, step.name, status="failed", minutes=minutes,
                    missing=[_rel(p) for p in faltan])
        log.error("'%s' terminó sin error pero no generó %s.",
                  step.name, _rel(faltan[0]))
        return False

    update_step(cfg, step.name, status="done", minutes=minutes)
    log.info("== '%s' listo (%.1f min) ==", step.name, minutes)
    return True


def main():
    p = argparse.ArgumentParser(description="Pipeline FraudGNN, reanudable.")
    p.add_argument("--status", action="store_true",
                   help="Mostrar en qué va y salir (no ejecuta nada)")
    p.add_argument("--only", metavar="PASO", help="Ejecutar solo ese paso")
    p.add_argument("--from", dest="from_step", metavar="PASO",
                   help="Empezar desde ese paso")
    p.add_argument("--force", nargs="*", metavar="PASO",
                   help="Rehacer esos pasos aunque estén listos "
                        "(sin argumentos: todos)")
    args = p.parse_args()

    cfg = load_config()
    ensure_dirs(cfg)

    for nombre in filter(None, [args.only, args.from_step, *(args.force or [])]):
        if nombre not in BY_NAME:
            p.error(f"paso desconocido: {nombre}. Válidos: "
                    f"{', '.join(BY_NAME)}")

    if not state_path(cfg).exists():
        log.info("Primera corrida — se crea el archivo de estado en %s",
                 state_path(cfg))

    if args.status:
        show_status(cfg)
        return 0

    pasos = STEPS
    if args.only:
        pasos = [BY_NAME[args.only]]
    elif args.from_step:
        pasos = STEPS[[s.name for s in STEPS].index(args.from_step):]

    forzados = (set(BY_NAME) if args.force == []
                else set(args.force or []))

    show_status(cfg)
    for step in pasos:
        if step.name in forzados:
            log.info("== %s (forzado) ==", step.title)
            extra = ["--force"] if step.name == "gnn" else []
            step = Step(step.name, step.title, step.module, step.outputs,
                        step.args + extra)
        elif step.is_done(cfg):
            log.info("-- %s: YA ESTÁ HECHO, se salta.", step.title)
            update_step(cfg, step.name, status="done", title=step.title)
            continue
        if not run_step(step, cfg):
            return 1

    log.info("Pipeline completo. Reportes en %s", resolve(cfg, "reports_dir"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
