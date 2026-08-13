"""
Runner del pipeline, con memoria de lo que YA está hecho.

Antes de cada paso mira si sus archivos de salida existen y lo salta si están.
Un corte (Ctrl-C, batería, desalojo de VM spot) no obliga a rehacer nada: se
relanza el mismo comando y sigue donde quedó — incluso a mitad de una GNN, que
retoma desde su última época. El avance vive en artifacts/pipeline_state.json.

El disco manda: borra una salida y ese paso vuelve a estar pendiente.

Las 7 etapas (`--steps` las explica una por una):
  1 download    Kaggle -> data/raw/                       ~1 min
  2 preprocess  CSV -> parquet + split de 6 meses         ~1 min
  3 graph       parquet -> grafo PyG (~22M aristas)       ~4 min
  4 gnn         GraphSAGE vs GAT, 6 corridas          2-4 h GPU  <- el caro
  5 cl          mes 6 semana a semana + fine-tuning       ~15 min
  6 xgboost     baseline tabular CONGELADO                ~10 min
  7 final       comparación GNN+CL vs baseline            ~1 min

XGBoost va en 6 y no en su [3] nominal: solo depende de preprocess y su salida
solo la usa final, así que el paso caro arranca antes.

Uso:
  python -m src.pipeline                 # corre lo que falte
  python -m src.pipeline --steps         # qué hace cada etapa (no ejecuta)
  python -m src.pipeline --status        # en qué va ahora   (no ejecuta)
  python -m src.pipeline --only gnn,cl   # SOLO esos          (coma)
  python -m src.pipeline --skip xgboost  # todo MENOS esos    (coma)
  python -m src.pipeline --from gnn      # desde ahí en adelante
  python -m src.pipeline --force gnn cl  # rehacer            (espacio)
  python -m src.pipeline --force         # rehacer TODO
"""
import argparse
import subprocess
import sys
import time
from dataclasses import dataclass, field, replace
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
    desc: str = ""                  # resumen de qué hace, ver --steps
    acepta_force: bool = False      # el módulo entiende --force por su cuenta

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
         [("raw_dir", "train_transaction.csv"), ("raw_dir", "train_identity.csv")],
         acepta_force=True,
         desc="Baja train_transaction.csv (~650 MB) + train_identity.csv de Kaggle. "
              "El test de la competencia NO trae etiquetas, por eso los 6 meses se "
              "cortan dentro del train. Único paso que necesita red. ~1 min."),
    Step("preprocess", "[1] Preprocesamiento + split temporal",
         "src.data.preprocessing",
         [("processed_dir", "full.parquet"), ("processed_dir", "split_masks.parquet"),
          ("processed_dir", "feature_cols.json")],
         desc="Une ambos CSV por TransactionID, codifica categóricas e imputa "
              "faltantes (mapas y medianas ajustados SOLO con train, sin fuga) y "
              "parte 6 meses por TransactionDT: 1-4 entrenan, 5 valida, 6 es test. "
              "~1 min."),
    Step("graph", "[2] Construcción del grafo (PyG)",
         "src.data.build_graph",
         [("graph_dir", "graph.pt")],
         desc="Construye el grafo: nodos = transacciones, aristas = comparten "
              "entidad (tarjeta / email / dispositivo) dentro de 30 días, con tope "
              "de 50 aristas por nodo para evitar hubs. ~4 min, ~22M aristas."),
    # Este paso tiene reanudación propia por seed y por época: aunque se corte
    # a la mitad, al relanzarlo sigue desde la última época guardada.
    Step("gnn", "[4-5] GraphSAGE vs GAT (3 seeds c/u) + selección",
         "src.gnn.compare_gnns",
         [("models_dir", "selected_model.json")],
         acepta_force=True,
         desc="Entrena GraphSAGE y GAT con 3 semillas cada uno = 6 corridas, con "
              "neighbor sampling 15-10-5 (nunca ve el grafo entero). Elige la mejor "
              "por AUC walk-forward semanal. EL PASO CARO: 2-4 h con GPU. "
              "Reanudable por corrida Y por época."),
    Step("cl", "[7] Ciclo de Continual Learning (mes 6 por semanas)",
         "src.continual_learning.cl_orchestrator",
         [("reports_dir", "cl_cycles.json"),
          ("reports_dir", "gnn_cl_test_scores.npz"),
          ("graph_dir", "graph_scored.pt")],
         desc="Simula el mes 6 semana a semana. Por cada una: mide qué fraudes se "
              "escaparon, dispara el gatillo si hay patrón nuevo, hace fine-tuning "
              "40/60 (nuevos + replay buffer) con LR diferenciado por capa, y valida "
              "que aprendió SIN olvidar. Solo despliega si pasa ambas. ~15 min."),
    # XGBoost va aquí, no en su posición nominal [3]: solo necesita
    # `preprocess` (lee full.parquet, no toca el grafo) y su salida la consume
    # únicamente `final`. Ponerlo al final deja que las 6 corridas GNN —lo caro
    # y lo que puede fallar— arranquen cuanto antes. Los títulos conservan la
    # numeración del capstone, que es lógica y no de ejecución.
    Step("xgboost", "[3] Baseline XGBoost (SMOTE + Optuna) — queda CONGELADO",
         "src.baseline_xgboost.train_xgboost",
         [("models_dir", "xgboost_baseline.json"),
          ("reports_dir", "xgboost_val_metrics.json")],
         desc="Baseline tabular sobre las MISMAS features: SMOTE en train + "
              "búsqueda bayesiana con Optuna (30 trials). Queda CONGELADO, nunca se "
              "reentrena: es el punto de referencia contra el que se mide todo. "
              "~10 min en GPU."),
    Step("final", "[8] Comparación final GNN+CL vs XGBoost (OE4)",
         "src.comparison.final_comparison",
         [("reports_dir", "final_comparison.json")],
         desc="Compara GNN+CL contra el baseline congelado sobre el mes 6: a "
              "threshold fijo, a IGUAL presupuesto de alertas y a IGUAL precisión. "
              "Las dos últimas son las comparables — con calibraciones distintas, el "
              "threshold fijo mide agresividad, no detección. ~1 min."),
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


def show_steps(cfg):
    """Mapa de las etapas: qué hace cada una y qué deja en disco."""
    import textwrap
    print(f"\nPipeline FraudGNN — {len(STEPS)} etapas, en orden de ejecución\n")
    for i, step in enumerate(STEPS, 1):
        estado = "LISTO" if step.is_done(cfg) else "pendiente"
        print(f"{i}. {step.name}  [{estado}]")
        print(f"   {step.title}")
        for linea in textwrap.wrap(step.desc, 72):
            print(f"     {linea}")
        print(f"     salidas: {', '.join(_rel(p) for p in step.output_paths(cfg))}")
        print()
    print("Los pasos se saltan solos si sus salidas ya existen. El disco es la")
    print("verdad: borra una salida y ese paso vuelve a estar pendiente.\n")


def run_step(step: Step, cfg, i: int = 1, n: int = 1) -> bool:
    """Ejecuta un paso como subproceso. True si terminó bien."""
    cmd = [sys.executable, "-m", step.module, *step.args]
    log.info("[%d/%d] %-10s %s", i, n, step.name, step.title)
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
    log.info("        %s listo en %.1f min", step.name, minutes)
    return True


def main():
    p = argparse.ArgumentParser(
        description="Pipeline FraudGNN, reanudable.",
        epilog="Pasos: " + ", ".join(BY_NAME) + "\n"
               "--only y --skip aceptan varios separados por coma y se pueden "
               "combinar con --from. Ejemplos:\n"
               "  --only gnn,cl          solo esos dos\n"
               "  --skip xgboost         todo menos ese\n"
               "  --from gnn --skip cl   desde gnn en adelante, sin cl",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--status", action="store_true",
                   help="Mostrar en qué va y salir (no ejecuta nada)")
    p.add_argument("--steps", action="store_true",
                   help="Explicar qué hace cada etapa y salir (no ejecuta nada)")
    p.add_argument("--only", metavar="PASOS",
                   help="Ejecutar SOLO estos pasos (separados por coma)")
    p.add_argument("--skip", metavar="PASOS",
                   help="Ejecutar todo MENOS estos pasos (separados por coma)")
    p.add_argument("--from", dest="from_step", metavar="PASO",
                   help="Empezar desde ese paso")
    p.add_argument("--force", nargs="*", metavar="PASO",
                   help="Rehacer esos pasos aunque estén listos "
                        "(sin argumentos: todos)")
    args = p.parse_args()

    cfg = load_config()
    ensure_dirs(cfg)

    def lista(v):
        """'gnn, cl' -> ['gnn', 'cl']"""
        return [x.strip() for x in v.split(",") if x.strip()] if v else []

    solo, omitir = lista(args.only), lista(args.skip)
    for nombre in [*solo, *omitir, *filter(None, [args.from_step]),
                   *(args.force or [])]:
        if nombre not in BY_NAME:
            p.error(f"paso desconocido: {nombre}. Válidos: "
                    f"{', '.join(BY_NAME)}")

    if not state_path(cfg).exists():
        log.info("Primera corrida — se crea el archivo de estado en %s",
                 state_path(cfg))

    if args.steps:
        show_steps(cfg)
        return 0

    if args.status:
        show_status(cfg)
        return 0

    pasos = STEPS
    if solo:
        # Se respeta el orden del pipeline, no el que escriba el usuario:
        # los pasos dependen unos de otros.
        pasos = [s for s in STEPS if s.name in solo]
    elif args.from_step:
        pasos = STEPS[[s.name for s in STEPS].index(args.from_step):]
    if omitir:
        pasos = [s for s in pasos if s.name not in omitir]

    if not pasos:
        log.warning("La selección no deja ningún paso por ejecutar.")
        return 0
    if solo or omitir:
        log.info("Pasos seleccionados: %s", ", ".join(s.name for s in pasos))

    forzados = (set(BY_NAME) if args.force == []
                else set(args.force or []))

    listos = [s.name for s in STEPS if s.is_done(cfg)]
    log.info("Estado: %d/%d listos%s", len(listos), len(STEPS),
             "" if len(listos) == len(STEPS)
             else " | faltan: " + ", ".join(s.name for s in STEPS
                                            if not s.is_done(cfg)))

    n = len(pasos)
    for i, step in enumerate(pasos, 1):
        if step.name in forzados:
            # Borrar las salidas es lo que hace que --force funcione IGUAL en
            # todos los pasos: sin esto, módulos como download ven sus archivos
            # y se saltan solos aunque el pipeline los haya marcado forzados.
            borradas = [p for p in step.output_paths(cfg) if p.exists()]
            for ruta in borradas:
                ruta.unlink()
            if borradas:
                log.info("   forzado: borradas %d salidas previas de '%s'",
                         len(borradas), step.name)
            if step.acepta_force:
                step = replace(step, args=step.args + ["--force"])
        elif step.is_done(cfg):
            log.info("[%d/%d] %-10s ya hecho, se salta", i, n, step.name)
            update_step(cfg, step.name, status="done", title=step.title)
            continue
        if not run_step(step, cfg, i, n):
            return 1

    log.info("Pipeline completo. Reportes en %s", resolve(cfg, "reports_dir"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
