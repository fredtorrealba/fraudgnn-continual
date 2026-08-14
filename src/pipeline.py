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
  python -m src.pipeline --only gnn --force   # rehacer SOLO ese paso
  python -m src.pipeline --skip cl --force   # rehacer todo menos ese
  python -m src.pipeline --force             # rehacer TODO
  python -m src.pipeline --archive "1capa"   # archivar y dejar en cero
"""
import argparse
import json
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
         [("graph_dir", "graph.pt"),
          ("processed_dir", "graph_features.parquet")],
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
    Step("oof", "[5a] gnn_score honesto por validación cruzada (meses 1-4)",
         "src.hybrid.oof", [("processed_dir", "gnn_oof_train.parquet"),
                            ("reports_dir", "oof_train.json")],
         args=["--window", "train"],
         desc="Entrena K redes dejando un trozo fuera cada vez y puntúa el "
              "excluido. Sin esto, la columna gnn_score sobre los meses 1-4 "
              "reflejaría lo que la red MEMORIZÓ, no lo que acierta, y la "
              "cabeza aprendería a copiarla. ~20 min."),
    Step("hybrid", "[5b] Cabeza XGBoost del sistema híbrido (variantes)",
         "src.hybrid.train_head",
         [("models_dir", "hybrid_head_440.json"),
          ("reports_dir", "hybrid_variants.json")],
         args=["--window", "train"],
         desc="Entrena la cabeza con 431 / 439 / 440 columnas —y con el "
              "embedding entero de la GNN si hybrid.usar_embedding— para separar "
              "cuánto aporta la estructura del grafo y cuánto la red. Optuna "
              "corre UNA vez y las tres comparten hiperparámetros. ~12 min."),
    Step("refit", "[6] Refit del ganador sobre train + validación",
         "src.gnn.refit",
         [("models_dir", "refit_model.pt"), ("reports_dir", "refit.json")],
         desc="Refit estándar: la validación ya eligió arquitectura y número de "
              "épocas, así que se reentrena DESDE CERO con todos los datos hasta "
              "el mes 5 (+21%). Sin early stopping — las épocas se heredan del "
              "pico de la corrida ganadora. El mes 6 sigue intacto. ~35 min GPU."),
    Step("oof_refit", "[6a] gnn_score honesto sobre meses 1-5",
         "src.hybrid.oof", [("processed_dir", "gnn_oof_trainval.parquet"),
                            ("reports_dir", "oof_trainval.json")],
         args=["--window", "trainval"],
         desc="Lo mismo que `oof` pero sobre la ventana del refit. ~25 min."),
    Step("hybrid_refit", "[6b] Cabeza de producción + umbral operativo",
         "src.hybrid.train_head",
         [("models_dir", "hybrid_head_prod.json"),
          ("reports_dir", "hybrid_thresholds.json")],
         args=["--window", "trainval"],
         desc="Reentrena TODAS las variantes con meses 1-5 (sin Optuna, heredan "
              "los hiperparámetros) y fija el umbral por presupuesto de alertas "
              "sobre el mes 5. ~3 min."),
    Step("cl", "[7] Ciclo de Continual Learning (mes 6 por semanas)",
         "src.continual_learning.cl_orchestrator",
         [("reports_dir", "cl_cycles.json"),
          ("reports_dir", "gnn_cl_test_scores.npz"),
          ("reports_dir", "hybrid_cl_test_scores.npz"),
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


# Salidas que NO se mueven al archivar: se COPIAN y se quedan donde están.
# El baseline XGBoost está congelado por diseño y no depende del grafo ni de
# gnn.hidden_dims — es la MISMA constante contra la que se miden todas las
# configuraciones. Reentrenarlo en cada corrida no solo cuesta 10 minutos:
# introduciría variación en la vara de medir.
COMPARTIDAS = [("models_dir", "xgboost_baseline.json"),
               ("reports_dir", "xgboost_val_metrics.json")]


def archivar(cfg, nombre: str | None = None) -> Path:
    """
    MUEVE los resultados de la corrida actual a historial/<fecha>_<nombre>/.

    Se archivan models/, reports/ y artifacts/ junto con una copia del
    config.yaml que los produjo — sin esa copia, un resultado antiguo no se
    puede interpretar (¿cuántas capas tenía? ¿qué batch_size?).

    Y se BORRA data/ entero: archivar cierra la corrida y deja el repositorio
    en cero. No se copia porque pesa GB y es determinista — con el config
    archivado se reconstruye igual, y la huella del grafo lo verifica.
    """
    import re, shutil, subprocess
    from datetime import datetime

    sello = datetime.now().strftime("%Y%m%d-%H%M")
    slug = re.sub(r"[^a-z0-9]+", "-", (nombre or "").lower()).strip("-")
    destino = resolve(cfg, "history_dir") / (f"{sello}_{slug}" if slug else sello)
    destino.mkdir(parents=True, exist_ok=True)

    compartidas = {resolve(cfg, k) / n for k, n in COMPARTIDAS}
    movidos, copiados = 0, 0
    for clave in ("models_dir", "reports_dir", "artifacts_dir"):
        origen = resolve(cfg, clave)
        sub = destino / origen.name
        for f in sorted(origen.iterdir()):
            if f.name == ".gitkeep" or f.name.startswith("."):
                continue
            sub.mkdir(parents=True, exist_ok=True)
            if f in compartidas:
                shutil.copy2(f, sub / f.name)   # el archivo queda autocontenido
                copiados += 1                    # pero el original NO se toca
            else:
                shutil.move(str(f), str(sub / f.name))
                movidos += 1

    if movidos == 0:
        destino.rmdir()
        log.warning("No había nada que archivar (models/, reports/ y "
                    "artifacts/ están vacíos).")
        return destino

    shutil.copy2(ROOT / "config" / "config.yaml", destino / "config.yaml")

    # feature_cols.json (5 KB) define QUÉ vio el modelo: sin él un resultado
    # archivado no se puede reinterpretar. Barato y necesario.
    fc = resolve(cfg, "processed_dir") / "feature_cols.json"
    if fc.exists():
        shutil.copy2(fc, destino / "feature_cols.json")

    # graph_scored.pt lo PRODUCE la corrida (paso cl) pero cae en data/, así que
    # se mueve aquí igual que los reports. graph.pt NO: es determinista a partir
    # de full.parquet + la sección graph: del config, y pesa 1.3 GB.
    gs = resolve(cfg, "graph_dir") / "graph_scored.pt"
    if gs.exists():
        shutil.move(str(gs), str(destino / "graph_scored.pt"))
        movidos += 1

    # El log se MUEVE (no se copia): así cada archivo conserva el suyo y la
    # corrida siguiente arranca con pipeline.log limpio, sin arrastrar el
    # historial de la anterior.
    for log_file in (ROOT / "pipeline.log", Path.cwd() / "pipeline.log"):
        if log_file.exists():
            shutil.move(str(log_file), str(destino / "pipeline.log"))
            movidos += 1
            break

    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                                cwd=ROOT, capture_output=True, text=True,
                                check=True).stdout.strip()
    except Exception:
        commit = None

    g = cfg["gnn"]
    meta = {
        "fecha": datetime.now().isoformat(timespec="seconds"),
        "nombre": nombre,
        "commit": commit,
        "archivos": movidos,
        "config": {
            "capas": len(g["hidden_dims"]), "hidden_dims": g["hidden_dims"],
            "fanouts": g["fanouts"], "batch_size": g["batch_size"],
            "epochs": g["epochs"], "patience": g["patience"],
            "num_workers": g.get("num_workers"), "seeds": g["seeds"],
            "n_jobs": (cfg.get("compute") or {}).get("n_jobs"),
            "xgboost_device": cfg["xgboost"].get("device"),
        },
        "grafo": _huella_grafo(cfg),
        "resumen": _resumen_metricas(destino),
    }
    with open(destino / "meta.json", "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    log.info("Archivados %d archivos en %s", movidos + copiados, _rel(destino))
    if copiados:
        log.info("   %d copiadas (no movidas): el baseline XGBoost se conserva "
                 "para que no haya que reentrenarlo", copiados)

    # data/ se borra ENTERO: archivar significa cerrar la corrida y dejar el
    # repositorio en cero. Es reproducible porque el config.yaml viaja con el
    # archivo, y la huella del grafo (nodos/aristas/features) permite verificar
    # que el reconstruido es el mismo.
    borrados, bytes_ = 0, 0
    for clave in ("raw_dir", "processed_dir", "graph_dir"):
        for f in resolve(cfg, clave).iterdir():
            if f.name == ".gitkeep" or f.name.startswith("."):
                continue
            bytes_ += f.stat().st_size
            f.unlink()
            borrados += 1
    if borrados:
        log.info("data/ vaciado: %d archivos, %.1f GB liberados",
                 borrados, bytes_ / 1073741824)

    log.info("Todo limpio. La corrida siguiente arranca desde cero:")
    log.info("  bash scripts/run_pipeline.sh          (~7 min de datos + entrenamiento)")
    return destino


def _huella_grafo(cfg) -> dict:
    """
    Identidad del grafo usado, SIN copiarlo (pesa 1.3 GB y es determinista a
    partir de full.parquet + la sección graph: del config). Con estos números
    se comprueba si dos corridas archivadas partieron del mismo grafo; si no,
    se reconstruye desde el config que acompaña al archivo.
    """
    ruta = resolve(cfg, "graph_dir") / "graph.pt"
    out = {"config_graph": cfg.get("graph"), "existe": ruta.exists()}
    if not ruta.exists():
        return out
    out["bytes"] = ruta.stat().st_size
    try:
        import torch
        d = torch.load(ruta, weights_only=False, map_location="cpu")
        out.update(nodos=int(d.num_nodes), aristas=int(d.edge_index.shape[1]),
                   features=int(d.x.shape[1]))
    except Exception as e:
        out["error"] = str(e)
    return out


def _resumen_metricas(dir_archivo: Path) -> dict:
    """Extrae los números clave del archivo recién creado, para comparar luego."""
    out = {}

    def leer(rel):
        f = dir_archivo / rel
        if not f.exists():
            return None
        try:
            with open(f) as fh:
                return json.load(fh)
        except Exception:
            return None

    sel = leer("models/selected_model.json")
    if sel and "selection" in sel:
        s = sel["selection"]
        out["gnn"] = {"ganadora": s.get("selected"), "seed": s.get("seed"),
                      "auc_mean": s.get("auc_mean"), "auc_std": s.get("auc_std"),
                      "kpi_ok": s.get("kpi_auc_ok")}
        for arq, r in (sel.get("results") or {}).items():
            out.setdefault("por_arquitectura", {})[arq] = {
                "auc_mean": r.get("auc_mean"), "pr_auc": r.get("pr_auc_mean")}

    fin = leer("reports/final_comparison.json")
    if fin:
        g = (fin.get("month6_overall") or {})
        for k, etq in (("xgboost_frozen", "xgboost"),
                       ("gnn_continual_learning", "gnn_cl")):
            if k in g:
                out.setdefault("mes6", {})[etq] = {
                    "roc_auc": g[k].get("auc_roc"), "pr_auc": g[k].get("pr_auc"),
                    "recall": g[k].get("recall"), "precision": g[k].get("precision")}
        mb = fin.get("matched_budget") or {}
        if mb:
            out.setdefault("mes6", {})["igual_presupuesto"] = {
                "alertas": mb.get("presupuesto_referencia"),
                "recall_xgboost": mb.get("recall_xgboost_ref"),
                "recall_gnn_cl": mb.get("recall_gnn_cl_ref")}

    ciclos = leer("reports/cl_cycles.json")
    if isinstance(ciclos, list):
        out["cl"] = {"ciclos": len(ciclos),
                     "desplegados": sum(1 for c in ciclos
                                        if (c.get("verdict") or {}).get("deploy"))}
    return out


def listar_historial(cfg):
    """Corridas archivadas, de la más reciente a la más antigua."""
    raiz = resolve(cfg, "history_dir")
    dirs = sorted([d for d in raiz.iterdir() if d.is_dir()], reverse=True) \
        if raiz.exists() else []
    if not dirs:
        print("\nNo hay corridas archivadas. Crea una con --archive [nombre]\n")
        return
    print(f"\n{len(dirs)} corrida(s) archivada(s) en {_rel(raiz)}\n")
    for d in dirs:
        meta = {}
        if (d / "meta.json").exists():
            with open(d / "meta.json") as f:
                meta = json.load(f)
        c = meta.get("config", {})
        print(f"  {d.name}")
        if c:
            print(f"    capas={c.get('capas')} {c.get('hidden_dims')} "
                  f"fanouts={c.get('fanouts')} batch={c.get('batch_size')}")
        r = meta.get("resumen", {})
        if "gnn" in r:
            g = r["gnn"]
            print(f"    GNN: gana {g.get('ganadora')} | AUC {g.get('auc_mean')}")
        if "mes6" in r:
            m = r["mes6"]
            for k in ("xgboost", "gnn_cl"):
                if k in m:
                    print(f"    mes6 {k:8s} PR-AUC {m[k].get('pr_auc')} | "
                          f"ROC {m[k].get('roc_auc')}")
        print()


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
               "--only y --skip aceptan varios separados por coma y se combinan "
               "con --from. --force actúa sobre lo que ellos dejen:\n"
               "  --only gnn,cl            solo esos dos, lo que falte\n"
               "  --only gnn,cl --force    solo esos dos, rehaciéndolos\n"
               "  --skip xgboost --force   rehace todo menos xgboost\n"
               "  --force                  rehace TODO",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--status", action="store_true",
                   help="Mostrar en qué va y salir (no ejecuta nada)")
    p.add_argument("--steps", action="store_true",
                   help="Explicar qué hace cada etapa y salir (no ejecuta nada)")
    p.add_argument("--archive", nargs="?", const="", metavar="NOMBRE",
                   help="Archivar la corrida en historial/ (con el config que "
                        "la produjo) y dejar models/, reports/, artifacts/ y "
                        "data/ en cero. Sale sin ejecutar nada")
    p.add_argument("--history", action="store_true",
                   help="Listar las corridas archivadas y salir")
    p.add_argument("--only", metavar="PASOS",
                   help="Ejecutar SOLO estos pasos (separados por coma)")
    p.add_argument("--skip", metavar="PASOS",
                   help="Ejecutar todo MENOS estos pasos (separados por coma)")
    p.add_argument("--from", dest="from_step", metavar="PASO",
                   help="Empezar desde ese paso")
    p.add_argument("--force", action="store_true",
                   help="Rehacer los pasos seleccionados aunque ya estén "
                        "listos. Actúa sobre lo que dejen --only/--skip/--from; "
                        "solo, rehace TODO")
    args = p.parse_args()

    cfg = load_config()
    ensure_dirs(cfg)

    def lista(v):
        """'gnn, cl' -> ['gnn', 'cl']"""
        return [x.strip() for x in v.split(",") if x.strip()] if v else []

    solo, omitir = lista(args.only), lista(args.skip)
    for nombre in [*solo, *omitir, *filter(None, [args.from_step])]:
        if nombre not in BY_NAME:
            p.error(f"paso desconocido: {nombre}. Válidos: "
                    f"{', '.join(BY_NAME)}")

    if not state_path(cfg).exists():
        log.info("Primera corrida — se crea el archivo de estado en %s",
                 state_path(cfg))

    if args.history:
        listar_historial(cfg)
        return 0

    if args.archive is not None:
        archivar(cfg, args.archive or None)
        return 0

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

    # --force no elige pasos: fuerza los que ya eligieron --only/--skip/--from.
    # Así "--only graph --force" rehace SOLO el grafo, en vez de rehacer el
    # grafo y además arrastrar cualquier otro paso pendiente.
    forzados = {p.name for p in pasos} if args.force else set()

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
