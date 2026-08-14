"""
Paso 7 — ORQUESTADOR del ciclo completo de Continual Learning.

Evaluación de adaptación simulada temporalmente (protocolo del capstone):
el mes 6 (test) se divide en semanas tratadas como "futuro que va llegando".
El etiquetado humano se simula con las etiquetas reales del dataset.
Walk-forward estricto — nunca se mira hacia adelante.

Por cada semana:
  OPERACIÓN     el modelo vigente scorea las transacciones de la semana
  ETIQUETADO    (simulado) el "equipo humano" confirma los fraudes reales
  GATILLO       fraudes confirmados con score <0.5 se acumulan; los
                ejecutores (conteo 50 / tasa 30%) deciden el disparo
  SPLIT         70% adaptación / 30% verificación
  FINE-TUNING   40/60 con replay buffer, LR diferenciado, pos_weight ~1.2
  VALIDACIÓN    ¿mejor que el modelo anterior en lo nuevo Y en lo viejo?
                si falla -> dial estabilidad-plasticidad -> reintento
  DESPLIEGUE    el modelo nuevo reemplaza al anterior (production_model.pt)
  ACTUALIZAR    adaptación -> buffer | verificación -> control (regla de oro)

Se reporta recall antes/después por ciclo + tiempo de adaptación, y se
guardan los scores del baseline-en-el-tiempo para la comparación final.

Uso:
  python -m src.continual_learning.cl_orchestrator
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.omp import guard_omp

# ANTES de importar torch o xgboost: este proceso carga los dos y en macOS sus
# runtimes de OpenMP se pisan y el intérprete muere con SIGSEGV.
guard_omp()

import json
import time

import numpy as np
import torch

from src.continual_learning.control_set import ControlSet
from src.continual_learning.finetune import finetune
from src.continual_learning.replay_buffer import ReplayBuffer
from src.continual_learning.splitter import split_new_pattern
from src.continual_learning.trigger import ConfirmedCase, NoveltyQueue
from src.continual_learning.validate import dial_overrides, score_nodes, validate_cycle
from src.gnn.models import build_model
from src.hybrid.system import HybridSystem, cargar_cabeza, cargar_struct, cargar_umbral
from src.utils.common import ensure_dirs, get_logger, load_config, resolve, set_seed
from src.utils.metrics import full_report

log = get_logger("cl.orchestrator")


def load_selected_model(cfg, data):
    from src.gnn.train_gnn import ruta_modelo_operativo
    ruta, etiqueta = ruta_modelo_operativo(cfg)
    ckpt = torch.load(ruta, weights_only=False)
    cfg["gnn"]["in_dim"] = ckpt["in_dim"]
    model = build_model(ckpt["model_name"], cfg)
    model.load_state_dict(ckpt["state_dict"])
    log.info("Modelo en operación: %s", etiqueta)
    return model, ckpt["model_name"]


def init_memory_sets(cfg, data, model):
    """Buffer (10K) y control (5K) iniciales desde el train, disjuntos."""
    train_idx = torch.where(data.train_mask)[0].numpy()
    y = data.y[data.train_mask].numpy().astype(int)
    months = data.month[data.train_mask].numpy()
    scores = score_nodes(model, data, train_idx, cfg)

    buffer = ReplayBuffer(cfg)
    buffer.build_initial(train_idx, y, scores, months)

    control = ControlSet(cfg)
    control.build_initial(train_idx, y, months,
                          buffer_nodes=set(int(i) for i in buffer.node_indices()))
    return buffer, control


def run():
    cfg = load_config()
    ensure_dirs(cfg)
    set_seed(42)
    reports_dir = resolve(cfg, "reports_dir")
    models_dir = resolve(cfg, "models_dir")
    artifacts_dir = resolve(cfg, "artifacts_dir")

    data = torch.load(resolve(cfg, "graph_dir") / "graph.pt", weights_only=False)
    current_model, model_name = load_selected_model(cfg, data)
    buffer, control = init_memory_sets(cfg, data, current_model)

    # --- sistema en operación: híbrido si la cabeza existe, GNN sola si no ---
    # Que degrade a GNN sola no es un apaño: permite correr el ciclo antiguo sin
    # tocar nada, que es como se comparan los dos sistemas.
    struct = cargar_struct(cfg)
    current_head = cargar_cabeza(cfg)
    thr_hibrido = cargar_umbral(cfg)
    hibrido = current_head is not None and struct is not None
    if hibrido:
        log.info("Sistema en operación: HÍBRIDO (GNN + %d columnas del grafo + "
                 "cabeza XGBoost) | umbral %.4f", struct.shape[1], thr_hibrido)
    else:
        log.info("Sistema en operación: GNN sola")

    def sistema(gnn, head):
        return HybridSystem(gnn, head if hibrido else None,
                            struct, cfg, thr_hibrido if hibrido else None)

    queue = NoveltyQueue(cfg, persist=False)
    thr = thr_hibrido if hibrido else cfg["gnn"]["threshold"]
    history = []
    pattern_counter = 0

    test_idx_all = torch.where(data.test_mask)[0].numpy()

    for week in range(1, cfg["data"]["test_weeks"] + 1):
        week_mask = (data.test_mask &
                     (data.week_in_month == week))
        week_idx = torch.where(week_mask)[0].numpy()
        if len(week_idx) == 0:
            continue
        log.info("========== SEMANA %d del mes de test (%d txn) ==========",
                 week, len(week_idx))

        # --- OPERACIÓN: el sistema vigente scorea la semana ---
        # El gatillo debe mirar lo que el sistema REAL habría alertado, así que
        # aquí se usa el híbrido completo, no solo la GNN.
        scores = sistema(current_model, current_head).score(data, week_idx)
        y_week = data.y[torch.tensor(week_idx)].numpy().astype(int)
        pre_report = full_report(y_week, scores, thr)
        log.info("Recall de la semana ANTES de adaptar: %.4f", pre_report["recall"])

        # --- ETIQUETADO HUMANO (simulado con etiquetas reales) + GATILLO ---
        fired = False
        for i, node in enumerate(week_idx):
            fired = queue.ingest(ConfirmedCase(
                tid=int(node), score=float(scores[i]), is_fraud=int(y_week[i])))
            if fired:
                break

        cycle_log = {"week": week, "recall_before": pre_report["recall"],
                     "n_txn": int(len(week_idx)), "fired": bool(fired)}

        if fired:
            t_fire = time.time()
            pattern_counter += 1
            pattern_id = f"pattern_{pattern_counter}"
            novel = queue.drain()
            novel_nodes = np.array([c.tid for c in novel])
            log.info("GATILLO disparado: %d fraudes no detectados (%s)",
                     len(novel_nodes), pattern_id)

            # --- SPLIT 70/30 ---
            adapt_nodes, verif_nodes = split_new_pattern(
                novel_nodes.tolist(), cfg, seed=42 + week)
            adapt_nodes, verif_nodes = np.array(adapt_nodes), np.array(verif_nodes)

            # --- FINE-TUNING + VALIDACIÓN con dial de reintentos ---
            overrides, verdict, new_model = None, None, None
            retries = cfg["continual_learning"]["validation"]["max_retries"]
            for attempt in range(1, retries + 1):
                log.info("--- Intento %d/%d (overrides=%s) ---",
                         attempt, retries, overrides)
                new_model, ft_info = finetune(
                    current_model, data, adapt_nodes,
                    buffer.node_indices(), cfg, overrides)

                # La cabeza se adapta con la MISMA mezcla y la MISMA semilla que
                # acaba de usar la GNN: ambas piezas ven las filas idénticas.
                new_head, head_info = current_head, None
                if hibrido:
                    from src.hybrid.head_cl import warm_start
                    new_head, head_info = warm_start(
                        current_head, new_model, data, struct, adapt_nodes,
                        buffer.node_indices(), cfg, overrides)
                    ft_info["cabeza"] = head_info

                elapsed_h = (time.time() - t_fire) / 3600
                verdict = validate_cycle(sistema(new_model, new_head).scorer(data),
                                         sistema(current_model, current_head).scorer(data),
                                         data, verif_nodes, control.node_indices(),
                                         elapsed_h, cfg, threshold=thr)
                verdict["finetune_info"] = ft_info
                if verdict["deploy"]:
                    break
                overrides = dial_overrides(verdict["dial"], cfg)
                if overrides is None:   # deep_retrain: fuera del loop automático
                    break

            cycle_log["pattern_id"] = pattern_id
            cycle_log["n_novel"] = int(len(novel_nodes))
            cycle_log["verdict"] = verdict

            # --- diagnóstico cuando NO se despliega (queda en la bitácora) ---
            if verdict and not verdict["deploy"]:
                dial = verdict.get("dial")
                if dial == "plasticity":
                    diag = ("Se agotaron los reintentos sin aprender el patrón "
                            "nuevo: probablemente usa relaciones que el grafo "
                            "no modela — requiere ARISTAS NUEVAS (revisar "
                            "entidades de conexión: IP, comercio, monto, etc.).")
                elif dial == "stability":
                    diag = ("Olvido persistente pese a la receta de "
                            "estabilidad: revisar composición del replay "
                            "buffer (frontera y piso histórico).")
                else:  # deep_retrain
                    diag = ("El patrón nuevo contradice los viejos: se "
                            "programa REENTRENAMIENTO PROFUNDO (correr "
                            "src.continual_learning.deep_retrain).")
                    # programarlo de verdad: dejar el pendiente en artifacts
                    pending = {
                        "pattern_id": pattern_id,
                        "adapt_nodes": [int(n) for n in adapt_nodes],
                        "verif_nodes": [int(n) for n in verif_nodes],
                        "fired_at_week": int(week),
                    }
                    with open(artifacts_dir / "pending_deep_retrain.json",
                              "w") as f:
                        json.dump(pending, f, indent=2)
                cycle_log["diagnosis"] = diag
                log.warning(diag)

            if verdict and verdict["deploy"]:
                # --- DESPLIEGUE: reemplaza al sistema anterior ---
                # Las dos piezas se despliegan JUNTAS: la cabeza se afinó sobre
                # los scores de esta GNN, así que separarlas las descalibraría.
                current_model = new_model
                if hibrido and new_head is not None:
                    current_head = new_head
                    current_head.save_model(
                        str(models_dir / "production_head.json"))
                torch.save({"model_name": model_name,
                            "in_dim": cfg["gnn"]["in_dim"],
                            "state_dict": {k: v.cpu() for k, v
                                           in current_model.state_dict().items()},
                            "deployed_after": pattern_id},
                           models_dir / "production_model.pt")
                log.info("DESPLIEGUE: modelo actualizado (%s) en producción.",
                         pattern_id)

                # --- ACTUALIZAR CONJUNTOS (regla de oro) ---
                y_all = data.y.numpy().astype(int)
                # El buffer guarda el score de la GNN, no el del sistema: su
                # semántica es "dificultad para la red", que es lo que decide
                # qué casos frontera conviene conservar para el repaso.
                s_adapt = score_nodes(current_model, data, adapt_nodes, cfg)
                buffer.update_with_adaptation(
                    [{"node_idx": int(n), "y": int(y_all[n]),
                      "score": float(s)} for n, s in zip(adapt_nodes, s_adapt)],
                    pattern_id)
                m_all = data.month.numpy()
                control.update_with_verification(
                    [{"node_idx": int(n), "y": int(y_all[n]),
                      "month": int(m_all[n])} for n in verif_nodes],
                    pattern_id,
                    buffer_nodes=set(int(i) for i in buffer.node_indices()))

                # recall de la MISMA semana con el sistema ya adaptado
                s_after = sistema(current_model, current_head).score(data, week_idx)
                cycle_log["recall_after"] = full_report(y_week, s_after, thr)["recall"]
                log.info("Recall de la semana DESPUÉS de adaptar: %.4f "
                         "(antes: %.4f)", cycle_log["recall_after"],
                         pre_report["recall"])

        history.append(cycle_log)

    # --- scores finales del sistema GNN+CL sobre TODO el mes 6 ---
    # (para la comparación OE4 contra XGBoost congelado)
    # Se emiten los scores de AMBOS sistemas sobre el mes 6, para que la
    # comparación final pueda medir la GNN sola y el híbrido con los mismos
    # datos y el mismo modelo final. El .npz de la GNN mantiene su nombre y su
    # formato: nada de lo que ya consume ese archivo se entera del cambio.
    y_test = data.y[torch.tensor(test_idx_all)].numpy()
    final_scores = score_nodes(current_model, data, test_idx_all, cfg)
    np.savez_compressed(reports_dir / "gnn_cl_test_scores.npz",
                        node_idx=test_idx_all, scores=final_scores,
                        y=y_test)
    if hibrido:
        s_hib = sistema(current_model, current_head).score(data, test_idx_all)
        np.savez_compressed(reports_dir / "hybrid_cl_test_scores.npz",
                            node_idx=test_idx_all, scores=s_hib, y=y_test,
                            umbral=np.array([thr_hibrido]))
        log.info("Scores del híbrido sobre el mes 6 -> hybrid_cl_test_scores.npz")
    # persistir el grafo con los fraud_score poblados: el nodo queda con
    # features + isFraud + fraud_score
    data.fraud_score[torch.tensor(test_idx_all, dtype=torch.long)] = \
        torch.tensor(final_scores, dtype=torch.float)
    torch.save(data, resolve(cfg, "graph_dir") / "graph_scored.pt")

    with open(reports_dir / "cl_cycles.json", "w") as f:
        json.dump(history, f, indent=2)
    log.info("Historia de ciclos en %s | scores finales en %s",
             reports_dir / "cl_cycles.json",
             reports_dir / "gnn_cl_test_scores.npz")


if __name__ == "__main__":
    run()
