#!/usr/bin/env bash
# =============================================================================
# FraudGNN — Pipeline completo del capstone, en orden y REANUDABLE.
#
#   bash scripts/run_pipeline.sh            # corre lo que falte
#   bash scripts/run_pipeline.sh --steps    # qué hace cada etapa (no ejecuta)
#   bash scripts/run_pipeline.sh --status   # en qué va ahora (no ejecuta)
#
# ---------------------------------------------------------------------------
# LAS 7 ETAPAS
# ---------------------------------------------------------------------------
#  #  paso        qué hace                                          tiempo
#  -  ----------  ------------------------------------------------  ---------
#  1  download    Baja el dataset IEEE-CIS de Kaggle. El test de     ~1 min
#                 la competencia no trae etiquetas, así que los 6
#                 meses se cortan dentro del train.
#                 -> data/raw/train_{transaction,identity}.csv
#
#  2  preprocess  Une los CSV, codifica categóricas e imputa (solo   ~1 min
#                 con train, sin fuga) y parte 6 meses por fecha:
#                 1-4 entrenan | 5 valida | 6 test.
#                 -> data/processed/full.parquet + feature_cols
#
#  3  graph       Nodos = transacciones. Aristas = comparten         ~4 min
#                 tarjeta, email o dispositivo dentro de 30 días,
#                 con tope de 50 por nodo (anti-hub). ~22M aristas.
#                 -> data/graph/graph.pt
#
#  4  gnn         GraphSAGE vs GAT, 3 semillas cada uno = 6         2-4 h GPU
#                 corridas. Neighbor sampling 15-10-5: la red        <- EL CARO
#                 nunca ve el grafo entero. Elige la mejor por
#                 AUC walk-forward semanal.
#                 -> models/selected_model.json + 6 checkpoints
#
#  5  cl          Recorre el mes 6 semana a semana: detecta los     ~15 min
#                 fraudes que se escaparon, dispara el gatillo,
#                 hace fine-tuning 40/60 con replay buffer y solo
#                 despliega si aprendió SIN olvidar lo viejo.
#                 -> reports/cl_cycles.json + gnn_cl_test_scores
#
#  6  xgboost     Baseline tabular con las MISMAS features: SMOTE   ~10 min
#                 + Optuna. Queda CONGELADO, nunca se reentrena.
#                 -> models/xgboost_baseline.json
#
#  7  final       GNN+CL vs baseline sobre el mes 6, a threshold     ~1 min
#                 fijo y —lo que de verdad compara— a IGUAL
#                 presupuesto de alertas e IGUAL precisión.
#                 -> reports/final_comparison.json
#
# XGBoost va en la posición 6 y no en su [3] nominal: solo depende de
# `preprocess` y su salida solo la usa `final`, así que ponerlo al final deja
# que las 6 corridas GNN arranquen cuanto antes. Los títulos del log conservan
# la numeración del capstone, que es lógica y no de ejecución.
#
# ---------------------------------------------------------------------------
# REANUDABLE
# ---------------------------------------------------------------------------
# Antes de cada paso mira si sus archivos de salida ya existen y lo salta si
# están. Si el proceso muere (Ctrl-C, batería, desalojo de VM spot), relanza
# ESTE MISMO comando y sigue donde quedó — incluso a mitad del entrenamiento
# de una GNN, que retoma desde su última época guardada.
# El avance vive en artifacts/pipeline_state.json (se crea solo).
# El disco manda: si borras una salida, ese paso vuelve a estar pendiente.
#
# ---------------------------------------------------------------------------
# ELEGIR QUÉ CORRER
# ---------------------------------------------------------------------------
#   --from gnn        desde ese paso en adelante
#   --only gnn,cl     SOLO esos pasos            (separados por COMA)
#   --skip xgboost    todo MENOS esos pasos      (separados por COMA)
#   --force gnn cl    rehacer aunque estén listos (separados por ESPACIO)
#   --force           rehacer TODO desde cero
#
# Pasos: download, preprocess, graph, gnn, cl, xgboost, final
# --only y --skip se combinan con --from y respetan siempre el orden real.
#
# ---------------------------------------------------------------------------
# RENDIMIENTO — no se pasa por línea de comandos, vive en config/config.yaml
# ---------------------------------------------------------------------------
#   compute.n_jobs     núcleos de CPU (XGBoost, OpenMP de pyg-lib, BLAS).
#                      En contenedores nproc MIENTE: mira la cuota real con
#                      `cat /sys/fs/cgroup/cpu.max` y pon ese número.
#   xgboost.device     auto | cuda | cpu
#   gnn.num_workers    procesos de muestreo en paralelo
#   gnn.pin_memory     copia CPU->GPU más rápida
#
# En macOS, para que el equipo no se suspenda a mitad de una corrida larga:
#   caffeinate -is bash scripts/run_pipeline.sh 2>&1 | tee pipeline.log
# =============================================================================
set -e
cd "$(dirname "$0")/.."
exec python -m src.pipeline "$@"
