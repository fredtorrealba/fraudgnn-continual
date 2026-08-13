#!/usr/bin/env bash
# =============================================================================
# FraudGNN — Pipeline completo del capstone, en orden y REANUDABLE.
#
# Correr desde la raíz del proyecto:  bash scripts/run_pipeline.sh
#
# Ya no es una lista de comandos a ciegas: src/pipeline.py revisa, paso por
# paso, si sus archivos de salida ya existen y salta los que están hechos.
# Si el proceso muere (Ctrl-C, batería, kernel panic), relanza ESTE MISMO
# comando y sigue donde quedó — incluso a mitad del entrenamiento de una GNN,
# que se reanuda desde su última época guardada.
#
# El avance vive en artifacts/pipeline_state.json (se crea solo).
#
# Otras formas de invocarlo (los argumentos pasan derecho al runner):
#   bash scripts/run_pipeline.sh --status        # ver en qué va, sin ejecutar
#   bash scripts/run_pipeline.sh --from gnn      # desde ese paso en adelante
#   bash scripts/run_pipeline.sh --only gnn,cl   # SOLO esos pasos (coma)
#   bash scripts/run_pipeline.sh --skip xgboost  # todo MENOS esos pasos (coma)
#   bash scripts/run_pipeline.sh --force xgboost # rehacer ese paso
#   bash scripts/run_pipeline.sh --force         # rehacer TODO desde cero
#
# Pasos: download, preprocess, graph, xgboost, gnn, cl, final
# --only/--skip se combinan con --from y respetan siempre el orden real del
# pipeline, no el orden en que los escribas.
#
# El paralelismo NO se pasa por línea de comandos: vive en config/config.yaml
#   compute.n_jobs      núcleos de CPU (XGBoost, OpenMP de pyg-lib, BLAS)
#   xgboost.device      auto | cuda | cpu
#   gnn.num_workers     procesos de muestreo en paralelo
#   gnn.pin_memory      copia CPU->GPU más rápida
#
# En macOS, para que el equipo no se suspenda a mitad de una corrida larga:
#   caffeinate -is bash scripts/run_pipeline.sh 2>&1 | tee pipeline.log
# =============================================================================
set -e
cd "$(dirname "$0")/.."
exec python -m src.pipeline "$@"
