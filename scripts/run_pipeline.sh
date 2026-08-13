#!/usr/bin/env bash
# =============================================================================
# FraudGNN — pipeline del capstone. Reanudable: salta lo que ya está hecho.
#
#   bash scripts/run_pipeline.sh            corre lo que falte
#   bash scripts/run_pipeline.sh --steps    qué hace cada etapa (no ejecuta)
#   bash scripts/run_pipeline.sh --status   en qué va ahora    (no ejecuta)
#
# ETAPAS
#   1 download    Kaggle -> data/raw/                      ~1 min
#   2 preprocess  CSV -> parquet + split de 6 meses        ~1 min
#   3 graph       parquet -> grafo PyG (~22M aristas)      ~4 min
#   4 gnn         GraphSAGE vs GAT, 6 corridas         2-4 h GPU  <- el caro
#   5 cl          mes 6 semana a semana + fine-tuning      ~15 min
#   6 xgboost     baseline tabular CONGELADO               ~10 min
#   7 final       comparación GNN+CL vs baseline           ~1 min
#
# ELEGIR QUÉ CORRER
#   --only gnn,cl     SOLO esos pasos             (COMA)
#   --skip xgboost    todo MENOS esos             (COMA)
#   --from gnn        desde ahí en adelante
#   --force gnn cl    rehacer aunque estén listos (ESPACIO)
#   --force           rehacer TODO
#   --archive "1capa" guarda y limpia
#    --history        lista lo archivado
#
# --force BORRA las salidas del paso antes de relanzarlo. En `download` eso
# significa volver a bajar 650 MB de Kaggle.
#
# RENDIMIENTO — no va por línea de comandos, vive en config/config.yaml:
#   compute.n_jobs   núcleos de CPU. En contenedores `nproc` MIENTE: usa la
#                    cuota real de `cat /sys/fs/cgroup/cpu.max`.
#   xgboost.device   auto | cuda | cpu
#   gnn.num_workers  procesos de muestreo en paralelo
#   gnn.log_every    0 = una línea por época; >0 = avance cada N batches
#
# En macOS, para que no se suspenda a mitad de una corrida larga:
#   caffeinate -is bash scripts/run_pipeline.sh 2>&1 | tee pipeline.log
# =============================================================================
set -e
cd "$(dirname "$0")/.."
exec python -m src.pipeline "$@"
