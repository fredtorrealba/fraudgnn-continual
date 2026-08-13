#!/usr/bin/env bash
# =============================================================================
# FraudGNN — pipeline del capstone. Reanudable: salta lo que ya está hecho.
#
#   bash scripts/run_pipeline.sh            corre lo que falte
#   bash scripts/run_pipeline.sh --steps    qué hace cada etapa (no ejecuta)
#   bash scripts/run_pipeline.sh --status   en qué va ahora    (no ejecuta)
#
# Toda corrida se guarda automáticamente en pipeline.log (anexando). Verlo:
#   tail -f pipeline.log                     en vivo desde otra terminal
#   grep "Época" pipeline.log | tail -20     solo el avance de las épocas
#   grep -E "WARNING|ERROR" pipeline.log     solo lo que salió mal
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
#   --archive "1capa" archiva la corrida y deja TODO en cero
#   --history         lista lo archivado
#   --force           fuerza los que dejen --only/--skip/--from.
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
set -o pipefail
cd "$(dirname "$0")/.."

# Las consultas no ejecutan nada: salen por pantalla y no ensucian el log.
for arg in "$@"; do
    case "$arg" in
        --status|--steps|--history|--archive|--help|-h)
            exec python -m src.pipeline "$@" ;;
    esac
done

# Todo lo demás SÍ es una corrida: se guarda siempre, sin tener que acordarse
# del `| tee`. Se anexa (nunca se pisa) porque el pipeline es reanudable y una
# corrida puede continuar otra. Ruta configurable con FRAUDGNN_LOG.
LOG="${FRAUDGNN_LOG:-pipeline.log}"
{
    echo
    echo "════════════════════════════════════════════════════════════════"
    echo "  $(date '+%Y-%m-%d %H:%M:%S')   run_pipeline.sh $*"
    echo "════════════════════════════════════════════════════════════════"
} >> "$LOG"

# -u (sin búfer): al pasar por la tubería, Python bufearía la salida y el
# avance aparecería a tirones en vez de línea a línea.
python -u -m src.pipeline "$@" 2>&1 | tee -a "$LOG"
exit "${PIPESTATUS[0]}"
