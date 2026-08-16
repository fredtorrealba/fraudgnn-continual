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
#   1  download      Kaggle -> data/raw/                        ~1 min
#   2  preprocess    CSV -> parquet + split de 6 meses          ~1 min
#   3  graph         grafo PyG + 8 columnas estructurales       ~5 min
#   4  gnn           GraphSAGE vs GAT, 6 corridas          30 min-4 h  <- el caro
#   5  oof           gnn_score honesto (K redes, meses 1-4)     ~20 min
#   6  hybrid        cabeza XGBoost, 3 variantes                ~12 min
#   7  refit         GNN reentrenada con meses 1-5              ~10 min
#   8  oof_refit     gnn_score honesto sobre meses 1-5          ~25 min
#   9  hybrid_refit  cabeza de producción + umbral operativo     ~3 min
#  10  cl            mes 6 semana a semana; adaptan AMBOS       ~15 min
#  11  xgboost       baseline tabular CONGELADO (viene en git)   se salta
#  12  final         baseline vs GNN sola vs híbrido             ~1 min
#
# EL SISTEMA HÍBRIDO
# La cabeza XGBoost recibe 431 features + 8 columnas del grafo + gnn_score, y
# emite la probabilidad final. Sin la GNN falta una columna: la red queda
# dentro del sistema. Las etapas `oof` existen porque la GNN memorizó los meses
# que entrenó: usar su score sobre ellos enseñaría a la cabeza a copiarla.
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
