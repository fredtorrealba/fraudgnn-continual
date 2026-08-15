#!/usr/bin/env bash
# =============================================================================
# Smoke test end-to-end con datos SINTÉTICOS, en un sandbox aislado.
#
#   bash scripts/smoke_test.sh
#
# Ejercita las 7 etapas en minutos. A diferencia de la receta antigua, NO fuerza
# CPU: deja device/CUDA y num_workers como en producción, porque el objetivo es
# comprobar que la GPU y el sampler nativo funcionan ANTES de gastar horas.
#
# Reduce solo lo que no cambia los caminos de código:
#   epochs 2 · seeds [42] · optuna_trials 2
#
# El sandbox va a /tmp: no toca models/, reports/ ni data/ del proyecto.
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."
PROY="$PWD"
SB="${SMOKE_DIR:-/tmp/smoke_fraudgnn}"

echo "== [1/4] Sandbox en $SB =="
rm -rf "$SB"; mkdir -p "$SB"
cp -R src scripts config requirements.txt "$SB"/
mkdir -p "$SB"/data/{raw,processed,graph} "$SB"/{models,reports,artifacts,historial}
cd "$SB"

echo "== [2/4] Config reducido (CUDA y workers INTACTOS) =="
python3 - <<'PY'
import yaml, pathlib
p = pathlib.Path("config/config.yaml"); c = yaml.safe_load(p.read_text())
c["gnn"]["epochs"] = 2
c["gnn"]["seeds"] = [42]
c["gnn"]["patience"] = 2
c["gnn"]["optuna_trials"] = 2
c["xgboost"]["optuna_trials"] = 2
# `oof_folds` ya no se usa: la etapa `oof` la sustituyó `embed`, que entrena
# UNA sola red. Las ventanas NO se tocan: el sintético tiene los mismos 6 meses
# y 4 semanas que el dataset real, así que ejercita el mismo reparto que
# producción. Si no cuadraran, `verificar()` aborta diciendo por qué.
p.write_text(yaml.safe_dump(c, sort_keys=False, allow_unicode=True))
print(f"  epochs={c['gnn']['epochs']} seeds={c['gnn']['seeds']} "
      f"optuna={c['gnn']['optuna_trials']}/{c['xgboost']['optuna_trials']} "
      f"trials_xgb={c['xgboost']['optuna_trials']}")
print(f"  INTACTOS -> device={c['xgboost']['device']} "
      f"num_workers={c['gnn']['num_workers']} n_jobs={c['compute']['n_jobs']} "
      f"batch={c['gnn']['batch_size']}")
PY

echo "== [3/4] Datos sintéticos =="
# 40.000 y no 12.000: con las ventanas, el bloque `examen` es 1/24 del total.
# Con 12.000 se quedaba en ~17 fraudes y el PR-AUC era puro ruido; con 40.000
# son ~57, suficiente para que todas las etapas ejerciten sus caminos.
python3 scripts/make_synthetic_demo.py --n 40000

echo "== [4/4] Pipeline completo =="
t0=$(date +%s)
bash scripts/run_pipeline.sh --skip download
echo
echo "────────────────────────────────────────────────────────────"
echo "  Smoke test OK en $(( $(date +%s) - t0 ))s"
echo "  Resumen: $SB/reports/resumen.json"
echo "  El proyecto en $PROY NO se tocó."
echo "────────────────────────────────────────────────────────────"
