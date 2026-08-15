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
#   epochs 2 · seeds [42] · optuna_trials 2 · oof_folds 2
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
c["hybrid"]["oof_folds"] = 2
p.write_text(yaml.safe_dump(c, sort_keys=False, allow_unicode=True))
print(f"  epochs={c['gnn']['epochs']} seeds={c['gnn']['seeds']} "
      f"optuna={c['gnn']['optuna_trials']}/{c['xgboost']['optuna_trials']} "
      f"folds={c['hybrid']['oof_folds']}")
print(f"  INTACTOS -> device={c['xgboost']['device']} "
      f"num_workers={c['gnn']['num_workers']} n_jobs={c['compute']['n_jobs']} "
      f"batch={c['gnn']['batch_size']}")
PY

echo "== [3/4] Datos sintéticos =="
python3 scripts/make_synthetic_demo.py --n 12000

echo "== [4/4] Pipeline completo =="
t0=$(date +%s)
bash scripts/run_pipeline.sh --skip download
echo
echo "────────────────────────────────────────────────────────────"
echo "  Smoke test OK en $(( $(date +%s) - t0 ))s"
echo "  Informe: $SB/reports/exploracion_mes5.json"
echo "  El proyecto en $PROY NO se tocó."
echo "────────────────────────────────────────────────────────────"
