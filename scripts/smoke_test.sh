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

echo "== [1/5] Sandbox en $SB =="
rm -rf "$SB"; mkdir -p "$SB"
cp -R src scripts config tests requirements.txt "$SB"/
mkdir -p "$SB"/data/{raw,processed,graph} "$SB"/{models,reports,artifacts,historial}
cd "$SB"

echo "== [2/5] Config reducido (CUDA y workers INTACTOS) =="
python3 - <<'PY'
import yaml, pathlib
p = pathlib.Path("config/config.yaml"); c = yaml.safe_load(p.read_text())
c["gnn"]["epochs"] = 2
c["gnn"]["seeds"] = [42]
c["gnn"]["patience"] = 2
c["gnn"]["optuna_trials"] = 2
c["xgboost"]["optuna_trials"] = 2
# El presupuesto por tiempo MANDA sobre optuna_trials: si se deja en 60, el
# smoke se pondría a buscar una hora por arquitectura en vez de hacer 2 trials.
c["gnn"]["optuna_presupuesto_min"] = 0
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

echo "== [3/5] Datos sintéticos =="
# 40.000 y no 12.000: con las ventanas, el bloque `examen` es 1/24 del total.
# Con 12.000 se quedaba en ~17 fraudes y el PR-AUC era puro ruido; con 40.000
# son ~57, suficiente para que todas las etapas ejerciten sus caminos.
python3 scripts/make_synthetic_demo.py --n 40000

echo "== [4/5] Pipeline completo =="
t0=$(date +%s)
bash scripts/run_pipeline.sh --skip download

# Los INVARIANTES, sobre los artefactos sintéticos recién construidos. El
# pipeline de arriba solo demuestra que nada revienta; estos cazan lo peor:
# respuestas equivocadas SIN reventar (ver tests/run.sh). Corren aquí dentro
# del sandbox, así que en el pod los dos tests que exigen `pyg-lib` corren
# COMPLETOS antes de gastar horas en la corrida real. `set -e` corta el smoke
# si alguno falla — es exactamente lo que se quiere.
echo "== [5/5] Invariantes sobre los artefactos sintéticos =="
bash tests/run.sh
# Los resultados se COPIAN al proyecto, a una subcarpeta propia. La corrida
# sigue siendo aislada —si escribiera en reports/ directamente, el pipeline
# creería que las etapas están hechas y se las saltaría CON DATOS SINTÉTICOS—
# pero así no hay que ir a buscarlos a /tmp, que además se borra al parar el pod.
DEST="$PROY/reports/smoke"
rm -rf "$DEST" && mkdir -p "$DEST"
cp -R "$SB"/reports/. "$DEST"/ 2>/dev/null
cp "$SB"/models/selected_model.json "$DEST"/ 2>/dev/null
cp "$SB"/data/graph/graph_meta.json "$DEST"/ 2>/dev/null

echo
echo "────────────────────────────────────────────────────────────"
echo "  Smoke test OK en $(( $(date +%s) - t0 ))s"
echo
echo "  Resultados copiados a  reports/smoke/"
ls -1 "$DEST" | sed 's/^/     /'
echo
echo "     cat reports/smoke/resumen.json"
echo
echo "  El pipeline real no se tocó: reports/ y models/ siguen intactos."
echo "────────────────────────────────────────────────────────────"
