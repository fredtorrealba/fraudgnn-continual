#!/usr/bin/env bash
# =============================================================================
# FraudGNN — preparar una VM Linux con GPU (Azure NC/ND) para entrenar las GNN.
#
# Pensado para Ubuntu 22.04/24.04 con driver NVIDIA ya instalado (las imágenes
# "Ubuntu Server + NVIDIA GPU Driver Extension" del Marketplace lo traen).
# Correr una sola vez, desde la raíz del proyecto:
#
#   bash scripts/setup_azure.sh
#
# Qué hace: entorno virtual, torch con CUDA, dependencias del proyecto y —lo
# importante para el rendimiento— el sampler nativo (pyg-lib / torch-sparse),
# que en Linux sí tiene ruedas precompiladas. Al final verifica que todo esté
# activo ANTES de que gastes horas de VM.
#
# Si el sampler nativo no queda instalado, el proyecto igual funciona: cae al
# fallback en Python (src/gnn/sampling.py), pero el muestreo es mucho más lento.
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PY:-python3}
TORCH_VERSION=${TORCH_VERSION:-}     # ej: TORCH_VERSION=2.5.1 bash scripts/setup_azure.sh

echo "== [1/5] Paquetes del sistema =="
sudo apt-get update -qq
sudo apt-get install -y -qq python3-venv python3-dev build-essential tmux unzip

echo "== [2/5] Entorno virtual =="
[ -d .venv ] || $PY -m venv .venv
source .venv/bin/activate
pip install -qU pip wheel

echo "== [3/5] PyTorch con CUDA =="
# En Linux la rueda por defecto de PyPI ya viene con CUDA.
if [ -n "$TORCH_VERSION" ]; then
    pip install -q "torch==$TORCH_VERSION"
else
    pip install -q torch
fi
TORCH=$(python -c "import torch;print(torch.__version__.split('+')[0])")
CUDA=$(python -c "import torch;print('cu'+torch.version.cuda.replace('.','')[:3] if torch.version.cuda else 'cpu')")
echo "   torch $TORCH ($CUDA)"

echo "== [4/5] Dependencias del proyecto + sampler nativo =="
pip install -q -r requirements.txt
# El sampler nativo es lo que evita el fallback en Python puro. Las ruedas van
# por detrás de las versiones de torch: si no hay para esta combinación, se
# avisa y se sigue (el proyecto funciona igual, más lento).
WHL="https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html"
echo "   buscando ruedas en $WHL"
if pip install -q pyg-lib torch-sparse -f "$WHL"; then
    echo "   sampler nativo instalado"
else
    echo "   !! sin ruedas para torch $TORCH + $CUDA."
    echo "      Opción: reinstalar con una versión de torch que sí tenga ruedas,"
    echo "      p.ej.:  TORCH_VERSION=2.5.1 bash scripts/setup_azure.sh"
    echo "      Ver las combinaciones disponibles en https://data.pyg.org/whl/"
fi

echo "== [5/5] Verificación =="
python - <<'PYCHECK'
import torch, sys
sys.path.insert(0, ".")
from src.gnn.sampling import _has_pyg_sampler
from src.utils.common import get_device

print(f"  torch            : {torch.__version__}")
print(f"  CUDA disponible  : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  GPU              : {torch.cuda.get_device_name(0)}")
print(f"  dispositivo       : {get_device().type}")
nativo = _has_pyg_sampler()
print(f"  sampler nativo   : {'SÍ' if nativo else 'NO (fallback en Python, lento)'}")
if not torch.cuda.is_available():
    print("\n  !! Sin CUDA: revisa el driver (nvidia-smi) antes de entrenar.")
    sys.exit(1)
PYCHECK

cat <<'FIN'

Listo. Antes de entrenar:
  1. Credenciales de Kaggle:  cp .env.example .env  &&  nano .env
  2. Lanzar dentro de tmux para que no muera al cerrar la sesión SSH:
       tmux new -s fraudgnn
       bash scripts/run_pipeline.sh 2>&1 | tee pipeline.log
       (Ctrl-B luego D para salir; 'tmux attach -t fraudgnn' para volver)
  3. Vigilar desde otra terminal:
       tail -f pipeline.log
       bash scripts/run_pipeline.sh --status
       watch -n 5 nvidia-smi
FIN
