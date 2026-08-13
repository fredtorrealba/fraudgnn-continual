#!/usr/bin/env bash
# =============================================================================
# FraudGNN — preparar un pod de RunPod (o Vast.ai / Lambda) para entrenar.
#
# Diferencias con setup_azure.sh, que es por lo que este script existe aparte:
#   - La imagen de RunPod YA trae torch con CUDA. Reinstalarlo es tirar 2.5 GB
#     y arriesgar que las ruedas de pyg-lib dejen de calzar. Aquí se detecta la
#     versión instalada y se instalan las ruedas que le corresponden.
#   - Eres root en el contenedor: no hay sudo.
#   - Solo /workspace sobrevive a un stop/start del pod. El proyecto va ahí.
#
# Correr desde /workspace/fraudgnn:
#   bash scripts/setup_runpod.sh
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== [1/4] Paquetes del sistema =="
apt-get update -qq
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq tmux unzip git curl >/dev/null

echo "== [2/4] Comprobando el torch que ya trae la imagen =="
python -c "import torch" 2>/dev/null || { echo "!! Esta imagen no trae torch. Usa una plantilla PyTorch de RunPod."; exit 1; }
TORCH=$(python -c "import torch;print(torch.__version__.split('+')[0])")
CUDA=$(python -c "import torch;print('cu'+torch.version.cuda.replace('.','')[:3] if torch.version.cuda else 'cpu')")
echo "   torch $TORCH ($CUDA)  <- NO se reinstala"

echo "== [3/4] Dependencias del proyecto + sampler nativo =="
# --no-deps en torch para que pip no intente "arreglar" la versión de la imagen.
pip install -q --no-cache-dir $(grep -vE '^\s*#|^\s*$|^torch' requirements.txt | tr '\n' ' ')
WHL="https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html"
echo "   buscando ruedas en $WHL"
pip install -q torch-geometric
if pip install -q pyg-lib torch-sparse -f "$WHL"; then
    echo "   sampler nativo instalado"
else
    echo "   !! sin ruedas para torch $TORCH + $CUDA — se usará el fallback en Python (lento)"
    echo "      Combinaciones disponibles: https://data.pyg.org/whl/"
fi

echo "== [4/4] Verificación =="
python - <<'PYCHECK'
import torch, sys
sys.path.insert(0, ".")
from src.gnn.sampling import _has_pyg_sampler
from src.utils.common import get_device

print(f"  torch            : {torch.__version__}")
print(f"  CUDA disponible  : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    vram = props.total_memory / 1024**3
    print(f"  GPU              : {props.name}")
    print(f"  VRAM             : {vram:.1f} GB")
    # GAT retiene ~12 GB de activaciones con batch_size 1024 sobre el dataset
    # completo (~1.8M aristas/batch, tensor [aristas, heads, canales] por capa).
    if vram < 20:
        print("  !! Menos de 20 GB: baja gnn.batch_size a 512 en config/config.yaml")
        print("     ANTES de la primera corrida, y déjalo fijo para las 6 seeds.")
    else:
        print("  VRAM suficiente para batch_size 1024 sin tocar la config.")
print(f"  dispositivo      : {get_device().type}")
print(f"  sampler nativo   : {'SÍ' if _has_pyg_sampler() else 'NO (fallback, lento)'}")
if not torch.cuda.is_available():
    print("\n  !! Sin CUDA: revisa que el pod tenga GPU asignada (nvidia-smi).")
    sys.exit(1)
PYCHECK

cat <<'FIN'

Listo. Antes de entrenar:
  1. Credenciales de Kaggle:  cp .env.example .env  &&  nano .env
  2. Lanzar dentro de tmux (si se cae el SSH, el proceso sigue):
       tmux new -s fraudgnn
       bash scripts/run_pipeline.sh 2>&1 | tee -a pipeline.log
       (Ctrl-B luego D para salir; 'tmux attach -t fraudgnn' para volver)
  3. Vigilar desde otra terminal:
       tail -f pipeline.log
       watch -n 5 nvidia-smi

  OJO: solo /workspace sobrevive si detienes el pod. Verifica que estés ahí:
       pwd   ->  debe empezar con /workspace
FIN
