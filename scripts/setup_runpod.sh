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
pip install -q --no-cache-dir $(grep -vE '^\s*#|^\s*$|^torch|^xgboost' requirements.txt | tr '\n' ' ')
# XGBoost aparte y desalojando lo que traiga la imagen: suele venir una versión
# preinstalada (a veces el paquete `xgboost-cpu`, que instala el MISMO módulo y
# sobrevive a --force-reinstall). Se usa el pin exacto de requirements.txt, que
# está fijado por compatibilidad de CUDA con el driver del pod — ver el
# comentario allí. Síntoma de saltarse esto: "No visible GPU is found" y los 30
# trials de Optuna del paso `hybrid` pasan de ~7 min a ~50. Lo verifica [4/4].
pip uninstall -yq xgboost xgboost-cpu 2>/dev/null || true
pip install -q --no-cache-dir "$(grep '^xgboost' requirements.txt)"

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
    # La VRAM que hace falta depende de la PROFUNDIDAD: cada capa es un salto,
    # y el subgrafo por batch crece con el fanout acumulado. Medido con
    # batch_size 1024 sobre el dataset completo:
    #   1 capa  -> < 1 GB     3.4K nodos por batch
    #   2 capas -> ~4 GB     12.3K nodos
    #   3 capas -> ~17 GB    23.7K nodos, 1.8M aristas (GAT retiene ~12 GB
    #                        solo en tensores de atención)
    import yaml
    with open("config/config.yaml") as fh:
        capas = len(yaml.safe_load(fh)["gnn"]["hidden_dims"])
    necesita = {1: 2, 2: 6, 3: 20}.get(capas, 20)
    print(f"  capas (config)   : {capas}  -> necesita ~{necesita} GB con batch 1024")
    if vram < necesita:
        print(f"  !! {vram:.1f} GB puede no bastar para {capas} capas: baja")
        print("     gnn.batch_size a 512 ANTES de la primera corrida, y déjalo")
        print("     fijo para las 6 seeds.")
    else:
        print("  VRAM suficiente para batch_size 1024 sin tocar la config.")
print(f"  dispositivo      : {get_device().type}")
print(f"  sampler nativo   : {'SÍ' if _has_pyg_sampler() else 'NO (fallback, lento)'}")
if torch.cuda.is_available():
    # Que torch vea la GPU no implica que XGBoost la vea: son runtimes CUDA
    # distintos. Se comprueba entrenando 2 árboles sobre datos de juguete.
    import subprocess, xgboost as xgb
    # Que torch vea la GPU no implica que XGBoost la vea: son runtimes CUDA
    # distintos. Y las dos comprobaciones "obvias" fallan:
    #   - build_info()["USE_CUDA"] dice True aunque el wheel no arranque con
    #     este driver (medido: xgboost 3.4.0 con driver 570).
    #   - train(device="cuda") NO lanza excepción: avisa y cae a CPU.
    # Lo único fiable es entrenar de verdad en otro proceso y buscar el aviso.
    sonda = ("import xgboost as xgb, numpy as np;"
             "X=np.random.rand(2000,8); y=(np.random.rand(2000)>.5).astype(int);"
             "xgb.train({'device':'cuda','tree_method':'hist'},"
             "xgb.DMatrix(X,label=y),3)")
    r = subprocess.run([sys.executable, "-c", sonda], capture_output=True,
                       text=True)
    if "No visible GPU" in (r.stdout + r.stderr) or r.returncode != 0:
        print(f"  !! xgboost {xgb.__version__}: NO usa la GPU")
        print("     El paso `hybrid` correrá en CPU: ~50 min en vez de ~7.")
        print("     Causa habitual: el wheel se compiló contra un CUDA MÁS")
        print("     NUEVO que el driver del pod. Mira tu driver con:")
        print("       nvidia-smi --query-gpu=driver_version --format=csv")
        print("     Verificado OK: xgboost 3.0.x con driver 570 (CUDA 12.8).")
        print("     Con driver 580+ puedes soltar el tope de requirements.txt.")
    else:
        print(f"  xgboost {xgb.__version__:<9}: GPU OK")

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
