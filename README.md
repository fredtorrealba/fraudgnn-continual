# FraudGNN — Detección Adaptativa de Fraude con Grafos

Código del Capstone **"FraudGNN: Detección Adaptativa con Grafos"** (MIA UAI, Grupo 18 — Freddy Torrealba).
Sistema de detección de fraude en transacciones card-not-present usando Graph Neural Networks
(GraphSAGE / GAT) con Continual Learning, comparado contra un baseline XGBoost congelado,
sobre el dataset IEEE-CIS.

El proyecto es **experimental**: todo corre localmente con PyTorch Geometric sobre el grafo
serializado. No incluye capa de servicio (API, base de grafos ni despliegue).

---

## 1. Estructura del proyecto

```
fraudgnn/
├── config/
│   └── config.yaml              # TODOS los hiperparámetros del sistema en un solo lugar
├── src/
│   ├── pipeline.py              # runner del pipeline completo, reanudable (sabe qué falta)
│   ├── data/                    # descarga, preprocesamiento y construcción del grafo
│   │   ├── download_ieee_cis.py
│   │   ├── preprocessing.py
│   │   └── build_graph.py
│   ├── baseline_xgboost/        # baseline tabular (caja aislada, se entrena UNA vez)
│   │   ├── smote_pipeline.py
│   │   └── train_xgboost.py
│   ├── gnn/                     # las dos redes y el comparador
│   │   ├── models.py            # GraphSAGE y GAT (misma columna vertebral)
│   │   ├── sampling.py          # neighbor sampling 15-10-5 (con fallback sin torch-sparse)
│   │   ├── train_gnn.py
│   │   └── compare_gnns.py
│   ├── continual_learning/      # todo el ciclo de adaptación
│   │   ├── trigger.py           # gatillo de novedad (ejecutores por conteo y por tasa)
│   │   ├── splitter.py          # split 70/30 adaptación/verificación
│   │   ├── replay_buffer.py     # buffer de repaso (~10K, estratificado, frontera)
│   │   ├── control_set.py       # set de control congelado (~5K, disjunto del buffer)
│   │   ├── finetune.py          # fine-tuning 40/60 con LR diferenciado por capa
│   │   ├── validate.py          # doble validación (aprendió / no olvidó) + dial
│   │   ├── cl_orchestrator.py   # orquesta el mes 6 semana a semana
│   │   └── deep_retrain.py      # reentrenamiento profundo (última carta del dial)
│   ├── comparison/
│   │   └── final_comparison.py  # OE4: GNN+CL vs XGBoost congelado
│   └── utils/                   # config, logging, seeds, dispositivo, métricas
├── scripts/
│   ├── run_pipeline.sh          # corre todo el pipeline en orden
│   ├── make_synthetic_demo.py   # demo sintética para probar sin el dataset real
│   ├── graph_explorer.py        # explorador interactivo del grafo (HTML navegable)
│   └── plot_embeddings.py       # t-SNE del espacio latente (antes vs después del CL)
├── requirements.txt
└── README.md                    # este archivo
```

**Directorios que se crean al correr** (no van en el zip): `data/`, `models/`, `reports/`, `artifacts/`.

---

## 2. Instalación

Requiere Python 3.10+.

```bash
pip install -r requirements.txt
```

Notas:
- `torch` conviene instalarlo primero según tu máquina (CPU o CUDA): https://pytorch.org/get-started/locally/
  En macOS basta `pip install torch` (la rueda de PyPI ya trae soporte MPS).
- **macOS**: XGBoost necesita el runtime de OpenMP → `brew install libomp`. Sin eso, el
  paso 3 falla al importar `xgboost` con un error de `libomp.dylib` no encontrada.
- `torch-sparse` / `pyg-lib` son **opcionales**: aceleran el neighbor sampling de PyG.
  Si no están, el código usa un sampler propio (`src/gnn/sampling.py`) que implementa
  el mismo protocolo 15-10-5 en numpy. Para el dataset completo (590K nodos) sí conviene
  instalarlos.

### Entrenar en una VM con GPU (Azure y similares)

En CPU, una época de GAT sobre el dataset completo tarda ~70 min (401 batches ×
~9,5 s): las 6 corridas del paso 4-5 se van a días. El costo dominante es la
atención de GAT sobre los ~1,8M de aristas de cada batch, que es exactamente lo
que una GPU resuelve bien. Con GPU + sampler nativo el paso completo baja a
horas.

SKU de referencia en Azure: `Standard_NC8as_T4_v3` (1× T4 16 GB, 8 vCPU, 56 GB
RAM), imagen *Ubuntu Server + NVIDIA GPU Driver Extension*, disco de 128 GB.
Como el pipeline es reanudable, una VM **spot** es buena idea: si te desalojan,
relanzas el mismo comando y retoma en la época que iba.

```bash
git clone <repo> && cd fraudgnn
bash scripts/setup_azure.sh        # venv + torch CUDA + deps + sampler nativo
cp .env.example .env && nano .env  # KAGGLE_USERNAME / KAGGLE_KEY
tmux new -s fraudgnn               # que no muera al cerrar el SSH
bash scripts/run_pipeline.sh 2>&1 | tee pipeline.log
```

El script verifica al final que `torch.cuda.is_available()` sea `True` y que el
sampler nativo esté activo — conviene mirarlo antes de gastar horas de VM.

**Importante para la comparación del OE2:** si migras de máquina, reentrena las
**6** corridas allá, no solo las que faltan. CPU y GPU dan resultados
numéricamente distintos y el paso 5 exige que ambas arquitecturas se midan en
igualdad de condiciones (ver `get_device()` en `src/utils/common.py`). Lo mismo
vale si activas el sampler nativo: cambia el muestreo respecto del fallback.
Basta con partir de `models/` y `reports/` limpios.

### Dataset

#### Opción A — IEEE-CIS real

**Este repositorio no incluye credenciales.** Lo configurable no secreto está en
`config/config.yaml` → sección `kaggle:` (competencia, archivos a bajar, dónde
buscar e instalar la credencial). El secreto va en un `.env` que **no se versiona**:

```bash
cp .env.example .env      # y editar
python -m src.data.download_ieee_cis --check
```

| # | Vía | Cómo | Caduca |
|---|---|---|---|
| 1 | **Usuario + API key** (recomendada) | `KAGGLE_USERNAME` y `KAGGLE_KEY` en `.env` o como variables de entorno. Salen del `kaggle.json` de [kaggle.com/settings](https://www.kaggle.com/settings) → API → *Create New Token* | No |
| 2 | Token de sesión | `kaggle auth login`, o pegar el `KGAT_...` en `.env` como `KAGGLE_API_TOKEN` | **Sí, en horas** |
| 3 | Archivo ya instalado | `~/.kaggle/kaggle.json` o `~/.kaggle/access_token` | Según cuál |

Si das el token por `.env` o por entorno, el script lo **instala solo** en
`~/.kaggle/access_token` con permisos 600 — no hay que correr ningún comando a mano.

Además hay que **aceptar las reglas de la competencia una vez** con esa misma cuenta:
[ieee-fraud-detection/rules](https://www.kaggle.com/competitions/ieee-fraud-detection/rules).
Sin eso la API responde **403** aunque las credenciales sean válidas — es el error
más común al empezar.

```bash
python -m src.data.download_ieee_cis           # solo los 2 archivos que usa el pipeline
python -m src.data.download_ieee_cis --all     # además test_* y sample_submission
python -m src.data.download_ieee_cis --force   # re-descargar
```

Por defecto baja **solo** `train_transaction.csv` (~650 MB) y `train_identity.csv`
(~26 MB): el test de Kaggle no trae etiquetas, así que los 6 "meses" del split
temporal se cortan dentro de `train_transaction.csv`. Si algo falla, el script
imprime exactamente qué configurar.

También puedes bajar esos dos archivos a mano desde la pestaña *Data* y dejarlos
en `data/raw/` — el resto del pipeline no nota la diferencia.

#### Opción B — demo sintética (sin Kaggle, en segundos)
```bash
python scripts/make_synthetic_demo.py --n 20000
```
Genera 6 "meses" con un patrón de fraude conocido (meses 1-5) y un patrón
**emergente solo en el mes 6** (banda de features que en el entrenamiento fue siempre
legítima + anillo nuevo de tarjetas), justamente para gatillar el continual learning.
Útil para validar el flujo completo antes de invertir horas en el dataset real.

---

## 3. Cómo ejecutar cada módulo (en orden)

Todo se corre desde la raíz del proyecto. El pipeline completo:

```bash
bash scripts/run_pipeline.sh
```

**El pipeline es reanudable: no repite lo que ya está hecho.** Antes de cada
paso revisa si sus archivos de salida existen y, si están, lo salta. Si el
proceso muere (Ctrl-C, batería, kernel panic, cierre de sesión SSH), se
relanza *el mismo comando* y sigue donde quedó — incluso a mitad del
entrenamiento de una GNN, que retoma desde su última época guardada. El
avance queda en `artifacts/pipeline_state.json`, que se crea solo.

```bash
bash scripts/run_pipeline.sh --status        # ver en qué va, sin ejecutar nada
bash scripts/run_pipeline.sh --from gnn      # desde ese paso en adelante
bash scripts/run_pipeline.sh --only cl       # un solo paso
bash scripts/run_pipeline.sh --force xgboost # rehacer ese paso aunque esté listo
bash scripts/run_pipeline.sh --force         # rehacer TODO desde cero
```

Pasos válidos para `--only/--from/--force`: `download`, `preprocess`, `graph`,
`xgboost`, `gnn`, `cl`, `final`. En macOS conviene lanzarlo a prueba de
suspensión: `caffeinate -is bash scripts/run_pipeline.sh 2>&1 | tee pipeline.log`.

O paso a paso:

### Paso 1 — Preprocesamiento y split temporal
```bash
python -m src.data.preprocessing
```
Une transaction+identity, codifica categóricas e imputa (ajustado SOLO con train,
sin fuga temporal), y asigna el split: **meses 1-4 train / mes 5 validación / mes 6 test**
(el mes 6 además queda dividido en 4 semanas para el walk-forward del CL).
Salidas: `data/processed/full.parquet`, `feature_cols.json`, `split_masks.parquet`.

### Paso 2 — Construcción del grafo
```bash
python -m src.data.build_graph
```
Grafo homogéneo: nodos = transacciones, aristas = entidad compartida
(huella de tarjeta / email+card1 / dispositivo), ventana de 30 días y tope de
50 aristas por nodo (anti-hub). Salida: `data/graph/graph.pt` (PyTorch Geometric).

### Paso 3 — Baseline XGBoost (caja aislada)
```bash
python -m src.baseline_xgboost.train_xgboost
```
SMOTE **solo aquí** (los sintéticos tabulares no tienen aristas, por eso no aplica a la GNN)
+ búsqueda de hiperparámetros con Optuna maximizando AUC en validación.
El modelo queda **congelado** en `models/xgboost_baseline.json` y no se toca nunca más:
representa al sistema tradicional que no se adapta.

### Pasos 4-5 — Entrenar las dos GNN y compararlas
```bash
python -m src.gnn.compare_gnns          # entrena graphsage y gat, 3 seeds c/u
# o individual:
python -m src.gnn.train_gnn --model graphsage --seed 42
```
Ambas arquitecturas comparten la columna vertebral (432→256→128→64 + MLP head);
solo cambia la agregación (MEAN vs atención). Entrenan con `BCEWithLogitsLoss` y
`pos_weight` = razón real de desbalance del train (~27.6 en IEEE-CIS), neighbor
sampling 15-10-5 y early stopping por AUC de validación.
La comparación es **walk-forward × 3**: además del AUC del mes de validación
completo, cada modelo se evalúa semana a semana dentro del mes 5 (el "futuro
que va llegando"), y la selección usa el AUC promedio de las semanas × seeds —
premia consistencia temporal, no solo el promedio. El criterio de desempate se
mantiene (delta < 0.005 = empate técnico y **gana GraphSAGE** por costo de
inferencia fijo e inductividad); todo queda en `models/selected_model.json`.

**Si el proceso se cae, se retoma solo.** Cada corrida tarda ~2 h, así que el
entrenamiento guarda su avance y sabe dónde quedó:

- `artifacts/pipeline_state.json` — archivo de estado, sección `runs`. Se
  **crea solo** en la primera corrida con las 6 combinaciones (2 modelos ×
  3 seeds) en `pending`, y se actualiza a `running` (con la última época y el
  mejor AUC) y a `done`. La sección `steps` del mismo archivo lleva el avance
  de los pasos del pipeline.
- `models/{modelo}_seed{seed}_resume.pt` — checkpoint de época: pesos,
  optimizador, mejor estado, contador de paciencia y las semillas de todos los
  RNG (incluido el del neighbor sampler). Se escribe de forma atómica al
  terminar cada época y se **borra** cuando la corrida cierra bien.

Tras un corte (Ctrl-C, batería, kernel panic) basta con **relanzar el mismo
comando**: salta las seeds terminadas y retoma la que quedó a medias desde la
época siguiente. Al restaurar los RNG, la corrida reanudada da resultados
idénticos a una sin interrupción. Para ignorar todo y reentrenar desde cero:
`--force`.

```bash
python -m src.gnn.compare_gnns              # entrena lo que falte y compara
python -m src.gnn.compare_gnns --skip-train # solo comparar lo ya entrenado
python -m src.gnn.compare_gnns --force      # reentrenar las 6 desde cero
cat artifacts/pipeline_state.json           # ver en qué va
```

En macOS conviene lanzarlo a prueba de suspensión:
`caffeinate -is nohup python -m src.gnn.compare_gnns > gat.log 2>&1 &`

### Paso 6 — Ciclo de Continual Learning (el corazón del capstone)
```bash
python -m src.continual_learning.cl_orchestrator
```
Simula el mes 6 semana a semana:
1. El modelo opera sobre la semana y se mide el recall "antes".
2. Los fraudes confirmados por analistas (simulados con las etiquetas reales) con
   **score bajo** entran a la cola de novedad — esa es la señal de patrón emergente.
3. El gatillo dispara por conteo (50 casos) o por tasa de escape (>30%).
4. Split 70/30: adaptación / verificación (la verificación NUNCA se entrena).
5. Fine-tuning: lotes 40% casos nuevos + 60% replay buffer, LR diferenciado por capa
   (capa 1 congelada, capa 2 = 1e-5, capa 3 = 1e-4, clasificador = 1e-3),
   BatchNorm congelado, 5-10 épocas.
6. Doble validación contra el modelo ANTERIOR: aprendió (recall verificación ≥ 70%)
   y no olvidó (caída en el set de control ≤ 0.02). Si falla, el **dial
   estabilidad-plasticidad** ajusta la receta y reintenta (máx. 3).
7. Solo si pasa ambas: despliegue (`models/production_model.pt`) y actualización
   del buffer (con casos de adaptación) y del control (con casos de verificación).
   **Regla de oro**: datos entrenados → buffer; datos nunca entrenados → control.
   Jamás se cruzan. En el buffer la evicción sale en este orden: primero
   **redundantes** (misma clase + mismo origen + score casi idéntico: aportan
   lo mismo al repaso, se conserva un representante), luego **fáciles** (score
   extremo); la frontera es intocable y el piso histórico se respeta — si eso
   impide expulsar lo suficiente, se recorta la entrada (el buffer es de
   tamaño FIJO).
8. Si se agotan los reintentos sin desplegar, el diagnóstico queda en la
   bitácora: si no aprendió pese a la plasticidad, el patrón probablemente usa
   relaciones que el grafo no modela (**requiere aristas nuevas**); si fallan
   ambos frentes, se **programa el reentrenamiento profundo** dejando
   `artifacts/pending_deep_retrain.json`.

Salidas: `reports/cl_cycles.json` (bitácora con diagnósticos), los recalls
antes/después, y `data/graph/graph_scored.pt` — el grafo con el atributo
`fraud_score` poblado por nodo (features + isFraud + fraud_score).

### Paso 6b — Reentrenamiento profundo (cuando el fine-tuning no basta)
```bash
python -m src.continual_learning.deep_retrain
```
Lee el pendiente que dejó el orquestador y reentrena la arquitectura
seleccionada **desde cero** sobre el train original completo + los casos de
adaptación del patrón conflictivo (pos_weight recalculado sobre esa unión).
Valida con la misma vara del ciclo normal (verificación ≥70% y control sin
caída vs el modelo vigente); solo si pasa, despliega y actualiza los conjuntos
con la regla de oro. Es lento (horas) — por eso es la última carta del dial.

### Paso 7 — Comparación final (OE4)
```bash
python -m src.comparison.final_comparison
```
Mismo mes 6, mismo threshold 0.5: GNN+CL vs XGBoost congelado, global y sobre
**patrones emergentes** (fraudes que el GNN original pre-CL no detectaba).
KPI: gap de recall ≥ 20 puntos sobre emergentes + estimación de impacto en USD.
Salida: `reports/final_comparison.json`.

---

## 3b. Visualización (herramientas de apoyo, no son parte del pipeline)

### Explorador interactivo del grafo
```bash
python scripts/graph_explorer.py                 # semilla: el fraude más conectado del mes 6
python scripts/graph_explorer.py --tid 3003456   # una transacción concreta
python scripts/graph_explorer.py --offline       # embebe la librería: funciona sin internet
```
Genera `reports/grafo_interactivo.html` y lo abre en el navegador. Parte de un solo
nodo y se navega expandiendo: **doble clic** expande vecinos, **clic** muestra la ficha
(tid, mes, split, etiqueta, `fraud_score`, grado, vecinos ocultos), y el **borde punteado**
marca los nodos que aún tienen vecinos sin destapar. El buscador por `TransactionID`
permite saltar a otra componente conexa.

Requiere el paso 2. Con `--graph data/graph/graph_scored.pt` (paso 6) las fichas
muestran además el score del modelo.

### t-SNE del espacio latente
```bash
python scripts/plot_embeddings.py                # un panel, modelo seleccionado
python scripts/plot_embeddings.py --auto         # ANTES vs DESPUÉS del CL
```
Proyecta en 2D los embeddings de 64 dims (salida de `conv3`, capturados con un
forward hook sobre `classifier[0]` — no modifica los modelos). El modo `--auto`
dibuja dos paneles con los mismos nodos del mes 6: pre-CL y post-CL, marcando con
**X amarilla** los fraudes emergentes (los que el modelo pre-CL dejaba pasar con
score < 0.5). Es la evidencia visual del OE3: se ve cómo pasan de estar diluidos
entre las legítimas a agruparse.

Requiere los pasos 4-5; el modo `--auto` requiere además el paso 6.

---

## 4. Flujo completo del sistema (cómo conversan los módulos)

```
                         ┌─────────────────────────────────────────┐
                         │   IEEE-CIS (o demo sintética)           │
                         └───────────────┬─────────────────────────┘
                                         ▼
        preprocessing.py ──► build_graph.py ──► graph.pt (PyG)
              │                                        │
              ▼                                        ▼
   train_xgboost.py (SMOTE+Optuna)          compare_gnns.py (SAGE vs GAT, 3 seeds)
   modelo CONGELADO ──────────┐                        │ selecciona
                              │                        ▼
                              │             cl_orchestrator.py (mes 6, semana a semana)
                              │             gatillo → split 70/30 → fine-tuning →
                              │             doble validación → despliegue/buffer/control
                              │                        │
                              ▼                        ▼
                        final_comparison.py  ◄─  production_model.pt
                        (recall global + emergentes + USD)
```

Todo el flujo es local: `graph.pt` en disco, PyTorch Geometric en memoria y
neighbor sampling 15-10-5 por batch. La red nunca ve el grafo completo de una vez,
así que el mismo mecanismo escalaría a un grafo servido por una base de datos —
pero eso queda fuera del alcance de este repositorio.

---

## 5. Configuración

Todo está en `config/config.yaml`, comentado. Lo que más se toca:

| Sección | Qué controla |
|---|---|
| `data.split` | meses de train/val/test y semanas del walk-forward |
| `graph` | entidades de arista, ventana 30d, tope 50 aristas |
| `gnn` | dims, fanouts 15-10-5, épocas, seeds, KPI AUC 0.93 |
| `xgboost` | SMOTE ratio, trials de Optuna |
| `continual_learning` | gatillo (50 / 30%), split 70/30, buffer 10K, control 5K, mezcla 40/60, LRs por capa, umbrales de validación (0.70 / −0.02), dial y reintentos |

Para una corrida rápida de humo: bajar `gnn.epochs`, `gnn.seeds` a `[42]`,
`xgboost.optuna_trials` a 5 y `continual_learning.trigger.min_cases` a ~15.

**Dispositivo de cómputo:** se elige solo (CUDA si hay NVIDIA, si no CPU) y se
anuncia en la primera línea del log. En Apple Silicon, MPS está disponible pero
**no se activa por defecto** — el cuello de botella es el neighbor sampling, que
corre en CPU igual. Para probarlo: `FRAUDGNN_DEVICE=mps python -m src.gnn.compare_gnns`.
Una corrida completa debe usar siempre el mismo dispositivo.

---

## 6. Resultados del smoke test (demo sintética, 20K txn)

Sirven para verificar que el mecanismo funciona — **no son los números del dataset
real**. Los artefactos completos están en `reports/smoke_synthetic/`.

- **Baseline XGBoost** (mes 5): AUC 0.9687, recall 0.757, precisión 0.321, PR-AUC 0.558.
- **Comparador** (mes 5, walk-forward × 3 seeds): GraphSAGE 0.9629 ± 0.0033 vs
  GAT 0.9491 ± 0.0048 → GraphSAGE por mayor AUC (Δ=+0.0138, no hubo empate técnico).
  KPI AUC ≥ 0.93 cumplido.
- **Ciclo CL** (mes 6): la semana 1 disparó por tasa de escape (55% > 30%) con recall
  0.5135. El **primer intento falló** ("no aprendió"), el dial de plasticidad ajustó la
  receta (mezcla 50/50, LR ×2, capa 2 descongelada, 12 épocas) y el segundo intento pasó
  la doble validación → despliegue. Recall de la semana: 0.5135 → 1.0000.
  Semanas 2-4 sin disparos (0.979 / 1.000 / 0.960).
- **Comparación final** (mes 6, threshold 0.5):

  | | XGBoost congelado | GNN + CL |
  |---|---|---|
  | Recall | 0.316 | 0.984 |
  | Precisión | 0.285 | 0.394 |
  | AUC-ROC | 0.931 | 0.978 |
  | PR-AUC | 0.429 | 0.809 |
  | Recall sobre **emergentes** | 0.010 | 0.981 |

  Gap sobre emergentes **+0.971** (KPI ≥ 0.20 cumplido). 101 fraudes adicionales.

Dos salvedades honestas sobre estos números: el split 70/30 dejó **solo 3 casos** en
verificación (con 11 fraudes escapados), así que el KPI de "aprendió ≥ 0.70" no tiene
poder estadístico — valida el mecanismo, no la magnitud. Y el patrón B de la demo fue
diseñado para ser aprendible por fine-tuning, lo que introduce cierta circularidad.
Ambas cosas desaparecen con el IEEE-CIS real.

---

## 7. Decisiones de diseño que conviene recordar (para la defensa)

- **SMOTE solo en el baseline tabular**: un ejemplo sintético no tiene aristas; en la
  GNN el desbalance se maneja con `pos_weight` en la loss (27.6 original, ~1.2 en
  fine-tuning porque la mezcla 40/60 ya viene balanceada; en inferencia no existe).
- **El gatillo mira score BAJO en fraude confirmado**: si el modelo ya le daba score
  alto no hay nada nuevo que aprender; el patrón emergente es el que se le escapa.
- **Buffer ≠ control**: el buffer es memoria de entrenamiento (casos ya entrenados,
  priorizando frontera 0.4-0.7); el control es un examen sorpresa congelado (casos
  jamás entrenados). Si se cruzan, la validación de olvido queda contaminada.
- **BatchNorm congelado en fine-tuning**: con lotes chicos y sesgados a fraude, dejar
  que los BN actualicen sus estadísticas colapsa el modelo (lo vimos empíricamente
  en el smoke test: recall de control 0.88 → 0.00 sin congelar, 0.88 → 0.88 congelando).
- **Validación contra el modelo anterior**: el nuevo debe ser mejor donde el viejo
  fallaba (verificación) sin empeorar donde funcionaba (control). Las dos cosas, no una.
- **Redundancia en el buffer**: dos casos de la misma clase, mismo origen y score
  casi idéntico son informacionalmente equivalentes para el repaso — se conserva
  uno y el resto sale primero en la evicción. Criterio simple y explicable, sin
  costo de cómputo extra.
- **Selección walk-forward**: elegir la arquitectura por su AUC promedio semana a
  semana (y no solo por el mes completo) evita premiar un modelo que rinde bien
  "en promedio" pero se degrada hacia el final del mes de validación.
