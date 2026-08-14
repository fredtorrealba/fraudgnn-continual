# FraudGNN — Detección Adaptativa de Fraude con Grafos

Capstone MIA UAI, Grupo 18 — Freddy Torrealba.

Detección de fraude en transacciones *card-not-present* con **Graph Neural Networks**
(GraphSAGE vs GAT) y **Continual Learning**, medido contra un **XGBoost congelado**
sobre el dataset IEEE-CIS.

Proyecto experimental: todo corre sobre el grafo serializado en disco. No incluye
capa de servicio (API, base de grafos ni despliegue).

---

## La idea en tres frases

1. **El fraude no viene solo, viene en redes.** Cada transacción es un nodo; se
   conecta con las que comparten tarjeta, email o dispositivo dentro de 30 días.
2. **El fraude cambia.** El mes 6 se recorre semana a semana: cuando aparecen
   fraudes que el modelo no detecta, se dispara un ciclo de adaptación que
   reentrena sin olvidar lo anterior.
3. **Hay que demostrarlo.** Un XGBoost entrenado con las mismas columnas y nunca
   reentrenado sirve de vara de medir.

El split es **temporal estricto**: meses 1-4 entrenan, el 5 decide, el 6 es el
examen. Nunca al revés.

---

## El sistema híbrido

Medimos que la GNN sola **no supera** al baseline tabular, pero que el grafo
**sí aporta**: quitarle las aristas le cuesta el 55% del PR-AUC, y detecta 132
fraudes que XGBoost no ve. La conclusión: el grafo tiene señal, pero una GNN
end-to-end es peor extrayendo datos tabulares que un ensemble de árboles.

El sistema entregado le da a XGBoost la señal del grafo ya digerida:

```
transacción
   ├─► 431 columnas originales
   ├─► 8 columnas estructurales del grafo   (contadas, sin etiquetas)
   └─► gnn_score                            (la opinión de la GNN)
                    ↓
            cabeza XGBoost  →  P(fraude)
```

Sin la GNN falta una de las 440 columnas. Y tres variantes separan de dónde
viene cada mejora: **431** (referencia) / **439** (+ estructura) / **440**
(+ la red).

## Las 12 etapas

| # | Etapa | Qué hace | Produce |
|---|---|---|---|
| 1 | `download` | Baja IEEE-CIS de Kaggle (590K txn etiquetadas) | `data/raw/*.csv` |
| 2 | `preprocess` | Une, codifica, imputa y parte los 6 meses | `data/processed/full.parquet` |
| 3 | `graph` | Nodos = txn, aristas = entidad compartida, **+ 8 columnas estructurales** | `graph.pt`, `graph_features.parquet` |
| 4 | `gnn` | 2 arquitecturas × 3 semillas = 6 corridas, y elige | `models/selected_model.json` |
| 5 | `oof` | **`gnn_score` honesto** por validación cruzada (meses 1-4) | `gnn_oof_train.parquet` |
| 6 | `hybrid` | Cabeza XGBoost, 3 variantes | `hybrid_head_{431,439,440}.json` |
| 7 | `refit` | Reentrena al ganador añadiendo el mes 5 | `models/refit_model.pt` |
| 8 | `oof_refit` | `gnn_score` honesto sobre meses 1-5 | `gnn_oof_trainval.parquet` |
| 9 | `hybrid_refit` | Cabeza de producción + **umbral operativo** | `hybrid_head_prod.json` |
| 10 | `cl` | Mes 6 semana a semana; **adaptan las dos piezas** | `reports/cl_cycles.json` |
| 11 | `xgboost` | Baseline tabular **congelado** (viene versionado en git) | `models/xgboost_baseline.json` |
| 12 | `final` | baseline vs GNN sola vs híbrido, a igual presupuesto | `reports/final_comparison.json` |

### Por qué existen las etapas `oof`

La GNN **memorizó** las transacciones con las que entrenó. Si su score sobre
esas mismas filas se usara como columna de XGBoost, la cabeza vería una columna
casi perfecta y aprendería a copiarla — y en el mes 6, donde la red no ha
memorizado nada, se rompería.

`oof` parte los meses de entrenamiento en 4 trozos y entrena 4 redes, cada una
dejando un trozo fuera. Cada transacción recibe el score de una red **que nunca
la vio**. Las 4 se descartan: solo sobrevive la columna.

Los folds son **aleatorios estratificados por (mes, clase)**, no temporales: con
folds temporales `gnn_score` quedaría correlacionado con el calendario y
XGBoost lo usaría como proxy de la fecha.

### El umbral no es 0.5

La GNN entrena con `pos_weight ≈ 27` y sus scores están inflados; la cabeza
devuelve probabilidades calibradas (~3%). Con un umbral fijo el híbrido no
alertaría casi nada. El umbral es el **cuantil que produce
`hybrid.alert_budget_pct` de alertas**, medido sobre el mes 5.

Las **3 semillas** existen porque entrenar tiene azar: con una sola corrida no
sabrías si una arquitectura ganó por buena o por suerte.

```
download → preprocess → graph ─┬─► gnn → oof → hybrid → refit → oof_refit ─┐
                               │                                            │
                               │                          hybrid_refit → cl ┤
                               │                                            ├─► final
                               └────────────► xgboost (congelado) ──────────┘
```

---

## Estructura

```
fraudgnn/
├── config/config.yaml           TODOS los hiperparámetros, en un solo archivo
├── src/
│   ├── pipeline.py              runner reanudable: sabe qué falta y lo salta
│   ├── data/
│   │   ├── download_ieee_cis.py credenciales: entorno > .env > ~/.kaggle
│   │   ├── preprocessing.py     codificación e imputación ajustadas SOLO con train
│   │   └── build_graph.py       aristas por entidad compartida, ventana 30 días
│   ├── gnn/
│   │   ├── models.py            GraphSAGE y GAT; profundidad = len(hidden_dims)
│   │   ├── sampling.py          neighbor sampling (con fallback sin pyg-lib)
│   │   ├── train_gnn.py         entrenamiento reanudable POR ÉPOCA
│   │   ├── compare_gnns.py      las 6 corridas y la selección walk-forward
│   │   └── refit.py             reentrena al ganador con meses 1-5
│   ├── continual_learning/
│   │   ├── trigger.py           gatillo: fraude confirmado con score BAJO
│   │   ├── splitter.py          70% adaptación / 30% verificación
│   │   ├── replay_buffer.py     memoria de repaso (~10K, prioriza frontera)
│   │   ├── control_set.py       examen sorpresa congelado (~5K, disjunto)
│   │   ├── finetune.py          mezcla 40/60 con LR diferenciado por capa
│   │   ├── validate.py          ¿aprendió? ¿olvidó? + dial estabilidad-plasticidad
│   │   ├── cl_orchestrator.py   orquesta el mes 6
│   │   └── deep_retrain.py      última carta del dial
│   ├── baseline_xgboost/
│   │   ├── smote_pipeline.py    SMOTE, solo sobre train
│   │   └── train_xgboost.py     Optuna + modelo CONGELADO
│   ├── hybrid/                  el sistema híbrido GNN + XGBoost
│   │   ├── features.py          8 columnas estructurales (sin etiquetas)
│   │   ├── oof.py               gnn_score honesto por validación cruzada
│   │   ├── head.py              ensamblado de las 440 columnas + IO
│   │   ├── train_head.py        entrena la cabeza (3 variantes)
│   │   ├── head_cl.py           warm start de la cabeza en cada ciclo
│   │   └── system.py            GNN + cabeza en operación
│   ├── comparison/final_comparison.py
│   └── utils/                   config, logging, semillas, dispositivo, métricas
├── scripts/
│   ├── run_pipeline.sh          punto de entrada
│   ├── setup_runpod.sh          instala el entorno en un pod con GPU
│   ├── make_synthetic_demo.py   demo sin Kaggle, para probar el flujo
│   └── graph_explorer.py        explorador de vecindarios (HTML)
└── historial/                   corridas archivadas (ver --archive)
```

Se crean al correr: `data/`, `models/`, `reports/`, `artifacts/`.

---

## Ejecutar en RunPod

### 1. Variables de entorno del pod

Al desplegar, en **Environment variables** del template:

```
KAGGLE_USERNAME = tu_usuario
KAGGLE_KEY      = tu_api_key
```

Salen del `kaggle.json` de [kaggle.com/settings](https://www.kaggle.com/settings) → API →
*Create New Token*. Y hay que **aceptar las reglas** de la competencia una vez, o la
API responde 403 aunque la credencial sea válida.

El código las busca en este orden: **entorno del proceso → `.env` → `~/.kaggle/`**.
En el pod basta con las variables; no hace falta crear ningún archivo.

### 2. Clonar

```bash
cd /workspace                          # lo único que sobrevive a un Stop
git clone https://github.com/fredtorrealba/fraudgnn-continual.git fraudgnn
cd fraudgnn
```

### 3. Averiguar los núcleos REALES

En un contenedor, `nproc` reporta los del host (96), no tu cuota:

```bash
if [ -r /sys/fs/cgroup/cpu.max ] && [ "$(cut -d' ' -f1 /sys/fs/cgroup/cpu.max)" != "max" ]; then
    n=$(awk '{print int($1/$2)}' /sys/fs/cgroup/cpu.max)
elif [ -r /sys/fs/cgroup/cpu/cpu.cfs_quota_us ] && [ "$(cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us)" -gt 0 ]; then
    n=$(( $(cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us) / $(cat /sys/fs/cgroup/cpu/cpu.cfs_period_us) ))
else n=$(nproc); fi
echo "n_jobs: $n | num_workers: $(( n>3 ? n-3 : 1 ))"
```

Con más hilos que núcleos el kernel **congela el contenedor** periódicamente
(*cgroup throttling*): no va un poco más lento, se apaga a ratos.

### 4. Configurar

```bash
nano config/config.yaml
```

```yaml
compute:
  n_jobs: 15                     # el primer número del paso 3
gnn:
  num_workers: 12                # el segundo
  hidden_dims: [256]             # 1 capa (descomenta la que quieras)
```

### 5. Instalar

```bash
bash scripts/setup_runpod.sh
```

Debe terminar con `CUDA disponible: True` y `sampler nativo: SÍ`. Si dice
`NO (fallback, lento)`, para: sin el sampler nativo el muestreo va varias veces
más lento.

### 6. Ejecutar

```bash
tmux new -s fraudgnn             # que no muera al cerrar el SSH
bash scripts/run_pipeline.sh
```

Ctrl-B luego D para salir dejándolo corriendo; `tmux attach -t fraudgnn` para
volver. **Si se cae el SSH, el proceso sigue** — solo se cae tu terminal.

Desde otra terminal:

```bash
tail -f pipeline.log             # el log se escribe SOLO, sin tee
watch -n 2 nvidia-smi            # si GPU-Util < 50%, el cuello es el sampler
```

### 7. Bajar resultados y archivar

**La vía limpia**: archiva primero en el pod y copia una sola carpeta, ya
autocontenida y con el `meta.json` de métricas hecho.

```bash
# en el pod
bash scripts/run_pipeline.sh --archive "1capa-sin-aristas"

# en tu máquina
scp -i ~/.ssh/id_ed25519 -P <puerto> -r \
    root@<ip>:/workspace/fraudgnn/historial/* historial/
```

**Si prefieres copiar en crudo** (porque vas a lanzar otra configuración y no
quieres que `--archive` borre `data/`), son cuatro sitios, no tres:

```bash
scp -i ~/.ssh/id_ed25519 -P <puerto> -r \
    root@<ip>:/workspace/fraudgnn/{reports,models,artifacts,pipeline.log,config} .

scp -i ~/.ssh/id_ed25519 -P <puerto> \
    root@<ip>:/workspace/fraudgnn/data/processed/feature_cols.json data/processed/
```

| Qué | Por qué |
|---|---|
| `models/` `reports/` `artifacts/` | los resultados |
| **`config/config.yaml`** | **qué configuración los produjo** — se edita en el pod y es lo más fácil de perder |
| `pipeline.log` | tiempos por época, warnings, decisiones del CL |
| **`data/processed/feature_cols.json`** | **qué 431 columnas vio el modelo** — 5 KB, y sin él un resultado no se puede reinterpretar |
| `data/graph/graph_scored.pt` | opcional, 1.4 GB. Solo lo usa `graph_explorer.py` |

⚠️ Las opciones de `scp` van **antes** de las rutas. En macOS, un `-i` al final
no se interpreta como opción: pasaría a ser el destino.

Verifica que abren **antes** de terminar el pod: *Terminate* borra `/workspace`
sin vuelta atrás.

---

## Ejecutar en local

```bash
pip install -r requirements.txt        # macOS: brew install libomp (XGBoost)
cp .env.example .env && nano .env      # KAGGLE_USERNAME / KAGGLE_KEY
bash scripts/run_pipeline.sh          # el log queda en pipeline.log
```

Sin GPU funciona, pero el paso `gnn` se va a días: en CPU una época de GAT sobre
el dataset completo tarda ~70 min. Para probar el flujo sin esperar:

```bash
python scripts/make_synthetic_demo.py --n 20000
bash scripts/run_pipeline.sh
```

---

## El pipeline es reanudable

Antes de cada etapa mira si sus archivos de salida existen y la salta si están.
Si el proceso muere (Ctrl-C, batería, desalojo de VM spot), relanzas **el mismo
comando** y sigue donde quedó — incluso a mitad de una GNN, que retoma desde su
última época guardada.

**El disco manda**: borra una salida y esa etapa vuelve a estar pendiente.

```bash
bash scripts/run_pipeline.sh --steps          # qué hace cada etapa
bash scripts/run_pipeline.sh --status         # en qué va ahora
bash scripts/run_pipeline.sh --only gnn,cl    # SOLO esas          (coma)
bash scripts/run_pipeline.sh --skip xgboost   # todo MENOS esas    (coma)
bash scripts/run_pipeline.sh --from gnn       # desde ahí
bash scripts/run_pipeline.sh --only gnn --force   # rehacer lo seleccionado
```

`--force` no elige etapas: **fuerza las que dejen `--only`/`--skip`/`--from`**.
Borra sus salidas antes de relanzarlas.

---

## El log

Toda corrida escribe en **`pipeline.log`** en la raíz del proyecto, sin que haya
que acordarse del `| tee`. Se **anexa**, nunca se pisa: como el pipeline es
reanudable, una corrida puede continuar a otra y conviene conservar el hilo.
Cada arranque queda separado por una cabecera con la fecha y los argumentos.

Las consultas (`--status`, `--steps`, `--history`, `--archive`) no escriben nada:
no ejecutan un entrenamiento.

```bash
tail -f pipeline.log                        # en vivo, desde otra terminal
grep "Época" pipeline.log | tail -20        # solo el avance de las épocas
grep -E "WARNING|ERROR" pipeline.log        # solo lo que salió mal
grep "LISTA —" pipeline.log                 # el resumen de cada corrida GNN
grep -c "^════" pipeline.log                # cuántas veces se ha relanzado
```

Cambiar la ruta: `FRAUDGNN_LOG=/otro/sitio.log bash scripts/run_pipeline.sh`

El log sirve para mirar el avance; **la evidencia real está en `reports/`**, que
no depende de él.

---

## Archivar y comparar corridas

```bash
bash scripts/run_pipeline.sh --archive "1capa"   # guarda y deja todo en cero
bash scripts/run_pipeline.sh --history           # lista lo archivado
```

Mueve a `historial/<fecha>_<nombre>/`:

| Se mueve | |
|---|---|
| `models/` `reports/` `artifacts/` | los resultados de la corrida |
| `pipeline.log` | así la corrida siguiente arranca con el log limpio |
| `data/graph/graph_scored.pt` | lo produce `cl` aunque viva en `data/` |

| Se copia (el original se queda) | |
|---|---|
| `config/config.yaml` | sin él, un resultado antiguo no se puede interpretar |
| `data/processed/feature_cols.json` | las 431 columnas que vio el modelo |
| `models/xgboost_baseline.json` + sus métricas | baseline congelado: no se reentrena nunca |

Y genera un `meta.json` con fecha, commit de git, la huella del grafo
(`nodos`/`aristas`/`features`) y las métricas ya extraídas.

Después borra `data/`: son 2 GB deterministas que se regeneran en ~7 min con el
config archivado.

Para comparar dos corridas, empieza por `meta.json`. Y comprueba que la **huella
del grafo** (`nodos`, `aristas`, `features`) coincida: si difiere, cambiaste el
grafo y estarías comparando cosas distintas.

---

## Configuración

Todo vive en `config/config.yaml`. Lo que más se toca:

```yaml
compute:
  n_jobs: -1              # núcleos de CPU. En contenedor, la cuota REAL
gnn:
  hidden_dims: [256]      # PROFUNDIDAD: cada capa es un SALTO en el grafo
  sin_aristas: false      # true = ablación: anula el grafo (mide qué aporta)
  fanouts: [15, 10, 5]    # vecinos por salto; se recortan a las capas
  batch_size: 1024
  num_workers: 4          # procesos de muestreo en paralelo
  log_every: 0            # 0 = una línea por época
  seeds: [42, 123, 2026]
xgboost:
  device: "auto"          # auto | cuda | cpu
hybrid:
  oof_folds: 4            # trozos para el gnn_score honesto
  variants: [431, 439, 440]
  optuna_on_variant: 431  # se afina sobre la MÁS PEQUEÑA a propósito
  alert_budget_pct: 2.0   # umbral por volumen de alertas, no fijo
```

**Sobre la profundidad**: la homofilia medida en este dataset dice que la señal
está a 1 salto (separación fraude/legítima 3.9x), se pierde a 2 (0.7x) y se
**invierte** a 3 (0.3x). Apilar capas promedia ruido — es *over-smoothing*.

---

## Decisiones de diseño (para la defensa)

- **SMOTE solo en el baseline.** Un ejemplo sintético no tiene aristas; en la GNN
  el desbalance se maneja con `pos_weight` en la loss.
- **El gatillo mira score BAJO en fraude confirmado.** Si el modelo ya le daba
  score alto no hay nada nuevo que aprender.
- **Buffer ≠ control.** El buffer es memoria de entrenamiento; el control es un
  examen sorpresa congelado. Si se cruzan, la validación de olvido se contamina.
- **BatchNorm congelado en fine-tuning.** Con lotes chicos y sesgados a fraude,
  dejar que actualice sus estadísticas colapsa el modelo (recall de control
  0.88 → 0.00 sin congelar; 0.88 → 0.88 congelando).
- **Doble validación.** El modelo nuevo debe mejorar donde el viejo fallaba
  *sin* empeorar donde funcionaba. Las dos cosas, no una.
- **Selección walk-forward.** Elegir por el AUC promedio semana a semana evita
  premiar a un modelo que va bien "en promedio" pero se degrada al final del mes.
- **Comparar a igual presupuesto de alertas.** A umbral fijo, dos modelos con
  calibraciones distintas no son comparables: el que tiene `pos_weight` alto
  alerta más y parece mejor en recall. `final_comparison.py` reporta también a
  igual número de alertas y a igual precisión.
- **`gnn_score` out-of-fold.** Un modelo no puede puntuar honestamente lo que
  entrenó. Sin validación cruzada, la cabeza aprendería a copiar una columna que
  en producción no acierta tanto — el fallo clásico del stacking.
- **Optuna una sola vez, sobre la variante más pequeña.** Si cada variante
  buscara sus hiperparámetros, una diferencia confundiría "más información" con
  "sorteo más afortunado". Afinando sobre 431 se le regala la ventaja a la
  referencia y el resultado del híbrido es una cota inferior conservadora.
- **Refit antes del test.** El mes 5 se gasta en decidir; una vez decidido, se
  reentrena al ganador desde cero con los meses 1-5. Las épocas se heredan del
  pico de la corrida original (sin validación no hay early stopping) — limitación
  asumida por el procedimiento.
