# FraudGNN — ¿aporta el grafo algo que las columnas no tengan ya?

Capstone MIA UAI, Grupo 18 — Freddy Torrealba.

Detección de fraude en transacciones *card-not-present* sobre IEEE-CIS con una
**GNN heterogénea** (GraphSAGE vs GATv2) cuyo embedding alimenta una cabeza
XGBoost, medida contra la **misma cabeza sin esas columnas**.

Proyecto experimental: todo corre sobre el grafo serializado en disco. No incluye
capa de servicio (API, base de grafos ni despliegue).

---

## La pregunta

El fraude no viene solo: llega en redes que comparten tarjeta, correo o
dispositivo. La hipótesis es que un grafo captura eso y las columnas tabulares
no.

**El problema:** las columnas que definen las aristas (`card1/2/3/5`, `addr1`,
`P_emaildomain`, `DeviceInfo`, `id_*`) **ya están entre las features**. Y las
familias C, D y V del dataset son agregados relacionales que Vesta precalculó
sobre historiales de entidad. Así que el grafo podría estar codificando
información que el modelo tabular ya tiene.

El experimento quita V, C y D **a todos por igual** —GNN incluida— y compara tres
cabezas idénticas salvo en sus columnas:

```
control            65 col tabulares          ¿cuánto se logra sin grafo?
solo_gnn           el embedding y nada más   ¿el grafo solo basta?
gnn_mas_tabular    65 + el embedding         ¿el grafo SUMA?
```

**La pregunta la responde `control` vs `gnn_mas_tabular`.** Misma ventana, mismo
SMOTE, mismos hiperparámetros, mismo umbral.

## El resultado

```
EXAMEN — 21.284 transacciones · 876 fraudes (4,12%)

                   PR-AUC   ROC-AUC   recall@2%
control            0.3396   0.8313      0.2489
gnn_mas_tabular    0.3339   0.8347      0.2443
solo_gnn           0.1027   0.6770      0.0856

APORTE DEL GRAFO   -0.0058   IC95 [-0.0209, +0.0089]   NO significativo
```

**El intervalo cruza el cero.** Con estos datos no se puede afirmar que el grafo
aporte ni que perjudique. Es un resultado nulo, medido bajo condiciones que
descartan las explicaciones alternativas —causalidad del muestreo verificada,
embedding de vecinos verificado, hiperparámetros compartidos, ablación
simétrica— y con su limitación temporal declarada (ver *Limitaciones*).

Lo que sí salió limpio: **GraphSAGE gana a GATv2** sin solape entre las tres
semillas, y las dos arquitecturas eligen la profundidad mínima (2 capas).

---

## El diseño temporal

Piloto sobre **2 meses partidos en 8 semanas**. Cada bloque tiene un trabajo y
ninguno se solapa — lo comprueba `utils/ventanas.py:verificar()`, que **aborta**
si detecta solape:

```
   MES 1                          MES 2
   S1    S2    S3    S4     S1    S2    S3    S4
   └─────────┘  └──┘  └──────────────┘  └──┘  └──┘
   gnn_entrena   gnn   cabezas_entrenan  cab.  EXAMEN
   56.059      valida  85.071          validan 21.284
               37.386                  21.006
```

```
gnn_entrena        la red aprende aquí, y solo aquí
gnn_valida         se elige arquitectura, hiperparámetros y épocas
cabezas_entrenan   las 3 cabezas XGBoost aprenden aquí
cabezas_validan    Optuna de las cabezas mide aquí; sale el umbral del 2%
examen             no se toca hasta el informe final
```

El **continual learning está desactivado** en esta fase: compara enfoques, y para
eso `examen` tiene que ser test puro. El código se conserva.

## El grafo es heterogéneo

```
       txn_A ───┬─ [uid]    card1 + addr1 + (día − D1)   el cliente real
                ├─ [card]   card1+card2+card3+card5
                ├─ [email]  P_emaildomain + card1
                ├─ [device] DeviceInfo + id_30/31/33
                └─ [net]    id_13/17/19/20
```

Las transacciones **no se conectan entre sí**: son vecinas si cuelgan del mismo
nodo de entidad. Los nodos de entidad **no tienen features propias** —entran en
ceros y su vector sale de agregar sus transacciones—, y eso es lo que hace el
modelo **inductivo**: una tarjeta vista por primera vez en el examen funciona
igual.

El muestreo baja **las 10 transacciones más recientes anteriores** de cada
entidad (`time_attr` + `temporal_strategy="last"`). Verificado sobre 4.517
vecinas: cero posteriores a su raíz, cero fuera de las 10 más recientes.

## Las 7 etapas

| # | Etapa | Qué hace | Produce |
|---|---|---|---|
| 0 | `download` | Baja IEEE-CIS de Kaggle | `data/raw/*.csv` |
| 1 | `preprocess` | Une, codifica, imputa; encoders sin fuga | `full.parquet` |
| 2 | `graph` | Grafo heterogéneo, 5 entidades, 10 tipos de arista | `graph.pt` |
| 3 | `gnn` | Optuna + 2 arquitecturas × 3 semillas + selección | `selected_model.json` |
| 5a | `embed` | UNA red describe todo lo que no entrenó | `gnn_embed.parquet` |
| 5b | `heads` | Las tres cabezas XGBoost | `heads_variantes.json` |
| 7 | `final` | Veredicto sobre `examen` + bootstrap emparejado | `final_comparison.json` |

La numeración salta porque quedan huecos del diseño anterior de 12 etapas.

### Por qué UNA red y no K

La GNN **memorizó** las transacciones con las que entrenó: su embedding sobre
esas filas es optimista. El diseño anterior lo resolvía con validación cruzada
(K redes, cada una describiendo el trozo que no vio), y falló por un motivo
distinto al esperado: **K redes aprenden K sistemas de coordenadas latentes**.
La dimensión 7 de una no significa lo mismo que la de otra, y mezclarlas hundía
las cabezas — la variante mixta cortó en **2 árboles** contra 517 del control.

Con las ventanas separadas basta una red: `gnn_entrena` queda excluido y todo lo
demás recibe un embedding honesto de la misma red.

### El umbral no es 0.5

La GNN entrena con `pos_weight` y sus scores están inflados; la cabeza devuelve
probabilidades calibradas. Con un umbral fijo el híbrido no alertaría casi nada.
El umbral es el **cuantil que produce `hybrid.alert_budget_pct` de alertas**.

Medido: el mismo modelo pasó de **F1 0.4356 a 0.5785** solo por corregir el punto
de operación, sin tocar un peso.

---

## Los invariantes

Siete comprobaciones que corren en segundos. Cada una guarda un fallo que **ya
ocurrió** y que no producía ningún síntoma: ni excepción, ni warning, ni número
raro. Solo un resultado plausible y falso.

```bash
bash tests/run.sh
```

```
E0   el embedding "solo vecinos" contiene vecinos
E1   la primera transacción de una entidad no recibe de ella
E2   el grafo tiene las aristas que dicen los datos
     ninguna entidad se cayó en silencio        (+ informe y diff)
     SMOTE solo sintetiza fraude y respeta el ratio
A2   el muestreo solo mira hacia atrás           (necesita pyg-lib)
     baja las N más recientes anteriores         (parcial sin pyg-lib)
```

El más caro de los que cazó: **E0 estuvo activo toda la primera fase**. Las
columnas que recibía la cabeza híbrida eran `constante + 4 proyecciones de las
features propias` y cero información del vecindario. No daba un error: daba una
respuesta equivocada a la pregunta de la tesis.

Y para probar el pipeline entero sin GPU ni Kaggle:

```bash
bash scripts/smoke_test.sh        # las 7 etapas con datos sintéticos, en sandbox
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
│   │   ├── preprocessing.py     codificación e imputación sin fuga temporal
│   │   └── build_graph.py       grafo heterogéneo + podas causales
│   ├── gnn/
│   │   ├── models.py            GraphSAGE y GATv2 sobre HeteroConv
│   │   ├── sampling.py          NeighborLoader temporal (exige pyg-lib)
│   │   ├── train_gnn.py         entrenamiento reanudable POR ÉPOCA
│   │   └── compare_gnns.py      Optuna, las 6 corridas y la selección
│   ├── hybrid/
│   │   ├── embed.py             UNA red describe lo que no entrenó
│   │   ├── head.py              ensamblado de columnas + umbral + IO
│   │   └── train_head.py        las tres cabezas
│   ├── comparison/
│   │   ├── final_comparison.py  el veredicto + bootstrap
│   │   └── resumen.py           las 6 métricas en los 3 puntos
│   ├── baseline_xgboost/        librería: SMOTE y el espacio de Optuna
│   ├── continual_learning/      DESACTIVADO, se conserva
│   └── utils/
│       └── ventanas.py          el reparto temporal, con verificación de solape
├── tests/
│   ├── run.sh                   los siete invariantes
│   └── test_*.py
├── scripts/
│   ├── run_pipeline.sh          punto de entrada
│   ├── smoke_test.sh            el pipeline con datos sintéticos
│   ├── setup_runpod.sh          instala el entorno en un pod con GPU
│   ├── make_synthetic_demo.py   genera esos datos
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

Salen del `kaggle.json` de [kaggle.com/settings](https://www.kaggle.com/settings)
→ API → *Create New Token*. Y hay que **aceptar las reglas** de la competencia
una vez, o la API responde 403 aunque la credencial sea válida.

### 2. Clonar

```bash
cd /workspace                          # lo único que sobrevive a un Stop
git clone <repo> fraudgnn && cd fraudgnn
```

### 3. Averiguar los núcleos REALES

`nproc` **miente** en un contenedor: reporta los del host. En el pod de
referencia dice 128 y la cuota real es 13,6.

```bash
N=$(if [ -f /sys/fs/cgroup/cpu.max ]; then awk '{print ($1=="max")?"0":int($1/$2)}' /sys/fs/cgroup/cpu.max; \
    elif [ -f /sys/fs/cgroup/cpu/cpu.cfs_quota_us ]; then \
      q=$(cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us); p=$(cat /sys/fs/cgroup/cpu/cpu.cfs_period_us); \
      [ "$q" -le 0 ] && echo 0 || echo $((q/p)); else echo 0; fi)
[ "$N" -eq 0 ] && N=$(nproc)
echo "núcleos reales: $N"
```

Con más hilos que núcleos el kernel **congela el contenedor** periódicamente
(*cgroup throttling*): no va un poco más lento, se apaga a ratos. Se comprueba
con `grep nr_throttled /sys/fs/cgroup/cpu/cpu.stat` dos veces con un minuto de
diferencia — si no sube, vas bien.

### 4. Instalar y verificar

```bash
bash scripts/setup_runpod.sh
bash tests/run.sh                # aquí sí corren los siete completos
```

`setup_runpod.sh` debe terminar con `CUDA disponible: True` y
`sampler nativo: SÍ`. **Sin `pyg-lib` el grafo heterogéneo no se puede muestrear**
y el pipeline aborta con mensaje claro.

### 5. Ejecutar

```bash
tmux new -s fraudgnn             # que no muera al cerrar el SSH
bash scripts/run_pipeline.sh
```

`Ctrl-B` luego `D` para salir dejándolo corriendo; `tmux attach -t fraudgnn`
para volver.

```bash
tail -f pipeline.log
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv -l 30
```

**Después de un Ctrl-C, comprueba que no queden huérfanos:**

```bash
nvidia-smi --query-compute-apps=pid,used_memory --format=csv
```

No debe quedar ninguno. (Los hijos llevan `PR_SET_PDEATHSIG`, pero conviene
mirarlo: hubo dos de 14 horas ocupando 1,5 GB de VRAM.)

### 6. Bajar resultados

```bash
# en el pod
bash scripts/run_pipeline.sh --archive "hetero-v2"

# en tu máquina
scp -i ~/.ssh/id_ed25519 -P <puerto> -r \
    root@<ip>:/workspace/fraudgnn/historial/* historial/
```

⚠️ Las opciones de `scp` van **antes** de las rutas. Y verifica que abren
**antes** de terminar el pod: *Terminate* borra `/workspace` sin vuelta atrás.

---

## Ejecutar en local

```bash
pip install -r requirements.txt        # macOS: brew install libomp (XGBoost)
cp .env.example .env && nano .env      # KAGGLE_USERNAME / KAGGLE_KEY
```

**En macOS no se puede entrenar la GNN**: `pyg-lib` no publica wheel para arm64
y sin él PyG cae a un sampler que solo sabe grafos homogéneos. Sí funciona todo
lo demás — preprocesado, construcción del grafo, las cabezas, los tests (cinco
de siete) y el análisis.

```bash
python -m src.data.build_graph         # el grafo, ~1 min
bash tests/run.sh                      # cinco completos, dos parciales
python tests/test_salud_grafo.py --informe
```

---

## El pipeline es reanudable

Antes de cada etapa mira si sus archivos de salida existen y la salta si están.
Si el proceso muere, relanzas **el mismo comando** y sigue donde quedó — incluso
a mitad de una GNN, que retoma desde su última época guardada.

**El disco manda**: borra una salida y esa etapa vuelve a estar pendiente.

```bash
bash scripts/run_pipeline.sh --steps          # qué hace cada etapa
bash scripts/run_pipeline.sh --status         # en qué va ahora
bash scripts/run_pipeline.sh --only heads     # SOLO esas          (coma)
bash scripts/run_pipeline.sh --from gnn       # desde ahí
bash scripts/run_pipeline.sh --only gnn --force
```

`--force` no elige etapas: **fuerza las que dejen `--only`/`--skip`/`--from`**.

> **Ojo con `--force` sobre `gnn`:** también borra el caché de Optuna
> (`reports/optuna_*.json` y `.db`), así que **relanza la búsqueda de
> hiperparámetros**. Al comparar cambios del grafo normalmente quieres
> conservarlos: borra a mano lo que toca y lanza **sin** `--force`. En el log
> debe salir `hiperparámetros ya buscados — se reutilizan`.

---

## Configuración

Todo vive en `config/config.yaml`. Lo que más se toca:

```yaml
compute:
  n_jobs: 14                    # la cuota REAL del cgroup, no `nproc`

ventanas:                       # el reparto temporal, declarativo
  gnn_entrena:      {mes: 1, semanas: [1, 2]}
  gnn_valida:       {mes: 1, semanas: [3]}
  cabezas_entrenan: [{mes: 1, semanas: [4]}, {mes: 2, semanas: [1, 2]}]
  cabezas_validan:  {mes: 2, semanas: [3]}
  examen:           {mes: 2, semanas: [4]}

graph:
  max_entity_degree: 0          # 0 = SIN poda; el muestreo ya limita a 10
  min_previas_entidad: 1        # sin anteriores no recibe: sería su propio eco
  vecinos_por_entidad: 10

gnn:
  optuna_trials: 100
  optuna_presupuesto_min: 0     # si > 0, MANDA sobre optuna_trials
  optuna_tope_trial_min: 10     # tope por trial, con indulto al que va ganando
  paralelo_optuna: 2
  paralelo_corridas: 2          # 6 no caben en 20 GB si gana `capas 3`
  seeds: [42, 123, 2026]

xgboost:
  excluir_prefijos: ["V", "C", "D"]   # la ablación, a TODOS por igual
  optuna_modo: "compartido"           # una búsqueda, las tres heredan
  por_cabeza:                          # DÉJALO VACÍO
    control: {}
    solo_gnn: {}
    gnn_mas_tabular: {}

hybrid:
  alert_budget_pct: 2.0         # umbral por volumen de alertas, no fijo
```

---

## Limitaciones declaradas

**El embedding caduca.** La GNN entrena en los días 1-15 y el examen está en los
53-60. En ese trayecto su poder discriminante cae un 38%:

```
gnn_valida        días 15-22    ROC 0.7387
cabezas_entrenan  días 23-45    ROC 0.6984
cabezas_validan   días 45-52    ROC 0.6535
examen            días 53-60    ROC 0.6467
```

La cabeza **aprende** donde el embedding vale 0.698 y **se examina** donde vale
0.647: le asigna un peso calibrado para una señal más fuerte de la que va a
encontrar.

Con 60 días no hay hueco para un refit sin romper el aislamiento —cualquier
ventana que la GNN reentrene se la quitas a la cabeza—, así que **el resultado es
una cota inferior** y el walk-forward refit va en la corrida de 6 meses.

**`uid` conecta poco.** 2,1 transacciones por entidad de media: la clave es tan
específica que el 62% de sus nodos tiene una sola transacción. Candidata a
revisión.

**`device` y `net` cubren el 20-27%** de las transacciones. Es una propiedad del
dataset, no un fallo.

---

## Decisiones de diseño (para la defensa)

- **Ablación simétrica.** V, C y D se quitan a las tres cabezas **y a la GNN**.
  Si la red las viera, las devolvería dentro del embedding por la puerta de atrás
  y la variante con grafo ganaría por copiarlas.
- **Optuna una sola vez, y las tres heredan.** Con búsqueda por cabeza,
  `gnn_mas_tabular` ganó en validación por 0.0004 y perdió en el examen: era la
  búsqueda sobreajustando la ventana donde se busca. Cualquier ventaja bajo
  hiperparámetros compartidos es una cota inferior conservadora.
- **Nada de hiperparámetros a mano para una cabeza y no para otra.** Tenerlo
  puesto en dos de tres dio un aporte de −0.0668 «significativo» que era
  íntegramente artefacto.
- **Comparar a igual presupuesto de alertas.** A umbral fijo, dos modelos con
  calibraciones distintas no son comparables.
- **El accuracy no compara.** Con 4,12% de fraude, no alertar nunca da 0.9588.
  El informe lo dice en cada tabla.
- **Bootstrap emparejado.** Sin intervalo, un delta de 0.005 parece un resultado.
- **La semilla de los trials de Optuna está FIJA.** Cada trial es hiperparámetros
  *y* inicialización; si las dos cambian, no se sabe cuál ganó. La robustez
  frente a la inicialización se mide después, con las 3 semillas.
- **SMOTE solo en las cabezas.** Un ejemplo sintético no tiene aristas; en la GNN
  el desbalance se maneja con `pos_weight` en la loss.
- **El muestreo mira solo hacia atrás, y está verificado.** No es una promesa de
  la configuración: hay un test que compara nodo a nodo lo que bajó el sampler
  contra lo que debería haber bajado.
