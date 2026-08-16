# CLAUDE.md

Guía para trabajar en este repositorio. El **README.md** explica qué es el
proyecto y cómo ejecutarlo; aquí está lo que hace falta para **modificarlo sin
romperlo**.

## Qué es

Capstone de MIA UAI: detección de fraude *card-not-present* sobre IEEE-CIS con
una GNN sobre grafo **heterogéneo**, comparada contra un XGBoost tabular bajo
condiciones idénticas. La GNN entra en el sistema como columnas de embedding
que consume una cabeza XGBoost.

**La pregunta del trabajo:** ¿aporta el grafo algo que las columnas tabulares no
tengan ya? La respuesta medida hasta hoy es **no de forma medible**
(ver `comparison.md`).

Todo el código, comentarios y logs están **en español**.

## Comandos

```bash
bash scripts/run_pipeline.sh --status      # en qué va       (no ejecuta)
bash scripts/run_pipeline.sh --steps       # qué hace cada etapa
bash scripts/run_pipeline.sh               # corre lo que falte
bash scripts/run_pipeline.sh --only heads --force
bash scripts/run_pipeline.sh --from gnn
bash scripts/run_pipeline.sh --archive "hetero-v2"   # archiva y deja en cero
bash scripts/run_pipeline.sh --history

bash tests/run.sh                          # los 7 invariantes, segundos
bash scripts/smoke_test.sh                 # las 7 etapas con datos sintéticos
```

`--force` es un **modificador**: rehace lo que dejen `--only/--skip/--from`.
Solo, rehace todo. **Ojo:** desde 2026-08 también borra el caché de Optuna
(`reports/optuna_*.json` y `.db`), así que una corrida forzada **vuelve a
buscar hiperparámetros**. Si quieres conservarlos —lo normal al comparar
cambios del grafo— borra a mano solo lo que toca y lanza **sin** `--force`.

## Arquitectura en una pantalla

Piloto sobre **2 meses partidos en 8 semanas**. Cada bloque tiene un trabajo y
ninguno se solapa:

```
   MES 1                          MES 2
   S1    S2    S3    S4     S1    S2    S3    S4
   └─────────┘  └──┘  └──────────────┘  └──┘  └──┘
   gnn_entrena   gnn   cabezas_entrenan  cab.  EXAMEN
   56.059      valida  85.071          validan 21.284
               37.386                  21.006
```

```
[0] download    Kaggle -> data/raw/
[1] preprocess  CSV -> parquet, semanas, encoders ajustados sin fuga
[2] graph       grafo HETEROGÉNEO: transacción <-> 5 entidades
[3] gnn         GraphSAGE vs GATv2 · Optuna · 3 semillas cada una
[5a] embed      UNA red describe todo lo que no entrenó
[5b] heads      las TRES cabezas XGBoost
[7] final       veredicto sobre `examen` + bootstrap emparejado
```

El continual learning está **desactivado** (`pipeline.py:CL_ACTIVO = False`):
esta fase compara enfoques y para eso `examen` tiene que ser test puro. El
código de `continual_learning/` se conserva intacto.

### Las tres cabezas

```
control            65 col tabulares            ¿cuánto se logra sin grafo?
solo_gnn           el embedding y nada más     ¿el grafo solo basta?
gnn_mas_tabular    65 + el embedding           ¿el grafo SUMA?
```

**La pregunta del capstone la responde `control` vs `gnn_mas_tabular`.** Las
tres reciben la misma ablación, la misma ventana y los mismos hiperparámetros.

## Convenciones

- **Español** en nombres, comentarios y logs.
- Los comentarios explican el **porqué**, no el qué. Si un comentario dice algo
  que ya se lee en el código, sobra; si documenta una decisión no obvia o un
  fallo que costó horas, es lo más valioso del archivo.
- **Nunca commitear ni pushear** sin que el usuario lo pida. Se le entregan los
  comandos y él los ejecuta.
- Antes de proponer código nuevo, buscar si ya existe: `objective_factory`,
  `apply_smote`, `full_report`, `umbral_por_presupuesto`, `mascara`,
  `cols_embedding` y `mezcla_40_60` se reutilizan a propósito.
- **Al cambiar código, actualizar su memoria en el mismo commit.**

## Reglas que ya se han roto

Cada una costó una corrida o una conclusión errónea.

1. **`node_idx` == índice de fila de `full.parquet`.** Contrato implícito, sin
   guardar en ningún sitio. Lo protegen asserts en `hybrid/head.py`. No los
   quites.
2. **El parquet del embedding debe cubrir todas las filas que usan las
   cabezas.** Las de `gnn_entrena` se excluyen a propósito (la red las
   memorizó); cualquier otra que falte entrena con NaN.
3. **`pipeline.py:STEPS` declara las salidas de cada etapa.** Si cambian y no se
   actualiza, la reanudación se rompe. Hay dos campos dinámicos —`outputs_dyn` y
   `limpiar_dyn`— para salidas cuyo nombre depende del config.
4. **Nada de hiperparámetros a mano para una cabeza y no para otra.**
   `xgboost.por_cabeza` se aplica como override FINAL, encima de Optuna. Tenerlo
   puesto solo en dos de tres cabezas dio un aporte de −0.0668 «significativo»
   que era íntegramente artefacto. Déjalo vacío.
5. **`guard_omp()` antes de importar torch/xgboost** en los módulos que cargan
   los dos. En macOS, si no, SIGSEGV.
6. **Umbral por presupuesto de alertas, nunca fijo.** El mismo modelo pasó de
   F1 0.4356 a 0.5785 solo por corregir el punto de operación.
7. **`sin_aristas` hay que propagarlo en cada loader** vía `loader_opts(cfg)`.
   No hacerlo produjo un −55% falso que sostuvo una conclusión errónea.
8. **Claves del checkpoint** (`model_name`, `seed`, `in_dim`, `best_epoch`,
   `state_dict`) las leen varios módulos. En el checkpoint final `state_dict`
   son los **mejores** pesos, no los de la última época.
9. **`selected_model.json`** lo escribe `compare_gnns` y lo lee `embed`.
10. **`config/config.yaml` está versionado**: `git pull` lo pisa. Después de
    cada pull, verificar:
    ```bash
    grep -nE "n_jobs:|paralelo_|optuna_|max_entity_degree:|min_previas_entidad:|por_cabeza:" config/config.yaml
    ```

## Invariantes científicos

Romperlos invalida resultados **sin que nada falle**. `tests/run.sh` guarda los
que se pueden comprobar en máquina.

- **`examen` no se toca** hasta la etapa `final`.
- **El muestreo solo mira hacia atrás.** `time_attr` + `temporal_strategy="last"`.
  Verificado sobre 123.980 vecinos: cero posteriores
  (`tests/test_causalidad_muestreo.py`).
- **El embedding «solo vecinos» tiene que contener vecinos.** Se captura en la
  ÚLTIMA capa: en la primera los nodos de entidad todavía son ceros
  (`tests/test_embedding_vecinos.py`).
- **Optuna corre una sola vez y las tres cabezas heredan**
  (`xgboost.optuna_modo: "compartido"`), para que la ventaja de las grandes sea
  una cota inferior conservadora. Con búsqueda por cabeza, `gnn_mas_tabular`
  ganó en validación por 0.0004 y perdió en el examen: era la búsqueda
  sobreajustando, no el grafo.
- **La ablación `["V","C","D"]` se aplica a TODOS por igual**, GNN incluida. Si
  la GNN viera esas columnas, las devolvería por la puerta de atrás dentro del
  embedding.
- **Toda comparación exige igualar la ventana de entrenamiento.** El +0.0626 del
  híbrido sobre el baseline resultó ser un mes extra de datos, no la GNN.
- **Buffer de replay y set de control jamás se cruzan** (cuando el CL vuelva).

## Limitación declarada: el embedding caduca

La GNN entrena en los días 1-15 y el examen está en los 53-60. En ese trayecto
el poder discriminante del embedding cae un 38%:

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
ventana que la GNN reentrene se la quitas a la cabeza—, así que **se documenta
como cota inferior** y el refit va en la corrida de 6 meses. No es un fallo
pendiente: es el límite del piloto, y está medido.

## Memorias por módulo

Leer la memoria del módulo **antes** de modificar cualquier archivo suyo: cada
una lleva el mapa de acoplamiento y los fallos ya cometidos.

| Memoria | Archivos que cubre |
|---|---|
| [`data.md`](.claude/memory/data.md) | `data/download_ieee_cis.py`, `data/preprocessing.py`, `data/build_graph.py` |
| [`gnn.md`](.claude/memory/gnn.md) | `gnn/models.py`, `gnn/sampling.py`, `gnn/train_gnn.py`, `gnn/compare_gnns.py`, `gnn/refit.py` |
| [`hybrid.md`](.claude/memory/hybrid.md) | `hybrid/embed.py`, `hybrid/head.py`, `hybrid/train_head.py`, `hybrid/features.py`, `hybrid/system.py`, `hybrid/oof.py`, `hybrid/head_cl.py` |
| [`comparison.md`](.claude/memory/comparison.md) | `comparison/final_comparison.py`, `comparison/resumen.py` · **y los resultados** |
| [`pipeline.md`](.claude/memory/pipeline.md) | `pipeline.py`, `scripts/*` |
| [`tests.md`](.claude/memory/tests.md) | `tests/*` — los siete invariantes y qué fallo guarda cada uno |
| [`utils.md`](.claude/memory/utils.md) | `utils/common.py`, `utils/metrics.py`, `utils/omp.py`, `utils/ventanas.py` |
| [`baseline_xgboost.md`](.claude/memory/baseline_xgboost.md) | `baseline_xgboost/smote_pipeline.py`, `baseline_xgboost/train_xgboost.py` |
| [`continual_learning.md`](.claude/memory/continual_learning.md) | `continual_learning/*` — desactivado, se conserva |
