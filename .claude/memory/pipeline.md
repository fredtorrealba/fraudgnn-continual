# pipeline y scripts — el orquestador

`src/pipeline.py` decide qué correr y qué saltarse. `scripts/` envuelve la
ejecución, el montaje del pod y la validación barata.

## Las 7 etapas

```
[0] download    Kaggle -> data/raw/                       ~1 min
[1] preprocess  CSV -> parquet + semanas                  ~1 min
[2] graph       grafo heterogéneo + graph_meta.json       ~1 min
[3] gnn         Optuna + 6 corridas + selección          EL CARO
[5a] embed      UNA red describe lo que no entrenó       ~0,5 min
[5b] heads      las tres cabezas XGBoost                  ~2-13 min
[7] final       veredicto + bootstrap                     ~0,5 min
```

La numeración salta porque quedan huecos del diseño anterior de 12 etapas. La
segunda pasada (`refit`/`oof_refit`/`heads_refit`) **se eliminó**: con ventanas
separadas cada bloque tiene un trabajo y no hay nada que reentrenar.

`CL_ACTIVO = False`: el continual learning sale del flujo por defecto porque
esta fase compara enfoques y para eso `examen` tiene que ser test puro. El
código de `continual_learning/` se conserva intacto.

## Cómo decide

**El disco manda.** Cada `Step` declara sus salidas; si existen, el paso se
salta. Borra una salida y vuelve a estar pendiente.
`artifacts/pipeline_state.json` guarda el avance pero **no es la autoridad**.

Un paso solo se marca `done` si **produjo sus salidas declaradas**.

### El contrato de `STEPS`, con dos campos dinámicos

```python
Step(nombre, título, módulo,
     [(clave_de_paths, "archivo"), ...],       # salidas fijas
     outputs_dyn=lambda cfg: [...],            # salidas cuyo NOMBRE depende del config
     limpiar_dyn=lambda cfg: [...],            # se borran con --force, NO marcan hecho
     acepta_force=True, desc="...")
```

Los dos dinámicos existen por el caché de Optuna:

- **`outputs_dyn`** → `reports/optuna_{arq}.json`. Son una por arquitectura, así
  que no se pueden escribir como constantes. Antes no figuraban como salidas:
  `--status` no las veía y `--force` no las borraba, de modo que una corrida
  «forzada» reutilizaba en silencio hiperparámetros de otra búsqueda.
- **`limpiar_dyn`** → `reports/optuna_{arq}.db`. No marcan el paso como hecho —un
  estudio a medias también los crea— pero `--force` tiene que llevárselos: si
  no, `load_if_exists=True` retoma la búsqueda anterior sin decirlo.

> **Consecuencia práctica:** `--force` sobre `gnn` **relanza la búsqueda de
> hiperparámetros**. Si quieres conservarlos —lo normal al comparar cambios del
> grafo— borra a mano solo lo que toca y lanza SIN `--force`:
> ```bash
> rm -f models/selected_model.json models/*_seed*.pt reports/*_seed*_val.json
> rm -f data/processed/gnn_embed.parquet reports/embed.json \
>       reports/heads_variantes.json reports/final_comparison.json reports/resumen.json
> bash scripts/run_pipeline.sh --from gnn        # sin --force
> ```
> Y comprueba en el log que sale `hiperparámetros ya buscados — se reutilizan`.

## Flags

```
--status              en qué va          (no ejecuta)
--steps               qué hace cada una  (no ejecuta)
--only a,b            SOLO esas
--skip a,b            todo MENOS esas
--from paso           desde ahí
--force               MODIFICADOR: rehace lo que dejen --only/--skip/--from
--archive "nombre"    archiva la corrida y deja todo en cero
--history             lista lo archivado
```

## scripts/ — qué queda y para qué

```
run_pipeline.sh          el envoltorio
setup_runpod.sh          monta el pod
smoke_test.sh            las 7 etapas con datos sintéticos, en sandbox
make_synthetic_demo.py   genera esos datos
graph_explorer.py        el grafo navegable en HTML
```

Los diagnósticos que vivían aquí (`inspeccionar_grafo.py`, `revisar_smote.py`)
**pasaron a `tests/`**. Ver `tests.md`.

### smoke_test.sh — la red de seguridad

Copia el proyecto a `/tmp`, reduce el config y corre las 7 etapas con 40.000
filas sintéticas. **Aislado**: si escribiera en `reports/` directamente, el
pipeline creería que las etapas están hechas con datos sintéticos.

Reduce `epochs 2 · seeds [42] · optuna_trials 2`, y **también
`optuna_presupuesto_min: 0`** — ese presupuesto MANDA sobre `optuna_trials`, así
que sin anularlo el smoke se pondría a buscar una hora por arquitectura.

Ha detectado bugs reales que habrían costado horas de GPU. El último: el caché
SQLite compartido entre los dos procesos de Optuna.

### setup_runpod.sh

**No reinstala torch**: usa el de la imagen y deriva de él la URL de las ruedas
de `pyg-lib`/`torch-sparse`. Sin ellas el sampler heterogéneo no existe y
`sampling.py` aborta con mensaje claro — es lo que impide correr la GNN en macOS.

Instala XGBoost **desalojando** lo que traiga la imagen (a veces `xgboost-cpu`,
que instala el mismo módulo y sobrevive a `--force-reinstall`). El paso final
**verifica de verdad** que ve la GPU entrenando en un subproceso y buscando el
aviso `No visible GPU`: ni `build_info()["USE_CUDA"]` ni un `try/except` lo
detectan.

## Config: los números que importan

```yaml
compute.n_jobs: 14            # la cuota real del cgroup, no `nproc`
gnn.paralelo_optuna: 2        # las 2 arquitecturas a la vez
gnn.paralelo_corridas: 2      # las 6 corridas de dos en dos
gnn.num_workers: 4
gnn.batch_size: 2048
```

**`nproc` MIENTE en contenedores.** En el pod dice 128 y la cuota real es 13,6:

```bash
cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us    # 1360000
cat /sys/fs/cgroup/cpu/cpu.cfs_period_us   #  100000  -> 13,6 núcleos
```

**`paralelo_corridas: 2` y no 6 es deliberado.** Las corridas heredan los
hiperparámetros ganadores; si Optuna elige `capas 3`, seis procesos a ~8 GB no
caben en 20 de VRAM. Y la captura de OOM de `compare_gnns` **solo cubre los
trials de Optuna**, no esta fase. Con el ganador habitual (`capas 2`, ~2 GB)
cabrían seis, pero no se puede saber antes de que la búsqueda termine.

Se puede editar **en caliente**: el valor se lee al empezar la etapa.

## Rendimiento: el cuello cambió

Con el grafo homogéneo el cuello era la CPU (neighbor sampling) y la GPU iba al
20-30%. Con `capas 3` en el heterogéneo se invierte:

```
GPU     100% ocupada · 101 W de 130 · sin throttling
CPU     2,0 de 13,6 núcleos (15%)
```

Los dos procesos padre quedan clavados en **un núcleo cada uno** y los 16
workers al 2%: el cuello es **lanzar kernels**. `HeteroConv` con 10 tipos de
arista × 3 capas son 30 convoluciones por paso, y las lanza el hilo principal en
serie. No hay config que lo arregle.

`utilization.gpu` al 100% no significa saturada: mide el **tiempo con algún
kernel corriendo**, no la ocupación. Con 101 W de 130 y `clocks_event_reasons`
en `0x0`, la tarjeta no se contiene — la carga no da para más.

## Si tocas esto, revisa

- **Cambiar las salidas de una etapa** → su entrada en `STEPS`, incluidos los
  dinámicos.
- **Añadir una etapa** → `STEPS`, el texto de `--steps` y `resumen.py`.
- **Renombrar un paso** → README y los ejemplos de `run_pipeline.sh`.
- **Cambiar el config del smoke** → si añades algo que MANDE sobre otra cosa
  (como el presupuesto de Optuna), anúlalo también ahí.

## Tiempos reales (RTX 4000 Ada, 13,6 núcleos, piloto de 2 meses)

```
download+preprocess+graph      ~3 min
gnn   Optuna 100 trials         ~5 h    (2 arquitecturas en paralelo)
      6 corridas                ~40 min (paralelo 2)
embed                          ~0,5 min
heads Optuna 30 + 3 cabezas    ~2-13 min  según profundidad que gane
final                          ~0,5 min
```

Con 30 trials la búsqueda baja a ~1,5 h. El tope por trial (10 min) acota el
peor caso: sin él, un solo trial se comió 29 minutos.
