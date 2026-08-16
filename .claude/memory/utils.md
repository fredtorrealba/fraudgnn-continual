# utils — lo que usa todo el mundo

Cuatro archivos pequeños de los que depende el resto. Cambiar algo aquí toca
todos los módulos.

## ventanas.py — quién entrena con qué

El reparto temporal del proyecto entero. Antes vivía disperso en
`preprocessing` como «meses 1-4 / 5 / 6»; ahora es **declarativo** en el config
y cinco bloques de semanas.

```python
mascara(cfg, bloque, meses, semanas)  -> np.ndarray booleano
verificar(cfg, meses, semanas, log)   -> dict, y ABORTA si hay solape
mascaras_grafo(cfg, data, log)        -> lo mismo como tensores, desde el grafo
```

`verificar()` no devuelve las máscaras y ya: **comprueba que ningún bloque se
solapa y aborta si lo hace**. Es la única barrera entre el diseño y una fuga
silenciosa, porque un solape no produce ningún error — produce un número mejor.

Estado verificado del piloto:

```
bloque             semanas          días    filas    fraudes    %fr
gnn_entrena        M1S1 M1S2        1-15    56.059     1.521   2,71%
gnn_valida         M1S3            15-22    37.386       923   2,47%
cabezas_entrenan   M1S4 M2S1 M2S2  23-45    85.071     2.641   3,10%
cabezas_validan    M2S3            45-52    21.006       909   4,33%
examen             M2S4            53-60    21.284       876   4,12%

solapes: ninguno · cobertura 220.806 de 220.806 (100%)
```

**El fraude sube con el tiempo** (2,71% → 4,12%): el examen tiene un 52% más de
fraude que la ventana donde entrena la red. Es drift real y parte de por qué el
problema es difícil.

## common.py

- `load_config(path=None)` — lee `config/config.yaml` **y llama a
  `_apply_compute()`**. Cargar la config tiene ese efecto secundario a
  propósito: fija `OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS` antes de que las
  librerías arranquen sus pools.
- `_apply_compute()` **respeta un `OMP_NUM_THREADS` ya fijado**: si algo lo puso
  antes fue por una razón (ver `omp.py`). Pisarlo resucitaba el SIGSEGV.
- `get_device()` — CUDA si existe, si no CPU. **MPS no se activa solo**: el
  cuello es el sampling (CPU) y GAT tiene soporte irregular. Opt-in con
  `FRAUDGNN_DEVICE=mps`. Anuncia el dispositivo una vez por proceso.
- `resolve(cfg, clave)` / `ensure_dirs(cfg)` — rutas desde `config.paths`.

### `set_seed(seed, determinista=True)`

```python
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
torch.use_deterministic_algorithms(True, warn_only=True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

**Verificado:** dos smokes seguidos dieron salida bit-idéntica — PR-AUC época a
época, número de árboles y el IC del bootstrap incluidos.

Lo único que puede mover el resultado sin que nadie toque nada es el **número de
hilos OMP**, que `_limitar_hilos` reparte según `paralelo_*`. Si los números se
mueven en el cuarto decimal tras cambiar el paralelismo, es eso.

### `_update_section` va con `flock`

```python
with open(lock, "w") as fl:
    fcntl.flock(fl, fcntl.LOCK_EX)
```

Con `paralelo_corridas > 1` hay varios procesos escribiendo
`artifacts/pipeline_state.json` a la vez. Sin el lock, dos escrituras
concurrentes dejan el JSON corrupto y la reanudación deja de funcionar. La
escritura además es atómica (temporal + rename).

### Las variables de entorno se leen UNA vez

`OMP_NUM_THREADS` y compañía las lee OpenBLAS/libgomp **al cargarse**. Si
importas numpy antes de `load_config()`, ya es tarde.

El pipeline no sufre esto porque lanza cada etapa como **subproceso**, que hereda
el entorno al arrancar. Un script suelto sí: hay que fijar las variables antes de
`import numpy`. Síntoma: `OpenBLAS warning: precompiled NUM_THREADS exceeded`, y
en el peor caso `BLAS : Bad memory unallocation!` con core dump.

## metrics.py

`full_report(y, s, thr)` — AUC-ROC, PR-AUC, recall, precisión, F1, accuracy. El
AUC solo si hay ambas clases. **La validación siempre sobre distribución real,
sin balanceo.** Lo usan baseline, GNN, híbrido y comparación: cambiar sus claves
rompe todos los informes.

## omp.py

`guard_omp()` — fija `OMP_NUM_THREADS=1` en **macOS**, donde torch y XGBoost
traen runtimes de OpenMP distintos y con los dos multihilo el intérprete muere
con SIGSEGV al cargar un modelo. En Linux no hace nada.

**Asignación directa, no `setdefault`**: el pipeline padre exporta
`OMP_NUM_THREADS` desde `compute.n_jobs` y el subproceso lo hereda, así que un
`setdefault` no llegaría a aplicarse nunca. Aquí no es un valor por defecto, es
un requisito para no segfaultear.

Va como **primera línea ejecutable**, antes de importar torch o xgboost.

## Si tocas esto, revisa

- **`ventanas.mascara`** → lo usan `preprocessing`, `build_graph`, `embed`,
  `train_head`, `final_comparison` y cuatro tests. Es el módulo más compartido.
- **`full_report`** → todos los informes JSON del proyecto.
- **`get_device`** → una corrida completa debe usar **siempre** el mismo
  dispositivo: CPU y GPU dan resultados numéricamente distintos y la comparación
  entre arquitecturas exige igualdad de condiciones.
- **`set_seed`** → si quitas el determinismo, dos corridas iguales dejan de serlo
  y ya no se puede atribuir un cambio a lo que se tocó.
- **`_apply_compute`** → el equilibrio con `guard_omp()` es delicado; no lo
  simplifiques sin releer los dos comentarios.
