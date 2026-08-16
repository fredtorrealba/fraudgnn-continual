# gnn — las redes, su muestreo y la búsqueda

GraphSAGE y GATv2 sobre el grafo **heterogéneo**. Dos capas como mínimo, porque
llegar de una transacción a otra cuesta dos saltos: `transacción -> entidad ->
transacción`.

## Artefactos

| Archivo | Lo escribe | Lo lee |
|---|---|---|
| `models/{modelo}_seed{n}.pt` | `train_gnn.py` | `compare_gnns`, `hybrid/embed` |
| `models/{modelo}_seed{n}_resume.pt` | `train_gnn.py` cada época | `train_gnn` al reanudar; se borra al terminar |
| `reports/{modelo}_seed{n}_val.json` | `train_gnn.py` | `compare_gnns` |
| `models/selected_model.json` | `compare_gnns.py` | `hybrid/embed.py` |
| `reports/optuna_{modelo}.json` | `compare_gnns.py` | él mismo (caché) |
| `reports/optuna_{modelo}.db` | Optuna vía SQLite | él mismo (reanudar) |

Los dos últimos son **salidas declaradas de la etapa `gnn`**
(`pipeline.py:outputs_dyn` / `limpiar_dyn`). Antes no lo eran, y un `--force`
reutilizaba en silencio hiperparámetros de otra búsqueda.

## models.py

### Los nodos de entidad entran en CEROS

No es un relleno perezoso: es lo que hace el modelo **inductivo**. Si cada
entidad tuviera su propio embedding aprendido, una tarjeta vista por primera vez
en el examen no tendría vector. Así su contenido sale **enteramente** de agregar
sus transacciones.

Consecuencia directa, y es la que costó el fallo E0:

```
CAPA 1   txn -> entidad     la entidad se llena
         entidad -> txn     pero estaba VACÍA al empezar: llegan ceros
CAPA 2   entidad -> txn     AQUÍ llega el resumen de verdad
```

### `encode(..., solo_vecinos=True)` — E0

Devuelve además el término que NO viene del propio nodo, para que XGBoost no
reciba dos veces las features de la transacción.

**Se captura en la ÚLTIMA capa** (`i == len(self.convs) - 1`). Capturarlo en la
primera —como estuvo toda la primera fase— daba un vector que **no contenía ni
un vecino**: matemáticamente `constante + 4 proyecciones lineales de x_txn`.

Medido antes del arreglo: cambiar las vecinas de 0.0 a 9.0 movía el embedding
**0.000000**. Después: 2.137227.

Y `_termino_vecinos(conv, h_txn, x_txn)` resta **las cinco** aristas entrantes,
no una:

```python
resto = h_txn
for et, sub in conv.convs.items():
    if et[2] == TXN:
        resto = resto - sub.lin_r(x_txn)     # antes había un `return` aquí
```

`HeteroConv` aplica una convolución por tipo de arista y las suma con
`aggr="sum"`, así que la salida lleva **cinco** términos `lin_r(x_i)`, uno por
entidad. La versión con `return` dentro del bucle quitaba el primero y dejaba
cuatro copias de las features propias dentro del embedding que existe justamente
para quitarlas.

Lo guarda `tests/test_embedding_vecinos.py`.

> **Contraintuitivo, y hay que tenerlo presente:** el embedding con MÁS señal por
> dimensión es el que hunde el modelo. `emb_` (completo) tiene AUC mediana 0.675
> por dimensión y al dárselo a `gnn_mas_tabular` da **−0.0325**; `embv_` tiene
> 0.526 y da +0.0053. AUC alto por dimensión significa *redundante con lo
> tabular*, no *útil*.

### Otros contratos

- `aggr=["mean","max","std"]` en GraphSAGE. La media destruye la dispersión del
  vecindario, y la dispersión es lo único que no está ya en el dataset: C, D y V
  son conteos, deltas y agregados — ninguno trae varianza.
- GATv2 con `concat=False, add_self_loops=False`. Sin auto-atención no hay
  término de raíz que descontar, por eso su `_termino_vecinos` devuelve `h_txn`.
- `dim_embedding` = `mlp_head_dim`, la capa oculta del clasificador. **No es
  `hidden_dims[-1]`.** Con `mlp_head_dim: 16` el embedding tiene 16 columnas,
  no 64.
- Menos de 2 capas lanza `ValueError`: con una sola, los nodos de entidad llegan
  a la transacción todavía en ceros y el grafo no aporta nada. Verificado: la
  diferencia con y sin aristas era exactamente 0.000000.

## sampling.py

### La dirección de los fanouts

```python
cuantos = por_entidad if et[0] == TXN else 1
```

Para el tipo `(src, rel, dst)`, el fanout dice cuántos **src** se traen al
expandir un **dst**:

```
('transaction','en_uid','uid')          -> 10 transacciones por uid
('uid','tiene_uid','transaction')       ->  1 uid por transacción
```

**Estuvo invertido**, y con eso 2048 semillas daban 2049 nodos: la red entrenaba
con el vecindario vacío en todas las corridas heterogéneas. Ahora da ×16.

### Causalidad y recencia

```python
time_attr="time",              # solo vecinos ANTERIORES
temporal_strategy="last",      # y de esos, los más recientes
```

Verificado en el pod sobre 4.517 vecinas muestreadas: **cero** posteriores a su
raíz y **cero** fuera de las 10 más recientes
(`tests/test_seleccion_vecinos.py`, modo completo).

`time_attr` activa `disjoint` automáticamente: cada semilla arrastra su propio
subgrafo y el tensor `batch` dice de qué semilla viene cada nodo. Eso permite
comprobar la causalidad **por semilla**, no contra el máximo del lote.

### Workers por tamaño, no por constante

```python
n_w = int(opts["num_workers"]) if (shuffle or n_semillas >= 20_000) else 0
```

Levantar 12 procesos cuesta ~60 s cuando el padre ya tiene contexto CUDA y el
grafo en memoria. Entrenar (`shuffle=True`) amortiza ese coste entre épocas;
puntuar una sola vez, no. Medido: un fold pasó de 61 s a 2 s al quitarlos.

`cerrar_loader()` es obligatorio tras cada uso: sin él, cada trial dejaba sus
workers vivos hasta el final del proceso.

## compare_gnns.py

### La semilla de los trials está FIJA

```python
set_seed(42)      # dentro de objetivo(), para TODOS los trials
```

Cada trial es dos cosas a la vez: unos hiperparámetros y una inicialización. Si
la inicialización cambia entre trials, cuando uno gana no se sabe cuál de las
dos fue. Y pesa más de lo que parece: los mismos hiperparámetros dieron 0.3077 y
0.2912 según cómo inicializaran. La robustez frente a la inicialización se mide
DESPUÉS, en las 6 corridas con semillas 42/123/2026.

### Semilla del sampler distinta por arquitectura

```python
semilla_tpe = 42 + sorted(cfg["gnn"]["arquitecturas"]).index(model_name)
```

Con la misma, los dos estudios sorteaban **los mismos hiperparámetros en el
mismo orden**: cuando a uno le tocaba la red grande, al otro también, y los picos
de VRAM coincidían en vez de turnarse.

### El presupuesto y sus dos relojes

```yaml
optuna_trials: 100            # manda si optuna_presupuesto_min es 0
optuna_presupuesto_min: 0     # si > 0, MANDA sobre optuna_trials
optuna_tope_trial_min: 10     # tope POR TRIAL
```

Se eligió **por trials** a propósito: el presupuesto por tiempo hace que el
resultado dependa de lo rápida que sea la máquina, y este proyecto tiene la
reproducibilidad como objetivo. Lo que motivaba el presupuesto por tiempo —que
una arquitectura se comiera 11 h con «los mismos 30 trials»— ya lo resuelve el
tope por trial.

**El tope no puede matar al que va ganando:**

```python
campeon = max((t.value for t in trial.study.trials if t.value is not None), default=0.0)
if mejor > campeon:
    tope = 0.0        # indultado, SOLO este trial
```

Sin eso, una red lenta pero superior quedaría `PRUNED` y Optuna solo elige entre
los `COMPLETE`: desaparecería sin dejar rastro. `tope` es una copia **local**;
tocar `tope_trial` directamente dejaría a los siguientes trials sin límite.

### Tres guardias que se complementan

```
5 primeros trials    el podador no los toca (n_startup_trials)
2 primeras épocas    de cada trial, tampoco (n_warmup_steps)
10 minutos           tope por trial — aplica desde el primero
va en cabeza         indulto: termina sus épocas
```

El podador mira **calidad**; el tope mira **reloj**. El agujero que tapó el tope:
`gatv2 trial 3` era uno de los cinco protegidos y se comió 29 minutos para salir
peor que la red mínima.

### Estudios persistidos, un archivo por arquitectura

```python
db = resolve(cfg, "reports_dir") / f"optuna_{model_name}.db"
```

**No uno compartido.** Con `paralelo_optuna: 2` los dos procesos crean el
esquema a la vez y el segundo muere con `table studies already exists` — lo cazó
el smoke. Y aunque no chocaran, SQLite serializa las escrituras y pelearían por
el lock en cada trial.

### Paralelismo

```python
ProcessPoolExecutor(..., initializer=_morir_con_el_padre)
```

`PR_SET_PDEATHSIG` en cada hijo: es lo único que aguanta un SIGKILL del padre.
Sin él, un Ctrl-C deja a los hijos vivos con `PPID 1` agarrando VRAM — hubo dos
de **14 horas** ocupando 1,5 GB de los 20 de la tarjeta.

Y spawn, no fork: un fork con contexto CUDA ya creado da comportamiento
indefinido.

`_limitar_hilos(cfg, par)` reparte `n_jobs` entre los procesos. **Muta el cfg
que recibe**, por eso `_buscar_todas` pasa una copia profunda por tarea.

## Si tocas esto, revisa

- **`encode`/`_termino_vecinos`** → `tests/test_embedding_vecinos.py` y todo
  `hybrid/`. Es el punto donde un fallo no produce ningún síntoma.
- **Los fanouts** → `tests/test_seleccion_vecinos.py` y `test_causalidad_muestreo.py`.
- **El espacio de búsqueda** → borra `reports/optuna_*` o seguirá usando el
  caché viejo.
- **`selected_model.json`** → lo lee `hybrid/embed.py`.
- **Añadir un loader** → propaga `sin_aristas` desde `loader_opts(cfg)`.

## Cifras de referencia

```
ganadores de dos búsquedas independientes:  ancho 64/128, capas 2
la única config de 256/3 que corrió entera:  0.0607  (peor que 64/2 con 0.0612)

corrida de 6 redes en paralelo:  16,5 min
expansión del muestreo:          8.192 semillas -> 132.172 nodos (x16,1)
```

`capas 3` nunca ha ganado, y con el grafo denso de E2 se sale del tope de 10 min.
