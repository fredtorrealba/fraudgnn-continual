# hybrid — del embedding a las tres cabezas

Donde la GNN entra en el sistema entregado: sus salidas son columnas que consume
una cabeza XGBoost.

## Artefactos

| Archivo | Lo escribe | Lo lee |
|---|---|---|
| `processed/gnn_embed.parquet` | `embed.py` | `head.cargar_tabla()`, `tests/test_smote.py` |
| `reports/embed.json` | `embed.py` | (informativo) |
| `models/hybrid_head_{variante}.json` | `train_head.py` | `comparison/final_comparison.py` |
| `reports/heads_variantes.json` | `train_head.py` | `comparison/*` |

## embed.py — UNA red, no K

**El fallo del cfg global (2026-08-17).** `embed_and_score_nodes` arma su
loader con `len(cfg.gnn.hidden_dims)` saltos. `embed.py` construía el modelo
con `cfg_arquitectura()` (la arquitectura real del checkpoint) pero pasaba el
**cfg global** al scorer: con la ganadora de 3 capas, la red describió con
vecindarios de 2 saltos — truncados, sin síntoma alguno. Latente mientras
Optuna eligió 2 capas; se activó la primera vez que ganó 256×3 y contaminó el
veredicto de esa corrida (−0.0201 «significativo»). Hoy `embed.py` pasa `c` y
`validate.py` tiene una guarda que revienta ante el mismatch. Si añades un
consumidor de `embed_and_score_nodes`/`score_nodes`, constrúyele el cfg con
`cfg_arquitectura(modelo, cfg, checkpoint)`.

Sustituye al OOF. Una sola red describe **todo lo que no entrenó**:

```python
usa = np.zeros(len(entrenadas), dtype=bool)
for n, m in v.items():
    if n != "gnn_entrena":
        usa |= m.numpy()
describir = np.where(usa & ~entrenadas)[0]
```

Con las ventanas separadas basta una red, y eso resuelve el fallo que hundió al
OOF: **K redes aprenden K sistemas de coordenadas latentes distintos**. La
dimensión 7 de la red A no significa lo mismo que la dimensión 7 de la red B, y
mezclarlas en las mismas columnas hundía las cabezas — medido: la cabeza mixta
cortó en **2 árboles** contra 517 del control.

`oof.py` se conserva pero **no está en el pipeline**. Si algún día hace falta
refit con ventanas cortas, el camino no es resucitar el OOF de embeddings sino
usar un **escalar calibrado** (`gnn_score`): una probabilidad sí es comparable
entre redes, las dimensiones crudas no.

El parquet lleva **tres bloques de columnas**:

```
gnn_score     el score del camino completo
emb_*         "yo + mi vecindario"   -> lo consume `solo_gnn`
embv_*        "solo mi vecindario"   -> lo consume `gnn_mas_tabular`
```

Cubre 164.747 filas de 220.806. Las 56.059 que faltan son `gnn_entrena`: la red
las memorizó y su embedding sería optimista. **Que falten es correcto**; que
falte cualquier otra, no.

## head.py

### `columnas(variante, ...)` — las tres cabezas

```
control            cols_base                 ¿cuánto sin grafo?
solo_gnn           emb_*                     ¿el grafo solo basta?
gnn_mas_tabular    cols_base + embv_*        ¿el grafo SUMA?
```

El ancho depende de `mlp_head_dim` que gane Optuna Y de cuántas derivadas traiga
el preprocess (flags `__na`, hora sin/cos, `__tiene_anterior`, `__grado_*`).
**No lo escribas como constante en ningún sitio**: `cols_base` sale de
`cargar_tabla` y el del booster de `booster.num_features()`.

Cada cabeza recibe el embedding que le corresponde y no es intercambiable:
`solo_gnn` no recibe nada más, así que sin las features propias no sabría nada de
la transacción; `gnn_mas_tabular` ya las tiene en su bloque tabular.

### `filtrar_prefijos` — la ablación

```yaml
xgboost.excluir_prefijos: ["V", "C", "D"]
```

Se aplica **a todas las cabezas y también a la GNN** (`build_graph` la importa de
aquí). Si la GNN viera V/C/D y las cabezas no, el embedding se las devolvería por
la puerta de atrás y la variante con grafo ganaría por copiarlas.

Son 368 de 433 columnas: los agregados relacionales que Vesta precalculó sobre
historiales de entidad, que son la hipótesis de por qué el grafo no aporta.
Quitarlas prueba esa hipótesis directamente.

### `cargar_tabla` — el contrato de posición

`node_idx` es el índice de FILA de `full.parquet`. La unión es **posicional**, no
un join por clave, así que si `build_graph` reordenara nodos todo quedaría
desalineado en silencio. Los asserts de aquí son la única red:

```python
assert ni.min() >= 0 and ni.max() < len(df)
assert len(np.unique(ni)) == len(ni)
```

Y avisa si quedan filas sin embedding dentro de las ventanas que usan las
cabezas.

**Desde 2026-08-16 añade los `__grado_*`** de `grados_entidad.parquet` (lo
escribe `build_graph`) a `cols_base`, para las TRES cabezas por igual.
`__grado_uid` es el `UID_FE` de los ganadores de Kaggle y la GNN ya lo veía
entre sus features: sin esto, `control` competía sin una columna que el híbrido
llevaba dentro del embedding. Va DESPUÉS de la ablación a propósito (no sale de
V/C/D) y también por posición, con su propio assert de longitud. Si el parquet
no está, WARNING y sigue sin ellos — pero esa corrida es asimétrica y no vale
para el veredicto.

### `umbral_por_presupuesto(scores, pct)`

**El umbral nunca es fijo.** Modelos con calibraciones distintas emiten
volúmenes de alerta incomparables: la GNN entrena con `pos_weight` y sus scores
están inflados, la cabeza devuelve probabilidades calibradas. A 0.5 el híbrido
daría casi cero alertas.

Medido: el mismo modelo pasó de **F1 0.4356 a 0.5785** solo por corregir el
punto de operación, sin tocar un peso.

## train_head.py — dos fases, y hay que distinguirlas

```
FASE 1   SMOTE + 30..100 trials de Optuna       modelos de usar y tirar
FASE 2   SMOTE + entrenar las 3 cabezas         los modelos que se reportan
```

Los trials **no producen ningún modelo que se use**: solo sobrevive la lista de
hiperparámetros. Por eso en el log aparecen 4 llamadas a SMOTE (una por la
búsqueda, tres por las cabezas) y por eso el `control` final sale con un número
de árboles que no aparece en ningún trial.

SMOTE se ejecuta **una vez por cabeza** porque interpola en el espacio de
columnas: los vecinos más cercanos de un fraude en 65 dimensiones no son los
mismos que en 81.

### `optuna_modo: "compartido"` — INVARIANTE

Optuna corre **una vez** sobre `control` y las tres heredan. Es el protocolo
declarado en `CLAUDE.md`, y volver a él costó una corrida entera.

Con `"por_cabeza"`, Optuna le dio a `gnn_mas_tabular` un `lr` 3× más alto que
converge en 106 árboles: ganaba por 0.0004 de AUC en la ventana donde se busca y
**perdía en el examen**. Eso no era el grafo, era la búsqueda sobreajustando la
validación.

### `por_cabeza` — déjalo VACÍO

```python
p.update((x.get("por_cabeza") or {}).get(variante, {}) or {})   # :72
```

Es un override **FINAL, encima de Optuna**. Solo las claves de `_ESPACIO_OPTUNA`
se siembran además como primer trial; `n_estimators` no está entre ellas, así que
ahí es un tope duro y nada más.

Tenerlo puesto en dos de las tres cabezas dio un aporte de **−0.0668
«significativo»** que era íntegramente artefacto: `gnn_mas_tabular` entrenaba con
`max_depth 6` y 500 árboles —topando en 500— mientras `control` usaba los suyos de
Optuna con techo de 1000 y paraba solo en 379.

El código ya avisaba y nadie leyó el WARNING. Si lo pones, ponlo en las **tres**.

## features.py

Las columnas estructurales que consume la cabeza sin cargar torch. Van a parquet
y no al grafo porque quien las consume es XGBoost.

## system.py / head_cl.py — dormidos con el CL

`HybridSystem.score()` y `head_cl.matriz_nodos()` arman la matriz de la cabeza en
**dos sitios distintos**. El ancho lo dicta `booster.num_features()`, nunca una
constante.

Ninguno de los dos está en el pipeline actual (`CL_ACTIVO = False`). Al
reactivarlos hay que revisarlos contra el esquema de tres cabezas: se escribieron
para el de cuatro variantes numeradas.

## Si tocas esto, revisa

- **Añadir o quitar una variante** → `config.hybrid.variantes`, `columnas()`,
  `train_head`, `final_comparison`, `resumen.py`, `system.py`, `head_cl.py`.
- **Cambiar el modo del embedding** → `cols_embedding(df, modo)` y qué cabeza
  recibe cuál. Son opuestos y no son intercambiables.
- **Cambiar `apply_smote` u `objective_factory`** → los comparte con
  `baseline_xgboost/`.
- **Tocar la ablación** → afecta a `build_graph` también.

## Cifras de referencia (examen, 21.284 txn · 876 fraudes)

```
                   PR-AUC   ROC      recall@2%   árboles
control            0.3396   0.8313     0.2489      379
gnn_mas_tabular    0.3339   0.8347     0.2443      360
solo_gnn           0.1027   0.6770     0.0856       13
gnn_sola           0.0810   0.6436     0.0753        —

aporte del grafo  -0.0058   IC95 [-0.0209, +0.0089]   no significativo
el embedding aporta el 3,2% de la ganancia (16 de 16 columnas usadas)
```

Ese **3,2%** es el dato que más dice. Antes de arreglar E0, las 64 columnas de
`embv_` se llevaban el **15,9%** — pero eran proyecciones lineales de las
features propias, que a un modelo de árboles le sirven como cortes oblicuos.
Cuando el embedding lleva información de vecinos de verdad, XGBoost la usa
**cinco veces menos**.

La información real del grafo vale menos que el artefacto que la sustituía.
