# data — de los CSV al grafo heterogéneo

Tres pasos sin vuelta atrás: descargar, preprocesar y construir el grafo. Lo que
se decide aquí condiciona todo lo demás, y algunos contratos son **implícitos**.

## Artefactos

| Archivo | Lo escribe | Lo lee |
|---|---|---|
| `raw/train_{transaction,identity}.csv` | `download_ieee_cis.py` | `preprocessing.py` |
| `processed/full.parquet` | `preprocessing.py` | `build_graph`, `hybrid/head`, `comparison`, `tests/` |
| `processed/feature_cols.json` | `preprocessing.py` | `build_graph`, `hybrid/head` |
| `processed/split_masks.parquet` | `preprocessing.py` | `utils/ventanas`, `hybrid/*` |
| `graph/graph.pt` | `build_graph.py` | `gnn/*`, `hybrid/embed`, `comparison`, `tests/` |
| `graph/graph_meta.json` | `build_graph.py` | `hybrid/head`, `tests/`, diagnóstico |

## EL contrato implícito

> **El índice de fila de `full.parquet` ES el índice de nodo del grafo.**

Nadie lo guarda en ningún sitio: se cumple porque `build_graph` construye los
nodos en el orden del parquet. Todo el sistema une columnas **por posición**, no
por clave. Si `build_graph` reordenara nodos o `preprocessing` cambiara el
`reset_index`, todo se desalinearía **en silencio**.

Lo protegen asserts en `hybrid/head.py:cargar_tabla()`. **No los quites.**

## preprocessing.py

- Mes = `TransactionDT // (86400*30)`, semana dentro del mes en `week_in_month`.
  **El reparto de bloques ya no vive aquí**: lo decide `config.ventanas` y lo
  resuelve `utils/ventanas.py`.
- `data.meses: []` en el config = solo los meses que alguna ventana usa.
  Descartarlos AQUÍ, y no más adelante, es lo que garantiza que ninguna
  estadística (mapa de categorías, mediana de imputación) se calcule con meses
  que el experimento no usa.
- Encoders e imputación se ajustan con `gnn_entrena | cabezas_entrenan`. Antes
  se ajustaban con «meses 1-4» y el examen caía dentro: fuga silenciosa.
- Lo no visto va a `-1` (**SMOTE rechaza NaN**, por eso el centinela).

### Dos features derivadas que hay que conocer

La ablación quita las familias C (conteos) y D (deltas), así que se fabrican
sustitutos **causales**:

```
__hora_dia         hora del día en [0,1]. TransactionDT se excluye por ser
                   identificador, así que sin esto nadie ve la hora. CÍCLICA:
                   las 3 de la mañana valen igual en enero que en junio, así
                   que se solapa entre bloques.
__delta_anterior   log1p(segundos desde la compra ANTERIOR de la misma card1),
                   -1 si no hay anterior. Es lo que daba D1. Causal (diff hacia
                   atrás) y comparable entre meses.
```

**Al fabricarlas se devuelve parte de lo que quita la ablación.** Hay que
declararlo al comparar corridas.

`_delta_tarjeta` usa `argsort(stable=True)`: el 3,4% de las transacciones
comparte `TransactionDT` y sin estabilidad el desempate cambia entre corridas.

## build_graph.py — el grafo HETEROGÉNEO

```
transacción  <--->  [uid] [card] [email] [device] [net]
```

Bipartito. **Las transacciones no se conectan entre sí**: son vecinas si cuelgan
del mismo nodo de entidad. Diez tipos de arista (cinco entidades × dos
sentidos).

### Cómo se fabrica una entidad

```yaml
uid:    {cols: ["card1", "addr1"], usa_d1: true}      # el cliente real
card:   {cols: ["card1","card2","card3","card5"]}
email:  {cols: ["P_emaildomain","card1"]}             # card1 evita el hub de gmail
device: {cols: ["DeviceInfo","id_30","id_31","id_33"]}
net:    {cols: ["id_13","id_17","id_19","id_20"]}
```

`usa_d1` añade `día − D1` a la clave. D1 es «días desde el primer uso de la
tarjeta», así que `día − D1` es **constante por cliente**: separa a dos personas
que colisionen en el mismo `card1`+`addr1`.

**Regla de nulos: si cualquier columna de la clave es nula, no hay nodo ni
arista.** Nunca se rellena con el texto `"nan"` — hacerlo convertía el patrón de
ausencia en parte de la identidad y partía a un cliente en dos grupos.

### Las dos podas, y por qué son asimétricas

```yaml
max_entity_degree: 0      # E2 · 0 = SIN poda (recomendado)
min_previas_entidad: 1    # E1 · grado mínimo, solo en la BAJADA
```

**`max_entity_degree: 0`.** La poda por grado era redundante con el muestreo:
`vecinos_por_entidad: 10` hace que el vector de una entidad se calcule siempre
con 10 transacciones, tenga 30 o 4.887. Y costaba caro — medido con el tope
en 500:

```
72 entidades de 10.106 (0,7%) dejaban sin arista al 24,6% del dataset
esas transacciones tenían MÁS fraude que la media (3,39% vs 3,11%)
__grado_card se les quedaba en 0, el MISMO valor que una fila sin card1
```

Esa última es la peor: la red no distinguía «no tengo tarjeta» de «mi tarjeta
lleva 3.000 compras». La feature quedaba **invertida** justo donde más historial
hay. Al quitar la poda, `__grado_card` pasó de 31,2% de ceros a 6,6% y su máximo
de `log1p(500)=6,22` a 8,49.

**`min_previas_entidad: 1`, y solo en una dirección:**

```
SUBIDA  transaction -> entidad    TODAS
        quitar la primera compra de un cliente dejaría a las
        siguientes sin saber de ella

BAJADA  entidad -> transaction    solo con `previas >= minimo`
        sin nadie delante, lo único que llega es el eco propio
```

En `uid` eso afectaba a 59.189 de sus 94.901 nodos (62%), o sea al 26,8% del
dataset: entidades de una sola transacción cuyo «vecindario» era esa misma
transacción.

Las dos podas son **causales**: cuentan las transacciones ANTERIORES, no las que
la entidad acabará teniendo. Si contaran el total, la actividad de junio
decidiría si una compra de enero tiene vecinos.

### El orden de las aristas NO es cosmético

```python
orden = torch.argsort(ei[0] * (int(t.max()) + 1) + t, stable=True)
```

`temporal_strategy="last"` coge un **sufijo** de las aristas de cada entidad. Si
no están ordenadas por (entidad, tiempo), «los 10 más recientes» devuelve
cualquier cosa y **nadie avisa**. `stable=True` porque hay empates de
`TransactionDT`.

Lo guarda `tests/test_salud_grafo.py`.

## Cifras de referencia (piloto de 2 meses, 220.806 transacciones)

```
entidad   nodos    subida   bajada   cob%   grado medio   grado máx
uid       94.901  197.058  102.157   89,2       2,1           177
card      10.106  216.343  206.237   98,0      21,4         4.887
email     24.647  184.999  160.352   83,8       7,5         2.109
device     2.840   44.056   41.216   19,9      15,5         2.249
net        4.040   59.491   55.451   26,9      14,7         1.247

features por nodo: 70   (65 tras la ablación + __hora_dia, __delta_anterior
                         y los 5 __grado_*)
transacciones sin recibir de ninguna entidad: 6.884 (3,1%)
```

**`uid` sigue siendo la entidad débil**: 2,1 transacciones por entidad de media.
La clave es tan específica que casi no conecta. Es candidata a revisión.

`device` y `net` cubren solo el 20-27%. Es correcto y hay que declararlo.

## Si tocas esto, revisa

- **Cambiar el orden de filas o nodos** → rompe el contrato de posición.
- **Cambiar `graph.entidades`** → cambian los tipos de arista, los `__grado_*`,
  `gnn.in_dim` y el ancho de las cabezas.
- **Cambiar `feature_cols` o la ablación** → `gnn.in_dim` y todas las variantes
  de `hybrid/head.py`.
- **Tocar las podas** → corre `tests/test_grado_minimo_entidad.py` y
  `tests/test_poda_grado_maximo.py`, que comprueban el grafo **contra
  `full.parquet`**, no contra sí mismo.
- **Cambiar el orden de las aristas** → `temporal_strategy` deja de funcionar en
  silencio.

## Aprendido a base de romperlo

- **`EDGE_RAW_COLS` estaba escrito a mano** y `device` y `net` acabaron con cero
  nodos sin que nada avisara. Ahora las columnas salen de `graph.entidades`.
- **`__grado_*` y la poda contaban el futuro.** Una fila de enero llevaba el
  grado que su tarjeta alcanzaría en junio. Se hicieron causales con
  `_previas_por_entidad`, que se calcula **una vez** y sirve para las tres cosas.
- **`np.ascontiguousarray()`**: `.T` y `[::-1]` devuelven vistas en orden
  Fortran, `torch.tensor()` preserva los strides y el sampler nativo falla con
  `Input should be contiguous`.
- **Un aviso que salta siempre enseña a ignorarlo.** El de «entidad muy GRUESA»
  era `media > tope/2`; con `tope = 0` eso es `media > 0` y saltaba en las cinco
  entidades, incluida `uid` con 2,1 —que es lo contrario de gruesa—.
