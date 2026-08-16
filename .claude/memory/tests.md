# tests — los invariantes que no dan síntomas

```bash
bash tests/run.sh
```

**Qué NO son:** un smoke test. `scripts/smoke_test.sh` comprueba que el pipeline
no revienta. Esto comprueba que no está devolviendo **respuestas equivocadas sin
reventar**, que es peor.

Cada test guarda un fallo que **ya ocurrió** y que no producía ningún síntoma:
ni excepción, ni warning, ni número raro. Solo un resultado plausible y falso.

## Los siete

| Test | Qué guarda | Necesita |
|---|---|---|
| `test_embedding_vecinos.py` | E0 · el embedding «solo vecinos» contiene vecinos | nada, 2 s |
| `test_grado_minimo_entidad.py` | E1 · la primera transacción de una entidad no recibe de ella | el grafo |
| `test_poda_grado_maximo.py` | E2 · el grafo tiene las aristas que dicen los datos | grafo + parquet |
| `test_salud_grafo.py` | ninguna entidad se cayó en silencio · **+ el informe** | el grafo |
| `test_smote.py` | SMOTE solo sintetiza fraude y respeta el ratio | la etapa `embed` |
| `test_causalidad_muestreo.py` | A2 · el muestreo solo mira hacia atrás | **`pyg-lib`** |
| `test_seleccion_vecinos.py` | baja las N más recientes anteriores | parcial sin `pyg-lib` |

## Tres estados, no dos

`run.sh` distingue:

```
OK        pasó
SALTADO   falta `pyg-lib` (macOS): no se puede llamar al sampler
PARCIAL   corrió, pero solo la mitad de las comprobaciones
```

El `PARCIAL` se añadió porque `test_seleccion_vecinos` **devolvía OK habiendo
comprobado solo la precondición**. Un test que dice OK sin haber mirado lo
importante es exactamente la mentira cómoda que estos tests existen para cazar.

## Qué comprueba cada uno, y por qué importa

### E0 — el embedding de vecinos

Construye un grafo de juguete, cambia las vecinas de `0.0` a `9.0` con la raíz
idéntica, y exige que el embedding se mueva.

```
código antiguo    0.000000     ← no reaccionaba
con el arreglo    2.137227
```

Este fallo estuvo activo **toda la primera fase**. `embv_` era
`constante + 4 proyecciones de las features propias` y cero información del
vecindario. El resultado del capstone se calcula con esas columnas, así que no
daba un error: daba **una respuesta equivocada a la pregunta de la tesis**.

Tres comprobaciones: reacciona al contenido de los vecinos, reacciona a cuántos
son, y se restan las **cinco** aristas entrantes (no una).

> Trampa que costó un rato: el `ModuleDict` de PyG devuelve claves distintas
> según cómo lo recorras. `for et in convs` da `'<transaction___en_uid___uid>'`
> (string serializado) y `.items()` da la tupla. Comparar `et[2] == "transaction"`
> sobre el string compara el **tercer carácter**.

### E1 y E2 — se comprueban contra `full.parquet`

Los dos recalculan lo que **debería** haber desde las columnas originales y lo
comparan con el grafo. **No verifican el grafo consigo mismo**: usar la misma
función que lo construyó aprobaría el mismo error dos veces.

E1 verifica el valor que haya en el config, **incluido `0`**. Una versión
anterior se saltaba el 0 en vez de comprobar que desactiva la poda, así que no
probaba lo que el comentario del config promete.

E2 verifica la cuenta exacta: `Σ min(minimo, grado)` por entidad para E1,
`previas <= tope` para E2. Comprobado en 0, 100, 500 y sin poda.

### salud del grafo — alarma, no termómetro

Umbrales **deliberadamente generosos**: salta si una entidad se queda sin
aristas o si más del 25% del dataset se desconecta, no si un número se mueve dos
décimas. Un test que salta con cada ajuste enseña a ignorarlo.

Trae además el **informe**, que es lo que sirve para decidir:

```bash
python tests/test_salud_grafo.py --informe
python tests/test_salud_grafo.py --informe --guardar antes.json
python tests/test_salud_grafo.py --informe --contra antes.json    # el diff
```

La decisión de E2 salió de ahí: `card +54.274 aristas`,
`__grado_card en cero -24,58 puntos`, `aisladas -5.396`.

### SMOTE

Que solo sintetice fraude (las legítimas exactamente iguales), que respete
`sampling_strategy` **leído del config**, sin NaN, sin colapso de duplicados y
con escala coherente.

Encontró al escribirlo que `solo_gnn` produce un **1,2% de sintéticos
duplicados**: dos transacciones con el mismo vecindario dan el mismo embedding,
así que interpolar entre ellas devuelve ese vector. No es un fallo —solo pasa con
columnas densas— pero está medido, y el umbral queda en 5%.

### A2 y recencia — solo completos en el pod

`pyg-lib` no tiene wheel para macOS arm64, y sin él PyG cae a un sampler que solo
sabe grafos homogéneos. No es velocidad ni GPU: es una librería que falta.

En el pod, verificados:

```
A2          123.980 vecinos comparados · 0 posteriores a su raíz
recencia      4.517 vecinas muestreadas · 0 fuera de las 10 más recientes
```

La comprobación de A2 es **por semilla**, no por lote: `time_attr` activa
`disjoint` y el tensor `batch` dice de qué semilla viene cada nodo. Contra el
máximo del lote la pasaría un muestreo que le diera a la transacción más antigua
los vecinos de la más reciente.

## Cómo escribir uno nuevo

1. **Que compruebe contra la fuente**, no contra el propio artefacto.
2. **Que falle de verdad.** Todos estos se validaron reintroduciendo el bug a
   propósito y comprobando que devuelven código 1. Un test que nunca ha fallado
   no sabes si funciona.
3. **Que diga dónde mirar**, no solo «MAL». El de E0 dice: *«se está capturando
   antes de que los nodos de entidad repartan su resumen; revisa `encode()`»*.
4. **Umbrales generosos.** Alarma, no termómetro.
5. **Que se salte con gracia** si le falta un artefacto o `pyg-lib`, sin marcar
   fallo — pero diciéndolo.

## Cuándo correrlos

- **Siempre antes de una corrida larga.** Segundos contra horas.
- **Después de tocar** `models.py:encode`, `build_graph.py`, `sampling.py` o las
  podas del config.
- **En el pod, después de cada `git pull`**: es donde A2 y recencia corren
  completos.
