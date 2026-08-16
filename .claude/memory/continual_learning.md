# continual_learning — desactivado, conservado

```python
CL_ACTIVO = False        # pipeline.py
```

Sale del flujo por defecto **a propósito**: esta fase compara enfoques, y para
eso `examen` tiene que ser test puro. Si el CL adaptara el modelo durante el
examen, las métricas dejarían de medir el enfoque y medirían la adaptación.

El código se conserva intacto y se reactiva cuando haya un ganador. Esta memoria
documenta cómo funciona y **qué hay que arreglar antes de encenderlo**.

## ANTES DE REACTIVARLO — deuda real

### 1. Dos módulos no compilan contra el `sampling.py` actual

```
finetune.py      make_neighbor_loader · fanouts · loader_opts
deep_retrain.py  make_neighbor_loader · fanouts · loader_opts
```

`sampling.py` exporta hoy `make_hetero_loader`, `fanouts_hetero` y
`loader_opts`. Los dos archivos llaman a nombres del grafo **homogéneo**, que ya
no existen. `finetune.py:46` tiene el import corregido con un
`# noqa (pendiente de portar)`, pero el cuerpo sigue usando la API vieja.

Portarlos exige más que renombrar: los fanouts heterogéneos son un `dict` por
tipo de arista, no una lista.

### 2. `system.py` y `head_cl.py` se escribieron para cuatro variantes numeradas

El esquema ahora es de **tres cabezas con nombre** (`control`, `solo_gnn`,
`gnn_mas_tabular`). `HybridSystem.score()` y `head_cl.matriz_nodos()` arman la
matriz de la cabeza en **dos sitios distintos**; el ancho lo dicta
`booster.num_features()`, nunca una constante.

### 3. El diseño temporal cambió de meses a semanas

El CL se escribió para «el mes 6, semana a semana». Ahora el examen es **una
semana** (M2S4, 21.284 filas, 876 fraudes). Con 4 ciclos de ~200 fraudes cada
uno el CL tenía poco margen; con uno solo, ninguno. **En el piloto de 2 meses el
CL no tiene sitio**: necesita los 6 meses.

## La regla de oro

```
ADAPTACIÓN (70%)   entrena  ->  después va SOLO al replay buffer
VERIFICACIÓN (30%) NUNCA entrena -> después va SOLO al set de control
```

**Buffer y control jamás se cruzan.** Si un caso entrenado cayera en el control,
la medición de olvido estaría contaminada y el sistema se autoaprobaría. Lo
garantiza `splitter.py` por construcción.

## Los módulos

| Archivo | Qué hace |
|---|---|
| `trigger.py` | fraude confirmado con score bajo = patrón que el modelo no tiene |
| `splitter.py` | el 70/30 que garantiza que buffer y control no se toquen |
| `replay_buffer.py` | 10.000 casos de *entrenamiento*, prioriza frontera (0.4-0.7) |
| `control_set.py` | 5.000 casos de *validación*, congelados el día 1, representativos |
| `mixture.py` | `mezcla_40_60` — 40% nuevos, 60% replay |
| `finetune.py` | LR por capa: el drift mueve la frontera, no la estructura |
| `validate.py` | la doble validación y `embed_and_score_nodes` |
| `cl_orchestrator.py` | el ciclo completo |
| `deep_retrain.py` | fuera del reintento automático, lo lanza el usuario |

`mezcla_40_60` está extraída para que **GNN y cabeza entrenen sobre filas
idénticas**, no sobre dos muestras parecidas. La usan `finetune.py` y
`hybrid/head_cl.py` con la misma semilla.

## La doble validación

```
¿APRENDIÓ?  recall sobre VERIFICACIÓN >= 0.70  Y mejor que el modelo anterior
¿OLVIDÓ?    recall sobre CONTROL no cae más de `control_max_drop`
despliega solo si pasa las DOS
```

El dial estabilidad-plasticidad es **un dial en dos direcciones**: si olvidó →
estabilidad (+buffer, −LR, congelar); si no aprendió → plasticidad (+nuevos,
+LR, descongelar). Si fallan **ambos**, el patrón contradice a los viejos y se
programa reentrenamiento profundo.

`validate_cycle` acepta un `nn.Module` **o un scorer ya preparado**, así que la
misma función valida la GNN sola y el híbrido sin ramas condicionales. Recibe el
**umbral como argumento**: los dos sistemas no comparten escala.

## `embed_and_score_nodes` — el punto más compartido

Devuelve los **dos** embeddings y el score en una sola pasada, porque el score
*es* `sigmoid(classifier(embedding))`.

- Reserva por nodos **únicos pedidos**, no por nodos del grafo: con 256
  dimensiones, dimensionarlo al grafo entero eran ~600 MB **por llamada**, y el
  CL la llama decenas de veces por semana.
- Pasa por `np.unique` + `inverse` porque `mezcla_40_60` puede traer duplicados:
  con un mapeo directo id→fila, las repeticiones quedaban en ceros **sin avisar**.

## El umbral se recalibra cada semana

No se congela: un equipo de revisión tiene capacidad constante, no un corte de
probabilidad constante. El umbral del mes 5 producía 1,07% de alertas en el mes 6
en vez del 2% configurado.

## Lo que este dataset dijo del CL

**0 de 4 ciclos desplegaron**, con y sin el umbral corregido. El motivo no es el
punto de operación: son **7-16 casos nuevos por semana**, muy pocos para mover
una red. Y el set de verificación son, por construcción, los fraudes que el
modelo **no** detectó — exigirle recall 0.70 sobre ellos tras 8 épocas con ~20
filas es una vara casi inalcanzable.

Es un hallazgo reportable, no un fallo.

## Si tocas esto, revisa

- **`embed_and_score_nodes`** → lo usan `hybrid/system.py` y `hybrid/head_cl.py`.
- **`mezcla_40_60`** → `finetune.py` y `hybrid/head_cl.py` **a la vez**, o
  dejarán de entrenar sobre las mismas filas.
- **Añadir un loader** → propaga `sin_aristas` desde `loader_opts(cfg)`.
- **Reactivar el CL** → los tres puntos de deuda de arriba, y el bloque `examen`
  deja de ser test puro: hay que declararlo.
