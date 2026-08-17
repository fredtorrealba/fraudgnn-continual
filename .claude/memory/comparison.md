# comparison — el veredicto y los resultados

Compara sobre `examen`, que **no ha entrenado ni validado nada**. Es donde una
comparación mal montada convierte un resultado en humo.

## Artefactos

| Archivo | Lo escribe | Lo lee |
|---|---|---|
| `reports/final_comparison.json` | `final_comparison.py` | `resumen.py` |
| `reports/resumen.json` | `resumen.py` | el humano |

`resumen.py` lo llama `final_comparison` al terminar: consolida las **seis
métricas uniformes** (recall, precisión, PR-AUC, ROC-AUC, F1, accuracy) en los
tres puntos —redes, cabezas, examen— más el recall@2% aparte.

## Las reglas de comparación

**1. A umbral fijo NO se compara.** Modelos con calibraciones distintas emiten
volúmenes de alerta incomparables. Se reporta el 0.5 como referencia histórica,
nunca como comparación.

**2. A igual presupuesto de alertas** — el coste de revisión es el mismo para
todos. Es la comparación con sentido operativo, y la que fija el umbral.

**3. A igual precisión** — a igual calidad de alerta, ¿quién recupera más fraude?

**4. Con la ventana de entrenamiento igualada.** Es la que más veces se olvida.
El +0.0626 del híbrido sobre el baseline resultó ser un mes extra de datos.

**5. Bootstrap emparejado** (1000 réplicas) sobre el examen. Sin el intervalo, un
delta de 0.005 parece un resultado; con él se ve que cruza el cero.

## El accuracy no compara, y el módulo lo dice

```
no alertar nunca da accuracy 0.9588
```

Esa línea sale en el informe a propósito. Con 4,12% de fraude, un modelo que no
alerta jamás saca 0.959. **No borres ese aviso.**

## guard_omp() primero

Este módulo carga torch **y** xgboost en el mismo proceso: en macOS hay que
llamarlo antes de los imports pesados o el intérprete muere con SIGSEGV.

## Si tocas esto, revisa

- **Añadir una variante** → entra en el barrido solo si está en el dict
  `sistemas`, y hay que añadirla también en `resumen.py`.
- **Cambiar `umbral_por_presupuesto`** → viene de `hybrid/head.py`.
- **Cambiar el bloque `examen`** → toda la sección de resultados de aquí queda
  obsoleta.

---

# RESULTADOS

## ⚠️ La corrida 2026-08-17 está CONTAMINADA — repetir embed→heads→final

El veredicto de abajo (−0.0201 significativo) se midió con un embedding
DEFECTUOSO: `embed.py` pasaba el cfg global a `embed_and_score_nodes`, así que
el loader muestreó los 2 saltos del config mientras la red ganadora (256×3,
primera vez que gana 3 capas) esperaba 3. Entrenó viendo 3 saltos y describió
viendo 2: la tercera capa agregó vecindarios truncados. Arreglado en embed.py
(pasa `c` de `cfg_arquitectura`) con guarda en `validate.py` que ahora revienta
ante el mismatch. **Los números de la GNN (sección 1) siguen siendo válidos**
— el bug solo afecta al parquet del embedding, no al entrenamiento ni a la
selección (`compare_gnns` ya hacía `cfg = c`). Rehacer solo embed→heads→final.

## El estado de la corrida 2026-08-17 (entrada v2 — veredicto PENDIENTE de repetir)

```
EXAMEN — 21.284 txn · 876 fraudes (4,12%)

                   PR-AUC   ROC      recall@2%
control            0.3341   0.8321     0.2614
gnn_mas_tabular    0.3140   0.8241     0.2352
solo_gnn           0.2078   0.7375     0.1884
gnn_sola           0.2024   0.7166     0.1667

APORTE DEL GRAFO   -0.0201   IC95 [-0.0349, -0.0043]   SIGNIFICATIVO (negativo)
                   P(delta > 0) = 0.013
```

**Dos hechos a la vez, y no se contradicen:**

1. **La normalización desbloqueó la RED.** `gnn_sola` pasó de ROC 0.512 (azar)
   a 0.717 en el examen y su PR-AUC ×4 (0.048→0.202); la validación de
   graphsage ×3 (PR-AUC 0.06→0.20). Y por primera vez Optuna eligió la red
   grande (256×3 capas, antes «capas 3 nunca ha ganado»): con la entrada sana,
   la capacidad extra por fin sirve de algo.
2. **Y el grafo como AÑADIDO pasó de neutro a dañino medible.** El embedding
   nuevo es 64 dims (mlp_head_dim de Optuna) y mucho más fuerte; la cabeza lo
   usa de verdad (46,7% de la ganancia, las 64 columnas) — y pierde en
   cabezas_validan (0.376 vs 0.413) Y en examen (0.314 vs 0.334). Es el patrón
   ya documentado en gnn.md: cuanto más señal por dimensión, más redundante con
   lo tabular y más arrastra el drift temporal (la cabeza calibra confianza en
   una señal que se degrada hacia el examen). El precedente exacto: el emb_ de
   64 cols dio −0.0325.

`control` apenas se movió (0.3396→0.3341) pese a sumar 20 flags `__na` y los 5
`__grado_*` — pero los USA: `id_03__na`, `dist1__na` y `__grado_uid` están en
su top-10 de ganancia. Reordenó su importancia sin mover el techo.

**Cabo suelto de la corrida:** `best_epoch: 1` en las 6 redes (antes 1-4, con
epochs 10 y patience 4). La red toca techo en la primera época. Candidatos: lr
de Optuna (3.6e-4) sobre entrada ya normalizada, o que 10 épocas con paciencia
4 corten antes de ver una segunda subida. Investigar antes del A/B de
RankGauss/LayerNorm.

## El estado anterior (referencia, hasta 2026-08-16)

```
                   PR-AUC   ROC      recall   prec     F1       recall@2%
control            0.3396   0.8313   0.2489   0.5117   0.3349     0.2489
gnn_mas_tabular    0.3339   0.8347   0.2443   0.5023   0.3287     0.2443
solo_gnn           0.1027   0.6770   0.0856   0.1761   0.1152     0.0856
gnn_sola           0.0810   0.6436   0.0753   0.1549   0.1014     0.0753

APORTE DEL GRAFO   -0.0058   IC95 [-0.0209, +0.0089]   NO significativo
```

## Cómo se llegó aquí: tres artefactos descartados

La historia importa porque el primer número parecía concluyente y era falso.

| corrida | aporte | IC95 | ¿significativo? |
|---|---|---|---|
| `por_cabeza` + overrides a mano | −0.0668 | [−0.0819, −0.0522] | **sí** |
| sin overrides (A1) | −0.0110 | [−0.0210, −0.0017] | **sí** |
| + `optuna_modo: compartido` (A3) | −0.0012 | [−0.0093, +0.0060] | no |
| + E0/E1/E2, hiperparámetros nuevos | −0.0058 | [−0.0209, +0.0089] | no |
| + entrada v2 (normalización causal, flags, grados, embedding 64d) | −0.0201 | [−0.0349, −0.0043] | **CONTAMINADA** (embed con 2 saltos para red de 3 capas) |

La última fila NO es un artefacto de medición como las dos primeras: la
simetría se mantuvo (misma ablación, Optuna compartido, grados a las tres). Lo
que cambió es el embedding — más fuerte, más ancho (64d) y con más drift que
descontar. El resultado desplaza la pregunta: ya no es «¿el grafo aporta?» sino
«¿por qué una señal de grafo 4× mejor RESTA al combinarla?». Las hipótesis
vivas: redundancia con lo tabular + caducidad temporal (medida) + 64 columnas
que XGBoost sobreajusta en su ventana. La poda por consistencia temporal
(MEJORAS punto 6) y un `mlp_head_dim` menor son los siguientes controles
baratos.

**El 98% del «daño del grafo» era artefacto de medición**: hiperparámetros
puestos a mano en dos de las tres cabezas, y búsqueda por cabeza sobreajustando
la ventana de validación.

## Lo que se puede afirmar

> Con muestreo causal verificado, embedding de vecinos verificado, grafo
> completo sin podas arbitrarias, hiperparámetros compartidos y ablación
> simétrica, **añadir el grafo a IEEE-CIS no produce diferencia medible**:
> −0.0058, IC95 [−0.0209, +0.0089].

Es un resultado nulo bien medido. No es lo que se buscaba, pero es defendible y
nadie puede decir que no se probó bien.

## Los tres datos que lo explican

**1. El grafo se deriva de columnas que XGBoost ya tiene.** Las claves de
entidad (`card1/2/3/5`, `addr1`, `P_emaildomain`, `DeviceInfo`, `id_*`) están
dentro de `feature_cols`, codificadas. El grafo codifica información ya
disponible para el modelo tabular.

**2. El embedding aporta el 3,2% de la ganancia** de `gnn_mas_tabular`, usando
sus 16 columnas. Antes de arreglar E0 se llevaba el 15,9% — pero eran
proyecciones lineales de las features propias, útiles a los árboles como cortes
oblicuos. **La información real del grafo vale menos que el artefacto que la
sustituía.**

**3. Y cuando el grafo llega de verdad, no ayuda:**

```
gnn_score   (camino completo, 1 col)     -0.0003
emb_        (camino completo, 64 cols)   -0.0325
embv_       (sin info de vecinos)        +0.0053   ← el único positivo, y no es grafo
```

## La pista que sigue viva

```
cabezas_validan   control 0.4136  ·  híbrido 0.4336    el híbrido GANA (+0.0200)
examen            control 0.3396  ·  híbrido 0.3339    el híbrido PIERDE
```

El híbrido gana claramente en validación y pierde en el examen. Y en el examen
gana a presupuestos amplios:

```
 5% alertas   control 0.3813  ·  híbrido 0.4155
25% alertas   control 0.7180  ·  híbrido 0.7432
ROC           control 0.8313  ·  híbrido 0.8347
```

Ordena mejor en conjunto; falla en la punta, que es donde vive el PR-AUC. Eso es
coherente con la limitación temporal declarada en `CLAUDE.md`: el embedding
pierde el 38% de su margen entre la ventana donde la cabeza aprende a confiar en
él y la ventana donde se examina.

## GraphSAGE vs GATv2

Este sí salió limpio, y en dos búsquedas independientes:

```
graphsage  PR-AUC walk-forward 0.0638 ± 0.0023
gatv2                          0.0546 ± 0.0008
```

Sin solape entre los grupos de tres semillas. Y las dos eligieron **`capas 2`**,
la profundidad mínima: ninguna red de 3 capas sobrevivió a la selección pese a
costar hasta 17 veces más.

## Lo que NO se ha medido

- **Refit dentro del piloto.** Con 60 días no hay hueco sin romper el
  aislamiento. Va en la corrida de 6 meses.
- **El eco parcial** (E1b): con `uid` de 3 transacciones, la raíz se recibe a sí
  misma diluida a 1/3. `min_previas_entidad: 3` lo bajaría al 25% a costa de más
  de la mitad de las aristas de `uid`. Sin medir.
- **Continual learning**: desactivado en esta fase.
