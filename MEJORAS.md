# Guía de mejoras — lo que enseñan los ganadores de Kaggle

Qué hicieron los que ganaron la competencia IEEE-CIS, qué de eso aplica aquí, y
en qué orden conviene probarlo.

Fuentes: [NVIDIA — Leveraging ML to Detect Fraud](https://developer.nvidia.com/blog/leveraging-machine-learning-to-detect-fraud-tips-to-developing-a-winning-kaggle-solution/)
· [1st Place Solution Part 2](https://www.kaggle.com/competitions/ieee-fraud-detection/writeups/fraudsquad-1st-place-solution-part-2)

---

## Lo que hicieron

```python
X['day'] = X.TransactionDT / (24*60*60)
X['UID'] = X.card1_addr1.astype(str) + '_' + np.floor(X.day - X.D1).astype(str)
```

**Es exactamente nuestra clave `uid`**: `card1 + addr1 + (día − D1)`, letra por
letra. La misma idea a la que llegó este proyecto por su cuenta.

Tres decisiones alrededor de esa clave:

1. **El UID crudo NO entró como feature** — *«to avoid overfitting»*. Solo se usó
   para agrupar.
2. **45+ features de agregación sobre el grupo**: `UID_FE` (conteo),
   `TransactionAmt_UID_mean`, `TransactionAmt_UID_std`, `D4_UID_mean`,
   `D9_UID_std`, `D10_UID_mean`, `C1_UID_mean` … `C14_UID_mean`,
   `UID_P_emaildomain_ct`, `UID_V314_ct`.
3. **Post-procesado**: *«replace all predictions from one UID value with that
   client's average prediction»*.

Y la conclusión de fondo, que es la que importa: **el fraude en este dataset es
de CLIENTE, no de transacción.** Lo dice el 2º lugar sin rodeos:

> *«Almost all my chains have mean fraud 0.0 or 1.0»* — Tony, sobre las cadenas
> de transacciones de una misma tarjeta física.

Ganar consistía en identificar al cliente y decidir sobre el grupo.

### Dos hallazgos que confirman nuestra hipótesis de la ablación

**Las columnas V SON agregaciones por cliente.** Tony desanonimizó `V307` como
`cumsum(TransactionAmt)` del titular, y encontró pares contador/suma entre las V:
`V307 = V310 + V317 + V320`, con contador `V294 + V298 + V285`.

Eso convierte nuestra hipótesis en un hecho documentado: quitar V/C/D **es**
quitar los agregados de cliente precalculados. La pregunta del capstone —¿puede
el grafo reconstruirlos?— está bien planteada, y ahora se puede citar.

---

## Las agregaciones no son estadísticos: son detectores de colisión

Esto es lo más fino de la solución ganadora y es fácil pasarlo por alto:

> *«If a specific uid has std=0 for D15n, then we know that all of its D15n are
> the same. **If std!=0 then that specific uid actually contains 2 or more
> clients and our model will split it up.**»*
>
> *«If uid_FE != AG(C13, uid, nunique) then this uid contains 2 or more clients
> and our model splits it up.»*

`D15n` (= `día − D15`) debería ser **constante dentro de un cliente**. Si su
desviación no es cero, el grupo `uid` no es un cliente: son dos que colisionaron
en la misma clave.

No están describiendo el grupo. Están dándole al modelo **una señal de calidad
del propio agrupamiento**, para que aprenda a desconfiar de los grupos sucios.

Nosotros no tenemos nada de eso: nuestro `uid` puede estar mezclando clientes y
ni el modelo ni nosotros nos enteramos.

## La idea que cuesta entender: predicción vs embedding

Es el punto clave, y merece la pena pararse.

**Lo que hace nuestra GNN:**

```
las 65 columnas de mis vecinos
        ↓  la red decide qué resumir
   16 números (el embedding)
        ↓
     XGBoost
```

La red tiene que **aprender, desde cero y a través de un cuello de 16
dimensiones, qué del vecindario importa** — y lo tiene que aprender *antes* de
saber para qué sirve, porque la cabeza XGBoost viene después.

**Lo que hicieron los ganadores:**

```
el modelo puntúa cada transacción      →   0.91  0.88  0.12  0.05
            ↓  promedio por grupo uid
todas las del grupo reciben            →   0.49  0.49  0.49  0.49
```

No resumen las *features* del vecindario. Resumen **la respuesta**. Y la
respuesta ya está en la escala correcta, ya sabe qué importa, y no pasa por
ningún cuello de botella.

**En una frase:** nosotros le pedimos a la red que adivine qué resumir; ellos
resumen directamente aquello que quieren saber.

Por eso la prueba número 3 de la lista es tan barata y tan informativa: si
promediar predicciones por uid mejora el resultado, entonces **la señal de
cliente está ahí y el problema es cómo la comprimimos**, no el grafo.

---

## Lo que ya estamos haciendo bien

- **La clave `uid` es la correcta.** `card1 + addr1 + (día − D1)`, idéntica.
- **El uid crudo no es una feature.** Verificado: no hay ninguna columna con
  `uid` en `feature_cols.json`. Solo se usa para construir aristas. ✔
- **Las agregaciones son causales.** `_previas_por_entidad` cuenta lo anterior,
  nunca el total. Los ganadores también insistieron en *«strictly adhering to
  temporal constraints»*.
- **`aggr=["mean","max","std"]`** en GraphSAGE es, conceptualmente, lo mismo que
  sus `_mean` y `_std` por grupo.

## Y el problema de fondo

```
94.901 nodos uid para 220.806 transacciones  →  2,1 de media
62% con UNA sola transacción
```

Los ganadores construyen todo sobre grupos con muchas transacciones. Con grupos
de 2, `TransactionAmt_UID_mean` es **casi la propia transacción**: no hay nada
que agregar.

No es un fallo de la clave —es la suya— sino de la ventana: ellos tenían 6 meses
y 590K transacciones, nosotros 2 meses y 220K. **Ese es el techo de todo lo
demás.**

---

# CHECKLIST — de lo más fácil a lo más difícil

## [~] 0 · Partir el examen en clientes CONOCIDOS y NUEVOS — PARCIAL

**Esfuerzo:** bajo. No reentrena nada: es reagrupar métricas que ya existen.

> **Ya reporta** `nuevo` (0 anteriores) / `conocido` (1-2) / `habitual` (3+)
> por `__grado_uid` causal (`final_comparison.py:GRUPOS_HISTORIAL`). Falta el
> grupo `dudoso` (entidad que colisiona), que depende de los detectores de
> colisión del punto 4.

Es lo más valioso que sacamos de los ganadores y no lo estábamos haciendo.

> Entrenando con 5 meses y prediciendo el último:
> · XGB fue el mejor en **UIDs conocidos**: AUC = 0.99723
> · LGBM fue el mejor en **UIDs desconocidos**: AUC = 0.92117
> · CAT fue el mejor en **UIDs dudosos**: AUC = 0.98834

**Ocho puntos de AUC** entre clientes vistos y no vistos. Y ellos elegían el
modelo por población, no en promedio.

Nosotros reportamos **un solo número promediado sobre las dos poblaciones**. Y
el grafo, por construcción, solo puede ayudar donde el cliente tiene historial:

```
62% de los uid tienen UNA sola transacción
```

Estamos midiendo el aporte del grafo mayormente donde el grafo **no puede hacer
nada**. Es evaluar a un traductor con páginas en blanco.

**Qué hacer:** en `comparison/final_comparison.py`, partir `examen` en tres
grupos según cuántas transacciones anteriores tiene el `uid` de cada fila —dato
que ya está en `__grado_uid`— y reportar las métricas por grupo:

```
nuevo        0 anteriores       el grafo no tiene nada que decir
conocido    >=1 anterior        aquí es donde puede aportar
dudoso      la entidad colisiona (ver el punto 5)
```

**Qué se aprende:**

```
el híbrido gana en "conocido" y pierde en "nuevo"
    → el grafo SÍ aporta, y el promedio lo estaba escondiendo
    → la conclusión del capstone cambia por completo

no gana ni en "conocido"
    → el resultado nulo es sólido y ya no admite esta objeción
```

Sea cual sea la respuesta, **es el análisis que hace defendible el resultado**.
Sin él, cualquier tribunal puede preguntar: *«¿y en los clientes con historial?»*
y hoy no sabemos contestar.

---

## [ ] 1 · Verificar que el uid crudo no se cuela — YA CUMPLIDO

**Esfuerzo:** ninguno, es una comprobación.

```bash
python3 -c "
import json; c=json.load(open('data/processed/feature_cols.json'))['feature_cols']
print([x for x in c if 'uid' in x.lower()])"     # debe salir []
```

Los ganadores excluyeron el UID crudo por sobreajuste: es un identificador casi
único, y un árbol que parte por él memoriza clientes en vez de aprender patrones.
En producción, un cliente nuevo no tiene ese valor y el modelo no sabe qué hacer.

Aquí nunca entró: el `uid` solo se usa para construir aristas. **Conviene dejarlo
escrito como decisión**, no como accidente — es un punto a favor en la defensa.

> **Cuidado si algún día se añade:** cualquier feature que identifique al cliente
> de forma casi única (el hash del uid, un embedding aprendido por entidad) tiene
> el mismo problema. Los nodos de entidad de este proyecto entran en ceros justo
> por eso, y es lo que mantiene el modelo inductivo.

---

## [x] 2 · Dar los `__grado_*` a XGBoost — HECHO 2026-08-16

**Esfuerzo:** bajo. Las columnas **ya están calculadas**, solo no llegan.

> **Implementado:** `build_graph` escribe `processed/grados_entidad.parquet` y
> `hybrid/head.py:cargar_tabla()` lo suma a `cols_base` para las TRES cabezas,
> después de la ablación y alineado por posición con su assert. Si el parquet
> falta, WARNING y la corrida queda asimétrica: reconstruir `graph`.

```
__grado_uid = log1p(transacciones ANTERIORES de ese uid)
```

**Eso es exactamente el `UID_FE` de los ganadores**, la más básica de sus 45
features de agregación. Y hoy:

```
la GNN         SÍ la ve   (va en los 70 features del nodo)
XGBoost        NO la ve   (no está en feature_cols.json, 433 columnas)
```

Hay cinco: `__grado_uid`, `__grado_card`, `__grado_email`, `__grado_device`,
`__grado_net`.

**Qué hacer:** que `build_graph` las escriba también a un parquet que lea
`hybrid/head.py:cargar_tabla()`, y añadirlas a `cols_base`.

**Por qué importa para el experimento:** ahora mismo `control` compite sin una
feature que el híbrido sí tiene (dentro del embedding). Dárselas a las **tres
cabezas** cierra ese hueco y hace la comparación más limpia — probablemente
suba `control` y baje el aporte aparente del grafo, pero será el número honesto.

**Riesgo:** ninguno para la validez; es simétrico.

---

## [ ] 3 · Post-procesado por grupo uid

**Esfuerzo:** bajo. ~20 líneas, **no reentrena nada**.

Es la técnica del 1º lugar, aplicada a los scores que ya tenemos.

```python
# para cada transacción, promediar su score con el de las ANTERIORES de su uid
for t in examen:
    hermanas = [s for s in uid_de(t) if tiempo(s) <= tiempo(t)]
    score_final[t] = mezcla(score[t], media(score[hermanas]))
```

**La versión causal es obligatoria aquí.** Los ganadores promediaron sobre todo
el grupo (tenían el test entero); nosotros solo podemos usar las anteriores, o
sería mirar el futuro.

**Qué se aprende, gane o pierda:**

```
mejora  →  la señal de cliente EXISTE y el cuello es el embedding de 16 dims
           → la vía es simplificar la salida de la GNN, no complicarla

no mejora → con grupos de 2,1 transacciones no hay nada que promediar
           → confirma que el problema es la ventana (punto 5)
```

**Se puede probar sobre los scores archivados**, sin tocar la GNN. Es la prueba
con mejor relación información/coste de toda la lista.

**Parámetro a explorar:** el peso de la mezcla. `0.5·propio + 0.5·grupo` es el
punto de partida; los ganadores reemplazaban del todo (peso 1 al grupo), pero
ellos tenían grupos grandes.

### DOS ADVERTENCIAS, y las dos vienen de que ya les pasó

**1. Post-procesar, NO reentrenar con ello.** CPMP probó exactamente esto como
feature y falló:

> *«Para cada UID, crear una variable de retardo igual al promedio de las
> predicciones antes del punto actual, y volver a entrenar. Esto es como apilar
> pero con desplazamiento temporal. **Esto no funcionó: el CV se disparó pero el
> LB bajó 0.01.**»*

El post-procesado —aplicado al final, sin reentrenar— sí funcionó (+0.0016 en
privado). La diferencia es la clave: **como feature, el modelo aprende a
confiar en ella y se sobreajusta; como post-proceso, solo suaviza la salida.**

**2. El peso se ajusta en `cabezas_validan`, NUNCA en el examen.** El aviso es
de sggpls, del mismo equipo:

> *«Realicé que la idea del post-procesado es mala y lleva a fuerte sobreajuste,
> porque algunos umbrales se ajustaron con LB probing. Eso explica por qué
> tenemos tantos envíos.»*
>
> *«Más de la mitad de los envíos se hicieron para calibrar el post-procesado.»*

Nosotros no tenemos leaderboard al que sondear — tenemos `examen`, y es de un
solo uso. Si el peso se elige mirándolo, el resultado deja de valer.

---

## [ ] 4 · Agregaciones explícitas de uid como columnas

**Esfuerzo:** medio. Toca `preprocessing.py` o `build_graph.py` y añade columnas.

En vez de esperar a que la GNN aprenda a resumir el vecindario, **calcular el
resumen a mano** y dárselo a XGBoost:

```
uid_n              cuántas anteriores tiene         (= __grado_uid, punto 2)
uid_amt_mean       media de TransactionAmt anterior
uid_amt_std        desviación
uid_amt_ratio      TransactionAmt / uid_amt_mean    ← "¿hoy gasta raro?"
uid_delta_mean     media de __delta_anterior
uid_nunique_email  cuántos emails distintos ha usado
```

Todas **causales**: solo con las transacciones anteriores, igual que
`_previas_por_entidad`.

**Este es EL control del experimento.** Separa dos hipótesis que hoy no podemos
distinguir:

```
si estas columnas ganan y el embedding no
    → el grafo tiene señal y la GNN la comprime mal
si ninguna de las dos gana
    → no hay señal de cliente que recuperar en 2 meses
```

Sea cual sea el resultado, **es un hallazgo publicable en la memoria** y responde
la pregunta del capstone mejor que el diseño actual.

**Añade los detectores de colisión.** No basta con `mean` y `std` como resumen:
la `std` de algo que debería ser constante por cliente es una **señal de calidad
del grupo**. Si `std != 0`, ese `uid` mezcla dos clientes y el modelo debe
saberlo:

```
uid_amt_std        dispersión del gasto        resumen
uid_ncard_nunique  ¿cuántas card2/3/5 usa?     >1 = colisión
uid_naddr_nunique  ¿cuántas addr1?             >1 = colisión
uid_nemail_nunique ¿cuántos emails?            >1 = sospechoso
```

Es la técnica exacta del 1º lugar: *«if uid_FE != AG(C13, uid, nunique) then
this uid contains 2 or more clients and our model splits it up»*. Y para
nosotros vale doble, porque alimenta el grupo «dudoso» del punto 0.

**Ojo con la ablación:** las mejores agregaciones de los ganadores eran sobre C y
D (`C1_UID_mean`, `D4_UID_mean`), que `excluir_prefijos` elimina. Con la ablación
puesta solo se pueden agregar las 65 columnas restantes. Hay que decidir —y
declarar— si estas agregaciones nuevas se calculan antes o después de la
ablación. **Lo consistente es después**, o estaríamos devolviendo por la ventana
lo que quitamos por la puerta.

---

## [ ] 5 · Más relaciones: seis meses

**Esfuerzo:** alto. Recalcular todo, ~3× de datos, todas las etapas más lentas.

Es el techo de todo lo anterior.

```
                     nosotros (2 meses)   ganadores (6 meses)
transacciones            220.806               590.540
uid: txn por grupo           2,1                  ~6
uid con una sola            62%                  mucho menos
```

**Con grupos de 2,1 el paso de mensajes no tiene sustrato.** Una transacción
pregunta a su uid *«¿cómo son las otras compras de mi dueño?»* y la respuesta es
*«no hay otras»* en el 62% de los casos.

Todo lo que proponen los ganadores —agregaciones, post-procesado, decidir a nivel
de cliente— **presupone que el cliente tiene historial**. Con 2 meses, casi
ninguno lo tiene.

**Y además desbloquea el refit** (ver `CLAUDE.md`, *el embedding caduca*): con 6
meses hay hueco para reentrenar la GNN más cerca del examen sin romper el
aislamiento, que hoy es imposible.

**Antes de lanzarlo, medir cuánto crece la entidad:**

```bash
python tests/test_salud_grafo.py --informe --guardar 2meses.json
# cambiar data.meses y graph.meses, reconstruir
python tests/test_salud_grafo.py --informe --contra 2meses.json
```

Si `uid` no pasa de ~4 transacciones por grupo, los seis meses tampoco arreglan
el problema y hay que replantear la entidad.

---

## [ ] 6 · Consistencia temporal de las columnas del embedding

**Esfuerzo:** bajo. Es un test, no un cambio.

Un truco de selección de features del 1º lugar, aplicable tal cual a nuestro
embedding:

> *«Entrenar un modelo con UNA sola feature en el primer mes y predecir el
> último. El 95% eran consistentes, pero **el 5% de las columnas perjudicaban**:
> tenían AUC de 0.60 en entrenamiento y **0.40 en validación**.»*

AUC 0.40 es **peor que el azar**: esa columna encontró un patrón que existía en
el presente y se invierte en el futuro. No es ruido, es daño activo.

Ya sabemos que nuestro embedding se degrada (ROC 0.7387 → 0.6467), pero lo
medimos **en bloque**. Este test lo mide **columna a columna**, y puede que unas
pocas dimensiones sean las que arrastran al resto.

```
para cada emb_i / embv_i:
    AUC de esa sola columna en cabezas_entrenan
    AUC de esa sola columna en cabezas_validan
    si baja de 0.5 en la segunda  ->  candidata a eliminar
```

Se hace sobre el parquet, sin tocar la GNN. Y si funciona, es una poda de
columnas que mejora sin reentrenar nada.

---

## [ ] 7 · Validación con hueco temporal

**Esfuerzo:** medio. Cambia `config.ventanas` y obliga a recorrer todo.

Los dos equipos dejaron un **hueco deliberado** entre entrenamiento y validación:

```
CPMP                          1º lugar
0 | 2 3 4 5 6                 entrena 4, salta 1, predice 1
0 1 | 3 4 5 6                 entrena 2, salta 2, predice 2
0 1 2 | 4 5 6                 entrena 1, salta 4, predice 1
0 1 2 3 | 5 6
```

> *«Intenté simular el hecho de que existe un intervalo de tiempo significativo
> entre el entrenamiento y la prueba»* — CPMP

Nuestras cinco ventanas son **contiguas**. Eso mide *«¿acierta la semana que
viene?»*, que es una pregunta legítima, pero distinta de *«¿acierta dentro de un
mes?»* — que es lo que pasa cuando un modelo lleva tiempo en producción.

Y encaja con nuestra limitación declarada: el embedding pierde el 38% de su
margen entre donde entrena la red y el examen. Con hueco, ese deterioro se
mediría **por diseño** en vez de descubrirse al final.

Con dos meses no cabe. **Es otra cosa que desbloquean los seis meses.**

---

# Resumen

| # | Cambio | Esfuerzo | Qué responde |
|---|---|---|---|
| 0 | Partir examen: conocidos vs nuevos | parcial ~ | **¿ayuda el grafo donde PUEDE ayudar?** (falta `dudoso`) |
| 1 | El uid crudo no es feature | ninguno ✔ | ya cumplido, documentarlo |
| 2 | `__grado_*` para XGBoost | hecho ✔ | cierra una asimetría del experimento |
| 3 | Post-procesado por uid | bajo | **¿el cuello es el embedding?** |
| 6 | Consistencia temporal por columna | bajo | ¿hay dimensiones que hacen daño? |
| 4 | Agregaciones explícitas + colisión | medio | **¿hay señal de cliente, sí o no?** |
| 7 | Validación con hueco | medio | mide el deterioro por diseño |
| 5 | Seis meses | alto | hace que la entidad `uid` exista |

**Empezar por el 0.** Es barato, no reentrena nada, y puede cambiar la
conclusión del trabajo: si el híbrido gana en clientes conocidos y pierde en
nuevos, el promedio actual está escondiendo el resultado.

Después el 3 y el 4, que responden si el cuello es la compresión o la ausencia
de señal. Y el 5 determina si algo de esto puede funcionar de verdad.

Y una lectura que conviene tener presente para la defensa: los ganadores
demostraron que **el fraude aquí es de cliente, no de transacción**. Eso no
invalida el enfoque de grafos — lo justifica. Un grafo con nodos de cliente es
la arquitectura natural para ese problema. Lo que este piloto muestra es que
necesita clientes **con historial**, y en dos meses no los hay.
