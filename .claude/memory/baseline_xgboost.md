# baseline_xgboost — ahora es una librería, no una etapa

**El módulo ya no produce ningún modelo propio.** La etapa `xgboost` salió del
pipeline y `models/xgboost_baseline.json` no existe ni está versionado.

## Por qué desapareció

El baseline congelado servía cuando se comparaba «GNN sobre meses 1-5» contra
«XGBoost sobre meses 1-4»: dos ventanas distintas, y la diferencia no era
atribuible a nada. Ese diseño ya dio un falso +0.0626 que resultó ser el mes
extra de datos.

Ahora **la vara de medir es la cabeza `control`**: mismas 65 columnas, misma
ventana `cabezas_entrenan`, mismo SMOTE, mismos hiperparámetros de Optuna y el
mismo umbral por presupuesto que las otras dos cabezas. La comparación
`control` vs `gnn_mas_tabular` cambia **una sola cosa**: las columnas del grafo.

Un baseline con distinto trato no es un baseline, es otro experimento.

## Qué se sigue usando de aquí

`hybrid/train_head.py` importa cuatro cosas. **No las dupliques:**

```python
apply_smote          # el mismo SMOTE para las tres cabezas
objective_factory    # el mismo espacio de búsqueda de Optuna
inferir_en_cpu       # pasa el booster a CPU para predecir
xgb_device           # auto | cuda | cpu
```

## smote_pipeline.py

SMOTE **solo sobre train**; validación y examen quedan con la distribución real.
Si apareciera un SMOTE después de la etapa `final`, sería un problema grave.

```yaml
sampling_strategy: 0.5    # la minoría llega al 50% de la mayoría
k_neighbors: 5
```

En el piloto: `2.641 → 41.215` fraudes, de 3,10% a 33,3%.

Se midió contra `scale_pos_weight` con los mismos hiperparámetros y ganó SMOTE
por 0.024 de PR-AUC, por eso se mantiene.

**Coste asumido, y hay que declararlo:** para `gnn_mas_tabular`, SMOTE interpola
también sobre las columnas del embedding, así que fabrica fraudes con vectores de
vecindario que son promedios de dos vecindarios reales — y eso no corresponde a
ningún subgrafo que exista. `control` solo interpola columnas tabulares. Es una
diferencia real entre las cabezas.

**SMOTE rechaza `NaN`.** Por eso el preprocesamiento usa `-1` como centinela.

**La GNN no puede usarlo**: un nodo sintético no tiene aristas. Usa `pos_weight`
en la loss, o `pos_weight_con_balanceo: 1.0` cuando las semillas ya vienen
balanceadas (si no, el desbalance se corrige dos veces).

Lo guarda `tests/test_smote.py`.

## train_xgboost.py

- `objective_factory(...)` — el objetivo de Optuna. **Lo reutiliza
  `hybrid/train_head.py`**: no dupliques el espacio de búsqueda.
- `xgb_device(cfg)` — con `auto` importa torch y mira `torch.cuda.is_available()`.
- `inferir_en_cpu(modelo)` — pasa el booster a CPU para **inferencia**.

### Por qué la inferencia va en CPU a propósito

Entrenar con `device="cuda"` deja el booster en `cuda:0`; al predecir sobre
arrays de numpy XGBoost avisa de *mismatched devices* y cae a construir un
`DMatrix` intermedio. Y recorrer árboles es ramificación pura y accesos
dispersos — justo lo que una GPU hace mal.

Es **exacto**: lo que difiere entre GPU y CPU es la *construcción* de histogramas
al entrenar, no la predicción sobre un árbol ya construido. Verificado:
predicciones bit-idénticas.

## Cargar modelos: usa `Booster`, no `XGBClassifier`

El wrapper de sklearn consulta `self._estimator_type`, que las versiones nuevas
de scikit-learn ya no definen, y `load_model` revienta con
`TypeError: _estimator_type undefined`. El `Booster` nativo es inmune. Lo hacen
así `hybrid/head.cargar()` y `comparison/final_comparison.py`.

## CUDA: la trampa del wheel

`requirements.txt` fija `xgboost>=3.0,<3.1` **por compatibilidad de driver**, no
por capricho. Las 3.4.x se compilan contra CUDA 13 y exigen driver 580+; los pods
de RunPod suelen traer 570.

Diagnosticar es engañoso:

- `build_info()["USE_CUDA"]` dice `True` aunque el wheel no arranque.
- `train(device="cuda")` **no lanza excepción**: avisa y cae a CPU en silencio.

Lo único fiable es entrenar de verdad y buscar el aviso `No visible GPU` — es lo
que hace `scripts/setup_runpod.sh`. Síntoma de no detectarlo: 30 trials de
Optuna pasan de ~7 min a ~50.

## Si tocas esto, revisa

- **`objective_factory`** → afecta a `hybrid/train_head.py`, que es su único
  consumidor real.
- **`apply_smote`** → las tres cabezas y `tests/test_smote.py`.
- **Resucitar la etapa `xgboost`** → antes pregúntate contra qué ventana va a
  entrenar. Si no es `cabezas_entrenan`, no es comparable con nada de este
  pipeline.
