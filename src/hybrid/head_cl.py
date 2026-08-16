"""
Adaptación de la cabeza XGBoost en cada ciclo de continual learning.

POR QUÉ WARM START Y NO REENTRENAR
La mezcla 40/60 de un ciclo son ~20 filas (los ciclos reales tuvieron 6-16
casos nuevos). Entrenar un XGBoost desde cero con 20 filas y 440 columnas no
produce un modelo, produce ruido. Y usar el buffer entero (10.000) tampoco:
está dimensionado para ACOMPAÑAR un fine-tuning, no para sustituir a las 470K
filas del entrenamiento original.

Warm start = conservar los árboles que ya hay y añadir unos pocos entrenados
sobre los casos nuevos, con learning rate reducido. Es el análogo exacto del
fine-tuning de la GNN: pocos pasos, tasa pequeña, partiendo del modelo vigente.

RIESGO ASUMIDO, Y CÓMO SE ACOTA
Un árbol ajustado sobre 20 filas parte un espacio de 440 dimensiones con muy
pocos puntos, y fuera de esas filas puede decir cualquier cosa. Se limita con
`max_depth` bajo y `min_child_weight` alto para que solo pueda hacer
correcciones groseras. Y sobre todo: la doble validación del ciclo mide el
sistema COMPLETO contra el set de control, así que si la cabeza se degrada el
ciclo no despliega. El mecanismo de seguridad ya existía.

IMPORTANTE: la GNN que se pasa aquí debe ser la RECIÉN AFINADA. La cabeza
aprende sobre los scores que va a recibir en producción, no sobre los del
modelo anterior.
"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.continual_learning.mixture import mezcla_40_60
from src.continual_learning.validate import embed_and_score_nodes
from src.utils.common import get_logger

log = get_logger("hybrid.head_cl")


def matriz_nodos(data, struct: np.ndarray, gnn, nodos: np.ndarray,
                 cfg, esperado: int | None = None) -> np.ndarray:
    """
    Columnas de esos nodos: features + estructurales + (gnn_score o embedding).

    `esperado` es el ancho que pide la cabeza (`booster.num_features()`). Con
    él se decide qué aporta la GNN, exactamente igual que en
    `HybridSystem.score`: si no coincidieran, el warm start entrenaría sobre
    una matriz con distinto número de columnas que el modelo que va a extender.
    """
    idx = np.asarray(nodos, dtype=np.int64)
    emb, g = embed_and_score_nodes(gnn, data, idx, cfg)
    base = data.x[idx].numpy().astype(np.float32)
    est = struct[idx]
    if esperado is None or esperado == base.shape[1] + est.shape[1] + 1:
        extra = g.reshape(-1, 1).astype(np.float32)
    else:
        extra = emb
    X = np.hstack([base, est, extra])
    assert esperado is None or X.shape[1] == esperado, (
        f"La cabeza espera {esperado} columnas y se le arman {X.shape[1]}")
    return X


def warm_start(booster, gnn, data, struct, adapt_nodes, buffer_nodes, cfg,
               overrides=None):
    """
    Añade rondas a la cabeza vigente sobre la mezcla 40/60. Devuelve
    (cabeza_nueva, info). Si algo impide adaptarla, devuelve la original.

    Usa `mezcla_40_60` con la semilla por defecto, la misma que acaba de usar
    `finetune()`: las dos piezas del sistema adaptan sobre FILAS IDÉNTICAS, no
    sobre dos muestras estadísticamente parecidas.
    """
    import xgboost as xgb

    ov = overrides or {}
    hc = (cfg.get("hybrid") or {}).get("cl_head", {})
    ft = cfg["continual_learning"]["finetune"]
    mix_new = ov.get("mix_new", ft["mix_new"])
    rondas = int(hc.get("rounds", 30))
    # El dial estabilidad-plasticidad también gobierna a la cabeza: si la GNN
    # tiene que frenar o acelerar, la cabeza la acompaña.
    lr = float(hc.get("lr_scale", 0.1)) * float(ov.get("lr_scale", 1.0))

    nodos = mezcla_40_60(np.asarray(adapt_nodes), np.asarray(buffer_nodes), mix_new)
    y = data.y.numpy()[nodos].astype(int)
    if len(np.unique(y)) < 2:
        log.warning("Mezcla con una sola clase (%d filas): la cabeza no se toca",
                    len(nodos))
        return booster, {"omitido": "una sola clase", "n_filas": int(len(nodos))}

    t0 = time.time()
    # El ancho lo manda la cabeza vigente: xgb_model= exige que la matriz de
    # las rondas nuevas tenga exactamente las mismas columnas que el modelo
    # que se extiende.
    X = matriz_nodos(data, struct, gnn, nodos, cfg, booster.num_features())
    dtrain = xgb.DMatrix(X, label=y)
    params = {"objective": "binary:logistic", "eval_metric": "auc",
              "tree_method": "hist", "learning_rate": lr,
              "max_depth": int(hc.get("max_depth", 3)),
              "min_child_weight": float(hc.get("min_child_weight", 5)),
              "nthread": 1, "seed": 42}
    nuevo = xgb.train(params, dtrain, num_boost_round=rondas, xgb_model=booster)

    info = {"n_filas": int(len(nodos)), "n_nuevos": int(len(adapt_nodes)),
            "rondas": rondas, "learning_rate": lr,
            "arboles_antes": booster.num_boosted_rounds(),
            "arboles_despues": nuevo.num_boosted_rounds(),
            "minutos": round((time.time() - t0) / 60, 2)}
    log.info("Cabeza: +%d árboles sobre %d filas (%d nuevos) | lr %.4f | %.2f min",
             info["arboles_despues"] - info["arboles_antes"], info["n_filas"],
             info["n_nuevos"], lr, info["minutos"])
    return nuevo, info
