"""
Paso 5 — el embedding de UNA sola red, para todas las filas que no entrenó.

QUÉ SUSTITUYE Y POR QUÉ
Esto ocupa el lugar de `oof.py`. El OOF entrenaba K redes y hacía que cada una
describiera el trozo que no había visto, para que ninguna fila llevara el
embedding de una red que la memorizó. La honestidad estaba bien resuelta —el
diagnóstico `auc_predecir_fold` salía 0.5000— pero el resultado era inservible:

    cada una de las K redes aprende SUS pesos, así que la dimensión 7 de su
    embedding mide algo distinto en cada una. La tabla que recibía XGBoost tenía
    las filas de los meses 1-4 escritas por 4 redes y las del mes 5 por una
    quinta: cinco idiomas en las mismas 32 columnas.

    MEDIDO: `gnn_mas_tabular` cortó por early stopping en 2 árboles (control:
    517) y su PR-AUC cayó de 0.4803 a 0.1159. `solo_gnn` (0.0614) quedó por
    DEBAJO de `gnn_sola` (0.1232), es decir, XGBoost sobre el embedding rendía
    menos que la propia red que lo produjo. Imposible si las columnas fueran
    coherentes.

LA SOLUCIÓN
Separar las ventanas: la GNN entrena en `gnn_entrena` y las cabezas en
`cabezas_entrenan`, que son bloques DISTINTOS. Entonces una sola red puede
describir todo lo que las cabezas ven —nunca lo memorizó— y las 32 columnas
significan lo mismo en toda la tabla. Sin K redes, sin folds, sin idiomas.

Es también lo que pasa en producción: una red describe, una cabeza decide.
Entrenar con cinco redes y servir con una era, además, un desajuste entre el
mundo del entrenamiento y el del despliegue.

Salida: data/processed/gnn_embed.parquet con node_idx, gnn_score y las dos
familias de 32 columnas (completo y solo-vecinos), igual que antes.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.gnn.models import TXN, build_model, cfg_arquitectura
from src.continual_learning.validate import embed_and_score_nodes
from src.utils.common import ensure_dirs, get_logger, load_config, resolve
from src.utils.ventanas import mascaras_grafo

log = get_logger("hybrid.embed")


def ruta_parquet(cfg) -> Path:
    return resolve(cfg, "processed_dir") / "gnn_embed.parquet"


def main():
    cfg = load_config()
    ensure_dirs(cfg)
    models_dir = resolve(cfg, "models_dir")

    with open(models_dir / "selected_model.json") as f:
        sel = json.load(f)["selection"]
    nombre, seed = sel["selected"], int(sel["seed"])
    ruta = models_dir / f"{nombre}_seed{seed}.pt"
    if not ruta.exists():
        raise SystemExit(f"Falta {ruta.name}: corre la etapa `gnn` primero.")

    data = torch.load(resolve(cfg, "graph_dir") / "graph.pt", weights_only=False)
    ck = torch.load(ruta, weights_only=False)
    c = cfg_arquitectura(nombre, cfg, ck)
    c["gnn"]["in_dim"] = ck["in_dim"]
    model = build_model(nombre, c, data.metadata())
    model.load_state_dict(ck["state_dict"])

    v = mascaras_grafo(cfg, data)
    entrenadas = v["gnn_entrena"].numpy()
    n_total = data[TXN].num_nodes

    # Se describen SOLO las ventanas que alguien usa, y nunca `gnn_entrena`:
    # esas filas llevarían un embedding memorizado. Los meses reservados para la
    # corrida grande no se tocan — describirlos costaba 4 veces más tiempo para
    # unas columnas que nadie lee.
    usa = np.zeros(len(entrenadas), dtype=bool)
    for n, m in v.items():
        if n != "gnn_entrena":
            usa |= m.numpy()
    describir = np.where(usa & ~entrenadas)[0]
    log.info("Red única: %s seed=%d | describe %d filas de %d",
             nombre, seed, len(describir), n_total)
    log.info("  %d entrenaron la red (memorizadas, se excluyen) | "
             "%d fuera de las ventanas (meses reservados)",
             int(entrenadas.sum()), int((~usa & ~entrenadas).sum()))

    t0 = time.time()
    emb, embv, sc = embed_and_score_nodes(model, data, describir, cfg)
    log.info("Embedding listo en %.1f min | %d dimensiones",
             (time.time() - t0) / 60, emb.shape[1])

    salida = {"node_idx": describir, "gnn_score": sc}
    salida.update({f"emb_{i}": emb[:, i] for i in range(emb.shape[1])})
    salida.update({f"embv_{i}": embv[:, i] for i in range(embv.shape[1])})
    pd.DataFrame(salida).to_parquet(ruta_parquet(cfg), index=False)

    # Cordura: el embedding debe tener escala parecida en los bloques que lo
    # usan. Aquí SIEMPRE la tiene, porque lo produce una sola red; el aviso
    # existe para detectar si alguien reintroduce varias.
    pos = {n: np.isin(describir, np.where(m.numpy())[0])
           for n, m in v.items() if n != "gnn_entrena"}
    log.info("Norma del embedding por bloque:")
    for n, sel_b in pos.items():
        if sel_b.any():
            nn = np.linalg.norm(emb[sel_b], axis=1)
            log.info("  %-18s mediana %.2f (p95 %.2f) sobre %d filas",
                     n, np.median(nn), np.percentile(nn, 95), int(sel_b.sum()))

    informe = {"modelo": nombre, "seed": seed,
               "n_descritas": int(len(describir)),
               "n_entrenaron_la_red": int(entrenadas.sum()),
               "dim_embedding": int(emb.shape[1]),
               "minutos": round((time.time() - t0) / 60, 1),
               "nota": ("UNA sola red describe todas las filas que no la "
                        "entrenaron. Sustituye al OOF: sin K redes no hay K "
                        "sistemas de ejes distintos en las mismas columnas.")}
    with open(resolve(cfg, "reports_dir") / "embed.json", "w") as f:
        json.dump(informe, f, indent=2, ensure_ascii=False)
    log.info("-> %s", ruta_parquet(cfg))


if __name__ == "__main__":
    main()
