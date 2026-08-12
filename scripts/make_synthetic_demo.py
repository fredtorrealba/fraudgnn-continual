"""
Generador de datos SINTÉTICOS de demo (no confundir con SMOTE).

Sirve para probar el pipeline completo end-to-end SIN descargar el IEEE-CIS
(útil para smoke tests, CI y para validar el flujo antes de correr con los
datos reales). Genera CSVs con el mismo esquema mínimo que el dataset real:
train_transaction.csv + train_identity.csv, con 6 "meses", entidades
compartidas (card/email/device) y un patrón de fraude que CAMBIA en el mes 6
(para gatillar el continual learning).

Uso: python scripts/make_synthetic_demo.py --n 20000
Luego el pipeline corre igual que con los datos reales.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.common import ensure_dirs, get_logger, load_config, resolve

log = get_logger("synthetic_demo")


def main(n: int, seed: int = 7):
    cfg = load_config()
    ensure_dirs(cfg)
    rng = np.random.default_rng(seed)
    spm = cfg["data"]["seconds_per_month"]

    # El pipeline calcula mes = DT//spm + 1 (igual que con el dataset real,
    # cuyo DT parte en el día 1). Acá DT parte en 0 -> meses 1..6.
    dt = np.sort(rng.integers(0, 6 * spm, size=n))
    n_cards, n_emails, n_devices = n // 20, n // 30, n // 25

    df = pd.DataFrame({
        "TransactionID": np.arange(3_000_000, 3_000_000 + n),
        "TransactionDT": dt,
        "TransactionAmt": np.round(rng.lognormal(4.0, 1.0, n), 2),
        "card1": rng.integers(1000, 1000 + n_cards, n),
        "P_emaildomain": rng.choice(
            [f"dom{k}.com" for k in range(n_emails)] + [None], n),
    })

    # Features numéricas de relleno (para acercarse al ancho real del dataset)
    n_extra = 60
    base = rng.normal(0, 1, (n, n_extra))
    for k in range(n_extra):
        df[f"V{k+1}"] = base[:, k].astype(np.float32)

    # --- fraude: patrón A (meses 1-5) + patrón B emergente (mes 6) ---
    is_fraud = np.zeros(n, dtype=int)
    signal_a = (base[:, 0] > 1.6) & (df["TransactionAmt"] > 150)
    is_fraud[signal_a] = 1
    m6 = (dt // spm + 1) == 6
    # Patrón NUEVO (solo mes 6): región de features que en los meses 1-5 fue
    # SIEMPRE legítima (valores dentro del rango normal, nada anómalo). El
    # modelo original aprendió que esa zona es legítima -> score bajo ->
    # exactamente el caso que gatilla el continual learning. El fine-tuning
    # sí puede aprender la banda V7/V8 con los casos confirmados.
    band = ((base[:, 6] > 0.8) & (base[:, 6] < 1.7) &
            (base[:, 7] > 0.8) & (base[:, 7] < 1.7))
    signal_b = m6 & band
    is_fraud[signal_b] = 1
    # ruido de fraude aleatorio hasta ~3.5%
    extra = rng.random(n) < max(0.0, 0.035 - is_fraud.mean())
    is_fraud[extra] = 1
    df["isFraud"] = is_fraud
    # correlacionar las features del patrón A con la etiqueta en meses 1-5
    df.loc[signal_a, "V1"] += 2.0

    # los fraudes del patrón A comparten entidades (el grafo aporta señal);
    # el patrón B usa un anillo NUEVO de tarjetas jamás visto en train —
    # ni las features ni la estructura lo delatan: el modelo original le da
    # score bajo y el gatillo del CL debe dispararse.
    idx_a = np.where((is_fraud == 1) & ~signal_b)[0]
    df.loc[idx_a, "card1"] = rng.integers(1000, 1000 + n_cards // 10, len(idx_a))
    idx_b = np.where(signal_b)[0]
    df.loc[idx_b, "card1"] = rng.integers(90000, 90000 + max(2, len(idx_b) // 4),
                                          len(idx_b))

    # La huella de tarjeta es estable por card1 (como en el dataset real:
    # una misma tarjeta repite card2/card3/card5/addr1 en sus transacciones)
    df["card2"] = (df["card1"] % 500 + 100).astype(float)
    df["card3"] = 150.0
    df["card5"] = 226.0
    df["addr1"] = (df["card1"] % 400 + 100).astype(float)


    tx_cols = [c for c in df.columns]
    df[tx_cols].to_csv(resolve(cfg, "raw_dir") / "train_transaction.csv", index=False)

    # identity: solo una parte de las transacciones tiene identidad
    has_id = rng.random(n) < 0.4
    ident = pd.DataFrame({
        "TransactionID": df.loc[has_id, "TransactionID"],
        "DeviceInfo": rng.choice([f"dev{k}" for k in range(n_devices)], has_id.sum()),
        "id_30": rng.choice(["Windows 10", "iOS 15", "Android 12", None], has_id.sum()),
        "id_31": rng.choice(["chrome", "safari", "firefox", None], has_id.sum()),
    })
    ident.to_csv(resolve(cfg, "raw_dir") / "train_identity.csv", index=False)
    log.info("Demo sintética: %d txn (%.2f%% fraude, patrón B solo en mes 6) "
             "en %s", n, 100 * is_fraud.mean(), resolve(cfg, "raw_dir"))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=20000)
    args = p.parse_args()
    main(args.n)
