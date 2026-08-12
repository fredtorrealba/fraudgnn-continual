"""
Radiografía de las ENTIDADES que generan las aristas del grafo.

El grafo conecta transacciones que comparten entidad. Este script muestra
exactamente CUÁLES son esas entidades y qué transacciones agrupan, para poder
responder "¿por qué esta transacción está conectada con esta otra?".

Reutiliza `entity_keys()` de src/data/build_graph.py — las claves que ves aquí
son literalmente las mismas que usó el constructor del grafo, no una
reimplementación que podría desincronizarse.

Las tres familias de entidad (definidas en config.yaml -> graph.edge_entities):
  card    huella completa: card1|card2|card3|card5|addr1
  email   P_emaildomain|card1  (el dominio solo crearía un hub gigante en gmail)
  device  DeviceInfo|id_30|id_31

Uso:
  python scripts/inspect_entities.py                     # resumen + rankings
  python scripts/inspect_entities.py --top 30
  python scripts/inspect_entities.py --type card         # solo tarjetas
  python scripts/inspect_entities.py --key "7919|321.0|150.0|226.0|299.0"
                                                         # las filas de un grupo
  python scripts/inspect_entities.py --month 6           # solo el mes de test
  python scripts/inspect_entities.py --csv reports/entidades.csv
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.build_graph import entity_keys                      # noqa: E402
from src.utils.common import load_config, resolve                 # noqa: E402


def load_processed(cfg, month: int | None, split: str | None) -> pd.DataFrame:
    path = resolve(cfg, "processed_dir") / "full.parquet"
    if not path.exists():
        sys.exit(f"No existe {path}. Corre antes: python -m src.data.preprocessing")
    df = pd.read_parquet(path)
    if month is not None:
        df = df[df["month"] == month]
    if split:
        df = df[df["split"] == split]
    if df.empty:
        sys.exit("El filtro (--month/--split) no dejó ninguna transacción.")
    return df.reset_index(drop=True)


def group_table(keys: pd.Series, df: pd.DataFrame, etype: str) -> pd.DataFrame:
    """Una fila por entidad: cuántas transacciones agrupa y cuántas son fraude."""
    valid = keys.notna() & ~keys.astype(str).str.startswith("nan")
    g = pd.DataFrame({"key": keys[valid], "isFraud": df.loc[valid, "isFraud"].values})
    t = g.groupby("key").agg(txn=("isFraud", "size"),
                             fraudes=("isFraud", "sum")).reset_index()
    t["tasa_fraude"] = t["fraudes"] / t["txn"]
    t["tipo"] = etype
    return t


def print_summary(tables: dict, df: pd.DataFrame, cfg):
    tasa_global = df["isFraud"].mean()
    print("=" * 78)
    print(f"{len(df):,} transacciones · {int(df['isFraud'].sum()):,} fraudes "
          f"({tasa_global * 100:.2f}%)")
    print("=" * 78)
    print(f"\n{'tipo':<8} {'entidades':>10} {'con 2+ txn':>11} {'txn cubiertas':>14} "
          f"{'tam.medio':>10} {'tam.max':>8} {'%fraude':>8}")
    print("-" * 78)
    for etype, t in tables.items():
        multi = t[t.txn > 1]
        cubiertas = int(multi.txn.sum())
        tasa = multi.fraudes.sum() / cubiertas if cubiertas else 0.0
        print(f"{etype:<8} {len(t):>10,} {len(multi):>11,} {cubiertas:>14,} "
              f"{multi.txn.mean() if len(multi) else 0:>10.1f} "
              f"{int(multi.txn.max()) if len(multi) else 0:>8,} {tasa * 100:>7.2f}%")
    print("-" * 78)
    print("  'con 2+ txn' son las entidades que efectivamente generan aristas:")
    print("  una entidad con una sola transacción no conecta con nadie.")
    print(f"  Tope anti-hub: {cfg['graph']['max_edges_per_node']} aristas por nodo · "
          f"ventana {cfg['graph']['window_days']} días")


def print_rankings(tables: dict, top: int, tasa_global: float):
    for etype, t in tables.items():
        multi = t[t.txn > 1]
        if multi.empty:
            print(f"\n### {etype.upper()} — sin grupos de 2+ transacciones")
            continue

        print(f"\n### {etype.upper()} — top {top} por TAMAÑO (los hubs)")
        print(f"{'entidad':<52} {'txn':>6} {'fraude':>7} {'tasa':>7}")
        for _, r in multi.nlargest(top, "txn").iterrows():
            print(f"{str(r.key)[:52]:<52} {int(r.txn):>6} {int(r.fraudes):>7} "
                  f"{r.tasa_fraude * 100:>6.1f}%")

        con_fraude = multi[multi.fraudes > 0]
        if con_fraude.empty:
            continue
        print(f"\n### {etype.upper()} — top {top} por CONCENTRACIÓN de fraude")
        print(f"{'entidad':<52} {'txn':>6} {'fraude':>7} {'tasa':>7} {'lift':>6}")
        rank = con_fraude.sort_values(["fraudes", "tasa_fraude"], ascending=False)
        for _, r in rank.head(top).iterrows():
            lift = r.tasa_fraude / tasa_global if tasa_global else 0
            print(f"{str(r.key)[:52]:<52} {int(r.txn):>6} {int(r.fraudes):>7} "
                  f"{r.tasa_fraude * 100:>6.1f}% {lift:>5.1f}x")
        print("  lift = cuántas veces la tasa base. Un lift alto significa que esa")
        print("  entidad concentra fraude: es la señal que la GNN propaga por aristas.")


def show_key(df: pd.DataFrame, keys_by_type: dict, key: str, cfg):
    """Las transacciones de una entidad concreta: el 'por qué' de sus aristas."""
    for etype, keys in keys_by_type.items():
        hit = keys == key
        if not hit.any():
            continue
        sub = df[hit.values]
        cols = ["TransactionID", "TransactionDT", "month", "week_in_month",
                "split", "isFraud"]
        raw = [c for c in df.columns if c.startswith("raw__")]
        print(f"\n=== Entidad {etype}: {key}")
        print(f"{len(sub)} transacciones · {int(sub.isFraud.sum())} fraudes")
        span = (sub.TransactionDT.max() - sub.TransactionDT.min()) / 86400
        print(f"Rango temporal: {span:.1f} días "
              f"(la ventana de conexión es {cfg['graph']['window_days']} días — "
              f"{'algunas parejas quedan fuera' if span > cfg['graph']['window_days'] else 'todas caben'})")
        with pd.option_context("display.max_columns", None, "display.width", 200):
            print(sub[cols].sort_values("TransactionDT").to_string(index=False))
            print("\nColumnas de entidad (crudas):")
            print(sub[raw].head(10).to_string(index=False))
        return
    sys.exit(f"No encontré la entidad '{key}'. Cópiala tal cual de los rankings.")


def export_transactions(df: pd.DataFrame, keys_by_type: dict, out: Path):
    """
    CSV por TRANSACCIÓN: una fila por txn con las columnas crudas que forman
    cada entidad, la clave compuesta resultante y cuántas transacciones la
    comparten. Sirve para responder, mirando una fila, con quién se conecta y
    por qué.
    """
    cols = {}
    for c in ("TransactionID", "TransactionDT", "month", "week_in_month",
              "split", "isFraud"):
        if c in df.columns:
            cols[c] = df[c].values

    # columnas crudas tal como las lee build_graph (sin el prefijo raw__)
    for c in [c for c in df.columns if c.startswith("raw__")]:
        cols[c.replace("raw__", "")] = df[c].values

    # clave compuesta por familia + tamaño del grupo que comparte esa clave
    for etype, keys in keys_by_type.items():
        k = keys.reset_index(drop=True)
        valid = k.notna() & ~k.astype(str).str.startswith("nan")
        sizes = k.where(valid).map(k[valid].value_counts()).fillna(0).astype(int)
        cols[f"{etype}_key"] = k.where(valid).values
        cols[f"{etype}_grupo"] = sizes.values          # 1 o 0 => no genera arista

    out.parent.mkdir(parents=True, exist_ok=True)
    tabla = pd.DataFrame(cols)
    tabla.to_csv(out, index=False)

    conectadas = (tabla[[f"{e}_grupo" for e in keys_by_type]] > 1).any(axis=1).sum()
    print(f"\nCSV por transacción -> {out}")
    print(f"  {len(tabla):,} filas × {len(tabla.columns)} columnas "
          f"({out.stat().st_size / 1e6:.1f} MB)")
    print(f"  {conectadas:,} transacciones ({conectadas / len(tabla) * 100:.1f}%) "
          f"comparten al menos una entidad -> son las que tienen aristas")
    print(f"  columnas: {', '.join(tabla.columns[:8])} ...")
    print("  Las *_grupo dicen cuántas transacciones comparten esa clave: "
          "1 = aislada por esa vía.")


def main():
    p = argparse.ArgumentParser(description="Entidades que generan las aristas")
    p.add_argument("--type", choices=["card", "email", "device"],
                   help="mirar solo una familia de entidad")
    p.add_argument("--month", type=int, help="filtrar por mes")
    p.add_argument("--split", choices=["train", "val", "test"], help="filtrar por split")
    p.add_argument("--top", type=int, default=15)
    p.add_argument("--key", help="ver las transacciones de una entidad concreta")
    p.add_argument("--csv", help="CSV por ENTIDAD: una fila por clave, con txn y fraudes")
    p.add_argument("--csv-txn",
                   help="CSV por TRANSACCIÓN: id + columnas de entidad + claves + "
                        "tamaño de grupo")
    p.add_argument("--by-month", action="store_true",
                   help="además del CSV completo, uno por cada mes "
                        "(los tamaños de grupo se recalculan dentro de cada mes)")
    args = p.parse_args()

    cfg = load_config()
    df = load_processed(cfg, args.month, args.split)

    keys_by_type = entity_keys(df, cfg)          # las MISMAS claves que build_graph
    if args.type:
        keys_by_type = {args.type: keys_by_type[args.type]}

    if args.key:
        show_key(df, keys_by_type, args.key, cfg)
        return

    if args.csv_txn:
        base = Path(args.csv_txn)
        base = base if base.is_absolute() else ROOT / base
        export_transactions(df, keys_by_type, base)

        if args.by_month and "month" in df.columns:
            # Los tamaños de grupo se recalculan DENTRO de cada mes: una tarjeta
            # con 300 transacciones repartidas en 6 meses no conecta 300 nodos
            # entre sí — la ventana de 30 días solo une lo cercano en el tiempo.
            for m in sorted(df["month"].unique()):
                sub = df[df["month"] == m].reset_index(drop=True)
                out_m = base.with_name(f"{base.stem}_mes{int(m)}{base.suffix}")
                export_transactions(sub, entity_keys(sub, cfg), out_m)

        if not args.csv:
            return

    tables = {e: group_table(k, df, e) for e, k in keys_by_type.items()}
    print_summary(tables, df, cfg)
    print_rankings(tables, args.top, df["isFraud"].mean())

    if args.csv:
        out = Path(args.csv)
        out = out if out.is_absolute() else ROOT / out
        out.parent.mkdir(parents=True, exist_ok=True)
        full = pd.concat(tables.values(), ignore_index=True)
        full = full[full.txn > 1].sort_values(["tipo", "fraudes", "txn"],
                                              ascending=[True, False, False])
        full[["tipo", "key", "txn", "fraudes", "tasa_fraude"]].to_csv(out, index=False)
        print(f"\nTabla de {len(full):,} entidades con aristas exportada a {out}")

    print("\nPara ver las transacciones de una entidad concreta:")
    print('  python scripts/inspect_entities.py --key "<pega la clave del ranking>"')


if __name__ == "__main__":
    main()
