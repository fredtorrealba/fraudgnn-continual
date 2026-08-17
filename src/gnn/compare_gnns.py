"""
Paso 5 — COMPARACIÓN GraphSAGE vs GAT (OE2).

Protocolo (walk-forward × 3):
- Cada arquitectura se entrena 3 veces (seeds 42/123/2026) con el mismo
  ventanas del config (entrena `gnn_entrena`, valida `gnn_valida`).
- Además del AUC sobre la ventana completa, cada modelo se evalúa SEMANA A
  SEMANA dentro del mes de validación (walk-forward: semanas 1→4, el
  "futuro que va llegando"). La selección usa el AUC promedio de las
  semanas × seeds — así la comparación premia consistencia temporal y no
  solo el promedio del mes.
- También se reportan recall/PR-AUC como métricas de apoyo.
- KPI del objetivo: AUC-ROC > 0.93.
- Se SELECCIONA la mejor y se registra en models/selected_model.json — a
  partir de ahí, esa es la red que entra en operación y en el ciclo de
  continual learning.

Criterio de desempate: si la diferencia de AUC promedio es < 0.005, gana
GraphSAGE por costo fijo de inferencia + carácter inductivo (producción).

REANUDABLE: el avance vive en artifacts/pipeline_state.json (se crea solo en la
primera corrida, con las 6 marcadas "pending"). Si el proceso muere, vuelve a
lanzar EL MISMO comando: salta las corridas terminadas y retoma la que quedó a
medias desde su última época. No hay que pasar ningún flag.

Uso:
  python -m src.gnn.compare_gnns              # entrena lo que falte + compara
  python -m src.gnn.compare_gnns --skip-train # solo compara reportes ya generados
  python -m src.gnn.compare_gnns --force      # reentrena las 6 desde cero
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.gnn.sampling import cerrar_loader  # noqa: E402
from src.utils.ventanas import mascaras_grafo  # noqa: E402
from src.utils.common import (ensure_dirs, get_device, get_logger, load_config, set_seed,
                              load_state, resolve, state_path, update_state)

log = get_logger("compare_gnns")

TIE_MARGIN = 0.005  # si el AUC difiere menos que esto, decide producción
# GAT va PRIMERO a propósito: es la arquitectura cara (atención con 4 cabezas
# sobre ~1.8M aristas por batch, ~12 GB de activaciones). Si va a fallar por
# memoria o a ir demasiado lenta, mejor saberlo en la primera corrida que tras
# hora y media de GraphSAGE. El orden NO afecta los resultados: train() llama a
# set_seed(seed) al inicio de cada corrida, así que cada una es independiente.
# Las arquitecturas salen del config (gnn.arquitecturas), no de una
# constante: cambiar de GAT a GATv2 no debería exigir tocar código.
MODELS = ("graphsage", "gatv2")   # valor por defecto si falta en config


def plan(cfg) -> list[tuple[str, int]]:
    """Las 6 corridas del protocolo, en orden fijo."""
    arqs = cfg["gnn"].get("arquitecturas", MODELS)
    return [(m, s) for m in arqs for s in cfg["gnn"]["seeds"]]


def init_state(cfg, force: bool = False):
    """Crea artifacts/pipeline_state.json si no existe (primera corrida) y deja
    cada corrida marcada según lo que YA hay en disco: done o pending."""
    from src.gnn.train_gnn import is_done, run_key

    fresh = not state_path(cfg).exists()
    if fresh:
        log.info("Sin archivo de estado — se crea %s", state_path(cfg))
    known = load_state(cfg)["runs"]
    for model, seed in plan(cfg):
        key = run_key(model, seed)
        if not force and is_done(model, seed, cfg):
            status = "done"
        elif not force and known.get(key, {}).get("status") == "running":
            status = "running"                     # quedó a medias: se retoma
        else:
            status = "pending"
        update_state(cfg, key, status=status, model=model, seed=seed)


def _seeds(lista) -> str:
    return ", ".join(str(s) for s in lista) if lista else "ninguna"


def show_runs(cfg, force: bool = False):
    """Resumen corto: por arquitectura, qué seeds están listas y cuáles faltan."""
    from src.gnn.train_gnn import is_done, resume_info

    pending = []
    hd = cfg["gnn"]["hidden_dims"]
    log.info("--- Corridas GNN (%d arquitecturas x %d seeds) ---",
             len(cfg["gnn"].get("arquitecturas", MODELS)),
             len(cfg["gnn"]["seeds"]))
    log.info("    arquitectura: %d capa(s) [%s] = %d salto(s) en el grafo",
             len(hd), ", ".join(map(str, hd)), len(hd))
    if cfg["gnn"].get("sin_aristas"):
        log.warning("    ABLACIÓN ACTIVA: sin_aristas=true -> el grafo se anula, "
                    "cada nodo queda aislado y el modelo es una MLP")
    for model in cfg["gnn"].get("arquitecturas", MODELS):
        listas, faltan = [], []
        for seed in cfg["gnn"]["seeds"]:
            (listas if is_done(model, seed, cfg) and not force
             else faltan).append(seed)
        log.info("%-10s listas: %-16s | faltan: %s",
                 model, _seeds(listas), _seeds(faltan))
        pending += [(model, s) for s in faltan]

    # ¿alguna quedó a mitad de camino?
    for model, seed in pending:
        info = None if force else resume_info(model, seed, cfg)
        if info:
            log.info("%s seed=%d quedó a medias en la época %d — se retoma.",
                     model, seed, info["epoch"])
    return pending


# ── PARALELISMO ────────────────────────────────────────────────────────────
# Las 8 tareas de esta etapa son INDEPENDIENTES: los 2 estudios de Optuna no se
# miran entre sí, y las 6 corridas solo cambian de semilla. En serie dejaban la
# máquina al 13% de CPU y 11% de GPU.
#
# Procesos y no hilos: el GIL serializaría el bucle de Python de HeteroConv, que
# es justo el cuello. Y con arranque "spawn" y no "fork", porque un fork con el
# contexto de CUDA ya creado en el padre da comportamiento indefinido.
#
# Cada proceso carga su copia de graph.pt (~1-2 GB) y usa ~1,5 GB de VRAM, así
# que los dos límites son la RAM y la VRAM. Se controlan desde el config.

def _limitar_hilos(cfg: dict, par: int) -> None:
    """
    Reparte los núcleos ENTRE los procesos paralelos.

    Cada hijo hereda OMP_NUM_THREADS del padre, así que con `compute.n_jobs: 14`
    y 3 procesos se pedirían 42 hilos sobre 16 vCPU. Eso no va más rápido: el
    kernel quema la cuota del cgroup en milisegundos y CONGELA los procesos el
    resto de cada ventana de 100 ms — el mismo fallo que documenta n_jobs.

    Se llama ANTES de importar torch: las variables de OpenMP las lee libgomp
    al primer uso, y una vez leídas ya no se pueden cambiar.
    """
    import os
    n = max(1, int((cfg.get("compute") or {}).get("n_jobs", 1)) // max(1, par))
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[v] = str(n)
    cfg.setdefault("compute", {})["n_jobs"] = n


def _entrenar_una(args):
    """Una corrida (modelo, semilla). A nivel de módulo para que spawn la pueda
    serializar."""
    model, seed, cfg, force, par = args
    _limitar_hilos(cfg, par)
    from src.gnn.train_gnn import train
    train(model, seed, cfg, force=force)
    return (model, seed)


def _buscar_una(args):
    """La búsqueda de Optuna de UNA arquitectura."""
    arq, cfg, par = args
    _limitar_hilos(cfg, par)
    return arq, buscar_hiperparametros(arq, cfg)


def _morir_con_el_padre():
    """
    Inicializador de cada hijo: que el kernel lo mate cuando muera el padre.

    Sin esto, un Ctrl-C o un OOM en el padre deja a los hijos VIVOS, adoptados
    por init (`PPID 1`) y agarrando la VRAM hasta que alguien los mate a mano.
    El 15/08 había dos de 14 horas ocupando 1,5 GB de los 20 de la tarjeta, y
    la corrida siguiente arrancaba con ese hueco sin que nada lo dijera.

    `PR_SET_PDEATHSIG` (prctl 1) es lo único que aguanta un SIGKILL del padre:
    un `atexit` o un handler de señales no llegan a ejecutarse en ese caso.
    Es de Linux; en macOS no existe y se ignora — allí no se entrena.
    """
    import platform
    import signal
    if platform.system() != "Linux":
        return
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").prctl(1, signal.SIGTERM)
    except Exception:                     # noqa: BLE001 - nunca debe tumbar al hijo
        pass


def _en_paralelo(fn, tareas: list, n: int):
    """Ejecuta `fn` sobre `tareas` con `n` procesos. Devuelve los resultados."""
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor, as_completed

    ctx = mp.get_context("spawn")
    salida = []
    with ProcessPoolExecutor(max_workers=n, mp_context=ctx,
                             initializer=_morir_con_el_padre) as ex:
        futuros = {ex.submit(fn, t): t for t in tareas}
        for fut in as_completed(futuros):
            # Si un proceso hijo revienta, la excepción sale AQUÍ y no en
            # silencio: mejor parar que quedarse con 5 corridas de 6.
            salida.append(fut.result())
    return salida


def _buscar_todas(arqs: list[str], cfg) -> dict[str, dict]:
    """Optuna de cada arquitectura, en paralelo si el config lo permite."""
    par = max(1, int(cfg["gnn"].get("paralelo_optuna", 1)))
    if par > 1 and len(arqs) > 1:
        log.info("Optuna de %d arquitecturas EN PARALELO (%d procesos)",
                 len(arqs), min(par, len(arqs)))
        n = min(par, len(arqs))
        # Copia por arquitectura: `_limitar_hilos` MUTA el cfg que recibe, y
        # compartir el mismo diccionario entre las dos tareas dividiría n_jobs
        # dos veces.
        tareas = [(a, json.loads(json.dumps(cfg)), n) for a in arqs]
        res = _en_paralelo(_buscar_una, tareas, n)
        return dict(res)
    return {a: buscar_hiperparametros(a, cfg) for a in arqs}


def run_all(cfg, force: bool = False):
    init_state(cfg, force)
    from src.gnn.train_gnn import resume_info, train

    pending = show_runs(cfg, force)
    total = len(plan(cfg))
    if not pending:
        log.info("Las %d corridas están listas — a la comparación.", total)
        return
    log.info("Faltan %d de %d corridas.", len(pending), total)

    # La búsqueda va ANTES de las semillas: las 3 corridas de cada arquitectura
    # tienen que compartir hiperparámetros o no serían réplicas de lo mismo.
    #
    # CADA ARQUITECTURA CON LOS SUYOS. Antes esto era un bucle que llamaba a
    # `aplicar_hiperparametros(cfg, ...)` sobre el MISMO cfg: los de gatv2 se
    # calculaban, se aplicaban, y los de graphsage los pisaban. Las 6 corridas
    # entrenaban con los de graphsage, la búsqueda de gatv2 se tiraba a la
    # basura, y la comparación entre arquitecturas quedaba inválida —gatv2
    # competía con hiperparámetros ajustados para su rival.
    arqs = sorted({m for m, _ in pending})
    mejores: dict[str, dict] = {}
    if int(cfg["gnn"].get("optuna_trials", 0)) > 0:
        mejores = _buscar_todas(arqs, cfg)

    par = max(1, int(cfg["gnn"].get("paralelo_corridas", 1)))
    tareas = []
    for model, seed in pending:
        c = json.loads(json.dumps(cfg))          # copia por corrida
        if model in mejores:
            aplicar_hiperparametros(c, mejores[model])
        tareas.append((model, seed, c, force, par))

    if par > 1 and len(tareas) > 1:
        # Nunca más procesos que tareas: con 2 corridas y paralelo_corridas=6
        # se levantaban 6 procesos para 2 trabajos.
        n_proc = min(par, len(tareas))
        log.info("Entrenando %d corridas con %d procesos EN PARALELO",
                 len(tareas), n_proc)
        _en_paralelo(_entrenar_una, tareas, n_proc)
    else:
        for i, (model, seed, c, f, _) in enumerate(tareas, 1):
            info = None if f else resume_info(model, seed, c)
            desde = (f"retoma en la época {info['epoch'] + 1} de {c['gnn']['epochs']}"
                     if info else f"desde cero (época 1 de {c['gnn']['epochs']})")
            log.info("=== [%d/%d] %s seed=%d — %s ===",
                     i, len(tareas), model, seed, desde)
            train(model, seed, c, force=f)


def buscar_hiperparametros(model_name: str, cfg) -> dict:
    """
    Búsqueda bayesiana para la GNN — lo que hasta ahora solo tenía XGBoost.

    La comparación era injusta en esfuerzo de ajuste: XGBoost recibía 30 trials
    de Optuna y la GNN corría con valores puestos a mano. No es paridad de
    cómputo (un trial de XGBoost cuesta ~13 s y uno de GNN ~5 min, 23x más) pero
    sí de protocolo, que es lo que se defiende: AMBOS modelos recibieron búsqueda
    bayesiana con el mismo número de trials y el mismo sampler.

    Cada trial entrena con `gnn_entrena` y se puntúa por PR-AUC sobre `gnn_valida` —
    la misma métrica y el mismo conjunto con que se elige la cabeza XGBoost.
    `MedianPruner` mata pronto los trials que van claramente peor que la mediana,
    que es lo que hace viable el presupuesto.

    El resultado se cachea en reports/optuna_{modelo}.json: la búsqueda es la
    parte cara y no debe repetirse al relanzar el pipeline.
    """
    import optuna
    import torch
    from sklearn.metrics import average_precision_score

    from src.gnn.models import build_model
    from src.gnn.train_gnn import evaluate, make_loader

    cache = resolve(cfg, "reports_dir") / f"optuna_{model_name}.json"
    if cache.exists():
        with open(cache) as f:
            prev = json.load(f)
        log.info("[%s] hiperparámetros ya buscados (PR-AUC %.4f) — se reutilizan",
                 model_name, prev.get("mejor_valor", float("nan")))
        return prev["mejores_params"]

    n_trials = int(cfg["gnn"].get("optuna_trials", 30))
    # PRESUPUESTO POR TIEMPO (D1). Con `ancho` y `capas` en el espacio, un trial
    # cuesta 20 veces más que otro: `ancho 64, capas 2` tarda 1,7 min y
    # `256/3` tarda 29. Repartir por NÚMERO de trials reparte el cómputo muy
    # desigual — medido el 15/08, "30 trials cada una" fueron 2 h para
    # graphsage y 11 h para gatv2, cinco veces más máquina por la misma
    # etiqueta. Con minutos, las dos reciben lo mismo de verdad, y "ambas
    # recibieron el mismo cómputo" es además más defendible que "ambas
    # recibieron 30 trials".
    presupuesto = cfg["gnn"].get("optuna_presupuesto_min")
    presupuesto = float(presupuesto) if presupuesto else None
    # Tope POR TRIAL (D2). El `timeout` de arriba no corta un trial ya
    # empezado: sin esto, uno solo se comió 29 minutos del presupuesto.
    tope_trial = float(cfg["gnn"].get("optuna_tope_trial_min", 0) or 0)
    data = torch.load(resolve(cfg, "graph_dir") / "graph.pt", weights_only=False)
    device = get_device()
    _v = mascaras_grafo(cfg, data)
    _m_tr, _m_va = _v["gnn_entrena"], _v["gnn_valida"]
    y_val = data["transaction"].y[_m_va].numpy()

    def objetivo(trial):
        # LA MISMA SEMILLA PARA TODOS LOS TRIALS, a propósito.
        #
        # Cada trial es dos cosas a la vez: unos hiperparámetros (los elige
        # Optuna) y una inicialización de pesos (azar). Si la inicialización
        # cambia entre trials, cuando uno gana no se sabe si fue por sus
        # hiperparámetros o porque le tocó mejor arranque.
        #
        # Y el arranque pesa MÁS que la señal: medido en el smoke, los mismos
        # hiperparámetros dieron 0.3077 y 0.2912 según cómo inicializaran —
        # 0.016 de diferencia, cuando entre trials suele haber menos.
        #
        # Fijándola, lo único que varía entre trials son los hiperparámetros y
        # la comparación queda limpia. La robustez frente a la inicialización se
        # mide DESPUÉS, en las 6 corridas finales con las semillas 42/123/2026:
        # cada fase hace un trabajo en vez de mezclarlos.
        #
        # (Antes tampoco se llamaba a set_seed aquí: cada trial arrancaba con el
        # estado que dejara el anterior, así que ni siquiera era reproducible.)
        set_seed(42)
        c = json.loads(json.dumps(cfg))          # copia profunda por trial
        g = c["gnn"]
        g["lr"] = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        g["dropout"] = trial.suggest_float("dropout", 0.0, 0.5)
        ancho = trial.suggest_categorical("ancho", [64, 128, 256])
        # MÍNIMO 2 capas: con una sola, los nodos de entidad llegan a la
        # transacción todavía en ceros y el grafo no aporta nada (models.py lo
        # rechaza con un ValueError). El rango 1-2 venía del grafo homogéneo,
        # donde una capa sí tenía sentido; aquí hacía fallar el primer trial.
        g["hidden_dims"] = [ancho] * trial.suggest_int("capas", 2, 3)
        # Con el ancho del embedding DECIDIDO (forzar_mlp_head_dim, ver config)
        # el trial no lo sortea: si lo sorteara, lr y dropout se afinarían bajo
        # un ancho que aplicar_hiperparametros va a pisar después, y la
        # búsqueda optimizaría una red que nunca se entrena. OJO: cambiar el
        # flag invalida el caché de Optuna (reports/optuna_*) por lo mismo.
        forzar = int(g.get("forzar_mlp_head_dim", 0) or 0)
        g["mlp_head_dim"] = forzar or trial.suggest_categorical(
            "mlp_head_dim", [16, 32, 64])
        g["in_dim"] = data["transaction"].x.shape[1]
        wd = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
        # épocas cortas: la búsqueda compara configuraciones, no exprime cada una
        epocas = max(3, cfg["gnn"]["epochs"] // 4)

        balancear = bool(g.get("balanceo_semillas", False))
        pw = (float(g.get("pos_weight_con_balanceo", 1.0)) if balancear else
              float((y_val == 0).sum() / max(1, (y_val == 1).sum())))
        modelo = build_model(model_name, c, data.metadata()).to(device)
        opt = torch.optim.Adam(modelo.parameters(), lr=g["lr"], weight_decay=wd)
        crit = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pw, device=device))

        tr = make_loader(data, _m_tr, c, True, balancear)
        va = make_loader(data, _m_va, c, False)
        # try/finally OBLIGATORIO: el pruner sale por excepción, y sin cerrar
        # los workers cada trial dejaría 12 procesos vivos hasta el final del
        # proceso. Con 30 trials x 2 arquitecturas son 720 procesos colgados.
        try:
            # `tope` es la copia LOCAL del tope: al indultar a un trial se
            # pone a 0 y no debe afectar a los siguientes.
            mejor, t_trial, tope = 0.0, time.time(), tope_trial
            for ep in range(1, epocas + 1):
                if tope and (time.time() - t_trial) / 60 > tope:
                    # El reloj NO puede matar al que va ganando. Un trial podado
                    # queda PRUNED, y Optuna elige el mejor solo entre los
                    # COMPLETE: sin esta salvedad, una red lenta pero superior
                    # se descartaría por tardar y no habría forma de saberlo.
                    # `t.value` es None en los podados, así que esto compara
                    # contra los que sí terminaron.
                    campeon = max((t.value for t in trial.study.trials
                                   if t.value is not None), default=0.0)
                    if mejor > campeon:
                        log.info("  [%s] trial %d pasa de %.0f min pero VA EN "
                                 "CABEZA (%.4f > %.4f) — se le deja terminar",
                                 model_name, trial.number, tope, mejor,
                                 campeon)
                        tope = 0.0              # indultado, solo este trial
                    else:
                        log.info("  [%s] trial %d CORTADO por tiempo en la época "
                                 "%d (%.1f min > %.0f, va %.4f contra %.4f)",
                                 model_name, trial.number, ep,
                                 (time.time() - t_trial) / 60, tope,
                                 mejor, campeon)
                        raise optuna.TrialPruned()
                t_ep = time.time()
                modelo.train()
                for batch in tr:
                    batch = batch.to(device)
                    n = batch["transaction"].batch_size
                    opt.zero_grad()
                    loss = crit(modelo(batch.x_dict, batch.edge_index_dict, batch)[:n],
                                batch["transaction"].y[:n])
                    loss.backward()
                    opt.step()
                yv, sv = evaluate(modelo, va, device)
                pr = float(average_precision_score(yv, sv))
                mejor = max(mejor, pr)
                # Una línea POR ÉPOCA. Sin esto, un trial son 12 épocas sin
                # imprimir nada: con la barra de Optuna avanzando solo al
                # terminar el trial, quedaban decenas de minutos a ciegas sin
                # saber si el proceso avanzaba o se había colgado.
                log.info("  [%s] trial %d · época %2d/%d · PR-AUC %.4f "
                         "(mejor %.4f) · %.1f min", model_name, trial.number,
                         ep, epocas, pr, mejor, (time.time() - t_ep) / 60)
                trial.report(mejor, ep)
                if trial.should_prune():
                    log.info("  [%s] trial %d PODADO en la época %d (%.1f min)",
                             model_name, trial.number, ep,
                             (time.time() - t_trial) / 60)
                    raise optuna.TrialPruned()
            log.info("  [%s] trial %d LISTO · PR-AUC %.4f · %.1f min",
                     model_name, trial.number, mejor,
                     (time.time() - t_trial) / 60)
            return mejor
        except torch.cuda.OutOfMemoryError:
            # B2. Una configuración que no cabe es un trial malo, no un fallo
            # del pipeline: se poda y la búsqueda sigue. Antes, un OOM en el
            # trial 29 tumbaba el proceso y se perdían los 29 anteriores —
            # que además vivían solo en memoria (ver el storage de abajo).
            # El pico es del ANCHO x CAPAS, así que Optuna aprende solo a
            # evitar esa zona: no hace falta tocarle el espacio de búsqueda.
            log.warning("  [%s] trial %d SIN MEMORIA en la GPU — se poda y se "
                        "sigue", model_name, trial.number)
            raise optuna.TrialPruned()
        finally:
            cerrar_loader(tr)
            cerrar_loader(va)
            del modelo
            # El caché del allocator NO se libera solo al morir el trial: los
            # bloques quedan reservados y el siguiente arranca con menos sitio.
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # SEMILLA DISTINTA POR ARQUITECTURA (B4). Con la misma, los dos estudios
    # sortean los MISMOS hiperparámetros en el mismo orden: cuando a una le toca
    # la red grande, a la otra también, y los picos de VRAM coinciden en vez de
    # turnarse. Se desacoplan sin perder nada — cada una recibe el mismo espacio
    # y el mismo presupuesto, que es lo que hace comparable el resultado.
    _arqs = sorted(cfg["gnn"].get("arquitecturas", MODELS))
    semilla_tpe = 42 + (_arqs.index(model_name) if model_name in _arqs else 0)

    # ESTUDIO PERSISTIDO (B1). Sin `storage` el estudio vive solo en memoria y
    # el JSON se escribe al FINAL: un accidente en el trial 29 se llevaba los 29
    # anteriores. Con SQLite cada trial se escribe al terminar y `load_if_exists`
    # deja retomar donde se quedó.
    #
    # UN ARCHIVO POR ARQUITECTURA, no uno compartido. Con `paralelo_optuna: 2`
    # los dos procesos crean el esquema a la vez y el segundo muere con
    # "table studies already exists" — lo cazó el smoke al aplicar esto. Además
    # SQLite serializa las escrituras, así que un solo archivo pondría a los dos
    # estudios a pelearse por el lock en cada trial. Separados no se tocan.
    db = resolve(cfg, "reports_dir") / f"optuna_{model_name}.db"
    db.parent.mkdir(parents=True, exist_ok=True)
    estudio = optuna.create_study(
        direction="maximize",
        study_name=f"gnn_{model_name}",
        storage=f"sqlite:///{db}",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=semilla_tpe),
        # n_warmup_steps=4, no 2. Con la entrada CRUDA las redes tocaban techo
        # en la época 1 y podar en la 2 no perdía nada. Con la entrada
        # normalizada aprenden de verdad, y una configuración con learning rate
        # bajo puede ir mediocre en la época 2 y ser la mejor en la 8: podarla
        # ahí mataría justo la que buscamos.
        #
        # Cuesta ~1,3 min más por trial podado (4 épocas en vez de 2 a 0,65
        # min/época). Con 30 trials son unos 25 minutos, que es barato comparado
        # con descartar la configuración buena.
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=4))

    hechos = len([t for t in estudio.trials
                  if t.state.name in ("COMPLETE", "PRUNED")])
    if hechos:
        log.info("[%s] el estudio ya tenía %d trials hechos — se retoman",
                 model_name, hechos)

    if presupuesto:
        log.info("[%s] Optuna: %.0f min de presupuesto, tope %.0f min/trial "
                 "(PR-AUC sobre gnn_valida)", model_name, presupuesto, tope_trial)
        estudio.optimize(objetivo, timeout=presupuesto * 60,
                         show_progress_bar=True)
    else:
        log.info("[%s] Optuna: %d trials (PR-AUC sobre gnn_valida)",
                 model_name, n_trials)
        estudio.optimize(objetivo, n_trials=max(0, n_trials - hechos),
                         show_progress_bar=True)

    best = dict(estudio.best_params)
    with open(cache, "w") as f:
        json.dump({"modelo": model_name, "n_trials": n_trials,
                   "mejor_valor": estudio.best_value, "mejores_params": best,
                   "podados": sum(1 for t in estudio.trials
                                  if t.state.name == "PRUNED")},
                  f, indent=2, ensure_ascii=False)
    log.info("[%s] mejor PR-AUC %.4f | %s", model_name, estudio.best_value, best)
    return best


def aplicar_hiperparametros(cfg, best: dict) -> dict:
    """Traduce lo que devuelve Optuna a la forma que espera `cfg["gnn"]`."""
    g = cfg["gnn"]
    for k in ("lr", "dropout", "mlp_head_dim"):
        if k in best:
            g[k] = best[k]
    if "ancho" in best:
        # El default es 2, no 1: si `capas` faltara (un estudio antiguo, un
        # enqueue_trial incompleto) se construiría una GNN de una capa, que en
        # el grafo heterogéneo no llega a propagar nada y muere en build_model.
        g["hidden_dims"] = [best["ancho"]] * max(2, int(best.get("capas", 2)))
    # EXPERIMENTO ancho-del-embedding (ver config). Se aplica AQUÍ y no en
    # cfg_arquitectura a propósito: esto decide cómo se ENTRENA una red nueva;
    # cfg_arquitectura reconstruye una YA entrenada y debe respetar su
    # checkpoint, o load_state_dict reventaría por dimensiones.
    forzar = int(g.get("forzar_mlp_head_dim", 0) or 0)
    if forzar:
        log.info("mlp_head_dim = %d por decisión adoptada (config: "
                 "forzar_mlp_head_dim; Optuna había elegido %s). Ver el "
                 "porqué medido en config.yaml.",
                 forzar, best.get("mlp_head_dim", "—"))
        g["mlp_head_dim"] = forzar
    return cfg


def weekly_val_aucs(model_name: str, seed: int, cfg) -> list[float]:
    """
    Walk-forward dentro del mes de validación: PR-AUC por semana (1..4).

    PR-AUC y no ROC-AUC, por coherencia con el resto del proyecto y porque con
    3,4% de fraude el ROC comprime las diferencias: la misma comparación que en
    PR-AUC da 0.0348 en ROC da 0.0073. Elegir arquitectura con una métrica que
    apenas distingue es pedir que la elección salga por azar.

    Por SEMANA y no sobre el mes entero porque el mes agregado se infla con la
    correlación temporal: medido en la ablación sin aristas, un modelo daba
    0.8524 sobre el mes completo y 0.6075 de media semanal — estaba separando
    por PERIODO, no por fraude. Dentro de una semana esa correlación no existe.
    """
    import torch
    from sklearn.metrics import average_precision_score

    from src.continual_learning.validate import score_nodes
    from src.gnn.models import build_model, cfg_arquitectura

    data = torch.load(resolve(cfg, "graph_dir") / "graph.pt", weights_only=False)
    ckpt = torch.load(resolve(cfg, "models_dir") / f"{model_name}_seed{seed}.pt",
                      weights_only=False)
    # La arquitectura sale del checkpoint (o del cache de Optuna), NO del cfg:
    # cada arquitectura tiene la suya y el cfg global no describe a ninguna.
    c = cfg_arquitectura(model_name, cfg, ckpt)
    c["gnn"]["in_dim"] = ckpt["in_dim"]
    model = build_model(model_name, c, data.metadata())
    model.load_state_dict(ckpt["state_dict"])
    cfg = c                                   # el scorer usa el mismo cfg

    # La selección se mide en `gnn_valida`, que la red NO vio.
    val_nodes = torch.where(mascaras_grafo(cfg, data)["gnn_valida"])[0].numpy()
    scores = score_nodes(model, data, val_nodes, cfg)
    y = data["transaction"].y.numpy()[val_nodes]
    weeks = data["transaction"].week_in_month.numpy()[val_nodes]

    aucs = []
    for w in sorted(np.unique(weeks)):
        m = weeks == w
        if y[m].sum() == 0 or y[m].sum() == m.sum():
            continue  # semana sin ambas clases: AUC indefinido, se omite
        aucs.append(float(average_precision_score(y[m], scores[m])))
    return aucs


def collect(cfg) -> dict:
    reports_dir = resolve(cfg, "reports_dir")
    results = {}
    for model in cfg["gnn"].get("arquitecturas", ["graphsage", "gatv2"]):
        runs = []
        for seed in cfg["gnn"]["seeds"]:
            f = reports_dir / f"{model}_seed{seed}_val.json"
            if not f.exists():
                log.warning("Falta %s — corre primero el entrenamiento.", f)
                continue
            with open(f) as fh:
                run = json.load(fh)
            # reportes viejos no traían estos campos
            run.setdefault("model", model)
            run.setdefault("seed", seed)
            # walk-forward: AUC por semana del mes de validación
            run["weekly_auc"] = weekly_val_aucs(model, seed, cfg)
            runs.append(run)
        if runs:
            all_weekly = [a for r in runs for a in r["weekly_auc"]]
            results[model] = {
                "runs": runs,
                # la selección usa el promedio walk-forward (semanas x seeds);
                # el AUC del mes completo queda como referencia
                "auc_mean": float(np.mean(all_weekly)),
                "auc_std": float(np.std(all_weekly)),
                "auc_month_mean": float(np.mean([r["auc_roc"] for r in runs])),
                "recall_mean": float(np.mean([r["recall"] for r in runs])),
                "pr_auc_mean": float(np.mean([r.get("pr_auc", 0) for r in runs])),
            }
    return results


def select(results: dict, cfg) -> dict:
    # Los nombres salen del CONFIG, no escritos a mano. Este bloque comparaba
    # results.get("gat") cuando la arquitectura pasó a llamarse "gatv2": la
    # búsqueda siempre devolvía None, se caía al else y GraphSAGE ganaba por
    # incomparecencia, con el mensaje "Única arquitectura disponible" aunque
    # las dos hubieran corrido. La comparación central del capstone estaba
    # desactivada en silencio.
    disponibles = [m for m in cfg["gnn"].get("arquitecturas", MODELS)
                   if results.get(m)]
    if not disponibles:
        raise SystemExit("Ninguna arquitectura tiene resultados: revisa el "
                         "paso `gnn`.")
    if len(disponibles) == 1:
        winner = disponibles[0]
        reason = (f"Única arquitectura con resultados: {winner}. Las demás "
                  f"({', '.join(m for m in cfg['gnn'].get('arquitecturas', MODELS) if m != winner)}) "
                  "no dejaron corridas.")
    else:
        orden = sorted(disponibles, key=lambda m: -results[m]["auc_mean"])
        primero, segundo = orden[0], orden[1]
        diff = results[primero]["auc_mean"] - results[segundo]["auc_mean"]
        if diff < TIE_MARGIN and "graphsage" in disponibles:
            # Empate técnico: decide producción, no el decimal de ruido.
            winner = "graphsage"
            reason = (f"Empate técnico (Δ={diff:.4f} < {TIE_MARGIN} entre "
                      f"{primero} y {segundo}). Gana GraphSAGE por costo de "
                      "inferencia fijo e inductividad (producción).")
        else:
            winner = primero
            reason = (f"Mayor PR-AUC walk-forward: {primero} {results[primero]['auc_mean']:.4f} "
                      f"contra {segundo} {results[segundo]['auc_mean']:.4f} (Δ={diff:+.4f}).")

    best_seed_runs = results[winner]["runs"]
    # la mejor seed también se elige por su promedio walk-forward.
    # OJO: la seed se lee del propio reporte — si faltara alguna corrida,
    # indexar cfg["gnn"]["seeds"] apuntaría a la seed equivocada.
    # PR-AUC semanal medio. El fallback a pr_auc mensual solo actúa si alguna
    # semana quedó sin ambas clases y no hubo curva que calcular.
    best_idx = int(np.argmax([np.mean(r["weekly_auc"]) if r.get("weekly_auc")
                              else r.get("pr_auc", 0.0) for r in best_seed_runs]))
    best_seed = best_seed_runs[best_idx]["seed"]

    # El KPI (0.93) es de AUC-ROC MENSUAL, así que se compara contra
    # `auc_month_mean`. Antes se comparaba contra `auc_mean`, que desde el
    # cambio a PR-AUC walk-forward vale ~0.29: el aviso saltaba SIEMPRE, aunque
    # el ROC mensual fuese 0.9435 y el KPI estuviera cumplido de sobra.
    kpi_ok = results[winner]["auc_month_mean"] > cfg["gnn"]["kpi_auc"]
    # La época del PICO de la corrida ganadora. Se anota aquí —y no solo dentro
    # del checkpoint— para que el paso `refit` pueda releerla sin cargar 2 MB de
    # pesos, y para que quede a la vista qué número se heredó.
    best_epoch = best_seed_runs[best_idx].get("best_epoch")
    return {
        "selected": winner,
        "seed": best_seed,
        "checkpoint": f"{winner}_seed{best_seed}.pt",
        "best_epoch": best_epoch,
        "reason": reason,
        "auc_mean": results[winner]["auc_mean"],
        "auc_std": results[winner]["auc_std"],
        "auc_month_mean": results[winner]["auc_month_mean"],
        "pr_auc_mean": results[winner].get("pr_auc_mean"),
        "kpi_auc_target": cfg["gnn"]["kpi_auc"],
        "kpi_auc_ok": bool(kpi_ok),
        "kpi_medido_sobre": "auc_roc mensual",
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--skip-train", action="store_true",
                   help="No entrenar; solo comparar reportes existentes")
    p.add_argument("--force", action="store_true",
                   help="Reentrenar las 6 corridas desde cero (ignora el estado)")
    args = p.parse_args()

    cfg = load_config()
    ensure_dirs(cfg)
    if not args.skip_train:
        run_all(cfg, force=args.force)

    results = collect(cfg)
    log.info("--- Resumen por arquitectura ---")
    for m, r in results.items():
        log.info("%-10s AUC walk-forward %.4f ± %.4f (mes: %.4f) | recall %.4f | PR-AUC %.4f",
                 m, r["auc_mean"], r["auc_std"], r["auc_month_mean"],
                 r["recall_mean"], r["pr_auc_mean"])

    selection = select(results, cfg)
    log.info("SELECCIONADA: %s (%s)", selection["selected"], selection["reason"])
    if not selection["kpi_auc_ok"]:
        log.warning("OJO: AUC-ROC mensual %.4f no supera el KPI %.2f — "
                    "revisar features/grafo. (El PR-AUC walk-forward, que es "
                    "lo que SELECCIONA, es %.4f: son métricas distintas.)",
                    selection["auc_month_mean"], selection["kpi_auc_target"],
                    selection["auc_mean"])

    out = resolve(cfg, "models_dir") / "selected_model.json"
    with open(out, "w") as f:
        json.dump({"selection": selection, "results": results}, f, indent=2)
    log.info("Selección registrada en %s", out)


if __name__ == "__main__":
    main()
