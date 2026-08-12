"""
CL Paso 1 — GATILLO del aprendizaje (detección de novedad / concept drift).

La señal: fraudes CONFIRMADOS por el equipo humano a los que el modelo dio
score BAJO (<0.5). Score alto = patrón conocido, nada que aprender.
Fraude real + score 0.18 = patrón que el modelo NO tiene.

Mecanismo (después de la validación humana):
1. Cada fraude confirmado se compara con su score original.
2. Los de score < 0.5 se acumulan en una cola (NoveltyQueue).
3. EJECUTORES según criterio de disparo:
   - CountExecutor: dispara al acumular N casos (default 50).
   - MissRateExecutor: dispara si >30% de los fraudes confirmados de la
     ventana no fueron detectados.
4. Cuando cualquiera dispara -> arranca el pipeline de fine-tuning.

El KPI <48h se mide desde que hay casos suficientes (disparo) hasta el
despliegue del modelo actualizado.
"""
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.common import get_logger, load_config, resolve

log = get_logger("cl.trigger")


@dataclass
class ConfirmedCase:
    """Un caso pasado por validación humana."""
    tid: int
    score: float          # score que dio el modelo en operación
    is_fraud: int         # etiqueta confirmada por el humano
    confirmed_at: float = field(default_factory=time.time)


class BaseExecutor:
    """Interfaz de ejecutor: decide si el estado de la cola amerita disparo."""
    name = "base"

    def should_fire(self, queue: "NoveltyQueue") -> bool:
        raise NotImplementedError


class CountExecutor(BaseExecutor):
    """Dispara por CONTEO: N fraudes no detectados acumulados."""
    name = "count"

    def __init__(self, min_cases: int):
        self.min_cases = min_cases

    def should_fire(self, queue):
        fire = len(queue.novel_cases) >= self.min_cases
        if fire:
            log.info("[executor:count] %d >= %d casos -> DISPARO",
                     len(queue.novel_cases), self.min_cases)
        return fire


class MissRateExecutor(BaseExecutor):
    """Dispara por TASA: >X% de los fraudes confirmados no fueron detectados."""
    name = "miss_rate"

    def __init__(self, miss_rate: float, min_confirmed: int = 20):
        self.miss_rate = miss_rate
        self.min_confirmed = min_confirmed  # evita disparos con muestras chicas

    def should_fire(self, queue):
        confirmed = queue.confirmed_frauds
        if confirmed < self.min_confirmed:
            return False
        rate = len(queue.novel_cases) / confirmed
        fire = rate > self.miss_rate
        if fire:
            log.info("[executor:miss_rate] tasa no detectados %.1f%% > %.0f%% -> DISPARO",
                     100 * rate, 100 * self.miss_rate)
        return fire


class NoveltyQueue:
    """
    Cola de novedad. Recibe casos post-validación humana y aplica el filtro:
    fraude confirmado + score bajo -> se acumula. El resto se descarta
    (score alto = ya lo sabe; legítima = no hay nada que aprender).
    Persiste a disco para sobrevivir reinicios.
    """

    def __init__(self, cfg: dict | None = None, persist: bool = True):
        self.cfg = cfg or load_config()
        t = self.cfg["continual_learning"]["trigger"]
        self.score_threshold = t["score_threshold"]
        self.executors: list[BaseExecutor] = [
            CountExecutor(t["min_cases"]),
            MissRateExecutor(t["miss_rate"]),
        ]
        self.novel_cases: list[ConfirmedCase] = []
        self.confirmed_frauds = 0   # todos los fraudes confirmados en la ventana
        self.persist = persist
        self.path = resolve(self.cfg, "artifacts_dir") / "novelty_queue.json"
        if persist and self.path.exists():
            self._load()

    def ingest(self, case: ConfirmedCase) -> bool:
        """
        Entrada post-validación humana. Devuelve True si algún ejecutor
        disparó el reentrenamiento.
        """
        if case.is_fraud == 1:
            self.confirmed_frauds += 1
            if case.score < self.score_threshold:
                self.novel_cases.append(case)
                log.debug("Patrón nuevo acumulado: tid=%d score=%.2f (%d en cola)",
                          case.tid, case.score, len(self.novel_cases))
        if self.persist:
            self._save()
        return any(ex.should_fire(self) for ex in self.executors)

    def drain(self) -> list[ConfirmedCase]:
        """Al disparar: entrega los casos acumulados y resetea la ventana."""
        cases, self.novel_cases, self.confirmed_frauds = self.novel_cases, [], 0
        if self.persist:
            self._save()
        return cases

    # --- persistencia ---
    def _save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w") as f:
            json.dump({"confirmed_frauds": self.confirmed_frauds,
                       "novel_cases": [asdict(c) for c in self.novel_cases]}, f)

    def _load(self):
        with open(self.path) as f:
            d = json.load(f)
        self.confirmed_frauds = d["confirmed_frauds"]
        self.novel_cases = [ConfirmedCase(**c) for c in d["novel_cases"]]
