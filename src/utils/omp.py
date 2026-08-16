"""
Guard de OpenMP para procesos que cargan PyTorch y XGBoost a la vez.

En macOS cada uno trae su propio runtime de OpenMP (torch empaqueta el suyo;
XGBoost usa el libomp de Homebrew). Con los dos multihilo simultáneamente el
intérprete muere con SIGSEGV al cargar un modelo. Limitar OpenMP a un hilo es
el único workaround que funciona (KMP_DUPLICATE_LIB_OK no basta), y TIENE QUE
aplicarse ANTES de importar torch o xgboost.

En Linux hay un solo runtime, así que no se toca nada y `compute.n_jobs` del
config sigue mandando.

Uso — como primerísima línea ejecutable del módulo, antes de los imports
pesados:

    from src.utils.omp import guard_omp
    guard_omp()

    import torch
    import xgboost as xgb
"""
import os
import sys


def guard_omp() -> bool:
    """
    Fija OMP_NUM_THREADS=1 en macOS. Devuelve True si lo aplicó.

    Asignación directa, NO setdefault: el pipeline padre exporta
    OMP_NUM_THREADS desde compute.n_jobs y el subproceso lo hereda, así que un
    setdefault no llegaría a aplicarse nunca. Aquí no es un valor por defecto
    sino un requisito para no segfaultear.
    """
    if sys.platform != "darwin":
        return False
    os.environ["OMP_NUM_THREADS"] = "1"
    return True
