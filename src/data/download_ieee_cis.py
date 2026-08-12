"""
Paso 0 — Descarga del dataset IEEE-CIS (Kaggle / Vesta Corp., 2019).
590.540 transacciones, 432 variables, 3.5% fraude (27.6:1).

TODO lo configurable vive en config/config.yaml, sección `kaggle:` — nombre de
la competencia, archivos a bajar y DÓNDE buscar la credencial. El token en sí
NO va ahí: config.yaml se versiona.

La credencial se busca en este orden:
  1. Variable de entorno (`kaggle.token_env`, por defecto KAGGLE_ACCESS_TOKEN),
     o el par KAGGLE_USERNAME + KAGGLE_KEY.
  2. Archivo `.env` en la raíz del proyecto (no versionado; ver .env.example).
  3. `kaggle.token_file` (~/.kaggle/access_token) o ~/.kaggle/kaggle.json ya
     instalados en el sistema.

Si la encuentra por (1) o (2) y todavía no está instalada, la escribe en
`token_file` con permisos 600 — equivale a hacer a mano:
    mkdir -p ~/.kaggle && echo "<token>" > ~/.kaggle/access_token
    chmod 600 ~/.kaggle/access_token

Además hay que ACEPTAR LAS REGLAS de la competencia una vez con esa misma
cuenta, o la API responde 403 aunque el token sea válido:
    https://www.kaggle.com/competitions/ieee-fraud-detection/rules

Uso:
  python -m src.data.download_ieee_cis            # los archivos de kaggle.files
  python -m src.data.download_ieee_cis --all      # además kaggle.files_extra
  python -m src.data.download_ieee_cis --force    # re-descargar aunque existan
  python -m src.data.download_ieee_cis --check    # solo verificar credenciales
"""
import argparse
import os
import stat
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.common import ROOT, ensure_dirs, get_logger, load_config, resolve

log = get_logger("download")


def _help_text(cfg) -> str:
    k = cfg["kaggle"]
    return f"""
No hay credenciales de Kaggle utilizables. Elige UNA opción:

  a) RECOMENDADA — usuario + API key en .env (no expira, no se versiona):
         cp .env.example .env
     y pon en .env las dos líneas que salen de descargar kaggle.json desde
     https://www.kaggle.com/settings -> API -> "Create New Token":
         KAGGLE_USERNAME=tu_usuario
         KAGGLE_KEY=tu_api_key

  b) Lo mismo por variables de entorno:
         export KAGGLE_USERNAME=tu_usuario
         export KAGGLE_KEY=tu_api_key

  c) Token de sesión (rápido, pero EXPIRA en pocas horas):
         kaggle auth login
     o pegándolo en .env como {k['token_env']}=KGAT_...

Y acepta las reglas de la competencia con esa misma cuenta (obligatorio, o la
API responde 403 aunque la credencial sea válida):
    https://www.kaggle.com/competitions/{k['competition']}/rules

Alternativa sin API: baja {', '.join(k['files'])} a mano desde la pestaña Data
y déjalos en data/raw/.
"""


def load_dotenv(path: Path) -> dict:
    """Parser mínimo de .env (KEY=VALUE), sin dependencias externas."""
    out = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        val = val.strip().strip('"').strip("'")
        if val:
            out[key.strip()] = val
    return out


def install_credentials(cfg) -> str | None:
    """
    Deja las credenciales listas para la API de Kaggle.
    Devuelve una etiqueta con la vía usada, o None si no encontró ninguna.
    """
    k = cfg["kaggle"]
    env_name = k.get("token_env", "KAGGLE_API_TOKEN")
    token_file = Path(os.path.expanduser(k.get("token_file", "~/.kaggle/access_token")))

    # (1) entorno del proceso  +  (2) .env del proyecto (sin pisar el entorno)
    values = dict(load_dotenv(ROOT / ".env"))
    values.update({key: v for key, v in os.environ.items() if v})

    # usuario + API key: la vía duradera (la librería los lee del entorno)
    if values.get("KAGGLE_USERNAME") and values.get("KAGGLE_KEY"):
        os.environ["KAGGLE_USERNAME"] = values["KAGGLE_USERNAME"]
        os.environ["KAGGLE_KEY"] = values["KAGGLE_KEY"]
        # Un token de sesión previo TIENE PRIORIDAD sobre usuario/key en el SDK:
        # si quedó uno viejo, la API responde 401 aunque la key sea correcta.
        # Se aparta (reversible) en vez de dejar un fallo incomprensible.
        if token_file.exists():
            disabled = token_file.with_suffix(token_file.suffix + ".disabled")
            token_file.rename(disabled)
            log.warning("Había un token de sesión en %s que tiene prioridad sobre "
                        "usuario/key y provoca 401. Lo aparté como %s.",
                        token_file, disabled.name)
        return f"KAGGLE_USERNAME/KAGGLE_KEY (usuario {values['KAGGLE_USERNAME']})"

    # token de acceso: si no está instalado, lo instalamos con permisos 600.
    # Se acepta también el alias KAGGLE_ACCESS_TOKEN por comodidad.
    token = next((values[n] for n in (env_name, "KAGGLE_API_TOKEN",
                                      "KAGGLE_ACCESS_TOKEN") if values.get(n)), None)
    if token:
        os.environ[env_name] = token          # el SDK lo lee directo de aquí
        current = token_file.read_text().strip() if token_file.exists() else None
        if current != token:
            token_file.parent.mkdir(parents=True, exist_ok=True)
            token_file.write_text(token + "\n", encoding="utf-8")
            token_file.chmod(stat.S_IRUSR | stat.S_IWUSR)          # 600
            log.info("Credencial instalada en %s (permisos 600).", token_file)
        return f"{env_name} -> {token_file}"

    # (3) ya instaladas en el sistema
    if token_file.exists():
        return f"{token_file} (ya presente)"
    kjson = Path.home() / ".kaggle" / "kaggle.json"
    if kjson.exists():
        return f"{kjson} (ya presente)"
    return None


def authenticate(cfg):
    """Devuelve una KaggleApi autenticada o termina con instrucciones claras."""
    origen = install_credentials(cfg)
    if origen is None:
        log.error("No encontré credenciales de Kaggle.")
        sys.exit(_help_text(cfg))
    log.info("Credenciales desde: %s", origen)

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        sys.exit("Falta el paquete de Kaggle:  pip install kaggle")

    api = KaggleApi()
    try:
        api.authenticate()
        api.competitions_list(search=cfg["kaggle"]["competition"])   # prueba real
    except SystemExit:
        # el SDK imprime su propia ayuda y aborta cuando el token no sirve
        log.error("Kaggle rechazó las credenciales (¿token expirado?).")
        sys.exit(_help_text(cfg))
    except Exception as e:                                        # noqa: BLE001
        log.error("Kaggle rechazó las credenciales: %s", e)
        sys.exit(_help_text(cfg))
    return api


def unzip_all(raw_dir: Path):
    """Kaggle entrega un .zip por archivo: extraer y limpiar."""
    for z in raw_dir.glob("*.zip"):
        with zipfile.ZipFile(z) as zf:
            zf.extractall(raw_dir)
        z.unlink()
        log.info("Extraído y eliminado %s", z.name)


def main():
    p = argparse.ArgumentParser(description="Descarga el dataset IEEE-CIS")
    p.add_argument("--all", action="store_true",
                   help="bajar también kaggle.files_extra (el pipeline no los usa)")
    p.add_argument("--force", action="store_true",
                   help="re-descargar aunque los archivos ya existan")
    p.add_argument("--check", action="store_true",
                   help="solo verificar credenciales, sin descargar nada")
    args = p.parse_args()

    cfg = load_config()
    ensure_dirs(cfg)
    k = cfg["kaggle"]
    raw_dir = resolve(cfg, "raw_dir")
    required = list(k["files"])
    wanted = required + (list(k.get("files_extra", [])) if args.all else [])

    if args.check:
        authenticate(cfg)
        log.info("Credenciales válidas. Listo para descargar.")
        return

    missing = [f for f in wanted if not (raw_dir / f).exists()]
    if not missing and not args.force:
        log.info("Los archivos ya están en %s — nada que hacer.", raw_dir)
        return

    api = authenticate(cfg)
    log.info("Descargando %d archivo(s) de '%s' en %s ...",
             len(missing), k["competition"], raw_dir)

    for fname in missing:
        log.info("  -> %s", fname)
        try:
            api.competition_download_file(k["competition"], fname,
                                          path=str(raw_dir), force=args.force,
                                          quiet=False)
        except Exception as e:                                    # noqa: BLE001
            msg = str(e)
            if "403" in msg or "Forbidden" in msg:
                log.error("403 al bajar %s. Las credenciales son válidas: lo que "
                          "falta es ACEPTAR LAS REGLAS de la competencia.", fname)
                sys.exit(f"https://www.kaggle.com/competitions/{k['competition']}/rules")
            log.error("No pude bajar %s (%s). Intento el paquete completo...",
                      fname, e)
            api.competition_download_files(k["competition"], path=str(raw_dir),
                                           force=args.force, quiet=False)
            break

    unzip_all(raw_dir)

    faltan = [f for f in required if not (raw_dir / f).exists()]
    if faltan:
        sys.exit(f"Terminó la descarga pero faltan: {faltan}. Revisa {raw_dir}")

    for f in required:
        log.info("  %-24s %7.1f MB", f, (raw_dir / f).stat().st_size / 1e6)
    log.info("Listo. Siguiente paso:  python -m src.data.preprocessing")


if __name__ == "__main__":
    main()
