# imagedaemon/__init__.py
import logging
from pathlib import Path

# ------------------------------------------------------------------
# default: log INFO+ to stdout
# ------------------------------------------------------------------
_console = logging.StreamHandler()
_console.setFormatter(logging.Formatter("[%(levelname).1s %(name)s] %(message)s"))
logging.basicConfig(
    level=logging.INFO,
    handlers=[_console],
    force=True,  # overwrite root handlers in notebooks
)

# expose a package‑level logger for convenience
log = logging.getLogger("imagedaemon")


# ------------------------------------------------------------------
# later: daemon can call e.g.  imagedaemon.add_file_logger("/var/log/…")
# ------------------------------------------------------------------
def add_file_logger(path: str | Path, level: int = logging.INFO):
    f = logging.FileHandler(path)
    f.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)s [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    root = logging.getLogger()
    root.addHandler(f)
    root.setLevel(min(root.level, level))
