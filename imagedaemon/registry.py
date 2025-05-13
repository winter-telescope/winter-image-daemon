"""
imagedaemon.registry
====================

Central lookup table that maps a *camera name* ("winter", "easycam", …)
to an *adapter instance* (its Pipelines class wired up with Meta).

Contract for a camera plug‑in package
------------------------------------
`cameras/<camera>/__init__.py` must expose **two** objects:

    Meta       – a pydantic / dataclass instance or *class* with attribute `.name`
    Pipelines  – a class that takes `meta` as first ctor arg

Example:

    from .meta import WinterMeta
    from .pipelines import WinterPipelines

    meta = WinterMeta()
    Pipelines = WinterPipelines

    __all__ = ["meta", "Pipelines"]

The registry will import the module and call:
    adapter = Pipelines(meta)
"""

from __future__ import annotations

import pkgutil
import sys
from importlib import import_module, metadata
from pathlib import Path
from types import ModuleType
from typing import Dict

_ADAPTERS: Dict[str, object] = {}  # name → Pipelines instance


# ----------------------------------------------------------------------
# 1. helpers
# ----------------------------------------------------------------------
def _register_module(mod: ModuleType):
    """Look for (meta, Pipelines) combo inside *mod* and register it."""
    meta = getattr(mod, "meta", None)

    # Some plug‑ins export a *class* instead of an instance.
    if meta is None:
        meta_cls = getattr(mod, "Meta", None)
        if meta_cls is not None:
            meta = meta_cls()  # instantiate

    pipelines_cls = getattr(mod, "Pipelines", None) or getattr(
        mod, "pipelines_cls", None
    )

    if meta is None or pipelines_cls is None:
        return  # not a valid plug‑in

    adapter = pipelines_cls(meta)
    _ADAPTERS[meta.name] = adapter


# ----------------------------------------------------------------------
# 2. built‑in cameras (src/imagedaemon/cameras/*)
# ----------------------------------------------------------------------
def _discover_builtin():
    pkg_path = Path(__file__).resolve().parent / "cameras"
    for modinfo in pkgutil.iter_modules([str(pkg_path)]):
        mod_name = f"imagedaemon.cameras.{modinfo.name}"
        try:
            module = import_module(mod_name)
            _register_module(module)
        except Exception as exc:
            print(f"[registry] failed to import {mod_name}: {exc}", file=sys.stderr)


# ----------------------------------------------------------------------
# 3. third‑party cameras via entry points
# ----------------------------------------------------------------------
def _discover_entry_points():
    for ep in metadata.entry_points(group="imagedaemon.cameras"):
        try:
            module = import_module(ep.module)
            _register_module(module)
        except Exception as exc:
            print(f"[registry] entry‑point {ep.name} failed: {exc}", file=sys.stderr)


# ----------------------------------------------------------------------
# 4. public API
# ----------------------------------------------------------------------
def get(name: str):
    """Return the Pipelines adapter for *name* (case‑insensitive)."""
    try:
        return _ADAPTERS[name.lower()]
    except KeyError as e:
        raise KeyError(
            f"Camera '{name}' not found. Available: {', '.join(_ADAPTERS)}"
        ) from e


def available():
    """List the registered camera names."""
    return list(_ADAPTERS)


# ----------------------------------------------------------------------
# 5. run discovery exactly once on import
# ----------------------------------------------------------------------
_discover_builtin()
_discover_entry_points()
