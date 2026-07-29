"""Explicit, trusted module loading for advanced RAG components.

Plugins are never discovered by scanning writable directories.  A deployment
must name importable modules in ``RAG_PLUGIN_MODULES``; each module exposes a
single ``register(registry)`` function and is treated as trusted application
code.
"""

from __future__ import annotations

import importlib
from collections.abc import Iterable
from typing import Any


def load_rag_plugin_modules(
    registry: Any,
    module_names: Iterable[str],
) -> list[dict[str, Any]]:
    loaded: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw_name in module_names:
        name = str(raw_name).strip()
        if not name or name in seen:
            continue
        seen.add(name)
        before = set(registry.pipeline_factories)
        module = importlib.import_module(name)
        register = getattr(module, "register", None)
        if not callable(register):
            raise ValueError(
                f"RAG plugin module {name} must expose register(registry)"
            )
        register(registry)
        added = sorted(set(registry.pipeline_factories) - before)
        loaded.append(
            {
                "module": name,
                "status": "loaded",
                "pipelines": added,
            }
        )
    return loaded


__all__ = ["load_rag_plugin_modules"]
