"""
Makes the frozen Python reference importable as `engine.*` for the dump tools.

engine_old was moved to engine/engine_old/, but agents/ (and engine_old's own
docstrings) still refer to it as the `engine` package. Rather than edit the frozen
reference, alias it in sys.modules.

ORDER MATTERS. Aliasing `engine` before the submodules are imported sends the
import machinery into infinite recursion: `engine` then resolves to the
engine_old package, which has no `engine_old` attribute of its own, and the
namespace-package path finder retries forever. So import every submodule under its
real name FIRST, then rebind.

Import this module before importing anything from `engine.*` or `agents.*`.
"""

import importlib
import os
import sys

_SUBMODULES = ("geometry", "state", "terrain", "setup", "placement", "movement", "battle", "buy",
               "collect", "turn")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def install():
    """Idempotent. Returns the repo root, already on sys.path."""
    if REPO_ROOT not in sys.path:
        sys.path.insert(0, REPO_ROOT)
    if sys.modules.get("engine") is not None and "engine.turn" in sys.modules:
        return REPO_ROOT

    # Step 1: import under the real package path, before any aliasing.
    real = {name: importlib.import_module(f"engine.engine_old.{name}") for name in _SUBMODULES}
    package = importlib.import_module("engine.engine_old")

    # Step 2: rebind. sys.modules entries for engine.engine_old.* survive, so the
    # relative imports inside the reference modules keep resolving.
    sys.modules["engine"] = package
    for name, module in real.items():
        sys.modules[f"engine.{name}"] = module
    return REPO_ROOT
