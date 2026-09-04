"""
C++ engine, exposed under the historical `engine` package name.

TRANSITIONAL SHIM (PLAN.md §1.2, §5). Re-exports the C++ engine so existing
callers - run.py, tournament.py and all twelve agents - keep working with
`from engine. import ...` unchanged. Deleted at M8 along with the bindings.

The frozen Python reference these replace is in engine/engine_old/.
"""

import os
import sys

# The compiled module lives in the build directory, which is not on sys.path.
# Look beside this file first (a copied/installed .pyd), then in build/.
_here = os.path.dirname(os.path.abspath(__file__))
for _candidate in (_here, os.path.join(_here, "build")):
    if _candidate not in sys.path:
        sys.path.insert(0, _candidate)

import oo_engine  # noqa: E402,F401

__all__ = ["oo_engine"]
