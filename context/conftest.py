"""
Context service test configuration.

Sets up the 'services' namespace package so test imports like
    from services.context.app.routes.build import _build_hints
resolve correctly against the monorepo layout on the local filesystem,
mirroring the Docker COPY path used at runtime.
"""

import sys
import types
from pathlib import Path

# context/ lives one level below the project root
PROJECT_ROOT = Path(__file__).parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if "services" not in sys.modules:
    _pkg = types.ModuleType("services")
    _pkg.__path__ = [str(PROJECT_ROOT)]
    _pkg.__package__ = "services"
    sys.modules["services"] = _pkg
