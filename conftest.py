"""
Root test configuration
========================
Replicates the Docker COPY layout so service packages can be imported
using the same paths the services use at runtime:

    Docker:  COPY context/app/  →  /app/services/context/app/
    Tests:   services.context.app  →  <project_root>/context/app/

Achieved by registering a 'services' namespace package whose __path__ is
the project root.  Python then resolves e.g. 'services.context.app' by
walking into the 'context/' directory, honouring all __init__.py files and
relative imports exactly as they run inside the container.

The project root is also added to sys.path so that 'shared.*' imports
(shared/contracts/…, shared/routing/…) resolve correctly.

Run all tests from the project root:
    pytest                      # all services
    pytest gateway/tests/       # one service
"""

import sys
import types
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent

# Ensure the project root is importable so 'shared.*' works.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Register a 'services' namespace package pointing at the project root.
# Python resolves:
#   services.context.app       →  PROJECT_ROOT/context/app/
#   services.experience.app    →  PROJECT_ROOT/experience/app/
#   etc.
if "services" not in sys.modules:
    _pkg = types.ModuleType("services")
    _pkg.__path__ = [str(PROJECT_ROOT)]
    _pkg.__package__ = "services"
    sys.modules["services"] = _pkg
