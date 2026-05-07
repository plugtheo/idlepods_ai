"""
Canonical agent role registry — shared across all services.

Any service that needs the list of trainable roles (scheduler, trainer
wrapper, etc.) should import from here rather than defining its own copy.
"""

from typing import List

CAPABILITIES: List[str] = [
    "coder",
    "debugger",
    "reviewer",
    "planner",
    "researcher",
    "critic",
]
