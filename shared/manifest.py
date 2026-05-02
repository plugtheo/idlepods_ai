"""
Shared manifest read/write helper with cross-platform file locking.
All writes go through write_manifest_locked to prevent lost-update races.
"""
from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from filelock import FileLock

from shared.contracts.manifest_schema import Manifest


class LegacyManifestError(RuntimeError):
    """Raised when manifest.json is schema_version < 2. Run scripts/migrate_manifest.py."""


def read_manifest(path: Path) -> Manifest:
    """
    Parse manifest.json and return a Manifest.
    Raises LegacyManifestError on schema_version < 2.
    Raises ValueError on schema_version > 2 (unknown future version).
    """
    raw = path.read_text(encoding="utf-8")
    import json
    data = json.loads(raw)
    sv = data.get("schema_version", 1)
    if sv < 2:
        raise LegacyManifestError(
            f"manifest.json is schema_version={sv}. "
            "Run: python scripts/migrate_manifest.py"
        )
    if sv > 2:
        raise ValueError(
            f"manifest.json has unknown schema_version={sv}. "
            "Upgrade this service to support the newer schema."
        )
    return Manifest.model_validate(data)


def write_manifest_locked(
    path: Path,
    mutator: Callable[[Manifest], None],
) -> None:
    """
    Acquire a cross-platform file lock, read (or create) the manifest,
    apply mutator in place, then atomically replace the file.
    Timeout after 30 s to prevent indefinite hangs.
    """
    lock_path = str(path) + ".lock"
    with FileLock(lock_path, timeout=30):
        if path.exists():
            m = read_manifest(path)
        else:
            m = Manifest(
                schema_version=2,
                updated_at=datetime.now(timezone.utc),
                adapters={},
            )
        mutator(m)
        m.updated_at = datetime.now(timezone.utc)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(m.model_dump_json(indent=2), encoding="utf-8")
        os.replace(tmp, path)
