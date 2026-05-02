"""
Cross-process locking test: 4 processes each increment a counter stored in
the manifest via write_manifest_locked. Final count must equal 4 (no lost writes).
"""
import json
import multiprocessing
import tempfile
from datetime import datetime, timezone
from pathlib import Path


def _increment(manifest_path_str: str) -> None:
    """Worker: increment the count stored in adapters.__counter.active_version."""
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(manifest_path_str).resolve().parents[3]))

    from shared.manifest import write_manifest_locked
    from shared.contracts.manifest_schema import AdapterEntry, HistoryEntry, Manifest

    _NOW = datetime.now(timezone.utc)

    def mutator(m: Manifest) -> None:
        existing = m.adapters.get("__counter")
        current = int(existing.active_version) if existing else 0
        dummy_history: list[HistoryEntry] = existing.history if existing else []
        m.adapters["__counter"] = AdapterEntry(
            schema_version=2,
            active_version=str(current + 1),
            active_path="/dev/null",
            previous_version=str(current),
            previous_path="/dev/null",
            backend="primary",
            updated_at=_NOW,
            history=dummy_history,
        )

    write_manifest_locked(Path(manifest_path_str), mutator)


def test_concurrent_counter_no_lost_writes():
    with tempfile.TemporaryDirectory() as td:
        mp = Path(td) / "manifest.json"
        n = 4
        with multiprocessing.Pool(processes=n) as pool:
            pool.map(_increment, [str(mp)] * n)

        from shared.manifest import read_manifest
        m = read_manifest(mp)
        count = int(m.adapters["__counter"].active_version)
        assert count == n, (
            f"Expected {n} but got {count} — lost write detected"
        )
