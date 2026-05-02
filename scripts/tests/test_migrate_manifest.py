"""
Tests for scripts/migrate_manifest.py:
- v1 → v2 migration produces expected snapshot
- --check prints diff and does not write
- idempotency: re-running on v2 is a no-op
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_MIGRATE = str(_PROJECT_ROOT / "scripts" / "migrate_manifest.py")

_V1_FIXTURE = {
    "schema_version": 1,
    "generated_at": "2026-03-27T00:00:00+00:00",
    "adapters": {
        "coding_lora": {
            "capability": "coding",
            "backend": "primary",
            "active_version": "1.0.0",
            "active_path": "/lora/coding_lora",
            "previous_version": "",
            "previous_path": "",
            "updated_at": "2026-03-27T00:00:00+00:00",
            "history": [
                {
                    "version": "1.0.0",
                    "status": "active",
                    "created_at": "2026-03-27T00:00:00+00:00",
                    "n_samples": 50,
                    "final_loss": 1.23,
                    "size_mb": 10.0,
                }
            ],
        }
    },
}


def _write_v1(path: Path) -> None:
    path.write_text(json.dumps(_V1_FIXTURE, indent=2), encoding="utf-8")


def _run_migrate(manifest_path: Path, extra_args=()) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, _MIGRATE, "--manifest", str(manifest_path), *extra_args],
        capture_output=True,
        text=True,
    )


def test_migration_produces_v2():
    with tempfile.TemporaryDirectory() as td:
        mp = Path(td) / "manifest.json"
        _write_v1(mp)
        result = _run_migrate(mp)
        assert result.returncode == 0, result.stderr

        data = json.loads(mp.read_text(encoding="utf-8"))
        assert data["schema_version"] == 2
        entry = data["adapters"]["coding_lora"]
        assert entry["schema_version"] == 2
        assert entry["active_version"] == "1.0.0"
        h = entry["history"][0]
        assert h["dataset_hash"] == "legacy"
        assert h["trainer_version"] == "legacy"
        assert "eval_metrics" in h
        assert h["n_samples"] == 50
        assert h["final_loss"] == pytest.approx(1.23)

        # Backup file should exist
        baks = list(Path(td).glob("manifest.json.v1.bak.*"))
        assert len(baks) == 1


def test_check_mode_does_not_write():
    with tempfile.TemporaryDirectory() as td:
        mp = Path(td) / "manifest.json"
        _write_v1(mp)
        original = mp.read_bytes()
        result = _run_migrate(mp, extra_args=["--check"])
        assert result.returncode == 0
        assert "---" in result.stdout or "+++" in result.stdout
        assert mp.read_bytes() == original


def test_idempotent_on_v2():
    with tempfile.TemporaryDirectory() as td:
        mp = Path(td) / "manifest.json"
        _write_v1(mp)
        # First migration
        _run_migrate(mp)
        v2_bytes = mp.read_bytes()
        # Second run — should be a no-op
        result = _run_migrate(mp)
        assert result.returncode == 0
        assert "already at v2" in result.stdout.lower()
        assert mp.read_bytes() == v2_bytes
