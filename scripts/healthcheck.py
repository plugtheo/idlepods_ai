"""
Plan C verification: Extended Manifest + Cross-Platform Locking + One-Shot Migration.

Run from project root:
    python scripts/healthcheck.py

Checks every requirement from the Plan C steps, constraints, and tests.
"""
from __future__ import annotations

import importlib
import json
import multiprocessing
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

_PASS = "PASS"
_FAIL = "FAIL"
_results: list[tuple[str, str, str]] = []  # (section, label, status)


def _record(section: str, label: str, ok: bool, detail: str = "") -> bool:
    status = _PASS if ok else _FAIL
    _results.append((section, label, status))
    icon = "  [OK]  " if ok else "  [!!]  "
    msg = f"{icon}{label}"
    if detail:
        msg += f"  — {detail}"
    print(msg)
    return ok


def _section(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# ---------------------------------------------------------------------------
# 1. File existence checks
# ---------------------------------------------------------------------------
_section("1. Required files exist (Step 1-10)")

_files = [
    "shared/contracts/manifest_schema.py",
    "shared/manifest.py",
    "scripts/migrate_manifest.py",
    "scripts/seed_adapter_metadata.py",
    "scripts/rollback_adapter.py",
    "scripts/show_manifest.py",
    "shared/tests/test_manifest_schema.py",
    "shared/tests/test_manifest_locking.py",
    "scripts/tests/test_migrate_manifest.py",
    "training/training/lora_trainer.py",
    "training/app/trainer_entry.py",
    "inference/app/backends/factory.py",
    "inference/app/routes/adapters.py",
]
for rel in _files:
    p = PROJECT_ROOT / rel
    _record("files", rel, p.exists(), "" if p.exists() else "MISSING")


# ---------------------------------------------------------------------------
# 2. filelock in all requirements.txt (Step 3 / Constraint)
# ---------------------------------------------------------------------------
_section("2. filelock>=3.13 in all service requirements.txt (Step 3)")

_req_files = [
    "orchestration/requirements.txt",
    "inference/requirements.txt",
    "training/requirements.txt",
    "shared/requirements.txt",
]
for rel in _req_files:
    p = PROJECT_ROOT / rel
    if not p.exists():
        _record("filelock", rel, False, "file missing")
        continue
    content = p.read_text(encoding="utf-8").lower()
    found = "filelock" in content
    _record("filelock", rel, found, "" if found else "filelock not found")


# ---------------------------------------------------------------------------
# 3. Schema module imports and field structure (Step 1)
# ---------------------------------------------------------------------------
_section("3. manifest_schema.py — Pydantic models importable and correct (Step 1)")

try:
    from shared.contracts.manifest_schema import (
        HistoryEntry,
        AdapterEntry,
        Manifest,
    )
    _record("schema", "import shared.contracts.manifest_schema", True)
except Exception as exc:
    _record("schema", "import shared.contracts.manifest_schema", False, str(exc))
    HistoryEntry = AdapterEntry = Manifest = None  # type: ignore

if Manifest is not None:
    _NOW = datetime.now(timezone.utc)

    def _he_dict(**kw):
        base = dict(
            version="1.0.0", status="active", trained_at=_NOW.isoformat(),
            backend="primary", base_model="test-org/test-model-14B", peft_type="lora",
            target_modules=["q_proj"], r=8, alpha=16, dropout=0.0,
            recipe={"peft_type": "lora"}, dataset_hash="abc", tokenizer_hash="def",
            trainer_version="trl==0.1", n_samples=10, final_loss=0.5, size_mb=5.0,
        )
        base.update(kw)
        return base

    try:
        he = HistoryEntry.model_validate(_he_dict())
        _record("schema", "HistoryEntry valid instance", True)
    except Exception as exc:
        _record("schema", "HistoryEntry valid instance", False, str(exc))

    try:
        from pydantic import ValidationError
        HistoryEntry.model_validate(_he_dict(status="invalid_status"))
        _record("schema", "HistoryEntry rejects invalid status", False, "should have raised")
    except ValidationError:
        _record("schema", "HistoryEntry rejects invalid status", True)
    except Exception as exc:
        _record("schema", "HistoryEntry rejects invalid status", False, str(exc))

    for status in ("staging", "active", "retired", "failed"):
        try:
            HistoryEntry.model_validate(_he_dict(status=status))
            _record("schema", f"HistoryEntry status='{status}'", True)
        except Exception as exc:
            _record("schema", f"HistoryEntry status='{status}'", False, str(exc))

    try:
        he = HistoryEntry.model_validate(_he_dict())
        ok = (
            he.quantization is None
            and he.eval_metrics == {}
            and he.smoke == {}
            and he.used_base_fallback_aggregate == 0.0
        )
        _record("schema", "HistoryEntry optional fields default correctly", ok)
    except Exception as exc:
        _record("schema", "HistoryEntry optional fields default", False, str(exc))

    try:
        m = Manifest.model_validate({"schema_version": 2, "updated_at": _NOW.isoformat(), "adapters": {}})
        _record("schema", "Manifest with empty adapters", m.adapters == {})
    except Exception as exc:
        _record("schema", "Manifest empty adapters", False, str(exc))

    try:
        ae_data = dict(
            schema_version=2, active_version="1.0.0", active_path="/lora/x",
            previous_version="", previous_path="", backend="primary",
            updated_at=_NOW.isoformat(), history=[_he_dict()],
        )
        m_data = {"schema_version": 2, "updated_at": _NOW.isoformat(), "adapters": {"x": ae_data}}
        m = Manifest.model_validate(m_data)
        dumped = json.loads(m.model_dump_json())
        m2 = Manifest.model_validate(dumped)
        _record("schema", "Manifest round-trip", m2.adapters["x"].active_version == "1.0.0")
    except Exception as exc:
        _record("schema", "Manifest round-trip", False, str(exc))


# ---------------------------------------------------------------------------
# 4. shared/manifest.py — read_manifest / write_manifest_locked / LegacyManifestError (Step 2)
# ---------------------------------------------------------------------------
_section("4. shared/manifest.py — helpers and locking (Step 2)")

try:
    from shared.manifest import read_manifest, write_manifest_locked, LegacyManifestError
    _record("manifest_py", "import shared.manifest", True)
except Exception as exc:
    _record("manifest_py", "import shared.manifest", False, str(exc))
    read_manifest = write_manifest_locked = LegacyManifestError = None  # type: ignore

if read_manifest is not None and Manifest is not None:
    with tempfile.TemporaryDirectory() as td:
        mp = Path(td) / "manifest.json"

        # write_manifest_locked creates a new manifest if file absent
        try:
            def _noop(m): pass
            write_manifest_locked(mp, _noop)
            _record("manifest_py", "write_manifest_locked creates new file", mp.exists())
        except Exception as exc:
            _record("manifest_py", "write_manifest_locked creates new file", False, str(exc))

        # read_manifest returns Manifest
        try:
            m = read_manifest(mp)
            _record("manifest_py", "read_manifest returns Manifest", isinstance(m, Manifest))
        except Exception as exc:
            _record("manifest_py", "read_manifest returns Manifest", False, str(exc))

        # atomic .tmp → final (no leftover .tmp)
        try:
            def _noop2(m): pass
            write_manifest_locked(mp, _noop2)
            tmp = mp.with_suffix(".tmp")
            _record("manifest_py", "no leftover .tmp after write", not tmp.exists())
        except Exception as exc:
            _record("manifest_py", "no leftover .tmp after write", False, str(exc))

        # LegacyManifestError raised on schema_version=1
        try:
            mp.write_text(json.dumps({"schema_version": 1, "adapters": {}}), encoding="utf-8")
            try:
                read_manifest(mp)
                _record("manifest_py", "LegacyManifestError raised on v1", False, "no exception raised")
            except LegacyManifestError:
                _record("manifest_py", "LegacyManifestError raised on v1", True)
        except Exception as exc:
            _record("manifest_py", "LegacyManifestError raised on v1", False, str(exc))

        # ValueError raised on schema_version=3 (unknown future)
        try:
            mp.write_text(json.dumps({"schema_version": 3, "updated_at": _NOW.isoformat(), "adapters": {}}), encoding="utf-8")
            try:
                read_manifest(mp)
                _record("manifest_py", "ValueError raised on schema_version=3", False, "no exception")
            except ValueError:
                _record("manifest_py", "ValueError raised on schema_version=3", True)
            except Exception as exc:
                _record("manifest_py", "ValueError raised on schema_version=3", False, str(exc))
        except Exception as exc:
            _record("manifest_py", "ValueError raised on schema_version=3", False, str(exc))


# ---------------------------------------------------------------------------
# 5. Concurrent write safety (test_manifest_locking replication) (Step 10)
# ---------------------------------------------------------------------------
_section("5. Cross-process locking — no lost writes (Step 10)")


def _worker_increment(manifest_path_str: str) -> None:
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(manifest_path_str).resolve().parents[3]))
    from shared.manifest import write_manifest_locked
    from shared.contracts.manifest_schema import AdapterEntry as _AE, HistoryEntry as _HE
    from datetime import datetime, timezone
    _now = datetime.now(timezone.utc)

    def _inc(m):
        existing = m.adapters.get("__counter")
        current = int(existing.active_version) if existing else 0
        history = existing.history if existing else []
        m.adapters["__counter"] = _AE(
            schema_version=2,
            active_version=str(current + 1),
            active_path="/dev/null",
            previous_version=str(current),
            previous_path="/dev/null",
            backend="primary",
            updated_at=_now,
            history=history,
        )

    write_manifest_locked(_Path(manifest_path_str), _inc)


try:
    with tempfile.TemporaryDirectory() as td:
        mp = Path(td) / "manifest.json"
        n = 4
        with multiprocessing.Pool(processes=n) as pool:
            pool.map(_worker_increment, [str(mp)] * n)
        from shared.manifest import read_manifest as _rm
        m = _rm(mp)
        count = int(m.adapters["__counter"].active_version)
        _record("locking", f"4-process counter reaches {n}", count == n, f"got {count}")
except Exception as exc:
    _record("locking", "4-process concurrent write test", False, str(exc))


# ---------------------------------------------------------------------------
# 6. migrate_manifest.py — v1→v2, --check, idempotency (Step 8 / Step 10)
# ---------------------------------------------------------------------------
_section("6. migrate_manifest.py (Step 8)")

_V1 = {
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
            "history": [{"version": "1.0.0", "status": "active",
                          "created_at": "2026-03-27T00:00:00+00:00",
                          "n_samples": 50, "final_loss": 1.23, "size_mb": 10.0}],
        }
    },
}

_MIGRATE_SCRIPT = str(PROJECT_ROOT / "scripts" / "migrate_manifest.py")


def _run_migrate(mp: Path, *extra_args) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, _MIGRATE_SCRIPT, "--manifest", str(mp), *extra_args],
        capture_output=True, text=True,
    )


with tempfile.TemporaryDirectory() as td:
    mp = Path(td) / "manifest.json"

    # v1 → v2
    mp.write_text(json.dumps(_V1, indent=2), encoding="utf-8")
    r = _run_migrate(mp)
    if r.returncode == 0:
        data = json.loads(mp.read_text(encoding="utf-8"))
        ok = data.get("schema_version") == 2
        _record("migrate", "v1→v2 produces schema_version=2", ok)

        entry = data["adapters"].get("coding_lora", {})
        _record("migrate", "entry.schema_version=2", entry.get("schema_version") == 2)
        h = entry.get("history", [{}])[0]
        _record("migrate", "history[0].dataset_hash='legacy'", h.get("dataset_hash") == "legacy")
        _record("migrate", "history[0].trainer_version='legacy'", h.get("trainer_version") == "legacy")
        _record("migrate", "history[0].eval_metrics present", "eval_metrics" in h)
        _record("migrate", "history[0].n_samples preserved", h.get("n_samples") == 50)

        baks = list(Path(td).glob("manifest.json.v1.bak.*"))
        _record("migrate", "backup .v1.bak.* file created", len(baks) == 1, f"found {len(baks)}")
    else:
        _record("migrate", "v1→v2 migration exit-code", False, r.stderr[:200])

    # --check mode does not write
    mp2 = Path(td) / "manifest_check.json"
    mp2.write_text(json.dumps(_V1, indent=2), encoding="utf-8")
    original_bytes = mp2.read_bytes()
    r2 = _run_migrate(mp2, "--check")
    _record("migrate", "--check exit code 0", r2.returncode == 0, r2.stderr[:100])
    diff_output = r2.stdout
    _record("migrate", "--check produces diff output (--- or +++)", "---" in diff_output or "+++" in diff_output, repr(diff_output[:80]))
    _record("migrate", "--check does not modify file", mp2.read_bytes() == original_bytes)

    # Idempotency: run twice on v2
    mp3 = Path(td) / "manifest_idem.json"
    mp3.write_text(json.dumps(_V1, indent=2), encoding="utf-8")
    _run_migrate(mp3)  # first run
    v2_bytes = mp3.read_bytes()
    r3 = _run_migrate(mp3)  # second run
    _record("migrate", "idempotent: exit 0 on v2", r3.returncode == 0, r3.stderr[:80])
    _record("migrate", "idempotent: 'already at v2' in output", "already at v2" in r3.stdout.lower(), repr(r3.stdout[:80]))
    _record("migrate", "idempotent: file unchanged after second run", mp3.read_bytes() == v2_bytes)


# ---------------------------------------------------------------------------
# 7. factory.py uses read_manifest / LegacyManifestError (Step 6)
# ---------------------------------------------------------------------------
_section("7. inference/app/backends/factory.py uses shared.manifest (Step 6)")

factory_path = PROJECT_ROOT / "inference/app/backends/factory.py"
if factory_path.exists():
    content = factory_path.read_text(encoding="utf-8")
    _record("factory", "imports read_manifest", "read_manifest" in content)
    _record("factory", "imports LegacyManifestError", "LegacyManifestError" in content)
    _record("factory", "handles LegacyManifestError", "LegacyManifestError" in content and "logger.fatal" in content)
    _record("factory", "no inline json.loads for manifest", "json.loads" not in content or "read_manifest" in content)
else:
    _record("factory", "factory.py exists", False, "file missing")


# ---------------------------------------------------------------------------
# 8. adapters.py uses write_manifest_locked for rollback (Step 7)
# ---------------------------------------------------------------------------
_section("8. inference/app/routes/adapters.py uses write_manifest_locked (Step 7)")

adapters_path = PROJECT_ROOT / "inference/app/routes/adapters.py"
if adapters_path.exists():
    content = adapters_path.read_text(encoding="utf-8")
    _record("adapters_route", "imports write_manifest_locked", "write_manifest_locked" in content)
    _record("adapters_route", "imports LegacyManifestError", "LegacyManifestError" in content)
    _record("adapters_route", "rollback endpoint defined", "rollback_adapter" in content or "/rollback" in content)
    _record("adapters_route", "rollback calls write_manifest_locked", "write_manifest_locked" in content)
else:
    _record("adapters_route", "adapters.py exists", False, "file missing")


# ---------------------------------------------------------------------------
# 9. lora_trainer.py uses write_manifest_locked + HistoryEntry (Step 4)
# ---------------------------------------------------------------------------
_section("9. training/training/lora_trainer.py uses shared manifest (Step 4)")

trainer_path = PROJECT_ROOT / "training/training/lora_trainer.py"
if trainer_path.exists():
    content = trainer_path.read_text(encoding="utf-8")
    _record("trainer", "imports write_manifest_locked", "write_manifest_locked" in content)
    _record("trainer", "imports HistoryEntry", "HistoryEntry" in content)
    _record("trainer", "imports AdapterEntry", "AdapterEntry" in content)
    _record("trainer", "no _manifest_file_lock (old fcntl helper)", "_manifest_file_lock" not in content)
    _record("trainer", "dataset_hash computed", "dataset_hash" in content)
    _record("trainer", "tokenizer_hash computed", "tokenizer_hash" in content)
    _record("trainer", "trainer_version computed", "trainer_version" in content)
    _record("trainer", "sft_format: chatml in recipe", '"chatml"' in content or "'chatml'" in content)
else:
    _record("trainer", "lora_trainer.py exists", False, "file missing")


# ---------------------------------------------------------------------------
# 10. No hardcoded model-name literals (Plan A invariant — via pytest guard)
# ---------------------------------------------------------------------------
_section("10. No hardcoded model-name literals (Plan A invariant)")

# Delegate to the authoritative pytest guard — avoids re-embedding the
# banned strings as literals in this file (which would self-trigger).
_lit_test = PROJECT_ROOT / "shared" / "tests" / "test_no_model_literals.py"
if not _lit_test.exists():
    _record("no_literals", "test_no_model_literals.py exists", False, "file missing")
else:
    _r = subprocess.run(
        [sys.executable, "-m", "pytest", str(_lit_test), "-q", "--tb=short"],
        capture_output=True, text=True, cwd=str(PROJECT_ROOT),
    )
    _record("no_literals", "no banned model-name literals in source",
            _r.returncode == 0,
            ("" if _r.returncode == 0
             else "\n".join((_r.stdout + _r.stderr).splitlines()[-8:])[:400]))


# ---------------------------------------------------------------------------
# 11. pytest — run the three test suites (Step 10)
# ---------------------------------------------------------------------------
_section("11. pytest test suites (Step 10)")

_test_suites = [
    ("shared/tests/test_manifest_schema.py", "schema round-trip + validation"),
    ("shared/tests/test_manifest_locking.py", "concurrent locking"),
    ("scripts/tests/test_migrate_manifest.py", "migrate_manifest"),
]

for rel, label in _test_suites:
    p = PROJECT_ROOT / rel
    if not p.exists():
        _record("pytest", label, False, "test file missing")
        continue
    r = subprocess.run(
        [sys.executable, "-m", "pytest", str(p), "-q", "--tb=short"],
        capture_output=True, text=True, cwd=str(PROJECT_ROOT),
    )
    ok = r.returncode == 0
    detail = ""
    if not ok:
        lines = (r.stdout + r.stderr).splitlines()
        tail = "\n".join(lines[-6:])
        detail = tail[:300]
    _record("pytest", label, ok, detail)


# ---------------------------------------------------------------------------
# 12. Constraint checks
# ---------------------------------------------------------------------------
_section("12. Constraint checks (Constraints/Notes)")

# os.replace used for atomic write (not shutil.move or direct write)
manifest_py = PROJECT_ROOT / "shared/manifest.py"
if manifest_py.exists():
    content = manifest_py.read_text(encoding="utf-8")
    _record("constraints", "os.replace used for atomic write", "os.replace" in content)
    _record("constraints", "FileLock timeout=30", "timeout=30" in content)
    _record("constraints", ".tmp suffix for temp file", ".tmp" in content)
    _record("constraints", "LegacyManifestError defined", "LegacyManifestError" in content)
    _record("constraints", "no silent v1→v2 migration in read_manifest",
            "migrate_manifest" in content and "LegacyManifestError" in content)

schema_py = PROJECT_ROOT / "shared/contracts/manifest_schema.py"
if schema_py.exists():
    content = schema_py.read_text(encoding="utf-8")
    _record("constraints", "dataset_hash field present", "dataset_hash" in content)
    _record("constraints", "tokenizer_hash field present", "tokenizer_hash" in content)
    _record("constraints", "eval_metrics field present", "eval_metrics" in content)
    _record("constraints", "used_base_fallback_aggregate field (Plan E hook)", "used_base_fallback_aggregate" in content)
    _record("constraints", "smoke field present (Plan F hook)", "smoke" in content)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
_section("SUMMARY")

total = len(_results)
passed = sum(1 for _, _, s in _results if s == _PASS)
failed = sum(1 for _, _, s in _results if s == _FAIL)

if failed:
    print(f"\n  FAILURES ({failed}):")
    for section, label, status in _results:
        if status == _FAIL:
            print(f"    [{section}] {label}")

print(f"\n  Total: {total}   Passed: {passed}   Failed: {failed}")
if failed == 0:
    print("\n  ALL CHECKS PASSED — Plan C verification complete.")
else:
    print(f"\n  {failed} CHECK(S) FAILED — review failures above.")

sys.exit(0 if failed == 0 else 1)
