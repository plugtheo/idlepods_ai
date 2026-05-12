"""
Trainer wrapper — one-shot container entrypoint.

Acquires a file-lock then either runs trainer_entry locally for each
capability or POSTs to a remote trigger URL.  Adapters land via the
hot-swap /adapters/load route — the trainer never touches the vLLM container.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple

import httpx

from shared.contracts.agent_prompts import ROLE_TO_BOOTSTRAP_CAP
from shared.contracts.roles import TRAINABLE_ROLES
from .config.settings import settings
from .utils.experience_reader import check_diversity, load_experiences

logging.basicConfig(level=logging.INFO, format="%(levelname)-8s | %(name)s | %(message)s")
log = logging.getLogger(__name__)

REMOTE_TRIGGER_LOG_PREVIEW_CHARS = 500


def _is_stale_lock(p: Path, stale_age_seconds: int) -> bool:
    """Return True when the lock at *p* belongs to a dead process or is too old."""
    try:
        age = time.time() - p.stat().st_mtime
        pid_text = p.read_text().strip()
        pid = int(pid_text) if pid_text.isdigit() else None
        if pid:
            try:
                os.kill(pid, 0)
                pid_alive = True
            except (ProcessLookupError, PermissionError):
                pid_alive = False
        else:
            pid_alive = False
        return age > stale_age_seconds or not pid_alive
    except OSError:
        return False


@contextmanager
def file_lock(path: str, stale_age_seconds: int = 90000):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists() and _is_stale_lock(p, stale_age_seconds):
        log.warning("Removing stale lock at %s", path)
        try:
            p.unlink(missing_ok=True)
        except OSError:
            pass
    try:
        fd = os.open(str(p), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        log.error("Lock file present at %s — another training run is active; aborting", path)
        sys.exit(1)
    try:
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
        yield
    finally:
        try:
            p.unlink(missing_ok=True)
        except OSError:
            pass


def _base_model_for(role: str) -> str:
    from shared.contracts.models import load_registry
    try:
        registry = load_registry(settings.models_yaml_path)
        backend = registry.backends[registry.default_backend]
        return backend.resolve_training_model_id()
    except Exception as exc:
        raise RuntimeError(f"Cannot resolve base model from registry: {exc}") from exc


def _role_adapter_dir(role: str) -> str:
    """Map a role name ('coder') to its adapter directory name ('coding_lora')."""
    cap = ROLE_TO_BOOTSTRAP_CAP.get(role, role)
    return f"{cap}_lora"


def _get_role_cursor(role: str, output_dir: str) -> Optional[datetime]:
    """
    Read the *last_trained_experience_timestamp* for *role* from its adapter's
    metadata.json. Returns None when no cursor is recorded (first retrain ever).
    """
    meta_path = Path(output_dir) / _role_adapter_dir(role) / "metadata.json"
    if not meta_path.exists():
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    raw = meta.get("last_trained_experience_timestamp")
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return None


def _parse_ts(ts_raw: Optional[str]) -> Optional[datetime]:
    if not ts_raw:
        return None
    try:
        return datetime.fromisoformat(str(ts_raw).replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return None


def _count_new_role_pairs(
    records: list[dict],
    role_name: str,
    cursor: Optional[datetime],
    min_score: float,
) -> Tuple[int, Optional[datetime]]:
    """
    Count post-cursor records with a qualifying contribution for *role_name*,
    plus the latest record timestamp seen (the new cursor candidate).

    Records lacking a timestamp are excluded from the gate — the system cannot
    tell whether they're new, and counting them risks an infinite-retrain loop
    on legacy data. Conservative: skip. They are still eligible for *training*,
    only the trigger gate ignores them.
    """
    n = 0
    max_ts: Optional[datetime] = None
    for rec in records:
        if float(rec.get("final_score", 0.0)) < min_score:
            continue
        ts = _parse_ts(rec.get("timestamp"))
        if ts is None:
            continue
        if cursor is not None and ts <= cursor:
            continue
        has_role_contrib = any(
            contrib.get("role") == role_name
            and contrib.get("messages")
            and contrib.get("full_output")
            for contrib in rec.get("contributions", [])
        )
        if not has_role_contrib:
            continue
        n += 1
        if max_ts is None or ts > max_ts:
            max_ts = ts
    return n, max_ts


def _run_local(records: list[dict]) -> None:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False, encoding="utf-8") as tmp:
        for rec in records:
            tmp.write(json.dumps(rec) + "\n")
        tmp_path = tmp.name

    platform_kwargs: dict = {}
    if sys.platform == "win32":
        platform_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        platform_kwargs["start_new_session"] = True

    def _kill(proc: subprocess.Popen) -> None:
        try:
            if sys.platform == "win32":
                subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], check=False)
            else:
                import signal
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except Exception:
            pass

    def _run_subprocess(cmd: list, label: str, timeout: int) -> int:
        """Run cmd, stream stdout/stderr to log, return exit code."""
        _t0 = time.monotonic()
        try:
            result = subprocess.run(
                cmd, check=False, timeout=timeout, **platform_kwargs,
            )
            duration_s = round(time.monotonic() - _t0, 1)
            if result.returncode == 0:
                log.info("%s done duration_s=%s", label, duration_s)
            else:
                log.error("%s exit=%s duration_s=%s", label, result.returncode, duration_s)
            return result.returncode
        except subprocess.TimeoutExpired as exc:
            if exc.process is not None:
                _kill(exc.process)
            log.error("%s timed_out timeout=%s", label, timeout)
            return -1
        except Exception as exc:
            log.error("%s error: %s", label, exc)
            return -1

    try:
        # ── Phase 1: per-role cursor + threshold + freeze gate ───────────────
        # Each role advances independently. A role is eligible for training
        # this tick only when it (a) is NOT frozen at rank cap, AND (b) has
        # accumulated >= min_new_experiences_per_role qualifying contributions
        # SINCE its adapter's last_trained_experience_timestamp cursor. This
        # prevents pointless retrains on quiet roles, lets high-traffic roles
        # progress independently, and stops any role whose rank cap was hit +
        # plateaued from continuing to spin (it's ready for merge instead).
        from training.app.rank_policy import is_frozen
        eligible: list[str] = []
        for capability in TRAINABLE_ROLES:
            adapter_dir = Path(settings.output_dir) / _role_adapter_dir(capability)
            if is_frozen(adapter_dir):
                log.info(
                    "skip role=%s reason=frozen_at_cap adapter_dir=%s",
                    capability, adapter_dir,
                )
                continue

            cursor = _get_role_cursor(capability, settings.output_dir)
            n, _ = _count_new_role_pairs(
                records, capability, cursor, settings.min_quality_score
            )
            cursor_disp = cursor.isoformat() if cursor else "none"
            log.info(
                "data_count_check role=%s qualifying_new_pairs=%s cursor=%s threshold=%s",
                capability, n, cursor_disp, settings.min_new_experiences_per_role,
            )
            if n < settings.min_new_experiences_per_role:
                log.info(
                    "skip role=%s reason=insufficient_new_data new=%s threshold=%s",
                    capability, n, settings.min_new_experiences_per_role,
                )
                continue
            eligible.append(capability)

        if not eligible:
            log.info("No roles cleared per-role retrain threshold — nothing to do this tick")
            return

        # ── Phase 2: probe each eligible role ────────────────────────────────
        # Trains 50 samples × 1 epoch in a temp dir, smoke-tests, exits 0/3.
        # Roles whose probe exits non-zero are skipped in Phase 3.
        probe_passed: list[str] = []
        for capability in eligible:
            base_model = _base_model_for(capability)
            probe_cmd = [
                sys.executable, "-m", "training.app.trainer_entry",
                "--data-path", tmp_path,
                "--base-model", base_model,
                "--output-dir", settings.output_dir,
                "--capability", capability,
                "--probe",
            ]
            log.info("probe_start role=%s", capability)
            code = _run_subprocess(probe_cmd, f"probe/{capability}", timeout=1800)
            if code == 0:
                probe_passed.append(capability)
                log.info("probe_pass role=%s", capability)
            else:
                log.error(
                    "probe_fail role=%s exit=%s — skipping full training for this role",
                    capability, code,
                )

        if not probe_passed:
            log.error("All probes failed — no full training runs will be launched")
            return

        # ── Phase 3: full training for probe-passed roles ────────────────────
        for capability in probe_passed:
            base_model = _base_model_for(capability)
            cmd = [
                sys.executable, "-m", "training.app.trainer_entry",
                "--data-path", tmp_path,
                "--base-model", base_model,
                "--output-dir", settings.output_dir,
                "--capability", capability,
            ]
            log.info("trainer_entry_start capability=%s base=%s", capability, base_model)
            _run_subprocess(cmd, f"train/{capability}", timeout=settings.training_timeout_seconds)
    finally:
        try:
            Path(tmp_path).unlink(missing_ok=True)
        except OSError:
            pass


def _run_remote() -> None:
    if not settings.training_trigger_url:
        log.error("TRAINING_TARGET=remote but TRAINING__TRAINING_TRIGGER_URL is empty")
        return
    log.info("Posting remote training trigger to %s", settings.training_trigger_url)
    try:
        resp = httpx.post(settings.training_trigger_url, timeout=settings.remote_trigger_timeout_seconds)
        resp.raise_for_status()
        log.info("Remote trigger accepted: %s", resp.text[:REMOTE_TRIGGER_LOG_PREVIEW_CHARS])
    except Exception as exc:
        log.error("Remote trigger failed: %s", exc)


def main() -> None:
    log.info("Training wrapper: target=%s", settings.training_target)

    records = load_experiences()
    passed, reason = check_diversity(records)
    if not passed:
        log.info("Threshold not met inside wrapper (%s) — exiting", reason)
        return
    log.info("Threshold met: %s", reason)

    with file_lock(settings.lock_path):
        if settings.training_target == "remote":
            _run_remote()
        else:
            _run_local(records)


if __name__ == "__main__":
    main()
