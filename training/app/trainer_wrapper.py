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
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import httpx

from .config.settings import settings
from .utils.experience_reader import check_diversity, load_experiences, to_training_records

logging.basicConfig(level=logging.INFO, format="%(levelname)-8s | %(name)s | %(message)s")
log = logging.getLogger(__name__)

CAPABILITIES = ["coder", "debugger", "reviewer", "planner", "researcher", "critic"]

REMOTE_TRIGGER_LOG_PREVIEW_CHARS = 500


@contextmanager
def file_lock(path: str):
    # No stale-TTL: a crashed trainer leaves the lock in place; remove it manually.
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
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
        return backend.training_model_id or backend.model_id
    except Exception as exc:
        raise RuntimeError(f"Cannot resolve base model from registry: {exc}") from exc


def _run_local(records: list[dict]) -> None:
    training_records = to_training_records(records)
    if not training_records:
        log.info("No quality-filtered records — skipping local training")
        return

    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False, encoding="utf-8") as tmp:
        for rec in training_records:
            tmp.write(json.dumps(rec) + "\n")
        tmp_path = tmp.name

    platform_kwargs: dict = {}
    if sys.platform == "win32":
        platform_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        platform_kwargs["start_new_session"] = True

    try:
        for capability in CAPABILITIES:
            base_model = _base_model_for(capability)
            cmd = [
                sys.executable, "-m", "services.training.app.trainer_entry",
                "--data-path", tmp_path,
                "--base-model", base_model,
                "--output-dir", settings.output_dir,
                "--capability", capability,
            ]
            log.info("trainer_entry capability=%s base=%s", capability, base_model)
            dataset_hash = None
            try:
                result = subprocess.run(
                    cmd,
                    check=False,
                    timeout=settings.training_timeout_seconds,
                    **platform_kwargs,
                )
                if result.returncode != 0:
                    log.error("trainer_entry non-zero exit capability=%s code=%s", capability, result.returncode)
            except subprocess.TimeoutExpired as exc:
                if exc.process is not None:
                    try:
                        if sys.platform == "win32":
                            subprocess.run(
                                ["taskkill", "/F", "/T", "/PID", str(exc.process.pid)],
                                check=False,
                            )
                        else:
                            import signal
                            os.killpg(os.getpgid(exc.process.pid), signal.SIGTERM)
                    except Exception:
                        pass
                log.error(
                    "training_timed_out role=%s dataset_hash=%s timeout=%s",
                    capability, dataset_hash, settings.training_timeout_seconds,
                )
            except Exception as exc:
                log.error("trainer_entry failed for %s: %s", capability, exc)
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
