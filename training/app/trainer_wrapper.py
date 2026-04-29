"""
Trainer wrapper — one-shot container entrypoint.

Acquires a file-lock, optionally stops vLLM (BLOCK mode), then either runs
trainer_entry locally for each capability or POSTs to a remote trigger URL.
Always restarts vLLM in `finally` when in BLOCK mode.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

import httpx

from .config.settings import settings
from .utils.experience_reader import check_diversity, load_experiences, to_training_records

logging.basicConfig(level=logging.INFO, format="%(levelname)-8s | %(name)s | %(message)s")
log = logging.getLogger(__name__)

CAPABILITIES = ["coder", "debugger", "reviewer", "planner", "researcher", "critic"]

REMOTE_TRIGGER_LOG_PREVIEW_CHARS = 500


def stop_vllm() -> None:
    """Thin abstraction — replace with k8s scale-to-zero when porting."""
    if not settings.vllm_services:
        return
    cmd = ["docker", "compose", "-f", settings.compose_file, "stop", *settings.vllm_services]
    log.info("stop_vllm: %s", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
    except Exception as exc:
        log.error("stop_vllm failed: %s", exc)


def start_vllm() -> None:
    """Thin abstraction — replace with k8s scale-up when porting."""
    if not settings.vllm_services:
        return
    cmd = ["docker", "compose", "-f", settings.compose_file, "start", *settings.vllm_services]
    log.info("start_vllm: %s", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
    except Exception as exc:
        log.error("start_vllm failed: %s", exc)


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


def _base_model_for(capability: str) -> str:
    return (
        settings.deepseek_model
        if any(x in capability for x in ("cod", "debug", "review"))
        else settings.mistral_model
    )


def _run_local(records: list[dict]) -> None:
    training_records = to_training_records(records)
    if not training_records:
        log.info("No quality-filtered records — skipping local training")
        return

    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False, encoding="utf-8") as tmp:
        for rec in training_records:
            tmp.write(json.dumps(rec) + "\n")
        tmp_path = tmp.name

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
            try:
                subprocess.run(cmd, check=False)
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
    log.info(
        "Training wrapper: target=%s exclusive=%s",
        settings.training_target, settings.training_exclusive_mode,
    )

    records = load_experiences()
    passed, reason = check_diversity(records)
    if not passed:
        log.info("Threshold not met inside wrapper (%s) — exiting", reason)
        return
    log.info("Threshold met: %s", reason)

    with file_lock(settings.lock_path):
        block = (
            settings.training_exclusive_mode == "BLOCK"
            and settings.training_target == "local"
        )
        if block:
            stop_vllm()
        try:
            if settings.training_target == "remote":
                _run_remote()
            else:
                _run_local(records)
        finally:
            if block:
                start_vllm()


if __name__ == "__main__":
    main()
