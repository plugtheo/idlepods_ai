"""
LoRA training launcher
========================
Prepares training data from experience records and launches the LoRA
trainer in a subprocess so the Training Service HTTP process stays responsive.

The subprocess runs ``training/lora_trainer.py`` (the existing monolith
trainer) via a thin adapter script that accepts CLI args, so there is no
tight coupling between the service and the trainer implementation.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import List

from ..config.settings import settings

logger = logging.getLogger(__name__)

# Guard: only one training job at a time
_training_lock = asyncio.Lock()
_training_running = False


async def launch_training(capability: str, training_records: List[dict]) -> None:
    """
    Write training_records to a temp JSONL file and invoke the LoRA trainer
    subprocess.  Returns immediately after the process is spawned (non-blocking
    from the caller's perspective; the lock prevents concurrent runs).
    """
    global _training_running

    if _training_running:
        logger.info("Training already running — skipping new trigger")
        return

    if not training_records:
        logger.warning("No training records after quality filter — skipping")
        return

    async with _training_lock:
        _training_running = True
        try:
            await _run_subprocess(capability, training_records)
        finally:
            _training_running = False


async def _run_subprocess(capability: str, records: List[dict]) -> None:
    # Write records to a temp file that the subprocess can read
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
    ) as tmp:
        for rec in records:
            tmp.write(json.dumps(rec) + "\n")
        tmp_path = tmp.name

    base_model = (
        settings.deepseek_model
        if any(x in capability for x in ("cod", "debug", "review"))
        else settings.mistral_model
    )

    cmd = [
        sys.executable,
        "-m",
        "services.training.app.trainer_entry",
        "--data-path",
        tmp_path,
        "--base-model",
        base_model,
        "--output-dir",
        settings.output_dir,
        "--capability",
        capability,
    ]

    env = os.environ.copy()
    if settings.hf_token:
        env["HF_TOKEN"] = settings.hf_token

    logger.info(
        "Launching LoRA training subprocess: capability=%s, records=%d, model=%s",
        capability,
        len(records),
        base_model,
    )

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        env=env,
    )

    # Stream logs from subprocess without blocking the event loop
    assert proc.stdout is not None
    async for line in proc.stdout:
        logger.info("[trainer] %s", line.decode().rstrip())

    await proc.wait()
    if proc.returncode == 0:
        logger.info("LoRA training completed successfully (capability=%s)", capability)
    else:
        logger.error("LoRA training failed with exit code %d", proc.returncode)

    # Clean up temp file
    try:
        Path(tmp_path).unlink(missing_ok=True)
    except Exception:
        pass


def is_training_running() -> bool:
    return _training_running
