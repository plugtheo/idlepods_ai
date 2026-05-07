"""
Async launcher for trainer_entry subprocesses.

Provides fire-and-forget training launches with an in-process guard to
prevent concurrent runs.  The REST trigger route and scheduler both use
this module to start training.
"""

from __future__ import annotations

import asyncio
import logging
import sys

from ..config.settings import settings as _settings

log = logging.getLogger(__name__)

_training_running: bool = False


def is_training_running() -> bool:
    return _training_running


def _get_base_model(capability: str) -> str:
    from ..config.settings import settings
    from shared.contracts.models import load_registry
    try:
        registry = load_registry(settings.models_yaml_path)
        return registry.backends[registry.default_backend].resolve_training_model_id()
    except Exception as exc:
        raise RuntimeError(f"Cannot resolve base model for capability '{capability}': {exc}") from exc


async def launch_training(capability: str, training_records: list) -> None:
    """
    Launch trainer_entry for the given capability in a subprocess.

    No-op if training is already running or no records are provided.
    Sets _training_running for the duration so callers can gate re-entry.
    """
    global _training_running
    if _training_running:
        log.info("launch_training: skipped — training already running")
        return
    if not training_records:
        log.info("launch_training: skipped — no training records")
        return

    _training_running = True
    try:
        base_model = _get_base_model(capability)
        cmd = [
            sys.executable, "-m", "services.training.app.trainer_entry",
            "--base-model", base_model,
            "--output-dir", _settings.output_dir,
            "--capability", capability,
            "--data-path", _settings.jsonl_path,
        ]  # type: ignore[attr-defined]

        log.info(
            "launch_training: spawning trainer_entry capability=%s model=%s",
            capability, base_model,
        )
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        if proc.stdout:
            async for line in proc.stdout:
                log.info("[trainer/%s] %s", capability, line.decode().rstrip())
        await proc.wait()
        if proc.returncode != 0:
            log.error(
                "launch_training: trainer_entry exited %d capability=%s",
                proc.returncode, capability,
            )
    finally:
        _training_running = False
