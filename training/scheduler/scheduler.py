"""
Training scheduler — lightweight interval cron.

On each tick:
  1. Read per-role cursor from Redis (cursor:<role> → {shard, offset}).
  2. Check diversity threshold on unprocessed records via iter_after(cursor).
  3. Launch the training container via docker compose with a timeout.
  4. Write a heartbeat file every 15 s during the run (external watchdog reads it).
  5. After successful exit (code 0), commit updated cursors to Redis.
  Failed runs leave cursors untouched — the next tick replays (idempotent via
  dataset_hash in Plan C).
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import schedule

# Reuse threshold logic + settings from the training service tree
sys.path.insert(0, "/app")

from app.config.settings import settings  # noqa: E402
from app.utils.experience_reader import (  # noqa: E402
    check_diversity,
    iter_after,
    iter_records,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)-8s | %(name)s | %(message)s")
log = logging.getLogger(__name__)

CAPABILITIES = ["coder", "debugger", "reviewer", "planner", "researcher", "critic"]

_redis_client = None


def _get_redis():
    global _redis_client
    if _redis_client is None:
        try:
            import redis
            _redis_client = redis.from_url(settings.redis_url, decode_responses=True)
        except Exception as exc:
            log.warning("Redis unavailable (%s) — cursor persistence disabled", exc)
    return _redis_client


def _read_cursor(role: str) -> Optional[dict]:
    r = _get_redis()
    if r is None:
        return None
    try:
        raw = r.get(f"cursor:{role}")
        return json.loads(raw) if raw else None
    except Exception as exc:
        log.warning("Failed to read cursor for %s: %s", role, exc)
        return None


def _write_cursor(role: str, shard: Path, offset: int) -> None:
    r = _get_redis()
    if r is None:
        return
    try:
        r.set(f"cursor:{role}", json.dumps({"shard": str(shard), "offset": offset}))
    except Exception as exc:
        log.warning("Failed to write cursor for %s: %s", role, exc)


def _snapshot_latest_cursor() -> dict[str, dict]:
    """Snapshot the current tail position for each capability from the reader."""
    latest: dict[str, dict] = {}
    for shard, offset, record in iter_records():
        role = record.get("capability") or record.get("role", "")
        if role:
            latest[role] = {"shard": str(shard), "offset": offset}
    return latest


def _heartbeat_loop(done_event: threading.Event) -> None:
    hb_path = Path(settings.heartbeat_path)
    hb_path.parent.mkdir(parents=True, exist_ok=True)
    while not done_event.is_set():
        try:
            hb_path.write_text(datetime.now(timezone.utc).isoformat())
        except OSError:
            pass
        time.sleep(15)


def tick() -> None:
    log.info("Scheduler tick")

    if Path(settings.lock_path).exists():
        log.info("Lock present at %s — skipping tick", settings.lock_path)
        return

    # Check diversity across unprocessed records for at least one role
    any_role_ready = False
    for role in CAPABILITIES:
        cursor = _read_cursor(role)
        role_records = [r for _, _, r in iter_after(cursor)
                        if (r.get("capability") or r.get("role", "")) == role]
        if role_records:
            passed, reason = check_diversity(role_records)
            if passed:
                log.info("Role %s threshold met: %s", role, reason)
                any_role_ready = True
                break
            else:
                log.info("Role %s threshold not met: %s", role, reason)

    if not any_role_ready:
        log.info("No role met diversity threshold — skipping tick")
        return

    # Snapshot tail positions before training so we can commit after success
    pre_snapshot = _snapshot_latest_cursor()

    log.info("Threshold met — launching training container")
    cmd = [
        "docker", "compose", "-f", settings.compose_file,
        "--profile", "training", "run", "--rm", "training",
    ]

    done_event = threading.Event()
    hb_thread = threading.Thread(target=_heartbeat_loop, args=(done_event,), daemon=True)
    hb_thread.start()

    success = False
    try:
        result = subprocess.run(
            cmd,
            check=False,
            timeout=settings.training_timeout_seconds,
        )
        if result.returncode == 0:
            success = True
        else:
            log.error("Training container exited with code %s", result.returncode)
    except subprocess.TimeoutExpired as exc:
        if exc.process is not None:
            try:
                exc.process.terminate()
            except Exception:
                pass
        log.error(
            "training_timed_out timeout=%s",
            settings.training_timeout_seconds,
        )
    except Exception as exc:
        log.error("docker compose run failed: %s", exc)
    finally:
        done_event.set()
        hb_thread.join(timeout=20)

    if success:
        for role, pos in pre_snapshot.items():
            _write_cursor(role, Path(pos["shard"]), pos["offset"])
            log.info("Cursor committed role=%s shard=%s offset=%s", role, pos["shard"], pos["offset"])


def main() -> None:
    interval = settings.scheduler_interval_hours
    log.info("Training scheduler started: interval=%dh lock=%s", interval, settings.lock_path)
    schedule.every(interval).hours.do(tick)
    # Evaluate once on boot so a freshly-deployed scheduler doesn't sit idle for hours.
    tick()
    while True:
        schedule.run_pending()
        time.sleep(settings.scheduler_poll_interval_seconds)


if __name__ == "__main__":
    main()
