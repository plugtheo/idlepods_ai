"""
Training scheduler — lightweight interval cron.

On each tick: check threshold via reused experience_reader logic,
then `docker compose run --rm training` if the lock-file is absent.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import time
from pathlib import Path

import schedule

# Reuse threshold logic + settings from the training service tree
sys.path.insert(0, "/app")

from app.config.settings import settings  # noqa: E402
from app.utils.experience_reader import check_diversity, load_experiences  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)-8s | %(name)s | %(message)s")
log = logging.getLogger(__name__)

COMPOSE_FILE = os.getenv("COMPOSE_FILE", settings.compose_file)


def tick() -> None:
    log.info("Scheduler tick")

    if Path(settings.lock_path).exists():
        log.info("Lock present at %s — skipping tick", settings.lock_path)
        return

    records = load_experiences()
    passed, reason = check_diversity(records)
    if not passed:
        log.info("Threshold not met: %s", reason)
        return

    log.info("Threshold met (%s) — running training container", reason)
    cmd = [
        "docker", "compose", "-f", COMPOSE_FILE,
        "--profile", "training", "run", "--rm", "training",
    ]
    try:
        subprocess.run(cmd, check=False)
    except Exception as exc:
        log.error("docker compose run failed: %s", exc)


def main() -> None:
    interval = settings.scheduler_interval_hours
    log.info("Training scheduler started: interval=%dh lock=%s compose=%s",
             interval, settings.lock_path, COMPOSE_FILE)
    schedule.every(interval).hours.do(tick)
    # Evaluate once on boot so a freshly-deployed scheduler doesn't sit idle for hours.
    tick()
    while True:
        schedule.run_pending()
        time.sleep(settings.scheduler_poll_interval_seconds)


if __name__ == "__main__":
    main()
