"""
Experience reader
==================
Reads the shared JSONL file that the Experience Service writes.
The Training Service mounts the same ``/data`` volume so this is a
direct local file read — no HTTP call needed.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Tuple

from ..config.settings import settings

logger = logging.getLogger(__name__)

PROMPT_FINGERPRINT_MAX_CHARS = 120  # normalised prompt characters kept for deduplication fingerprint


def load_experiences() -> List[dict]:
    """Return all experience records from the JSONL file."""
    path = Path(settings.jsonl_path)
    if not path.exists():
        return []
    records = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records


def check_diversity(records: List[dict]) -> Tuple[bool, str]:
    """
    Evaluate whether the experience batch is diverse enough for training.

    Three criteria (all must pass):
    1. Batch size  ≥ settings.min_batch_size
    2. Score spread  (max - min) ≥ settings.min_score_spread
    3. Unique fingerprint ratio  ≥ settings.min_diversity_ratio

    Returns (passes: bool, reason: str).
    """
    n = len(records)
    if n < settings.min_batch_size:
        return False, f"too few experiences: {n} < {settings.min_batch_size}"

    scores = [float(r.get("final_score", 0.0)) for r in records]
    spread = max(scores) - min(scores)
    if spread < settings.min_score_spread:
        return False, (
            f"score spread too small: {spread:.3f} < {settings.min_score_spread} "
            f"(range {min(scores):.2f}–{max(scores):.2f})"
        )

    fingerprints = {_fingerprint(r.get("prompt", "")) for r in records}
    diversity_ratio = len(fingerprints) / n
    if diversity_ratio < settings.min_diversity_ratio:
        return False, (
            f"low prompt diversity: {diversity_ratio:.2f} < {settings.min_diversity_ratio} "
            f"({len(fingerprints)} unique / {n} total)"
        )

    return True, f"criteria met: n={n}, spread={spread:.2f}, diversity={diversity_ratio:.2f}"


def _fingerprint(prompt: str) -> str:
    """Compact, case-insensitive prompt fingerprint for deduplication."""
    normalised = "".join(prompt.lower().split())
    return normalised[:PROMPT_FINGERPRINT_MAX_CHARS]


def to_training_records(records: List[dict]) -> List[dict]:
    """
    Return ExperienceEvent records that meet the minimum quality score.

    The full records are passed to the training subprocess so that
    trainer_entry._load_sft_pairs() can extract per-contribution SFT pairs
    using the messages and full_output fields.
    """
    return [
        r for r in records
        if float(r.get("final_score", 0.0)) >= settings.min_quality_score
    ]
