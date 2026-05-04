"""
Step 11 — test_experience_cursor.py

Write 3 dated shards; set cursor at shard-2/offset-5;
assert reader yields exactly the unread tail.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from training.app.utils.experience_reader import iter_after


def _write_shard(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")


@pytest.fixture()
def three_shards(tmp_path):
    shard1 = tmp_path / "experiences-20260101.jsonl"
    shard2 = tmp_path / "experiences-20260102.jsonl"
    shard3 = tmp_path / "experiences-20260103.jsonl"

    for i, shard in enumerate([shard1, shard2, shard3], start=1):
        records = [{"role": "coder", "line": f"shard{i}_line{j}"} for j in range(10)]
        _write_shard(shard, records)

    return shard1, shard2, shard3


def test_iter_after_cursor_yields_tail(three_shards, tmp_path):
    """Cursor at shard2/offset=5 → yields shard2[6..9] then all of shard3."""
    shard1, shard2, shard3 = three_shards

    cursor = {"shard": str(shard2), "offset": 5}

    with patch("training.app.utils.experience_reader.settings") as mock_settings:
        mock_settings.jsonl_path = str(tmp_path / "experiences.jsonl")

        results = list(iter_after(cursor))

    # Should yield shard2 lines 6-9 (4 records) + shard3 lines 0-9 (10 records) = 14
    assert len(results) == 14, f"Expected 14 records, got {len(results)}"

    # First yielded record should be shard2, offset 6
    first_shard, first_offset, first_rec = results[0]
    assert first_shard == shard2
    assert first_offset == 6
    assert first_rec["line"] == "shard2_line6"

    # Last record should be shard3, offset 9
    last_shard, last_offset, last_rec = results[-1]
    assert last_shard == shard3
    assert last_offset == 9
    assert last_rec["line"] == "shard3_line9"


def test_iter_after_none_cursor_yields_all(three_shards, tmp_path):
    """None cursor → all records from all shards."""
    shard1, shard2, shard3 = three_shards

    with patch("training.app.utils.experience_reader.settings") as mock_settings:
        mock_settings.jsonl_path = str(tmp_path / "experiences.jsonl")

        results = list(iter_after(None))

    assert len(results) == 30, f"Expected 30 records, got {len(results)}"


def test_iter_after_missing_cursor_shard_replays_all(three_shards, tmp_path):
    """If cursor shard no longer exists, replay everything."""
    shard1, shard2, shard3 = three_shards

    cursor = {"shard": str(tmp_path / "experiences-20250101.jsonl"), "offset": 5}

    with patch("training.app.utils.experience_reader.settings") as mock_settings:
        mock_settings.jsonl_path = str(tmp_path / "experiences.jsonl")

        results = list(iter_after(cursor))

    assert len(results) == 30
