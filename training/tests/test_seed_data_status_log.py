"""
Step 15 readiness test — test_seed_data_status_log.py

Invoke _load_curated_pairs("coder", tmp_path, 100, recipe) with no curated file;
assert seed_data_status log line emitted with curated_count=0 and exists=False.
"""
from __future__ import annotations

import io
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest


def _make_recipe():
    return SimpleNamespace(
        peft_type="rslora",
        use_rslora=True,
        use_dora=False,
        r=32,
        alpha=64,
        dropout=0.05,
        target_modules=["q_proj"],
        learning_rate=2e-4,
        num_epochs=3,
        max_seq_length=2048,
        sft_format="openai_messages",
        tool_call_style="openai_native",
    )


def test_seed_data_status_emitted_when_curated_missing(tmp_path, capsys):
    """_load_curated_pairs with missing curated file emits seed_data_status exists=False."""
    # tmp_path has no training_data_curated directory → curated file won't exist
    recipe = _make_recipe()

    from training.app.trainer_entry import _load_curated_pairs, _CAPABILITY_TO_CURATED

    # Ensure "coder" has a mapping so the path is attempted
    with patch.dict(_CAPABILITY_TO_CURATED, {"coder": "coder_curated.jsonl"}):
        result = _load_curated_pairs("coder", tmp_path, 100, recipe)

    assert result == [], "Expected empty list when curated file is absent"

    captured = capsys.readouterr()
    assert "seed_data_status" in captured.out, (
        f"Expected 'seed_data_status' in stdout. Got: {captured.out!r}"
    )
    assert "curated_count=0" in captured.out, (
        f"Expected 'curated_count=0' in stdout. Got: {captured.out!r}"
    )
    assert "exists=False" in captured.out, (
        f"Expected 'exists=False' in stdout. Got: {captured.out!r}"
    )


def test_seed_data_status_emitted_when_curated_present(tmp_path, capsys):
    """_load_curated_pairs with present curated file emits seed_data_status exists=True."""
    import json
    recipe = _make_recipe()

    curated_dir = tmp_path / "training_data_curated"
    curated_dir.mkdir()
    curated_file = curated_dir / "coder_curated.jsonl"
    curated_file.write_text(
        json.dumps({"instruction": "Write hello world", "response": "print('Hello, World!')"}) + "\n"
    )

    from training.app.trainer_entry import _load_curated_pairs, _CAPABILITY_TO_CURATED

    with patch.dict(_CAPABILITY_TO_CURATED, {"coder": "coder_curated.jsonl"}):
        result = _load_curated_pairs("coder", tmp_path, 100, recipe)

    captured = capsys.readouterr()
    assert "seed_data_status" in captured.out
    assert "exists=True" in captured.out
    assert "curated_count=" in captured.out
