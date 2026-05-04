"""Tests for trainer_entry._load_sft_pairs — sft_format branching."""
import json
import tempfile
from pathlib import Path

import pytest

from shared.contracts.training import AdapterRecipe


def _make_recipe(sft_format="openai_messages", tool_call_style="openai_native"):
    return AdapterRecipe(
        target_modules=["q_proj"],
        sft_format=sft_format,
        tool_call_style=tool_call_style,
    )


def _write_jsonl(path: Path, records):
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def _make_experience_record(role="coder", score=0.9):
    return {
        "final_score": score,
        "prompt": "Write a hello function",
        "final_output": "def hello(): return 'hi'",
        "contributions": [
            {
                "role": role,
                "full_output": "def hello(): return 'hi'",
                "quality_score": score,
                "iteration": 1,
                "messages": [
                    {"role": "system", "content": "You are a coder."},
                    {"role": "user", "content": "Write hello."},
                ],
            }
        ],
    }


def _call_load_sft_pairs(data_path, capability, recipe):
    from training.app.trainer_entry import _load_sft_pairs
    return _load_sft_pairs(str(data_path), capability, recipe)


def test_openai_messages_emits_messages_key(tmp_path):
    record = _make_experience_record()
    data_file = tmp_path / "data.jsonl"
    _write_jsonl(data_file, [record])
    recipe = _make_recipe(sft_format="openai_messages")

    pairs = _call_load_sft_pairs(data_file, "coder", recipe)

    assert len(pairs) >= 1
    pair = pairs[0]
    assert "messages" in pair, "openai_messages format must have 'messages' key"
    assert "instruction" not in pair, "openai_messages must not have 'instruction' key"
    msgs = pair["messages"]
    assert isinstance(msgs, list)
    roles = [m["role"] for m in msgs]
    assert "system" in roles
    assert "user" in roles
    assert roles[-1] == "assistant"


def test_fallback_record_openai_messages(tmp_path):
    """Records without per-contribution messages fall back to top-level prompt/output."""
    record = {
        "final_score": 0.8,
        "prompt": "Write a hello function",
        "final_output": "def hello(): return 'hi'",
        "contributions": [],
    }
    data_file = tmp_path / "data.jsonl"
    _write_jsonl(data_file, [record])
    recipe = _make_recipe(sft_format="openai_messages")

    pairs = _call_load_sft_pairs(data_file, "coder", recipe)

    assert len(pairs) >= 1
    pair = pairs[0]
    assert "messages" in pair
    msgs = pair["messages"]
    assert msgs[-1]["role"] == "assistant"
    assert msgs[-1]["content"] == "def hello(): return 'hi'"


def test_below_min_score_excluded(tmp_path):
    record = _make_experience_record(score=0.3)
    data_file = tmp_path / "data.jsonl"
    _write_jsonl(data_file, [record])
    recipe = _make_recipe()

    pairs = _call_load_sft_pairs(data_file, "coder", recipe)
    assert len(pairs) == 0
