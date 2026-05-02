"""Tests for LoRATrainer.train — messages-record path + sft_format propagation."""
import json
import sys
import tempfile
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest

from shared.contracts.training import AdapterRecipe
from shared.contracts.manifest_schema import Manifest


def _make_recipe(sft_format="openai_messages", **kwargs):
    return AdapterRecipe(
        target_modules=["q_proj"],
        sft_format=sft_format,
        **kwargs,
    )


def _inject_unsloth_mock():
    mock_unsloth = ModuleType("unsloth")
    mock_flm = MagicMock()
    mock_unsloth.FastLanguageModel = mock_flm
    sys.modules["unsloth"] = mock_unsloth
    mock_trl = ModuleType("trl")
    mock_sft_trainer = MagicMock()
    mock_sft_config = MagicMock()
    mock_trl.SFTTrainer = mock_sft_trainer
    mock_trl.SFTConfig = mock_sft_config
    sys.modules["trl"] = mock_trl
    mock_datasets = ModuleType("datasets")
    mock_hf_dataset = MagicMock()
    mock_hf_dataset.from_list.side_effect = lambda lst: lst
    mock_datasets.Dataset = mock_hf_dataset
    sys.modules["datasets"] = mock_datasets
    return mock_flm, mock_sft_trainer


def _reload_trainer():
    for mod in list(sys.modules.keys()):
        if "lora_trainer" in mod:
            del sys.modules[mod]
    from training.training.lora_trainer import LoRATrainer
    return LoRATrainer


def _write_dataset(path: Path, records):
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def test_messages_record_not_wrapped_with_legacy_markers(tmp_path):
    """A record with 'messages' key must not have [SYSTEM]/[USER]/[RESPONSE] injected."""
    mock_flm, mock_sft_trainer = _inject_unsloth_mock()
    mock_model = MagicMock()
    mock_model.named_modules.return_value = [
        ("model.layers.0.self_attn.q_proj", MagicMock()),
        ("model.layers.0.self_attn.k_proj", MagicMock()),
        ("model.layers.0.self_attn.v_proj", MagicMock()),
        ("model.layers.0.self_attn.o_proj", MagicMock()),
    ]

    def _fake_save(path, *args, **kwargs):
        from pathlib import Path as _P
        p = _P(path)
        p.mkdir(parents=True, exist_ok=True)
        (p / "adapter_model.safetensors").write_bytes(b"fake")

    mock_model.save_pretrained.side_effect = _fake_save
    mock_flm.from_pretrained.return_value = (mock_model, MagicMock())
    mock_flm.get_peft_model.return_value = mock_model

    captured_dataset = []

    def _capture_trainer(model, processing_class, train_dataset, args):
        if isinstance(train_dataset, list):
            captured_dataset.extend(train_dataset)
        instance = MagicMock()
        instance.state.log_history = [{"loss": 0.5}]
        return instance

    mock_sft_trainer.side_effect = _capture_trainer

    record = {
        "messages": [
            {"role": "system", "content": "You are a coder."},
            {"role": "user", "content": "Write a function."},
            {"role": "assistant", "content": "def f(): pass"},
        ],
        "score": 0.9,
    }
    dataset_file = tmp_path / "data.jsonl"
    _write_dataset(dataset_file, [record])

    LoRATrainer = _reload_trainer()
    trainer = LoRATrainer(base_model="test-model", output_dir=str(tmp_path / "out"))

    import asyncio
    recipe = _make_recipe()
    asyncio.run(trainer.train(dataset_path=dataset_file, recipe=recipe))

    assert len(captured_dataset) >= 1
    processed = captured_dataset[0]
    assert "messages" in processed, "messages key must be preserved"
    assert "prompt" not in processed or not processed.get("prompt", "").startswith("[SYSTEM]")
    for key in ("prompt", "completion"):
        if key in processed:
            for marker in ("[SYSTEM]", "[USER]", "[RESPONSE]"):
                assert marker not in processed.get(key, ""), \
                    f"Legacy marker {marker!r} must not appear in {key}"


def test_promote_to_active_uses_recipe_sft_format(tmp_path):
    """_promote_to_active must write recipe.sft_format from metadata history, not 'chatml'."""
    import json as _json
    from datetime import datetime, timezone

    from training.training.lora_trainer import _promote_to_active, _ADAPTER_TARGET_MODULES

    adapter_name = "coding_lora"
    checkpoint_dir = tmp_path
    save_path = checkpoint_dir / adapter_name
    save_path.mkdir()

    new_meta = {
        "name": adapter_name,
        "version": "1.1.0",
        "capability": "coding",
        "backend": "primary",
        "base_model": "test-model",
        "status": "staging",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "lora_r": 32,
        "lora_alpha": 64,
        "target_modules": _ADAPTER_TARGET_MODULES,
        "size_mb": 1.0,
        "dataset_hash": "abc",
        "tokenizer_hash": "def",
        "trainer_version": "test",
        "history": [
            {
                "version": "1.1.0",
                "status": "staging",
                "recipe": {"sft_format": "openai_messages", "peft_type": "rslora"},
            }
        ],
    }

    (save_path / "metadata.json").write_text(_json.dumps(new_meta))
    (save_path / "adapter_model.safetensors").write_bytes(b"fake")

    manifest_path = checkpoint_dir / "manifest.json"
    _promote_to_active(checkpoint_dir, "coding", new_meta)

    assert manifest_path.exists()
    manifest = _json.loads(manifest_path.read_text())
    history = manifest["adapters"][adapter_name]["history"]
    assert len(history) >= 1
    last = history[-1]
    assert last["recipe"]["sft_format"] == "openai_messages", \
        f"Expected 'openai_messages', got {last['recipe'].get('sft_format')!r}"
    assert "chatml" not in _json.dumps(last["recipe"]), \
        "The literal 'chatml' must not appear in the promoted history recipe"
