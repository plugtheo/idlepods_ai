"""Tests for apply_recipe — parametrised over peft_type variants."""
import sys
from types import ModuleType
from unittest.mock import MagicMock, patch
import pytest

from shared.contracts.training import AdapterRecipe


def _inject_unsloth():
    """Inject a minimal mock unsloth into sys.modules before import."""
    mock_unsloth = ModuleType("unsloth")
    mock_flm = MagicMock()
    mock_unsloth.FastLanguageModel = mock_flm
    sys.modules["unsloth"] = mock_unsloth
    return mock_flm


def _make_recipe(**kwargs) -> AdapterRecipe:
    base = dict(
        peft_type="lora",
        r=16,
        alpha=32,
        dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj"],
        use_rslora=False,
        use_dora=False,
        loftq_config=None,
        load_in_4bit=False,
        learning_rate=2e-4,
        num_epochs=3,
        max_seq_length=2048,
        sft_format="openai_messages",
        tool_call_style="openai_native",
    )
    base.update(kwargs)
    return AdapterRecipe(**base)


def _make_mock_model(modules=("q_proj", "k_proj", "v_proj")):
    """Return a mock model whose named_modules() yields the given leaf names."""
    model = MagicMock()
    model.named_modules.return_value = [(f"layer.0.{m}", MagicMock()) for m in modules]
    return model


@pytest.mark.parametrize("peft_type,use_rslora,use_dora", [
    ("lora", False, False),
    ("rslora", True, False),
    ("dora", False, True),
])
def test_apply_recipe_calls_get_peft_model(peft_type, use_rslora, use_dora):
    mock_flm = _inject_unsloth()
    mock_peft_model = MagicMock()
    mock_flm.get_peft_model.return_value = mock_peft_model

    recipe = _make_recipe(peft_type=peft_type, use_rslora=use_rslora, use_dora=use_dora)
    model = _make_mock_model()

    # Force re-import after injecting mock.
    if "training.training.lora_trainer" in sys.modules:
        del sys.modules["training.training.lora_trainer"]
    from training.training.lora_trainer import apply_recipe
    result = apply_recipe(model, recipe)

    mock_flm.get_peft_model.assert_called_once()
    _, kwargs = mock_flm.get_peft_model.call_args
    assert kwargs["r"] == recipe.r
    assert kwargs["lora_alpha"] == recipe.alpha
    assert kwargs["target_modules"] == recipe.target_modules
    assert kwargs["use_rslora"] == use_rslora
    assert kwargs["use_dora"] == use_dora
    assert result is mock_peft_model


def test_apply_recipe_fails_on_missing_target_module():
    _inject_unsloth()
    if "training.training.lora_trainer" in sys.modules:
        del sys.modules["training.training.lora_trainer"]
    from training.training.lora_trainer import apply_recipe

    recipe = _make_recipe(target_modules=["nonexistent_proj"])
    model = _make_mock_model(modules=["q_proj", "k_proj"])

    with pytest.raises(ValueError, match="not found in model"):
        apply_recipe(model, recipe)
