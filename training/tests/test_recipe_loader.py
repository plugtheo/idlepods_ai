"""Tests for AdapterRecipe YAML round-trip, precedence, and override."""
import textwrap
import tempfile
from pathlib import Path

import pytest

from shared.contracts.training import AdapterRecipe, load_recipes, lookup_recipe


@pytest.fixture()
def recipes_yaml(tmp_path: Path) -> Path:
    content = textwrap.dedent("""
        default:
          peft_type: lora
          r: 16
          alpha: 32
          dropout: 0.0
          target_modules: [q_proj, k_proj, v_proj]
          learning_rate: 2e-4
          num_epochs: 3
          max_seq_length: 2048
          sft_format: openai_messages
          tool_call_style: openai_native

        by_role:
          coder: {r: 32, alpha: 64, num_epochs: 4}
          consensus: {peft_type: none}

        by_backend_role:
          "secondary:coder": {peft_type: rslora, use_rslora: true, r: 8, alpha: 16}
    """)
    p = tmp_path / "recipes.yaml"
    p.write_text(content)
    return p


def _fresh_load(path: Path):
    # Clear lru_cache between tests.
    load_recipes.cache_clear()
    return load_recipes(str(path))


def test_default_round_trip(recipes_yaml):
    reg = _fresh_load(recipes_yaml)
    assert reg.default.peft_type == "lora"
    assert reg.default.r == 16
    assert reg.default.sft_format == "openai_messages"


def test_role_precedence_over_default(recipes_yaml):
    reg = _fresh_load(recipes_yaml)
    coder = reg.lookup("primary", "coder")
    assert coder.r == 32
    assert coder.alpha == 64
    assert coder.num_epochs == 4
    # defaults should still be inherited
    assert coder.sft_format == "openai_messages"


def test_backend_role_precedence_over_role(recipes_yaml):
    reg = _fresh_load(recipes_yaml)
    coder = reg.lookup("secondary", "coder")
    assert coder.peft_type == "rslora"
    assert coder.use_rslora is True
    assert coder.r == 8


def test_default_fallback_for_unknown_role(recipes_yaml):
    reg = _fresh_load(recipes_yaml)
    r = reg.lookup("primary", "unknown_role")
    assert r.r == 16
    assert r.peft_type == "lora"


def test_consensus_peft_none(recipes_yaml):
    reg = _fresh_load(recipes_yaml)
    r = reg.lookup("primary", "consensus")
    assert r.peft_type == "none"


def test_missing_yaml_returns_defaults():
    load_recipes.cache_clear()
    reg = load_recipes("/nonexistent/path/recipes.yaml")
    assert isinstance(reg.default, AdapterRecipe)
    assert reg.default.r == 16
