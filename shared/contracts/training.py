"""
Training Service contracts
===========================
Shapes used when the Experience Service notifies the Training Service
that a capability may have crossed the training threshold.
"""

import functools
import os
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import yaml
from pydantic import BaseModel, Field


class TrainingTriggerRequest(BaseModel):
    """
    Sent from Experience Service → Training Service after a new experience
    is stored.  The Training Service decides whether the threshold is met
    and starts a training job if so.
    """

    capability: str = Field(
        description="Agent capability to evaluate for training, e.g. 'coding'."
    )
    new_experience_count: int = Field(
        description=(
            "Number of new experiences that prompted this trigger call. "
            "Currently always sent as 1 (one experience per call). "
            "The Training Service does not use this field for decisions — "
            "it loads and counts experiences itself."
        )
    )
    session_id: Optional[str] = Field(
        default=None, description="Session that produced the triggering experience."
    )


class AdapterRecipe(BaseModel):
    peft_type: Literal["lora", "rslora", "dora", "qlora", "none"] = "lora"
    r: int = 16
    alpha: int = 32
    dropout: float = 0.0
    target_modules: List[str] = Field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])
    use_rslora: bool = False
    use_dora: bool = False
    resume_from_prev_adapter: bool = False
    # Optional path to a synthetic-data JSONL for this role.  Mixed in with
    # experience + curated pairs in trainer_entry.py.  Future synthesis pipeline
    # writes here; missing/empty path is a no-op so this is safe to ship now.
    synthetic_dataset_path: Optional[str] = None
    loftq_config: Optional[Dict[str, Any]] = None
    load_in_4bit: bool = False
    learning_rate: float = 2e-4
    num_epochs: int = 3
    max_seq_length: int = 2048
    sft_format: Literal["openai_messages"] = "openai_messages"
    tool_call_style: Literal["openai_native", "hermes", "none"] = "openai_native"
    tokenizer_pre_tokenizer: Optional[Literal["bytelevel", "metaspace"]] = None


class RecipeRegistry(BaseModel):
    default: AdapterRecipe = Field(default_factory=AdapterRecipe)
    by_role: Dict[str, AdapterRecipe] = Field(default_factory=dict)
    by_backend_role: Dict[str, AdapterRecipe] = Field(default_factory=dict)  # key="backend:role"

    def lookup(self, backend: str, role: str) -> AdapterRecipe:
        key = f"{backend}:{role}"
        if key in self.by_backend_role:
            return self.default.model_copy(update=self.by_backend_role[key].model_dump(exclude_unset=True))
        if role in self.by_role:
            return self.default.model_copy(update=self.by_role[role].model_dump(exclude_unset=True))
        return self.default


def _find_recipes_yaml() -> Optional[str]:
    candidates = [
        os.environ.get("RECIPES_YAML_PATH", ""),
        "/config/recipes.yaml",
        str(Path(__file__).resolve().parents[2] / "recipes.yaml"),
    ]
    for p in candidates:
        if p and Path(p).exists():
            return p
    return None


@functools.lru_cache(maxsize=1)
def load_recipes(path: Optional[str] = None) -> RecipeRegistry:
    resolved = path or _find_recipes_yaml()
    if not resolved or not Path(resolved).exists():
        return RecipeRegistry()
    raw = yaml.safe_load(Path(resolved).read_text())
    if not raw:
        return RecipeRegistry()

    default_data = raw.get("default", {})
    default = AdapterRecipe(**default_data) if default_data else AdapterRecipe()

    by_role: Dict[str, AdapterRecipe] = {}
    for role, overrides in (raw.get("by_role") or {}).items():
        merged = {**default_data, **(overrides or {})}
        by_role[role] = AdapterRecipe(**merged)

    by_backend_role: Dict[str, AdapterRecipe] = {}
    for key_str, overrides in (raw.get("by_backend_role") or {}).items():
        # Keys must be "backend:role" strings in YAML.
        merged = {**default_data, **(overrides or {})}
        by_backend_role[str(key_str)] = AdapterRecipe(**merged)

    return RecipeRegistry(default=default, by_role=by_role, by_backend_role=by_backend_role)


def lookup_recipe(backend: str, role: str) -> AdapterRecipe:
    recipe = load_recipes().lookup(backend, role)

    # --- Runtime validation: recipes.yaml must override max_seq_length ---
    if recipe.max_seq_length == 2048:
        # This means the YAML did NOT override it.
        raise RuntimeError(
            f"[recipes.yaml] Role '{role}' resolved to default max_seq_length=2048. "
            "This indicates the YAML override is missing or misspelled."
        )

    return recipe


class TrainingTriggerResponse(BaseModel):
    """Response from Training Service after evaluating a trigger request."""

    capability: str
    triggered: bool = Field(
        description="True if a training job was started."
    )
    reason: str = Field(
        description="Human-readable explanation of why training was or was not triggered."
    )
