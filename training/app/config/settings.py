"""
Training Service — configuration
"""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class TrainingSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="TRAINING__", env_nested_delimiter="__")

    # Shared data paths (same volumes as Experience Service)
    jsonl_path: str = Field(
        "/data/experiences.jsonl",
        description="Path to the JSONL experience file written by the Experience Service",
    )
    output_dir: str = Field(
        "/data/lora_checkpoints",
        description="Directory to write trained LoRA adapter checkpoints",
    )

    # Base model identifiers
    deepseek_model: str = Field(
        "deepseek-ai/deepseek-coder-6.7b-instruct",
        description="HuggingFace model ID for DeepSeek LoRA training",
    )
    mistral_model: str = Field(
        "mistralai/Mistral-7B-Instruct-v0.1",
        description="HuggingFace model ID for Mistral LoRA training",
    )

    # Diversity thresholds
    min_batch_size: int = Field(
        50,
        description="Minimum experience records before training is considered",
    )
    min_score_spread: float = Field(
        0.15,
        description="max_score - min_score must be >= this value (ensures varied quality labels)",
    )
    min_diversity_ratio: float = Field(
        0.60,
        description="Fraction of records with unique prompt fingerprints (deduplication guard)",
    )
    min_quality_score: float = Field(
        0.65,
        description="Minimum quality score for an experience to be included in training data",
    )

    # HuggingFace auth (optional; required for gated models)
    hf_token: str = Field("", description="HuggingFace Hub token for downloading gated models")

    # ── LoRA training hyperparameters ─────────────────────────────────────
    max_seq_length: int = Field(
        default=2048,
        description=(
            "Maximum token sequence length for LoRA fine-tuning. "
            "Controls Unsloth FastLanguageModel.from_pretrained max_seq_length "
            "and SFTTrainer max_seq_length. Must match or be smaller than the "
            "base model's native context window. "
            "Override with TRAINING__MAX_SEQ_LENGTH."
        ),
    )
    response_max_chars: int = Field(
        default=6000,
        description=(
            "Maximum character count for a training response before it is skipped "
            "as a 'wall-of-text' sample that likely exceeds the training context window. "
            "Rule of thumb: max_seq_length * chars_per_token ≈ 2048 * 3 = 6144. "
            "Override with TRAINING__RESPONSE_MAX_CHARS."
        ),
    )
    lora_num_epochs: int = Field(
        default=3,
        description=(
            "Number of SFT training epochs for LoRA adapter training. "
            "Applies to both the one-time bootstrap script (train_gpu_simple.py) "
            "and the online self-training path (trainer_entry.py). "
            "Override with TRAINING__LORA_NUM_EPOCHS."
        ),
    )
    lora_learning_rate: float = Field(
        default=2e-4,
        description=(
            "AdamW learning rate for LoRA fine-tuning. "
            "Override with TRAINING__LORA_LEARNING_RATE."
        ),
    )
    lora_rank: int = Field(
        default=16,
        description=(
            "LoRA rank (r). Higher values increase capacity but also VRAM usage. "
            "Override with TRAINING__LORA_RANK."
        ),
    )
    lora_alpha: int = Field(
        default=32,
        description=(
            "LoRA alpha scaling factor. Effective learning rate scales as alpha/rank. "
            "Override with TRAINING__LORA_ALPHA."
        ),
    )

    port: int = Field(8013, description="Listening port")


settings = TrainingSettings()
