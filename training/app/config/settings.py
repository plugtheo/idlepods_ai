"""
Training configuration
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class TrainingSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="TRAINING__", env_nested_delimiter="__", env_file=".env")

    # Shared data paths (same volumes as Experience Service)
    jsonl_path: str = Field(
        "/data/experiences.jsonl",
        description="Path to the JSONL experience file written by the Experience Service",
    )
    output_dir: str = Field(
        "/data/lora_checkpoints",
        description="Directory to write trained LoRA adapter checkpoints",
    )

    # Backend registry
    models_yaml_path: str = Field(
        default="/config/models.yaml",
        description="Path to the models.yaml registry file.",
    )

    # Diversity thresholds
    min_batch_size: int = Field(
        50,
        description="Minimum experience records before training is considered",
    )
    min_score_spread: float = Field(
        0.15,
        description="max_score - min_score must be >= this value",
    )
    min_diversity_ratio: float = Field(
        0.60,
        description="Fraction of records with unique prompt fingerprints",
    )
    min_quality_score: float = Field(
        0.65,
        description="Minimum quality score for an experience to be included in training data",
    )

    # HuggingFace auth
    hf_token: str = Field("", description="HuggingFace Hub token for downloading gated models")

    # ── LoRA training hyperparameters ─────────────────────────────────────
    max_seq_length: int = Field(default=2048)
    response_max_chars: int = Field(default=6000)
    lora_num_epochs: int = Field(default=3)
    lora_learning_rate: float = Field(default=2e-4)
    lora_rank: int = Field(default=16)
    lora_alpha: int = Field(default=32)

    # ── Scheduler / wrapper ──────────────────────────────────────────────
    training_target: Literal["local", "remote"] = Field(
        default="local",
        description="Where training runs: 'local' = in-process trainer_entry; 'remote' = HTTP POST to training_trigger_url.",
    )
    training_trigger_url: str = Field(
        default="",
        description="Remote training endpoint (used only when training_target='remote').",
    )
    training_exclusive_mode: Literal["BLOCK", "ALLOW"] = Field(
        default="BLOCK",
        description="BLOCK = stop vLLM during training (free GPU); ALLOW = run concurrently.",
    )
    scheduler_interval_hours: int = Field(
        default=4,
        description="Cron interval (hours) between scheduler ticks.",
    )
    scheduler_poll_interval_seconds: int = Field(
        default=60,
        description="Seconds between schedule.run_pending() calls in the scheduler main loop.",
    )
    remote_trigger_timeout_seconds: float = Field(
        default=60.0,
        description="HTTP timeout for the remote training trigger POST request.",
    )
    lock_path: str = Field(
        default="/data/training.lock",
        description="Filesystem lock to prevent concurrent training jobs (scheduler-side only).",
    )
    vllm_services: list[str] = Field(
        default_factory=lambda: ["vllm-primary"],
        description="Compose service names of local vLLM instances to stop/start in BLOCK mode.",
    )
    compose_file: str = Field(
        default="/compose/compose.yml",
        description="Path to compose.yml inside the wrapper/scheduler container (for docker compose stop/start).",
    )


settings = TrainingSettings()
