"""
Experience Service — configuration
"""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ExperienceSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="EXPERIENCE__", env_nested_delimiter="__")

    # Local storage
    jsonl_path: str = Field(
        "/data/experiences.jsonl",
        description="Append-only JSONL file for experience records",
    )
    chroma_api_key: str = Field(
        default="",
        description=(
            "ChromaDB Cloud API key. When set, uses CloudClient (api.trychroma.com). "
            "Takes precedence over chroma_host."
        ),
    )
    chroma_tenant: str = Field(
        default="",
        description="ChromaDB Cloud tenant ID (required when chroma_api_key is set).",
    )
    chroma_database: str = Field(
        default="",
        description="ChromaDB Cloud database name (required when chroma_api_key is set).",
    )
    chroma_host: str = Field(
        default="chromadb",
        description="ChromaDB HTTP server hostname (self-hosted). Ignored when chroma_api_key is set.",
    )
    chroma_port: int = Field(
        default=8000,
        description="ChromaDB HTTP server port (self-hosted mode).",
    )
    embedding_model: str = Field(
        "sentence-transformers/all-MiniLM-L6-v2",
        description="Sentence-transformers model for embedding experience prompts",
    )
    chroma_collection: str = Field(
        "experiences",
        description="ChromaDB collection name for experience vectors",
    )

    # Training trigger
    training_url: str = Field(
        "",
        description=(
            "Training Service base URL. "
            "Leave empty (or unset EXPERIENCE__TRAINING_URL) in API mode — "
            "the Training Service is not started and the notification is skipped."
        ),
    )

    # Diversity + batch thresholds (kept in sync with Training Service settings)
    min_batch_size: int = Field(
        50,
        description="Minimum stored experiences before a training trigger is attempted",
    )

    port: int = Field(8012, description="Listening port")


settings = ExperienceSettings()
