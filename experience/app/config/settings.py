"""
Experience Service — configuration

The Experience Service stores experience JSONL and upserts embeddings into
a local self-hosted ChromaDB instance.
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
    chroma_host: str = Field(
        default="chromadb",
        description="ChromaDB HTTP server hostname (self-hosted).",
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
        ...,
        description=(
            "Training Service base URL. "
            "This must be configured for local self-training mode."
        ),
    )

    # Diversity + batch thresholds (kept in sync with Training Service settings)
    min_batch_size: int = Field(
        50,
        description="Minimum stored experiences before a training trigger is attempted",
    )

    port: int = Field(8012, description="Listening port")


settings = ExperienceSettings()
