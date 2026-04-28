"""
Context Service — settings
===========================
The Context Service uses a local self-hosted ChromaDB instance for all
few-shot retrieval. It connects to the configured HTTP host and port.

Other variables
---------------
CONTEXT__CHROMA_COLLECTION   — collection name (must match Experience Service)
CONTEXT__EMBEDDING_MODEL     — sentence-transformers model for embeddings
CONTEXT__MAX_FEW_SHOTS       — max past examples to inject per request
CONTEXT__SIMILARITY_THRESHOLD — minimum cosine similarity for RAG results
CONTEXT__REPO_PATH           — local repo root used for code snippet scanning
CONTEXT__PORT                — port this service listens on
"""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ContextSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="CONTEXT__",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    chroma_host: str = Field(
        default="chromadb",
        description="ChromaDB HTTP server hostname (self-hosted).",
    )
    chroma_port: int = Field(
        default=8000,
        description="ChromaDB HTTP server port (self-hosted mode).",
    )
    chroma_collection: str = Field(
        default="experiences",
        description="ChromaDB collection name — must match the Experience Service.",
    )
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="Sentence-transformers model used for embedding prompts.",
    )
    max_few_shots: int = Field(
        default=4,
        description="Maximum number of past examples to include in BuiltContext.",
    )
    similarity_threshold: float = Field(
        default=0.68,
        ge=0.0,
        le=1.0,
        description="Minimum cosine similarity for a past example to be included.",
    )
    repo_path: str = Field(
        default=".",
        description="Root of the repo to scan for relevant code snippets.",
    )
    max_repo_snippets: int = Field(
        default=3,
        description="Maximum number of repo snippets to return.",
    )
    port: int = Field(default=8011, description="Port this service listens on.")


settings = ContextSettings()
