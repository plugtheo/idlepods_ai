"""
Inference Service — settings
=============================
All configuration is read from environment variables.
Backend identity is driven by models.yaml (INFERENCE__MODELS_YAML_PATH).
"""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from shared.contracts.inference import (
    GRPC_DEFAULT_MAX_TOKENS,
    GRPC_DEFAULT_TEMPERATURE,
    GRPC_DEFAULT_TOP_P,
)


class InferenceSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="INFERENCE__",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Backend registry ──────────────────────────────────────────────────
    models_yaml_path: str = Field(
        default="/config/models.yaml",
        description="Path to the models.yaml registry file.",
    )
    accept_legacy_backend_names: bool = Field(
        default=False,
        description="Deprecated: legacy backend alias translation has been removed; this field is kept for backwards-compatible env var parsing only.",
    )

    # ── HTTP client settings ──────────────────────────────────────────────
    request_timeout_seconds: float = Field(
        default=300.0,
        description="HTTP timeout for calls to vLLM servers.",
    )
    http_max_connections: int = Field(
        default=10,
        description="Max total connections in the httpx connection pool for vLLM backends.",
    )
    http_max_keepalive_connections: int = Field(
        default=5,
        description="Max idle keep-alive connections in the httpx connection pool for vLLM backends.",
    )

    # ── gRPC server-side sampling defaults ────────────────────────────────
    grpc_default_max_tokens: int = Field(
        default=GRPC_DEFAULT_MAX_TOKENS,
        ge=1,
        le=8192,
        description=(
            "Default max_tokens applied by the gRPC server when the proto request "
            "omits the field (HasField check). "
            "Override with INFERENCE__GRPC_DEFAULT_MAX_TOKENS. "
            "Must match the gRPC client default in shared/contracts/inference.py."
        ),
    )
    grpc_default_temperature: float = Field(
        default=GRPC_DEFAULT_TEMPERATURE,
        ge=0.0,
        le=2.0,
        description=(
            "Default temperature applied by the gRPC server when the proto request "
            "omits the field. Override with INFERENCE__GRPC_DEFAULT_TEMPERATURE. "
            "Must match the gRPC client default in shared/contracts/inference.py."
        ),
    )
    grpc_default_top_p: float = Field(
        default=GRPC_DEFAULT_TOP_P,
        ge=0.0,
        le=1.0,
        description=(
            "Default top_p applied by the gRPC server when the proto request "
            "omits the field. Override with INFERENCE__GRPC_DEFAULT_TOP_P. "
            "Must match the gRPC client default in shared/contracts/inference.py."
        ),
    )

    # ── Adapter discovery ─────────────────────────────────────────────────
    adapter_cache_ttl_seconds: int = Field(
        default=120,
        description="How often (seconds) to re-query vLLM /v1/models to discover newly trained adapters.",
    )

    # ── Adapter bootstrap ─────────────────────────────────────────────────
    lora_manifest_path: str = Field(
        default="/data/lora_checkpoints/manifest.json",
        description="Path to the LoRA adapter manifest written by the training service. Read at startup to restore adapter state after restart.",
    )

    # ── Adapter auto-rollback ─────────────────────────────────────────────
    adapter_fallback_rollback_threshold: int = Field(
        default=5,
        description="Number of base-fallback events within the window that triggers auto-rollback.",
    )
    adapter_fallback_window_seconds: int = Field(
        default=60,
        description="Sliding window (seconds) for counting adapter fallback events.",
    )

    # ── Service ports ─────────────────────────────────────────────────────
    port: int = Field(default=8010, description="HTTP port this service listens on.")
    grpc_port: int = Field(default=50051, description="gRPC port (runs alongside HTTP).")
    grpc_shutdown_grace_seconds: float = Field(
        default=5.0,
        description="Seconds to wait for in-flight gRPC RPCs to finish on shutdown.",
    )


settings = InferenceSettings()
