"""
Inference Service — settings
=============================
All configuration is read from environment variables.
Backend identity is driven by models.yaml (INFERENCE__MODELS_YAML_PATH).
"""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


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
        description="Accept legacy model-name strings as backend aliases (see _LEGACY_BACKEND_ALIASES in contracts).",
    )

    # ── HTTP client settings ──────────────────────────────────────────────
    request_timeout_seconds: float = Field(
        default=120.0,
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
        default=1024,
        ge=1,
        le=8192,
        description=(
            "Default max_tokens applied by the gRPC server when the proto request "
            "omits the field (HasField check). "
            "Override with INFERENCE__GRPC_DEFAULT_MAX_TOKENS."
        ),
    )
    grpc_default_temperature: float = Field(
        default=0.2,
        ge=0.0,
        le=2.0,
        description=(
            "Default temperature applied by the gRPC server when the proto request "
            "omits the field. Override with INFERENCE__GRPC_DEFAULT_TEMPERATURE."
        ),
    )
    grpc_default_top_p: float = Field(
        default=0.95,
        ge=0.0,
        le=1.0,
        description=(
            "Default top_p applied by the gRPC server when the proto request "
            "omits the field. Override with INFERENCE__GRPC_DEFAULT_TOP_P."
        ),
    )

    # ── Service ports ─────────────────────────────────────────────────────
    port: int = Field(default=8010, description="HTTP port this service listens on.")
    grpc_port: int = Field(default=50051, description="gRPC port (runs alongside HTTP).")
    grpc_shutdown_grace_seconds: float = Field(
        default=5.0,
        description="Seconds to wait for in-flight gRPC RPCs to finish on shutdown.",
    )


settings = InferenceSettings()
