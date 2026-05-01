"""
Inference Service — settings
=============================
All configuration is read from environment variables.

The inference service runs locally on a self-hosted vLLM server (Qwen/Qwen3-14B).
The only required parameters are the Qwen vLLM URL, model ID, gRPC defaults,
timeout, and ports.
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

    # ── Qwen backend config ───────────────────────────────────────────────
    qwen_backend: str = Field(
        default="local_vllm",
        description=(
            "Backend type for Qwen family: 'local_vllm' (default) or 'remote_vllm'. "
            "Set to 'remote_vllm' and point INFERENCE__QWEN_URL at any "
            "OpenAI-compatible endpoint to route Qwen agent calls there."
        ),
    )
    qwen_url: str = Field(
        default="http://vllm-qwen:8000",
        description=(
            "Base URL of the Qwen vLLM OpenAI-compatible server. "
            "Override to http://localhost:8000 for bare-metal local runs."
        ),
    )
    qwen_model_id: str = Field(
        default="Qwen/Qwen3-14B",
        description="HuggingFace model ID served by the Qwen vLLM server.",
    )
    qwen_auth_token: str = Field(
        default="",
        description=(
            "Optional Bearer token for secured remote Qwen endpoints. "
            "Only used when qwen_backend=remote_vllm."
        ),
    )
    qwen_ssl_verify: bool = Field(
        default=True,
        description="Verify SSL certificates for remote Qwen connections.",
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

    mode: str = Field(
        default="local",
        description="Inference service mode, currently always local self-hosted vLLM.",
    )

    # ── Service ports ─────────────────────────────────────────────────────
    port: int = Field(default=8010, description="HTTP port this service listens on.")
    grpc_port: int = Field(default=50051, description="gRPC port (runs alongside HTTP).")
    grpc_shutdown_grace_seconds: float = Field(
        default=5.0,
        description="Seconds to wait for in-flight gRPC RPCs to finish on shutdown.",
    )


settings = InferenceSettings()
