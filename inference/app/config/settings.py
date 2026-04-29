"""
Inference Service — settings
=============================
All configuration is read from environment variables.

The inference service runs locally on self-hosted vLLM servers.
The only required parameters are the DeepSeek and Mistral vLLM URLs,
model IDs, gRPC defaults, timeout, and ports.
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

    # ── DeepSeek backend config ───────────────────────────────────────────
    deepseek_backend: str = Field(
        default="local_vllm",
        description=(
            "Backend type for DeepSeek family: 'local_vllm' (default) or 'remote_vllm'. "
            "Set to 'remote_vllm' and point INFERENCE__DEEPSEEK_URL at any "
            "OpenAI-compatible endpoint to route DeepSeek agent calls there."
        ),
    )
    deepseek_url: str = Field(
        default="http://vllm-deepseek:8000",
        description=(
            "Base URL of the DeepSeek vLLM OpenAI-compatible server. "
            "Override to http://localhost:8000 for bare-metal local runs."
        ),
    )
    deepseek_model_id: str = Field(
        default="deepseek-ai/deepseek-coder-6.7b-instruct",
        description="HuggingFace model ID served by the DeepSeek vLLM server.",
    )
    deepseek_auth_token: str = Field(
        default="",
        description=(
            "Optional Bearer token for secured remote DeepSeek endpoints. "
            "Only used when deepseek_backend=remote_vllm."
        ),
    )
    deepseek_ssl_verify: bool = Field(
        default=True,
        description="Verify SSL certificates for remote DeepSeek connections.",
    )

    # ── Mistral backend config ────────────────────────────────────────────
    mistral_backend: str = Field(
        default="local_vllm",
        description=(
            "Backend type for Mistral family: 'local_vllm' (default) or 'remote_vllm'. "
            "Set to 'remote_vllm' and point INFERENCE__MISTRAL_URL at any "
            "OpenAI-compatible endpoint to route Mistral agent calls there."
        ),
    )
    mistral_url: str = Field(
        default="http://vllm-mistral:8001",
        description=(
            "Base URL of the Mistral vLLM OpenAI-compatible server. "
            "Override to http://localhost:8001 for bare-metal local runs."
        ),
    )
    mistral_model_id: str = Field(
        default="mistralai/Mistral-7B-Instruct-v0.1",
        description="HuggingFace model ID served by the Mistral vLLM server.",
    )
    mistral_auth_token: str = Field(
        default="",
        description=(
            "Optional Bearer token for secured remote Mistral endpoints. "
            "Only used when mistral_backend=remote_vllm."
        ),
    )
    mistral_ssl_verify: bool = Field(
        default=True,
        description="Verify SSL certificates for remote Mistral connections.",
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
    # Applied when a proto request omits optional sampling fields.
    # These mirror the shared/contracts/inference.py Pydantic defaults so
    # both HTTP and gRPC paths produce identical behaviour when callers omit
    # the fields. Centralised here so a single env-var change updates both.
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
