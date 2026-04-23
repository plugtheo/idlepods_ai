"""
Inference Service — settings
=============================
All configuration is read from environment variables.

Local mode  (INFERENCE__MODE=local):
  Each model family maps to a running vLLM server.
  INFERENCE__DEEPSEEK_URL   — URL of the DeepSeek vLLM server
  INFERENCE__MISTRAL_URL    — URL of the Mistral vLLM server

API mode  (INFERENCE__MODE=api):
  Uses LiteLLM to route to Anthropic (or any OpenAI-compatible provider).
  INFERENCE__API_PROVIDER   — e.g. "anthropic" (default)
  INFERENCE__API_MODEL      — e.g. "claude-3-5-haiku-20241022" (default)
  INFERENCE__API_KEY        — provider API key

Set via `.env` file or shell environment, prefixed with INFERENCE__.
"""

from typing import Dict, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class InferenceSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="INFERENCE__",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Mode ──────────────────────────────────────────────────────────────
    mode: str = Field(
        default="local",
        description="'local' = hit vLLM servers; 'api' = use LiteLLM/Anthropic.",
    )

    # ── Local vLLM URLs ───────────────────────────────────────────────────
    deepseek_url: str = Field(
        default="http://vllm-deepseek:8000",
        description=(
            "Base URL of the DeepSeek vLLM OpenAI-compatible server. "
            "Override to http://localhost:8000 for bare-metal local runs."
        ),
    )
    mistral_url: str = Field(
        default="http://vllm-mistral:8001",
        description=(
            "Base URL of the Mistral vLLM OpenAI-compatible server. "
            "Override to http://localhost:8001 for bare-metal local runs."
        ),
    )

    # ── Local model identifiers (used in vLLM /v1/chat/completions 'model' field) ─
    deepseek_model_id: str = Field(
        default="deepseek-ai/deepseek-coder-6.7b-instruct",
        description="HuggingFace model ID served by the DeepSeek vLLM server.",
    )
    mistral_model_id: str = Field(
        default="mistralai/Mistral-7B-Instruct-v0.1",
        description="HuggingFace model ID served by the Mistral vLLM server.",
    )

    # ── API (LiteLLM) settings ────────────────────────────────────────────
    api_provider: str = Field(
        default="anthropic",
        description=(
            "LiteLLM provider string, e.g. 'anthropic', 'openai', 'together_ai'. "
            "See https://docs.litellm.ai/docs/providers for the full list."
        ),
    )
    api_model: str = Field(
        default="claude-3-5-haiku-20241022",
        description=(
            "Model identifier passed to LiteLLM. "
            "Anthropic default: 'claude-3-5-haiku-20241022' (cost-efficient). "
            "For higher quality use 'claude-3-5-sonnet-20241022'."
        ),
    )
    api_key: Optional[str] = Field(
        default=None,
        description="API key for the chosen provider (e.g. ANTHROPIC_API_KEY value).",
    )
    # Per-role model overrides for API mode (dev/stage/prod).
    # Maps agent role → LiteLLM model string (fine-tuned endpoint or stronger model).
    # Roles not listed fall back to api_model.
    # Example in .env:
    #   INFERENCE__ROLE_MODEL_OVERRIDES={"coder":"ft:gpt-4o:org:id","planner":"claude-3-5-sonnet-20241022"}
    role_model_overrides: Dict[str, str] = Field(
        default_factory=dict,
        description="role → LiteLLM model string; overrides api_model per role in api mode.",
    )

    # ── HTTP client settings ──────────────────────────────────────────────
    request_timeout_seconds: float = Field(
        default=120.0,
        description="HTTP timeout for calls to vLLM servers or external APIs.",
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

    # ── Service ports ─────────────────────────────────────────────────────
    port: int = Field(default=8010, description="HTTP port this service listens on.")
    grpc_port: int = Field(default=50051, description="gRPC port (runs alongside HTTP).")


settings = InferenceSettings()
