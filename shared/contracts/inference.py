"""
Inference Service contracts
===========================
Shapes used when Orchestration Service calls the Inference Service.

A `Message` is one conversation turn (system/user/assistant/tool).
`GenerateRequest` carries the full messages list plus which backend
and LoRA adapter to use.
`GenerateResponse` wraps the model's output text (and any tool calls) back
to the caller.
"""

import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)

# Canonical defaults for gRPC sampling parameters.
# Both the gRPC client (orchestration) and gRPC server (inference) import these
# so the wire-elision optimisation (skip field when value equals default) stays
# consistent even as env-var overrides are applied.
GRPC_DEFAULT_MAX_TOKENS: int = 1024
GRPC_DEFAULT_TEMPERATURE: float = 0.2
GRPC_DEFAULT_TOP_P: float = 0.95


class Message(BaseModel):
    """One conversation turn."""

    role: str
    content: Optional[str] = None
    tool_calls: Optional[List] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None


class ToolDefinition(BaseModel):
    """OpenAI-compatible function tool definition."""

    type: str = "function"
    function: Dict[str, Any]


class GenerateRequest(BaseModel):
    """Request sent from Orchestration → Inference Service."""

    backend: str = Field(
        description="Opaque backend name from models.yaml, e.g. 'primary'."
    )
    role: str = Field(
        description="Agent role performing this call, e.g. 'coder', 'planner'."
    )
    messages: List[Message] = Field(
        description="Full conversation so far, including system prompt."
    )
    adapter_name: Optional[str] = Field(
        default=None,
        description=(
            "LoRA adapter name to activate on the vLLM server, "
            "e.g. 'coding_lora'. Pass None to use the base model."
        ),
    )
    max_tokens: int = Field(default=GRPC_DEFAULT_MAX_TOKENS, ge=1, le=8192)
    temperature: float = Field(default=GRPC_DEFAULT_TEMPERATURE, ge=0.0, le=2.0)
    top_p: float = Field(default=GRPC_DEFAULT_TOP_P, ge=0.0, le=1.0)
    session_id: Optional[str] = Field(
        default=None,
        description="Opaque string for log correlation across service calls.",
    )
    tools: Optional[List[ToolDefinition]] = Field(
        default=None,
        description="OpenAI-compatible tool definitions to pass to the model.",
    )
    thinking_enabled: bool = Field(
        default=False,
        description="Enable Qwen3 thinking mode (disabled by default).",
    )
    response_schema: Optional[Dict[str, Any]] = None  # JSON schema for guided decoding

    @field_validator("backend", mode="before")
    @classmethod
    def _validate_backend(cls, v: str) -> str:
        from shared.contracts.models import load_registry
        try:
            registry = load_registry()
            if v not in registry.backends:
                raise ValueError(
                    f"Unknown backend '{v}'. Known: {list(registry.backends)}"
                )
        except RuntimeError:
            # Registry file not yet mounted (e.g. during unit tests) — pass through.
            pass
        return v


class GenerateResponse(BaseModel):
    """Response returned from Inference Service → Orchestration."""

    content: str = Field(description="Generated text from the model.")
    backend: str = Field(description="Backend name that generated this response.")
    role: str = Field(description="Agent role that was used.")
    tokens_generated: int = Field(
        default=0, description="Number of tokens in the generated output."
    )
    session_id: Optional[str] = Field(default=None)
    tool_calls: Optional[List] = Field(
        default=None,
        description="OpenAI-format tool calls emitted by the model, if any.",
    )
    used_base_fallback: bool = Field(
        default=False,
        description="True when the base model served this response because the requested adapter was unavailable.",
    )
    parsed: Optional[Dict[str, Any]] = Field(
        default=None,
        description=(
            "If response_schema was provided in the request, the model's output parsed according to that schema. "
            "Otherwise None."
        ),
    )
