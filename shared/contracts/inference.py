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
import os
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)

# One-release legacy alias map — only active when INFERENCE__ACCEPT_LEGACY_BACKEND_NAMES=true.
# Keys are old model-name strings; values are registry backend names.
_LEGACY_BACKEND_ALIASES: Dict[str, str] = {
    "qwen":     "primary",
}
_legacy_warned: set[str] = set()


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
    max_tokens: int = Field(default=1024, ge=1, le=8192)
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    top_p: float = Field(default=0.95, ge=0.0, le=1.0)
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

    @field_validator("backend", mode="before")
    @classmethod
    def _resolve_backend(cls, v: str) -> str:
        accept_legacy = os.environ.get("INFERENCE__ACCEPT_LEGACY_BACKEND_NAMES", "false").lower() == "true"
        if accept_legacy and v in _LEGACY_BACKEND_ALIASES:
            mapped = _LEGACY_BACKEND_ALIASES[v]
            if v not in _legacy_warned:
                logger.warning("Legacy backend name %r → %r", v, mapped)
                _legacy_warned.add(v)
            return mapped
        # Validate against registry (import here to avoid circular at module load).
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
