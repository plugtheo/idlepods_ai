"""
Inference Service contracts
===========================
Shapes used when Orchestration Service calls the Inference Service.

A `Message` is one turn in a conversation (system/user/assistant/tool).
`GenerateRequest` carries the full messages list plus which model family
and LoRA adapter to use.
`GenerateResponse` wraps the model's output text (and any tool calls) back
to the caller.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


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

    model_family: str = Field(
        description="Which model family to use, e.g. 'qwen'."
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


class GenerateResponse(BaseModel):
    """Response returned from Inference Service → Orchestration."""

    content: str = Field(description="Generated text from the model.")
    model_family: str = Field(description="Model family that generated this response.")
    role: str = Field(description="Agent role that was used.")
    tokens_generated: int = Field(
        default=0, description="Number of tokens in the generated output."
    )
    session_id: Optional[str] = Field(default=None)
    tool_calls: Optional[List] = Field(
        default=None,
        description="OpenAI-format tool calls emitted by the model, if any.",
    )
