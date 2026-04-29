"""
Orchestration Service — settings
==================================
ORCHESTRATION__INFERENCE_URL     — base URL of Inference Service
ORCHESTRATION__REQUEST_TIMEOUT   — general HTTP timeout for internal calls
ORCHESTRATION__PORT              — port this service listens on
ORCHESTRATION__DEFAULT_MAX_ITER  — default max agent-chain iterations
ORCHESTRATION__CONVERGENCE_THRESHOLD — default quality score to stop looping

ORCHESTRATION__JSONL_PATH        — append-only experience file
ORCHESTRATION__MIN_BATCH_SIZE    — minimum records before training trigger
ORCHESTRATION__TRAINING_URL      — Training Service base URL
ORCHESTRATION__EMBEDDING_MODEL   — sentence-transformers model for embeddings
ORCHESTRATION__CHROMADB_HOST     — empty = local PersistentClient; set = HttpClient

Agent constants (AGENT_PROMPTS, role_max_tokens, role_model_family, role_adapter)
are also exported from this module so nodes.py has a single import point.
"""

from typing import Dict, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# ── Agent system prompts ──────────────────────────────────────────────────
# Imported from shared/contracts/agent_prompts.py — the single source of truth
# used by all services (orchestration at inference time, training at train time,
# eval scripts at evaluation time).  Do NOT redefine these strings here.
from shared.contracts.agent_prompts import AGENT_PROMPTS  # noqa: F401 (re-exported)


class OrchestrationSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="ORCHESTRATION__",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    inference_url: str = Field(
        default="http://inference:8010",
        description="Base URL of the Inference Service.",
    )
    request_timeout: float = Field(
        default=180.0,
        description="General HTTP timeout for Inference Service calls.",
    )
    default_max_iterations: int = Field(
        default=5,
        description="Maximum agent-chain iterations when not specified by the caller.",
    )
    convergence_threshold: float = Field(
        default=0.85,
        description="Quality score (0–1) at which the loop stops early.",
    )
    port: int = Field(default=8001, description="Port this service listens on.")

    # ── Inference optimization levers ─────────────────────────────────────
    # Both default to True (enabled).  Set to False to revert to baseline
    # behaviour for debugging or A/B comparisons.
    optimize_role_history_filter: bool = Field(
        default=True,
        description=(
            "Each agent receives only history from roles it semantically depends on. "
            "Reduces input token count per call. "
            "Disable with ORCHESTRATION__OPTIMIZE_ROLE_HISTORY_FILTER=false."
        ),
    )
    optimize_structured_extraction: bool = Field(
        default=True,
        description=(
            "Structured-output roles (reviewer, critic, debugger) store only "
            "extracted key fields in iteration_history instead of full prose. "
            "Disable with ORCHESTRATION__OPTIMIZE_STRUCTURED_EXTRACTION=false."
        ),
    )

    # ── gRPC transport settings ───────────────────────────────────────────
    # When inference_use_grpc=True the orchestration service calls the
    # Inference Service over gRPC instead of HTTP.  gRPC uses HTTP/2
    # multiplexing and binary protobuf serialisation — lower per-call
    # latency than REST for the high-frequency agent→inference path.
    inference_use_grpc: bool = Field(
        default=False,
        description=(
            "Use gRPC to call the Inference Service instead of HTTP. "
            "Requires proto stubs generated at build time. "
            "Enable with ORCHESTRATION__INFERENCE_USE_GRPC=true."
        ),
    )
    inference_grpc_host: str = Field(
        default="inference",
        description="Hostname of the Inference Service gRPC server.",
    )
    inference_grpc_port: int = Field(
        default=50051,
        description="Port of the Inference Service gRPC server.",
    )
    model_context_len: int = Field(
        default=4096,
        description=(
            "Token context window of the deployed vLLM models. "
            "Must match --max-model-len in compose.yml. "
            "Override with ORCHESTRATION__MODEL_CONTEXT_LEN=8192."
        ),
    )

    # ── Context-budget tuning ─────────────────────────────────────────────
    # All numeric token-related constants live here so they can be adjusted
    # via environment variables without touching source code.

    chars_per_token: int = Field(
        default=3,
        description=(
            "Conservative characters-per-token ratio used for fast token estimation. "
            "Real averages: ~3.5 EN prose, ~2.5 code. Lowering this value makes "
            "budget estimates more conservative (less content fits per call). "
            "Override with ORCHESTRATION__CHARS_PER_TOKEN."
        ),
    )
    context_safety_margin: int = Field(
        default=128,
        description=(
            "Tokens reserved as overhead for message-wrapping and role/name fields. "
            "Subtracted from the available context budget before any content is added. "
            "Override with ORCHESTRATION__CONTEXT_SAFETY_MARGIN."
        ),
    )
    context_budget_history_ratio: float = Field(
        default=0.70,
        description=(
            "Fraction of the available context budget allocated to iteration history. "
            "Remaining fraction is split between repo snippets and few-shot examples. "
            "Override with ORCHESTRATION__CONTEXT_BUDGET_HISTORY_RATIO."
        ),
    )
    context_budget_repo_ratio: float = Field(
        default=0.20,
        description=(
            "Fraction of the available context budget allocated to repository snippets "
            "(code-facing roles only). Override with ORCHESTRATION__CONTEXT_BUDGET_REPO_RATIO."
        ),
    )
    history_lookback_iterations: int = Field(
        default=1,
        description=(
            "Number of past pipeline iterations whose history entries are eligible for "
            "inclusion (e.g. 1 = last iteration only; 0 = all iterations). "
            "The budget trimming system further limits what actually fits in context. "
            "Set to 0 to let growing history accumulate across all iterations — useful "
            "when model_context_len is large enough to absorb it. "
            "Override with ORCHESTRATION__HISTORY_LOOKBACK_ITERATIONS."
        ),
    )
    context_budget_conv_history_ratio: float = Field(
        default=0.40,
        description=(
            "Fraction of the history budget allocated to cross-turn conversation history "
            "(loaded from Redis). Remaining fraction goes to within-run iteration history. "
            "Override with ORCHESTRATION__CONTEXT_BUDGET_CONV_HISTORY_RATIO."
        ),
    )
    max_conversation_turns: int = Field(
        default=10,
        description=(
            "Maximum number of prior conversation turns to inject into agent prompts. "
            "Most-recent turns are kept; oldest are dropped when this cap is exceeded. "
            "Override with ORCHESTRATION__MAX_CONVERSATION_TURNS."
        ),
    )
    max_conversation_history_tokens: int = Field(
        default=2048,
        description=(
            "Token cap for the full conversation history persisted to Redis per task_id. "
            "Oldest entries are trimmed before saving when this cap is exceeded. "
            "Override with ORCHESTRATION__MAX_CONVERSATION_HISTORY_TOKENS."
        ),
    )
    fewshot_problem_max_chars: int = Field(
        default=300,
        description=(
            "Maximum characters kept from each few-shot example's problem field. "
            "Override with ORCHESTRATION__FEWSHOT_PROBLEM_MAX_CHARS."
        ),
    )
    fewshot_solution_max_chars: int = Field(
        default=400,
        description=(
            "Maximum characters kept from each few-shot example's solution field. "
            "Override with ORCHESTRATION__FEWSHOT_SOLUTION_MAX_CHARS."
        ),
    )
    repo_snippet_max_chars: int = Field(
        default=400,
        description=(
            "Maximum characters kept from each repository snippet. "
            "Override with ORCHESTRATION__REPO_SNIPPET_MAX_CHARS."
        ),
    )
    role_max_tokens: Dict[str, int] = Field(
        default_factory=lambda: {
            "planner":    512,
            "researcher": 768,
            "coder":      1536,
            "debugger":   1024,
            "reviewer":   512,
            "critic":     384,
            "consensus":  1536,
        },
        description=(
            "Per-role output token budget reserved for model generation. "
            "When overriding via env var the entire map must be supplied. "
            "Set as JSON: "
            'ORCHESTRATION__ROLE_MAX_TOKENS=\'{\'coder\'":2048,\'consensus\'":2048,...}\''
        ),
    )
    role_model_family: Dict[str, str] = Field(
        default_factory=lambda: {
            "planner":    "mistral",
            "researcher": "mistral",
            "coder":      "deepseek",
            "debugger":   "deepseek",
            "reviewer":   "deepseek",
            "critic":     "mistral",
            "consensus":  "mistral",
        },
        description=(
            "Maps each agent role to the vLLM model family it should run on. "
            "Must align with the model families served by the Inference Service. "
            "Override with ORCHESTRATION__ROLE_MODEL_FAMILY."
        ),
    )
    role_adapter: Dict[str, Optional[str]] = Field(
        default_factory=lambda: {
            "planner":    "planning_lora",
            "researcher": "research_lora",
            "coder":      "coding_lora",
            "debugger":   "debugging_lora",
            "reviewer":   "review_lora",
            "critic":     "criticism_lora",
            "consensus":  None,
        },
        description=(
            "Maps each agent role to the LoRA adapter name activated on the vLLM server "
            "for that request. Use null/None to run on the base model (no adapter). "
            "Override with ORCHESTRATION__ROLE_ADAPTER."
        ),
    )
    structured_field_value_max_chars: int = Field(
        default=300,
        description=(
            "Maximum characters kept per extracted structured field value "
            "(SCORE, ISSUES, SUGGESTIONS, etc.) when structured_extraction is enabled. "
            "Override with ORCHESTRATION__STRUCTURED_FIELD_VALUE_MAX_CHARS."
        ),
    )

    # ── Experience recording (formerly the Experience Service) ────────────
    jsonl_path: str = Field(
        default="/data/experiences.jsonl",
        description="Append-only JSONL file for experience records.",
    )
    min_batch_size: int = Field(
        default=50,
        description=(
            "Minimum stored experiences before a training trigger is attempted. "
            "Single source of truth — replaces the duplicated value previously in "
            "both experience/ and training/ settings."
        ),
    )
    training_url: str = Field(
        default="http://training:8013",
        description="Training Service base URL for LoRA trigger notifications.",
    )

    # ── Context enrichment (formerly the Context Service) ─────────────────
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="Sentence-transformers model for embedding prompts.",
    )
    max_few_shots: int = Field(
        default=4,
        description="Maximum number of past examples to include in context.",
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

    # ── ChromaDB client (Task 3) ──────────────────────────────────────────
    # CHROMADB_HOST empty (default) → local PersistentClient at CHROMADB_PATH.
    # CHROMADB_HOST set             → HttpClient connecting to a ChromaDB server.
    chromadb_host: str = Field(
        default="",
        description=(
            "ChromaDB server hostname. Empty = use local PersistentClient. "
            "Set to 'chromadb' (and uncomment the chromadb service in compose.yml) "
            "to switch to a standalone ChromaDB server."
        ),
    )
    chromadb_port: int = Field(
        default=8000,
        description="ChromaDB HTTP server port (only used when chromadb_host is set).",
    )
    chromadb_ssl: bool = Field(
        default=False,
        description="Use SSL when connecting to the ChromaDB HTTP server.",
    )
    chromadb_path: str = Field(
        default="/data/vector_store",
        description=(
            "Local file path for ChromaDB PersistentClient storage. "
            "Only used when chromadb_host is empty."
        ),
    )
    chromadb_collection: str = Field(
        default="experiences",
        description="ChromaDB collection name for experience vectors.",
    )

    # ── Redis session store ───────────────────────────────────────────────────
    redis_url: str = Field(
        default="redis://redis:6379",
        description=(
            "Redis connection URL for the session history store. "
            "Override with ORCHESTRATION__REDIS_URL."
        ),
    )
    redis_session_ttl_s: int = Field(
        default=3600,
        description=(
            "TTL in seconds for session history keys in Redis. "
            "Override with ORCHESTRATION__REDIS_SESSION_TTL_S."
        ),
    )


settings = OrchestrationSettings()
