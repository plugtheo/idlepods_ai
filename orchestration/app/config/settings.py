"""
Orchestration Service — settings
==================================
ORCHESTRATION__INFERENCE_URL     — base URL of Inference Service
ORCHESTRATION__REQUEST_TIMEOUT   — general HTTP timeout for internal calls
ORCHESTRATION__PORT              — port this service listens on
ORCHESTRATION__DEFAULT_MAX_ITER  — default max agent-chain iterations
ORCHESTRATION__CONVERGENCE_THRESHOLD — default quality score to stop looping

ORCHESTRATION__JSONL_PATH        — append-only experience file
ORCHESTRATION__SPOOL_PATH        — spool file for in-flight experiences drained at shutdown
ORCHESTRATION__SHUTDOWN_DRAIN_TIMEOUT_S — seconds to wait for in-flight experience tasks at shutdown
ORCHESTRATION__EMBEDDING_MODEL   — sentence-transformers model for embeddings
ORCHESTRATION__CHROMADB_HOST     — empty = local PersistentClient; set = HttpClient

Agent constants (AGENT_PROMPTS, role_max_tokens, role_backend, role_adapter)
are also exported from this module so nodes.py has a single import point.
"""

from typing import Dict, List, Literal, Optional

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
    http_max_connections: int = Field(
        default=10,
        description="Max total connections in the httpx pool for Inference Service calls.",
    )
    http_max_keepalive_connections: int = Field(
        default=5,
        description="Max idle keep-alive connections in the httpx pool for Inference Service calls.",
    )
    default_max_iterations: int = Field(
        default=5,
        description="Maximum agent-chain iterations when not specified by the caller.",
    )
    convergence_threshold: float = Field(
        default=0.85,
        description="Quality score (0–1) at which the loop stops early.",
    )
    plateau_window_iterations: int = Field(
        default=3,
        description=(
            "Number of recent iteration scores compared when checking for a score plateau. "
            "Set to 0 to disable plateau early-stop entirely. "
            "Override with ORCHESTRATION__PLATEAU_WINDOW_ITERATIONS."
        ),
    )
    plateau_score_epsilon: float = Field(
        default=0.02,
        description=(
            "Maximum score spread (max − min) across the plateau window that is treated as "
            "stagnation and triggers an early stop before max_iterations is reached. "
            "Override with ORCHESTRATION__PLATEAU_SCORE_EPSILON."
        ),
    )
    plateau_min_iterations: int = Field(
        default=3,
        description=(
            "Minimum completed iterations before plateau early-stop becomes eligible. "
            "Prevents a false stop when only 1–2 scores exist. "
            "Override with ORCHESTRATION__PLATEAU_MIN_ITERATIONS."
        ),
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
        default=12288,
        description=(
            "Token context window of the deployed vLLM models. "
            "Must match --max-model-len in compose.yml. "
            "Override with ORCHESTRATION__MODEL_CONTEXT_LEN."
        ),
    )

    # ── Context-budget tuning ─────────────────────────────────────────────
    # All numeric token-related constants live here so they can be adjusted
    # via environment variables without touching source code.

    chars_per_token: float = Field(
        default=2.5,
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
        default=0,
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
        default=1000,   # Soft cap - token budget does heavy lifting
        description=(
            "Maximum number of prior conversation turns to inject into agent prompts. "
            "Most-recent turns are kept; oldest are dropped when this cap is exceeded. "
            "Override with ORCHESTRATION__MAX_CONVERSATION_TURNS."
        ),
    )
    max_conversation_history_tokens: int = Field(
        default=12288,
        description=(
            "Token cap for the full conversation history persisted to Redis per task_id. "
            "Oldest entries are trimmed before saving when this cap is exceeded. "
            "Override with ORCHESTRATION__MAX_CONVERSATION_HISTORY_TOKENS."
        ),
    )
    fewshot_problem_max_chars: int = Field(
        default=500,
        description=(
            "Maximum characters kept from each few-shot example's problem field. "
            "Override with ORCHESTRATION__FEWSHOT_PROBLEM_MAX_CHARS."
        ),
    )
    fewshot_solution_max_chars: int = Field(
        default=800,
        description=(
            "Maximum characters kept from each few-shot example's solution field. "
            "Override with ORCHESTRATION__FEWSHOT_SOLUTION_MAX_CHARS."
        ),
    )
    repo_snippet_max_chars: int = Field(
        default=1200,
        description=(
            "Maximum characters kept from each repository snippet. "
            "Override with ORCHESTRATION__REPO_SNIPPET_MAX_CHARS."
        ),
    )
    role_max_tokens: Dict[str, int] = Field(
        default_factory=lambda: {
            "planner":    512,
            "researcher": 768,
            "coder":      3072,
            "debugger":   2048,
            "reviewer":   512,
            "critic":     512,
            "consensus":  2048,
            "summarizer": 512,
            "router":     64,
        },
        description=(
            "Per-role output token budget reserved for model generation. "
            "When overriding via env var the entire map must be supplied. "
            "Set as JSON: "
            'ORCHESTRATION__ROLE_MAX_TOKENS=\'{\'coder\'":2048,\'consensus\'":2048,...}\''
        ),
    )
    role_backend: Dict[str, str] = Field(
        default_factory=lambda: {
            "planner":    "primary",
            "researcher": "primary",
            "coder":      "primary",
            "debugger":   "primary",
            "reviewer":   "primary",
            "critic":     "primary",
            "consensus":  "primary",
            "summarizer": "primary",
        },
        description=(
            "Maps each agent role to the models.yaml backend name it should run on. "
            "Override with ORCHESTRATION__ROLE_BACKEND."
        ),
    )
    role_adapter: Dict[str, Optional[str]] = Field(
        default_factory=lambda: {
            "planner":    None,
            "researcher": None,
            "coder":      None,
            "debugger":   None,
            "reviewer":   None,
            "critic":     None,
            "consensus":  None,
            "summarizer": "summarizer_lora",
        },
        description=(
            "Maps each agent role to the LoRA adapter name activated on the vLLM server "
            "for that request. Use null/None to run on the base model (no adapter). "
            "Override with ORCHESTRATION__ROLE_ADAPTER."
        ),
    )
    role_tools_enabled: Dict[str, List[str]] = Field(
        default_factory=lambda: {
            "planner":    ["web_search", "list_files"],
            "researcher": ["web_search", "read_file", "list_files"],
            "coder":      ["read_file", "write_file", "list_files", "run_command"],
            "debugger":   ["read_file", "write_file"],
            "reviewer":   [],
            "critic":     [],
            "consensus":  [],
            "summarizer": [],
        },
        description=(
            "Per-role tool allowlist. Empty list ⇒ role does not call tools. "
            "Override with ORCHESTRATION__ROLE_TOOLS_ENABLED."
        ),
    )
    non_streaming_roles: List[str] = Field(
        default_factory=lambda: ["reviewer", "critic", "review_critic", "router"],
        description=(
            "List of roles for which the inference call should use the blocking path even when a streaming response is available."
        ),
    )

    # ── Query router ─────────────────────────────────────────────────────────
    router_mode: str = Field(
        default="hybrid",
        description=(
            "Routing strategy: 'regex' (legacy keyword classifier), 'llm' "
            "(every prompt goes through a guided-JSON LLM call), or 'hybrid' "
            "(regex first, LLM only when regex confidence is below "
            "router_confidence_threshold). Override with ORCHESTRATION__ROUTER_MODE."
        ),
    )
    router_backend: str = Field(
        default="primary",
        description="Inference backend used for LLM-based routing.",
    )
    router_max_tokens: int = Field(
        default=64,
        description="Generation token budget for the router classification call.",
    )
    router_temperature: float = Field(
        default=0.0,
        description="Temperature for the router classification call (0.0 = deterministic).",
    )
    router_confidence_threshold: float = Field(
        default=0.6,
        description=(
            "Hybrid router only: regex confidence below this triggers an LLM "
            "classification call."
        ),
    )
    router_cache_size: int = Field(
        default=256,
        description="Maximum prompts cached in-process by LLMQueryRouter.",
    )

    # ── Few-shot retrieval scoping ───────────────────────────────────────────
    few_shot_scope: str = Field(
        default="task_exclude",
        description=(
            "Scope for few-shot RAG retrieval: 'global' (no filter — legacy "
            "behaviour, leaks across task_ids), or 'task_exclude' (exclude "
            "experiences from the current task_id, preventing in-flight echo "
            "and cross-task content leakage). Override with "
            "ORCHESTRATION__FEW_SHOT_SCOPE."
        ),
    )
    compaction_trigger_ratio: float = Field(
        default=0.85,
        description=(
            "Fraction of max_conversation_history_tokens at which compaction is triggered. "
            "Override with ORCHESTRATION__COMPACTION_TRIGGER_RATIO."
        ),
    )
    compaction_tool_output_threshold_tokens: int = Field(
        default=500,
        description=(
            "Tool-output messages larger than this many tokens are eligible for truncation. "
            "Override with ORCHESTRATION__COMPACTION_TOOL_OUTPUT_THRESHOLD_TOKENS."
        ),
    )
    compaction_retain_recent_messages: int = Field(
        default=6,
        description=(
            "Number of recent non-tool messages that are exempt from tool-output truncation. "
            "Override with ORCHESTRATION__COMPACTION_RETAIN_RECENT_MESSAGES."
        ),
    )
    compaction_retention_days: int = Field(
        default=30,
        description=(
            "TTL (days) for compacted summary entries stored in Redis. "
            "Override with ORCHESTRATION__COMPACTION_RETENTION_DAYS."
        ),
    )
    compaction_rollup_iteration_history: bool = Field(
        default=False,
        description=(
            "When true, applies LLM-based summarization to within-run iteration history "
            "in addition to cross-turn conversation history. "
            "Ship only after Phase 14 has been stable for one cycle. "
            "Override with ORCHESTRATION__COMPACTION_ROLLUP_ITERATION_HISTORY."
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

    # ── Pipeline dispatch strategy ────────────────────────────────────────
    pipeline_use_supervisor: bool = Field(
        default=False,
        description=(
            "When True, agent dispatch is driven by a supervisor node that reads "
            "plan state and intent instead of advancing a static agent_chain index. "
            "When False (default), the legacy chain-index path is used. "
            "Enable with ORCHESTRATION__PIPELINE_USE_SUPERVISOR=true."
        ),
    )
    pipeline_supervisor_max_steps: int = Field(
        default=8,
        description=(
            "Maximum number of plan-step dispatches the supervisor will make per "
            "iteration before forcing a convergence check.  Caps the recursion depth "
            "under the supervisor pipeline.  "
            "Override with ORCHESTRATION__PIPELINE_SUPERVISOR_MAX_STEPS."
        ),
    )
    pipeline_supervisor_strategy: Literal["deterministic", "llm", "hybrid"] = Field(
        default="deterministic",
        description=(
            "Supervisor routing strategy. 'deterministic' = pure rule-based, no LLM "
            "(default). 'llm' = single inference call per turn when the state is "
            "ambiguous; forced dispatches (R1/R1.5/R2a/R2b) bypass the LLM. "
            "'hybrid' = deterministic safety guards for R1-R3, LLM only for R4/R5 "
            "(no-plan open-set territory). "
            "Override with ORCHESTRATION__PIPELINE_SUPERVISOR_STRATEGY."
        ),
    )
    pipeline_supervisor_llm_calls_per_run: int = Field(
        default=8,
        description=(
            "Maximum LLM supervisor inference calls per pipeline run. When the cap is "
            "reached all further supervisor decisions fall back to deterministic rules. "
            "Only applies when pipeline_supervisor_strategy is 'llm' or 'hybrid'. "
            "Override with ORCHESTRATION__PIPELINE_SUPERVISOR_LLM_CALLS_PER_RUN."
        ),
    )
    pipeline_supervisor_backend: str = Field(
        default="primary",
        description=(
            "Inference backend used for LLM supervisor routing calls. "
            "Must match a backend name in models.yaml. "
            "Override with ORCHESTRATION__PIPELINE_SUPERVISOR_BACKEND."
        ),
    )
    pipeline_supervisor_max_tokens: int = Field(
        default=256,
        description=(
            "Output token budget for each LLM supervisor routing call. "
            "Kept small since the model only needs to emit a route_to() tool call. "
            "Override with ORCHESTRATION__PIPELINE_SUPERVISOR_MAX_TOKENS."
        ),
    )
    pipeline_supervisor_history_window: int = Field(
        default=6,
        description=(
            "Number of most-recent iteration_history entries included in the LLM "
            "supervisor prompt. Higher values give more context at the cost of more "
            "input tokens per routing call. "
            "Override with ORCHESTRATION__PIPELINE_SUPERVISOR_HISTORY_WINDOW."
        ),
    )
    pipeline_supervisor_max_decisions_per_run: int = Field(
        default=50,
        description=(
            "Hard cap on total supervisor decisions per pipeline run. When reached, "
            "supervisor_anchor forces check_convergence immediately without invoking "
            "the configured supervisor, preventing infinite routing loops. "
            "Set to 0 to disable. "
            "Override with ORCHESTRATION__PIPELINE_SUPERVISOR_MAX_DECISIONS_PER_RUN."
        ),
    )
    pipeline_supervisor_max_consecutive_same_role: int = Field(
        default=5,
        description=(
            "Maximum consecutive dispatches to the same agent role before the "
            "supervisor logs a WARNING and overrides the decision to check_convergence. "
            "Only applies to worker roles — tool_executor and check_convergence are "
            "exempt. Set to 0 to disable. "
            "Override with ORCHESTRATION__PIPELINE_SUPERVISOR_MAX_CONSECUTIVE_SAME_ROLE."
        ),
    )
    pipeline_supervisor_token_budget_per_run: int = Field(
        default=4096,
        description=(
            "Maximum total tokens generated by supervisor LLM routing calls per "
            "pipeline run. When exceeded, all further LLM calls fall back to "
            "deterministic rules. Only applies when strategy is 'llm' or 'hybrid'. "
            "Set to 0 to disable. "
            "Override with ORCHESTRATION__PIPELINE_SUPERVISOR_TOKEN_BUDGET_PER_RUN."
        ),
    )
    pipeline_supervisor_rollout_pct: int = Field(
        default=100,
        description=(
            "Percentage (0–100) of pipeline runs that use pipeline_supervisor_strategy. "
            "Sessions outside the window fall back to 'deterministic'. Uses an MD5 hash "
            "of session_id for consistent per-session routing. 100 = apply to all runs. "
            "Override with ORCHESTRATION__PIPELINE_SUPERVISOR_ROLLOUT_PCT."
        ),
    )

    # ── Tool runner ────────────────────────────────────────────────────────
    tool_output_truncate_chars: int = Field(
        default=4000,
        description="Tool output longer than this many characters is truncated before being passed to the next agent.",
    )

    # ── Experience recording (formerly the Experience Service) ────────────
    experience_dedupe_tail_lines: int = Field(
        default=100,
        description="Number of tail JSONL lines checked when deduplicating experience records on write.",
    )
    jsonl_dir: str = Field(
        default="/data",
        description="Directory where daily JSONL shards (experiences-YYYYMMDD.jsonl) are written.",
    )
    jsonl_path: str = Field(
        default="/data/experiences.jsonl",
        description="Legacy single-file path; still readable by the cursor until pruned.",
    )
    spool_path: str = Field(
        default="/data/experiences.spool.jsonl",
        description="Spool file holding experience records whose in-flight tasks did not finish before shutdown.",
    )
    shutdown_drain_timeout_s: float = Field(
        default=10.0,
        description="Seconds to wait for in-flight experience-recording tasks to complete during shutdown before spooling the rest.",
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
    
    p95_latency_budget_s: float = Field(
        default=90.0,
        description=(
            "A hard upper bound (in seconds) on the 95th percentile inference latency allowed during validation of a newly trained adapter."
        ),
    )
    context_trim_log_level: str = Field(
        default="warning",
        description=(
            "Controls how loudly the system logs when it trims conversation context to fit within the model’s maximum sequence length."
        ),
    )


settings = OrchestrationSettings()
