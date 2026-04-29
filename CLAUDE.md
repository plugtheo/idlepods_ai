# CLAUDE.md

## Services
- `gateway` :8080 — auth, routing → orchestration
- `orchestration` :8001 — LangGraph pipeline, context (in-process), experience (in-process)
- `inference` :8010/:50051 — vLLM backend + gRPC
- `training` :8013 — Unsloth LoRA fine-tune
- `shared/` — Pydantic contracts, gRPC stubs, routing

## Core Logic
| Concern | File |
|---|---|
| Pipeline | `orchestration/app/graph/pipeline.py` |
| Agent nodes | `orchestration/app/graph/nodes.py` |
| State schema | `orchestration/app/graph/state.py` |
| Context assembly | `orchestration/app/context/builder.py` |
| Backend dispatch | `inference/app/backends/factory.py` |
| Agent prompts | `shared/contracts/agent_prompts.py` |
| Cross-service models | `shared/contracts/` |
| Convergence scoring | `orchestration/app/utils/scoring.py` |
| ChromaDB session | `orchestration/app/db/chroma.py` |

## Invariants
- All ChromaDB access → `orchestration/app/db/chroma.py`
- Cross-service Pydantic models → `shared/contracts/` only; never redefined per-service
- Agent prompts → `shared/contracts/agent_prompts.py` only
- Token-budget constants → `orchestration/app/config/settings.py`; `ORCHESTRATION__MODEL_CONTEXT_LEN` must match `--max-model-len` in `docker/compose.yml`
- Health endpoints return `{"status": "ok", "service": "<name>"}`

## Conventions
- `snake_case` functions/vars; `PascalCase` classes
- All I/O wrapped in `try/except` with `log.error`
- Config via `<service>/app/config/settings.py` — never `os.environ` inline

## Workflows
- **New endpoint:** route in `<service>/app/routes/`, register in `main.py`, test in `<service>/tests/`
- **New agent role:** settings.py entry + prompt in `agent_prompts.py` + adapter name
- **New contract:** Pydantic model in `shared/contracts/<domain>.py`, export from `__init__.py`
- **Tests:** run from project root with `pytest`; `conftest.py` handles namespace packages

## Glossary
- **Agent role** — specialist (coder, debugger, planner…) selecting LoRA adapter + system prompt
- **Experience** — completed pipeline run → `experiences.jsonl` + ChromaDB
- **Context** — enriched prompt prefix (repo snippets, few-shot, hints) built by `context/builder.py`
- **AgentState** — TypedDict carrying full pipeline state through LangGraph nodes
