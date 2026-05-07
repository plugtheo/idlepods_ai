# IdlePods AI

A self-improving, self-hostable, multi-agent LLM assistant for everyday tasks — coding, debugging, research, planning, fact-checking, and general conversation. Built around a microservice architecture so it can run end-to-end on a single workstation (RTX 3090 or better) but is not locked to local inferencing: you can point it at any vLLM-compatible backend.

The system runs a **team of specialized agents** (planner, researcher, coder, debugger, reviewer, critic, consensus, summarizer) over a LangGraph state machine. Every successful run is recorded as an `ExperienceEvent` and, once enough diverse high-quality samples accumulate, the **training service** fine-tunes per-role rsLoRA adapters which are hot-swapped into the running vLLM server. Over time, your local model gets better at your kind of tasks. 

---

## Why this exists

- **Local-first, but not local-only.** Default deployment runs vLLM + adapters on one box. Swap models.yaml to point at a remote vLLM and the same code runs against shared infra.
- **Self-improvement loop.** No need to babysit fine-tuning. Records pile up; the scheduler runs cron-style; new adapters are validated by a smoke gate before being promoted.
- **Per-agent specialization via LoRA.** One base model (default `Qwen/Qwen3-8B`), many cheap rsLoRA adapters keyed by role. Per-role recipes live in `recipes.yaml`.
- **Scalable architecture.** Microservices behind a Gateway → Orchestration → Inference / Training data plane. gRPC on the hot path; HTTP elsewhere.

---

## Quick start

Requires Docker (with Compose), an NVIDIA GPU + recent drivers, and HuggingFace credentials for gated models.

```bash
cp .env.example .env
docker compose -f docker/compose.yml up
```

When all containers report healthy, send a prompt:

```bash
curl -X POST http://localhost:8080/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Write a Python function that debounces a callback"}'
```

Streaming endpoint (SSE):

```bash
curl -N -X POST http://localhost:8080/v1/chat/stream \
  -H "Accept: text/event-stream" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Plan a multi-region ingestion pipeline"}'
```

Bare-metal (no Docker)? Generate gRPC stubs first:

```bash
python scripts/generate_protos.py
```

---

## Services

| Service | Port | Responsibility |
| --- | --- | --- |
| [gateway](gateway/) | `8080` | External entry point. Auth (bearer key), regex prompt routing, proxy to orchestration. |
| [orchestration](orchestration/) | `8001` (host: `8014`) | LangGraph multi-agent pipeline. Owns context enrichment (RAG + repo scan) and experience recording in-process. |
| [inference](inference/) | `8010` HTTP, `50051` gRPC | Wraps the vLLM server(s). Loads/unloads/swaps LoRA adapters. Serves all agent roles. |
| [vllm-primary](docker/compose.yml) | `8000` | The actual vLLM model server (`Qwen/Qwen3-8B` by default). |
| [training](training/) | — | One-shot job (compose profile `training`). Reads experiences JSONL, builds SFT pairs, runs rsLoRA fine-tune via Unsloth, smoke-tests, and hot-swaps new adapter into vLLM. |
| [training-scheduler](training/scheduler/) | — | Lightweight cron. Every `SCHEDULER_INTERVAL_HOURS` (default `4`) checks diversity thresholds and launches the training profile. |
| [redis](docker/compose.yml) | `6379` | Session history, fingerprints, snippets, training cursors, compaction summaries. |
| [shared](shared/) | — | Pydantic contracts, models registry, recipe registry, gRPC stubs, query router. |

The data plane uses **two bind-mounted volumes** rooted at `../data/`:
- `vector_store/` — ChromaDB persistent client storage (few-shot RAG).
- `lora_checkpoints/` — adapter directories + `manifest.json`.

ChromaDB runs **embedded** by default (no separate container). To switch to a standalone server, uncomment the `chromadb` block + `ORCHESTRATION__CHROMADB_HOST` in `docker/compose.yml`.

---

## Agent roles

| Role | What it does | Default LoRA adapter |
| --- | --- | --- |
| `planner` | Break the task into ordered steps. Emits a JSON plan. | base (none) |
| `researcher` | Gather facts, prior art, best practices. May call `web_search`. | base |
| `coder` | Implement. Calls `read_file` / `write_file` / `list_files` / `run_command` via OpenAI tool calls. | base |
| `debugger` | Identify root cause, produce fixed code. | base |
| `reviewer` | JSON-schema review (issues, suggestions, score). | base |
| `critic` | JSON-schema verdict (blockers, improvements, score). | base |
| `review_critic` | Runs reviewer → critic in one chain slot (critic depends on reviewer's structured fields). | base |
| `consensus` | Final synthesizer. No adapter — always runs on base. | base |
| `summarizer` | Compaction tier-2 LLM summariser (tool-output + oldest-turn rollups). | `summarizer_lora` |
| `router` | Optional LLM-based prompt classifier (when `router_mode != regex`). | base |

The active chain for a request is decided by the query router (`shared/routing/query_router.py`) from `(intent, complexity)` — see the table in that file. Convergence threshold is `0.85` by default; max iterations is `5`.

---

## Self-improvement loop

```
[ user prompt ]
      │ HTTPS
      ▼
[ Gateway :8080 ] ── auth ── routes/chat.py classifies intent/complexity
      │ HTTP
      ▼
[ Orchestration :8001 ] ── builds context (few-shots from ChromaDB,
      │                                     repo snippets, hint text)
      │ gRPC
      ▼
[ Inference :50051 ] ── vLLM with Qwen3-8B + per-role rsLoRA adapter
      │
      ▼
agent loop — score each iteration, stop when ≥ 0.85 or max_iter
      │
      ▼
[ ExperienceEvent ] ─── fire-and-forget ────► JSONL shard (daily) + ChromaDB
                                                   │
            (cron)                                 │
              ▼                                    │
      [ training-scheduler ] ── reads ◄────────────┘
              │ if diversity thresholds met (≥50 records,
              │ ≥0.15 score spread, ≥60% unique fingerprints)
              ▼
      `docker compose run --rm training` (one-shot, profile-gated)
              │
              ├── load adapter recipe from recipes.yaml (per role)
              ├── build SFT pairs (curated + experience + synthetic)
              ├── pre-train backup of existing adapter
              ├── Unsloth + TRL SFT (rsLoRA by default)
              ├── stage as `<adapter>__staging`, run smoke gate
              └── on pass: POST /adapters/swap → vLLM hot-swaps
```

Scoring (`orchestration/app/utils/scoring.py`) prefers explicit `SCORE:` from reviewer/critic, blends with code-presence heuristics for coder/debugger, and penalizes detected pipeline-metadata leakage.

---

## Configuration map

All knobs are pydantic `BaseSettings` per service. Env-var prefixes: `GATEWAY__`, `ORCHESTRATION__`, `INFERENCE__`, `TRAINING__`. The full lists live in:

- `gateway/app/config/settings.py`
- `orchestration/app/config/settings.py` — by far the largest; owns context-budget knobs, role maps, router config, ChromaDB, Redis, compaction.
- `inference/app/config/settings.py` — HTTP timeouts, gRPC sampling defaults, adapter rollback thresholds.
- `training/app/config/settings.py` — diversity thresholds, scheduler interval, lock path, training timeout.

Backend identity (URL, model id, max ctx, parsers) lives in `models.yaml`. PEFT recipes live in `recipes.yaml`. `ORCHESTRATION__MODEL_CONTEXT_LEN` **must** match `--max-model-len` in `docker/compose.yml`. `MODELS_YAML_PATH` / `RECIPES_YAML_PATH` env vars override the lookup paths.

---

## Hardware

Validated on a single RTX 3090 (24GB). The defaults assume that:

- vLLM: `--gpu-memory-utilization 0.9385`, `--max-model-len 12288`, `--max-num-seqs 1`, `--kv-cache-dtype fp8`.
- Training: Unsloth 4-bit base, rsLoRA (`r=32, alpha=64`), `bf16=True`, `per_device_train_batch_size=1`, `gradient_accumulation_steps=8`.

Smaller GPU or different model? Edit `models.yaml` (and matching `--max-model-len`), `recipes.yaml`, and the vLLM `command:` block in `docker/compose.yml` together.

---

## Limitations

- **GPU required.** No CPU fallback for inference or training.
- **Training is long.** Per-role adapter retrains take 5–17 hours on a 3090 with the default recipe; see `recipes.yaml` to trade quality for speed (lower `r`, fewer epochs, packing tweaks).
- **Adapter manifest is the source of truth.** If you delete `data/lora_checkpoints/manifest.json`, vLLM falls back to base; training will rebuild from scratch on the next run.
- **Single-tenant.** No per-user quotas, no multi-tenant routing.
- **Web search uses DuckDuckGo** (`duckduckgo-search`) — install on demand, may rate-limit.

---

## Development

- Tests: `pytest` from the repo root. `conftest.py` handles namespace packaging.
- Lint: `ruff` (allowed via `run_command` tool).
- Type-check: `mypy` (allowed via `run_command` tool).
- Architecture overview: see [ARCHITECTURE.md](ARCHITECTURE.md).
- Service-level READMEs in each `<service>/README.md`.
