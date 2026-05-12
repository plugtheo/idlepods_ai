# IdlePods AI

A self-hostable, self-improving AI assistant that runs on the dusty RTX 3090 already sitting under your desk — so you can stop paying $200/month for the privilege of someone else owning your prompts.

It's a multi-agent setup (planner, researcher, coder, debugger, reviewer, critic, consensus, summarizer) wired together with LangGraph. Every successful run gets logged, and once enough good runs pile up, the system quietly fine-tunes per-role LoRA adapters and hot-swaps them into the running vLLM server. The longer you use it, the more it bends toward your kind of work.

You can run the whole thing on one machine, or point it at a remote vLLM backend if you ever outgrow the desktop. Either way, the data stays yours.

Currently still a work in progress but is in a stable state to train new adapters and collect experiences for self sufficiency.

---

## Why bother

- **Your hardware, your data, your weights.** Default deployment is vLLM + adapters on a single box. Flip a setting in `models.yaml` and you're hitting shared infra instead — same code, same agents.
- **It teaches itself while you sleep.** No babysitting fine-tunes. Records accumulate, a cron job notices, training kicks off, a smoke test gates promotion. If the new adapter is worse, it doesn't ship.
- **One base model, many cheap specialists.** Default is `Qwen/Qwen3-8B`. Each agent gets its own rsLoRA adapter via recipes in `recipes.yaml` — much cheaper than running eight separate models.
- **Microservices, but reasonable.** Gateway → Orchestration → Inference/Training. gRPC where latency matters, HTTP everywhere else. Nothing fancier than it needs to be.
- **Why Microservices?** So that one day if you decide to grab yourself a DGX spark or a beefy stack with RTX 6000, you can host your own inference server for better scalability/portability or whatever reasons you may want.

---

## Quick start

You'll need Docker (with Compose), an NVIDIA GPU with recent drivers, and HuggingFace creds if you want any gated models.

```bash
cp .env.example .env
docker compose -f docker/compose.yml up
```

Wait for everything to go healthy, then throw it a prompt:

```bash
curl -X POST http://localhost:8080/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Write a Python function that debounces a callback"}'
```

Streaming (SSE) if you'd rather watch it think:

```bash
curl -N -X POST http://localhost:8080/v1/chat/stream \
  -H "Accept: text/event-stream" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Plan a multi-region ingestion pipeline"}'
```

Running bare-metal without Docker? Generate the gRPC stubs first:

```bash
python scripts/generate_protos.py
```

---

## What's running where

| Service | Port | What it does |
| --- | --- | --- |
| [gateway](gateway/) | `8080` | Front door. Bearer-token auth, regex prompt routing, proxies to orchestration. |
| [orchestration](orchestration/) | `8001` (host: `8014`) | The LangGraph pipeline. Builds context (RAG + repo scan) and records experiences inline. |
| [inference](inference/) | `8010` HTTP, `50051` gRPC | Thin wrapper around vLLM. Loads, unloads, and swaps LoRA adapters per role. |
| [vllm-primary](docker/compose.yml) | `8000` | The actual model server — `Qwen/Qwen3-8B` by default. |
| [training](training/) | — | One-shot job (compose profile `training`). Reads experience JSONL, builds SFT pairs, fine-tunes with Unsloth, smoke-tests, hot-swaps. |
| [training-scheduler](training/scheduler/) | — | Tiny cron. Wakes up every `SCHEDULER_INTERVAL_HOURS` (default `4`), checks if there's enough diverse data, kicks off training if so. |
| [redis](docker/compose.yml) | `6379` | Sessions, fingerprints, snippets, training cursors, compaction summaries. The usual. |
| [shared](shared/) | — | Pydantic contracts, model + recipe registries, gRPC stubs, query router. |

Two bind-mounted volumes under `../data/` hold the stateful stuff:
- `vector_store/` — ChromaDB persistent client (used for few-shot RAG).
- `lora_checkpoints/` — adapter directories plus the `manifest.json` that vLLM trusts.

ChromaDB runs **embedded** by default — no extra container, no extra service to babysit. If you'd rather run it standalone, uncomment the `chromadb` block and `ORCHESTRATION__CHROMADB_HOST` in `docker/compose.yml`.

---

## The agents

| Role | Job | Default adapter |
| --- | --- | --- |
| `planner` | Breaks the task into ordered steps, emits JSON. | base (none) |
| `researcher` | Hunts down facts, prior art, best practices. Has `web_search`. | base |
| `coder` | Actually writes code. Has `read_file`, `write_file`, `list_files`, `run_command` via OpenAI tool calls. | base |
| `debugger` | Finds the root cause and patches it. | base |
| `reviewer` | Structured review — issues, suggestions, score. | base |
| `critic` | Structured verdict — blockers, improvements, score. | base |
| `review_critic` | Reviewer → critic in one chain slot (critic reads reviewer's fields). | base |
| `consensus` | Final synthesis. Always runs on base — no adapter, on purpose. | base |
| `summarizer` | Compaction tier-2 — rolls up tool output and oldest turns. | `summarizer_lora` |
| `router` | Optional LLM classifier when `router_mode != regex`. | base |

Which chain you get depends on `(intent, complexity)`, decided by `shared/routing/query_router.py`. The loop stops when iteration score hits `0.85` or you blow past `max_iter` (default `5`).

---

## The self-improvement loop

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

Scoring (`orchestration/app/utils/scoring.py`) prefers an explicit `SCORE:` from reviewer/critic, mixes in code-presence heuristics for coder/debugger, and docks points if it catches pipeline metadata leaking into the output.

---

## Configuration

Every service uses pydantic `BaseSettings`, with env-var prefixes `GATEWAY__`, `ORCHESTRATION__`, `INFERENCE__`, `TRAINING__`. Full knobs live in:

- `gateway/app/config/settings.py`
- `orchestration/app/config/settings.py` — the big one. Owns context budgets, role maps, router config, ChromaDB, Redis, compaction.
- `inference/app/config/settings.py` — HTTP timeouts, gRPC sampling defaults, adapter rollback thresholds.
- `training/app/config/settings.py` — diversity thresholds, scheduler interval, lock path, training timeout.

Backend identity (URL, model id, max ctx, parsers) lives in `models.yaml`. PEFT recipes live in `recipes.yaml`. Heads up: `ORCHESTRATION__MODEL_CONTEXT_LEN` **must** match `--max-model-len` in `docker/compose.yml`, or you'll get cryptic context-length errors at 2am. Override the YAML paths with `MODELS_YAML_PATH` / `RECIPES_YAML_PATH` if you need to.

---

## Hardware

Validated on a single RTX 3090 (24GB), which is the assumption baked into all the defaults:

- vLLM: `--gpu-memory-utilization 0.9385`, `--max-model-len 12288`, `--max-num-seqs 1`, `--kv-cache-dtype fp8`.
- Training: Unsloth 4-bit base, rsLoRA (`r=32, alpha=64`), `bf16=True`, `per_device_train_batch_size=1`, `gradient_accumulation_steps=8`.

Got a different GPU or want to swap the base model? Edit `models.yaml` (and matching `--max-model-len`), `recipes.yaml`, and the vLLM `command:` block in `docker/compose.yml` together — they're load-bearing on each other.

---

## Known limitations (the honest list)

- **GPU required.** No CPU fallback for inference or training. This is not the project for your 2015 MacBook.
- **Training is slow.** Per-role adapter retrains run 5–17 hours on a 3090 with the default recipe. `recipes.yaml` is where you trade quality for speed — lower `r`, fewer epochs, packing tweaks.
- **The manifest is the source of truth.** Nuke `data/lora_checkpoints/manifest.json` and vLLM happily falls back to base. Training will rebuild from zero on the next run.
- **Single-tenant.** No per-user quotas, no multi-tenant routing. Built for you, not your startup.
- **Web search is DuckDuckGo.** `duckduckgo-search`, installed on demand, occasionally rate-limited. It is what it is.

---

## Development

- Tests: `pytest` from the repo root. `conftest.py` handles namespace packaging.
- Lint: `ruff`.
- Type-check: `mypy`.
- Architecture deep-dive: [ARCHITECTURE.md](ARCHITECTURE.md).
- Each service has its own README in `<service>/README.md` if you want to drill in.