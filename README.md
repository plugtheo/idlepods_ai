# IdlePods AI

A self-improving multi-agent AI system built on a microservices architecture. Users submit prompts through a single API endpoint; a team of specialised AI agents collaborates to produce a high-quality answer; and every successful run feeds back into training data that gradually improves the agents over time.

---

## How the system works end to end

```
User
 │
 ▼
Gateway (port 8080)
 │  Checks auth, classifies intent + complexity, selects agent chain
 │
 ▼
Orchestration (port 8001)
 │  Fetches context → runs LangGraph agent pipeline → scores convergence
 │
 ├──► Context Service (port 8011)          [enriches the pipeline with past examples + repo code]
 │
 ├──► Inference Service (port 8010/50051)   [calls the language models for each agent]
 │         │
 │         ├──► vLLM DeepSeek (port 8000)  [local GPU — coder, debugger, reviewer]
 │         └──► vLLM Mistral  (port 8001)  [local GPU — planner, researcher, critic]
 │             (or a cloud API via LiteLLM in API mode)
 │
 └──► Experience Service (port 8012)       [records the result for memory + future training]
           │
           ▼
       Training Service (port 8013)        [fine-tunes models when enough data accumulates]
```

---

## Services

| Service | Port | README |
|---------|------|--------|
| [Gateway](gateway/) | 8080 | [gateway/README.md](gateway/README.md) |
| [Orchestration](orchestration/) | 8001 (8014 on host) | [orchestration/README.md](orchestration/README.md) |
| [Inference](inference/) | 8010 (HTTP), 50051 (gRPC) | [inference/README.md](inference/README.md) |
| [Context](context/) | 8011 | [context/README.md](context/README.md) |
| [Experience](experience/) | 8012 | [experience/README.md](experience/README.md) |
| [Training](training/) | 8013 | [training/README.md](training/README.md) |
| [Shared contracts](shared/) | — | [shared/README.md](shared/README.md) |

---

## Quick start

```bash
# API mode (no GPU required — uses a cloud AI provider like Anthropic)
cp .env.example .env        # fill in ANTHROPIC_API_KEY (or equivalent)
docker compose -f docker/compose.yml up

# Local GPU mode (requires NVIDIA GPU + CUDA drivers)
docker compose -f docker/compose.yml --profile local up
```

The system is ready when all health checks pass (Docker Compose reports all services healthy).

Test with:
```bash
curl -X POST http://localhost:8080/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Write a Python function that debounces a callback"}'
```

---

## Troubleshooting

**All services start but requests return 503 / connection refused**
Check that all containers reached the `healthy` state: `docker compose -f docker/compose.yml ps`. The Orchestration Service waits for Inference, Context, and Experience to be healthy before accepting traffic. If one is stuck, inspect its logs: `docker compose logs <service-name>`.

**`ModuleNotFoundError: No module named 'shared.grpc_stubs.inference_pb2'`**
The gRPC stubs are generated at Docker build time. If you are running tests or the Orchestration Service outside Docker, generate them first:
```bash
python scripts/generate_protos.py
```

**Training Service skipped — experiences accumulate but no LoRA training starts**
In API mode (`INFERENCE__MODE=api`) there are no local model weights, so the Training Service refuses all triggers. Set `EXPERIENCE__TRAINING_URL=` (empty) in `.env` to silence the spurious trigger attempts, or switch to local GPU mode.

**vLLM crashes with CUDA out-of-memory**
DeepSeek (6.7B) and Mistral (7B) each need a dedicated GPU. Running both on a single GPU will OOM. Either use two GPUs (set `CUDA_VISIBLE_DEVICES` per service in `.env`) or use API mode.

**Generated code has no spaces — words run together**
This is the Metaspace/ByteLevel tokenizer mismatch described in the [Adapter training critical notes](#adapter-training-and-inference--critical-notes) section below. The affected adapter was trained with the wrong pre-tokenizer and must be retrained. Existing adapters in `data/lora_checkpoints/` already have the correct `tokenizer.json`.

**Context Service returns no few-shot examples despite past runs**
The Experience and Context Services share the same ChromaDB collection. Verify both use the same `CHROMA_HOST`, `CHROMA_PORT`, and `CHROMA_COLLECTION` values. In self-hosted mode the default collection name is `experiences` for both services.

---

## Self-improvement loop

The system improves itself over time through a feedback loop:

1. **Run** — a user prompt is processed by the agent pipeline
2. **Record** — the result is saved to the Experience Service (JSONL + ChromaDB)
3. **Retrieve** — on the next similar request, the Context Service finds this past example and injects it as a reference
4. **Train** — once ≥50 diverse, high-quality experiences have accumulated, the Training Service fine-tunes LoRA adapters on the local models
5. **Deploy** — vLLM picks up the new adapters automatically; future runs use the improved model

---

## Known issues and technical debt

The following issues were identified during a thorough code review. They are documented here to make them easy to find and prioritise.

> **Status key:** ✅ Fixed | ⚠️ Won't fix / by design | 🔲 Open

### Bugs

| Severity | Status | Location | Issue |
|----------|--------|----------|-------|
| High | ✅ | `orchestration/app/routes/run.py` | The SSE streaming endpoint never emits a final `done` event — the stream just closes. Clients have no programmatic way to receive the final output over SSE. |
| High | ✅ | `gateway/app/clients/orchestration.py` | The SSE proxy adds one `\n` after each line instead of the required two (`\n\n`). Browser SSE clients may fail to parse events correctly. |
| High | ✅ | `experience/app/storage/vector_store.py` | The sentence-transformer embedding model was re-loaded from disk on every single write. Now a module-level singleton; ChromaDB init offloaded to thread pool. |
| Medium | ✅ | `experience/app/storage/vector_store.py` + `context/app/retrieval/few_shot.py` | The Experience Service stored the final solution under `"final_output"` but the Context Service read it as `"solution"`. Metadata keys are now aligned — RAG few-shot examples contain correct solutions. |
| Medium | ✅ | `shared/contracts/orchestration.py` + `docker/compose.yml` | `OrchestrationRequest.max_iterations` default was `5` in the Pydantic contract, overriding the `ORCHESTRATION__DEFAULT_MAX_ITERATIONS=2` Docker Compose setting. Default now reads from env via `settings`. |

### Dead / broken code

| Status | Location | Issue |
|--------|----------|-------|
| ✅ | `scripts/run_agent_experience_pipeline.py` | Deleted — imported from `app.*` modules that no longer exist (old monolithic architecture). |
| ✅ | `scripts/verify_system_prerequisites.py` | Deleted — same broken imports. |
| ✅ | `scripts/training_analysis_report.py` | Deleted — hardcoded a developer-machine-specific filename. |
| ✅ | `scripts/verify_synthetic_pipeline.py` | Deleted — same broken imports. |
| ✅ | `data/sandbox/` | Deleted — agent-generated sandbox artifacts that ended up in the data directory. |
| ✅ | `context/app/config/settings.py` → `experience_jsonl` | Config field defined but never read anywhere in the Context Service. Dead config. |
| ✅ | `orchestration/app/graph/pipeline.py` → `check_convergence` node | Was a no-op lambda. Replaced with named `_noop_convergence_anchor` function for clarity. |
| ✅ | `orchestration/app/routes/run.py` → `AgentStep.score` | Was always hardcoded to `0.0`. Now populated via `score_per_entry()` using real per-agent quality scores. |
| ⚠️ | `experience/app/clients/training.py` → `new_experience_count` | Always passed as `1`. Intentional simplification — the field is kept for future use. |

### Inconsistencies

| Status | Location | Issue |
|--------|----------|-------|
| ✅ | `gateway/app/routing/query_router.py` vs `orchestration/app/routing/query_router.py` | Both are now thin re-exports of `shared/routing/query_router.py` — a single canonical implementation guarantees identical routing regardless of entry point. |
| ⚠️ | `orchestration/app/agents/prompts.py` → `ROLE_ADAPTER["consensus"]` | `consensus` adapter is `None` (uses bare base model) by design — there is no consensus LoRA. |
| ✅ | `experience/app/routes/record.py` → `_infer_capability` | Now returns capability labels (`"coding"`, `"debugging"`) instead of agent role names. Training Service model selection is no longer fragile substring matching. |
| ✅ | `training/app/config/settings.py` + `experience/app/config/settings.py` | `min_batch_size = 50` was defined independently in both services. Comments now explicitly cross-reference each other; value is kept in sync. |
| ✅ | `context` vs `experience` embedding model name | Context used `"sentence-transformers/all-MiniLM-L6-v2"`, experience used `"all-MiniLM-L6-v2"`. Both now use the fully-qualified HuggingFace model ID. |

### Performance issues

| Status | Location | Issue |
|--------|----------|-------|
| ✅ | `context/app/retrieval/repo.py` | Declared `async` but file-walk was synchronous. Now uses `asyncio.to_thread()`. |
| ✅ | `context/app/utils/embedder.py` | `model.encode()` was synchronous and CPU-intensive on the event loop. Now offloaded to a thread pool. |
| ✅ | `experience/app/storage/vector_store.py` | Model loaded on every call. Now a singleton with async thread-pool offload. |
| ✅ | `orchestration/app/clients/context.py` | Created a new `httpx.AsyncClient` (full TCP handshake) on every pipeline request. Now a module-level persistent connection pool. |

### Deprecation

| Status | Location | Issue |
|--------|----------|-------|
| ✅ | All 5 services (`gateway`, `orchestration`, `experience`, `training`, `inference`) | Used deprecated `@app.on_event("startup"/"shutdown")`. Migrated to `@asynccontextmanager lifespan` as recommended by FastAPI. |
| ✅ | `gateway/app/middleware/auth.py` | Used `token != settings.api_key` — vulnerable to timing attacks. Replaced with `secrets.compare_digest`. |
| ✅ | `experience/app/config/settings.py` | Legacy `vector_store_path` field (ChromaDB replaced flat-file store). Field removed. |

---

## Adapter training and inference — critical notes

These are hard-won lessons from debugging persistent whitespace/token corruption in the `coding_lora` adapter. They apply to every LoRA adapter trained on DeepSeek or any model whose `tokenizer.json` uses a Metaspace pre-tokenizer while the BPE vocabulary uses the GPT-2 byte-level `Ġ` convention.

---

### 1. The Metaspace / ByteLevel pre-tokenizer mismatch

**Root cause of all silent space stripping.**

DeepSeek's `tokenizer.json` declares a `Metaspace(replacement="▁")` pre-tokenizer but its BPE merge rules and vocabulary use the GPT-2 byte-level convention — spaces encoded as `Ġ` (U+0120). These are incompatible:

- With `Metaspace`: `"def add(a, b)"` → `['def', 'add']` — spaces silently dropped, IDs `1551` and `1761`
- With `ByteLevel`: `"def add(a, b)"` → `['def', 'Ġadd']` — correct, IDs `1551` and `957`

Adapters trained with `Metaspace` active learn to generate spaceless token sequences. No amount of prompt engineering or post-processing fixes an adapter trained this way — it must be retrained.

**Fix — apply immediately after Unsloth loads the tokenizer, before any training:**

```python
from tokenizers.pre_tokenizers import ByteLevel
tokenizer.backend_tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
```

This is already applied in `training/training/lora_trainer.py`. Any future change to how the tokenizer is loaded must preserve this line.

---

### 2. `_fix_bpe_artifacts()` is required at both validation and inference time

A correctly trained adapter outputs raw GPT-2 byte-level tokens: `Ġ` for space (U+0120), `Ċ` for newline (U+010A), etc. These must be converted to readable text before any downstream use.

- **At inference**: `inference/app/backends/local_vllm.py` → `_fix_bpe_artifacts()`. Already present. Do not remove.
- **At validation**: `training/training/validate_adapter.py` must call `_fix_bpe_artifacts()` on raw adapter output _before_ running any checks. Without this, checks like "no BPE artifacts" are a false negative — they pass when the adapter is broken (spaceless ASCII) and fail when it is working correctly (proper Ġ tokens).

If you add new validation checks, always apply `_fix_bpe_artifacts()` first.

---

### 3. Save `tokenizer.json` from the backend, not via Unsloth's `save_pretrained`

Calling `tokenizer.save_pretrained()` or passing `tokenizer=tokenizer` to `model.save_pretrained()` through Unsloth writes a `tokenizer_config.json` with mojibake (UTF-8 special token strings double-encoded as Latin-1 bytes), causing vLLM to reject the file and fall back to the base model tokenizer.

However, **not** saving any tokenizer leaves vLLM loading the base model's original `tokenizer.json` which has `Metaspace(replacement="▁")` as its pre-tokenizer. The adapter was trained with `ByteLevel(add_prefix_space=False)`. This mismatches the input prompt tokenization at inference — different token IDs → different RoPE positions for the `[RESPONSE]` boundary the adapter learned.

The fix is to write **only** `tokenizer.json` from the in-memory backend tokenizer (which already has ByteLevel applied), bypassing Unsloth's save path entirely:

```python
tok_json_str = tokenizer.backend_tokenizer.to_str()
(adapter_dir / "tokenizer.json").write_text(tok_json_str, encoding="utf-8")
```

This is already done in `lora_trainer.py` immediately after `model.save_pretrained()`. The adapter directory should contain:

```
adapter_config.json
adapter_model.safetensors
metadata.json
tokenizer.json          ← corrected (ByteLevel pre-tokenizer, no mojibake)
```

Do NOT add `tokenizer_config.json` — that is the file that causes mojibake via Unsloth. Only `tokenizer.json` (the tokenizers-library native JSON with vocab + merges + pre_tokenizer) is needed and safe.

---

### 4. `shared/contracts/agent_prompts.py` must exist inside the training container

The training image predates `agent_prompts.py`. Until the image is rebuilt, the file must be copied into the running container after every restart:

```powershell
docker cp shared/contracts/agent_prompts.py docker-training-1:/app/shared/contracts/agent_prompts.py
```

**`docker compose run` spawns a fresh container** where this file does not exist. Always run validation and training directly in the named container (`docker exec docker-training-1 ...`) until the image is rebuilt with the file baked in.

---

### 5. All DeepSeek adapters trained before April 2026 are affected

`debugging_lora` and `review_lora` (on DeepSeek) were trained with the Metaspace pre-tokenizer active. They generate spaceless output. They must be retrained with the ByteLevel fix applied.

Mistral adapters (`planning_lora`, `research_lora`, `criticism_lora`) may or may not be affected depending on Mistral's tokenizer configuration — verify with a direct BPE token check before assuming they are clean.

To check whether an adapter is affected, inspect raw vLLM output before `_fix_bpe_artifacts()` is applied:

```python
# Healthy adapter — output contains Ġ/Ċ tokens, e.g.:
# "defĠfibonacci(n):ĊĠĠĠĠreturnĠn"
# Broken adapter — output is pure ASCII with no spaces, e.g.:
# "deffibonacci(n):returnn"
```

---

### 6. Proof-before-retrain checklist

Before committing to a retrain to fix an inference problem, run the following proof steps:

1. **Check vLLM is alive**: `curl http://localhost:8000/health` — OOM-kills (exit code 137) silently take vLLM offline and make adapters appear broken when the model is simply unreachable.
2. **Check tokenizer pre-tokenizer**: `tok.backend_tokenizer.pre_tokenizer` should be `ByteLevel`, not `Metaspace`.
3. **Test tokenization round-trip**: `tokenizer.decode(tokenizer.encode("def add(a, b):"))` should return `"def add(a, b):"` — if spaces are missing, the pre-tokenizer is wrong.
4. **Inspect raw adapter output** (before `_fix_bpe_artifacts`): must contain `Ġ`/`Ċ` characters, not spaceless ASCII.
5. **Run `validate_adapter.py`** in the named training container (`docker exec docker-training-1 ...`), not via `docker compose run`.

---

## Architecture decisions and trade-offs

**Why separate vLLM servers for DeepSeek and Mistral?**  
The two models were selected for complementary strengths: DeepSeek for code generation, Mistral for reasoning and language tasks. Running them on separate servers allows GPU memory to be allocated independently. The trade-off is that agent-to-agent communication necessarily goes through text (tokens) — each agent re-reads and re-tokenizes the full conversation history. For long chains with many iterations, this is computationally expensive. An architecture using latent-space communication between agents would be dramatically more efficient but would require all agents to share the same model, losing the specialisation benefit.

**Why gRPC between Orchestration and Inference but HTTP everywhere else?**  
The Orchestration → Inference path is the highest-frequency call in the system (one per agent node per iteration). gRPC's binary encoding reduces per-call overhead. All other inter-service calls are lower frequency, and HTTP is simpler for debugging and development.

**Why fire-and-forget for experience and training notifications?**  
Both the Experience Service write and the Training Service notification are detached from the user-facing response path. The user does not need to wait for memory to be recorded or for training to be evaluated. Decoupling these allows the pipeline response time to remain predictable regardless of downstream load.


## Ideas

I would like to consider introducing another agent that fact check a prompt (whether simple question, statement, code, article, .. pretty much anything in any topic) where it searches against various online sources that are plausible that can be used in any agent pipeline to generate altered responses by saying things like resources check against whatever, and chances of fact/information being true. So that it can be up to end users to determine the fact and not return any responses that poses as true fact since any inference can still lead to incorrect information which is of the biggest challenges in AI industry today.





## Refactor Plan: IdlePods AI → Local-First, Self-Contained

Guiding principle
The project only makes sense when the self-training loop is running. Anything that assumes cloud inference, cloud ChromaDB, or a path where training is optional is accidental complexity that should be removed. The target deployment is one GPU machine (min RTX 3090 / 24 GB VRAM, min 32 GB system RAM), optionally extended by adding remote inference nodes.

Phase 1 — Remove Cloud Inference (API mode)
Scope: inference/ service only. Zero behavior change to any other service.

Delete entirely:

inference/app/backends/api.py — the LiteLLM/Anthropic backend
inference/tests/test_api_backend.py — its tests
Simplify:

inference/app/backends/factory.py — remove the elif mode == "api" branch; the factory only ever returns LocalVLLMBackend. Remove the mode check entirely; the factory just constructs and returns LocalVLLMBackend.
inference/app/config/settings.py — delete mode, api_provider, api_model, api_key, role_model_overrides fields. Keep only deepseek_url, mistral_url, deepseek_model_id, mistral_model_id, gRPC defaults, timeout, ports.
inference/requirements.txt — remove litellm>=1.34.0.
docker/compose.yml — remove INFERENCE__MODE, INFERENCE__API_PROVIDER, INFERENCE__API_MODEL, INFERENCE__API_KEY env vars from the inference: block. Remove profiles: ["local"] gate from training: and vllm-* containers — they are always required.
shared/contracts/training.py — remove any mode guard in the training trigger contract.
experience/ settings — remove EXPERIENCE__TRAINING_URL being conditional; training URL is always required.
Risk: LOW. Pure deletion. The local path is already the default (INFERENCE__MODE=local). Removes ~250 lines and one heavyweight dependency.

Phase 2 — Remove Cloud ChromaDB Mode
Scope: context/, experience/, docker/compose.yml.

The compose file currently has two ChromaDB backends: self-hosted container (profile self-hosted) and cloud (Chroma API key). For a local tool, the self-hosted container is the only mode.

Changes:

docker/compose.yml — remove CONTEXT__CHROMA_API_KEY, CONTEXT__CHROMA_TENANT, CONTEXT__CHROMA_DATABASE, EXPERIENCE__CHROMA_API_KEY, etc. Make chromadb: container always-on (no profiles: ["self-hosted"]). Always-on alongside vllm-* and training:.
Context/Experience settings — strip cloud ChromaDB config fields; simplify to just chroma_host, chroma_port.
Remove the required: false conditional depends_on the ChromaDB service.
Risk: LOW. Forces self-hosted ChromaDB; removes dead config branches.

Phase 3 — GPU Coordinator (Critical Missing Piece)
This is the most important correctness fix. Currently nothing prevents training and inference from running simultaneously on the same GPU, which will cause an OOM crash.

The problem
Running all three simultaneously is guaranteed OOM.

Solution: Inference service exposes a GPU lease API
Add two admin endpoints to the inference service:

Training service flow becomes:

During paused state, orchestration's inference calls return 503 Service Unavailable. The Gateway or Orchestration service should surface this to the user as "model is training, try again in ~N minutes" rather than a generic error.

Add a GET /v1/admin/status endpoint on the Gateway that aggregates inference + training status so a client can show a meaningful message.

Files added/changed:

New: inference/app/routes/admin.py — pause/resume/status routes
New: inference/app/backends/lifecycle.py — subprocess management for vLLM
Modified: training/app/ — add pause/resume calls around training subprocess
Modified: docker/compose.yml — expose INFERENCE__ADMIN_SECRET for authenticated pause/resume calls
Risk: MEDIUM. The lifecycle management of vLLM subprocesses is new surface area. Needs careful testing: subprocess crash during training should still trigger resume so inference recovers.

Phase 4 — Inference Node Registry (Decentralization Ready)
Currently the inference service has two hardcoded endpoints: deepseek_url and mistral_url. Scaling to multiple machines or adding a third model requires code changes.

Replace hardcoded URLs with a NodeRegistry
The LocalVLLMBackend selects a node by model_family, health-checks available nodes, and routes round-robin among equal-priority ones. If a node is paused (Phase 3), it is excluded from routing.

Files changed:

New: inference/app/registry.py — NodeRegistry class with health-aware routing
Modified: inference/app/backends/local_vllm.py — use registry instead of direct URL config
Modified: inference/app/config/settings.py — replace deepseek_url/mistral_url with nodes: list[NodeConfig]
Backward compat: Default nodes value mirrors the current two hardcoded URLs, so existing deployments need no config changes.

Risk: MEDIUM. Changes routing logic inside the inference service. The existing test_local_vllm.py suite will need updating to mock the registry.

Phase 5 — Single-Model Mode for Single-GPU Deployments
Running two vLLM servers simultaneously on a 3090 uses 85% of VRAM and leaves almost no room for KV cache growth under load. The two models exist because different agent roles were mapped to different model families — but with LoRA adapters, one base model can serve all roles.

Recommendation: Make the two-model setup optional. Introduce a INFERENCE__MODEL_PROFILE setting:

Profile	VRAM usage	Use case
dual (current)	~20 GB	Two GPUs or future NVLink setup
single (new default for 3090)	~10 GB	Single 3090, leaves headroom for training
In single mode, the NodeRegistry only starts/uses vllm-deepseek. The model_family field in GenerateRequest is honored for LoRA adapter selection but routing ignores it (both families land on the same node). DeepSeek-Coder-6.7B covers all roles adequately with role-specific LoRA adapters.

In docker/compose.yml:

Risk: LOW for single-mode. The dual mode stays as-is. This is purely additive.

Phase 6 — Module Boundary Cleanup
6a. Move agent_prompts out of shared/
shared/contracts/agent_prompts.py is only used by the orchestration service. shared/ should contain only inter-service DTOs (Pydantic models) and generated gRPC stubs. System prompts are orchestration-internal logic.

Move to: orchestration/app/agents/prompts.py

Risk: LOW. No other service imports agent prompts.

6b. Enforce "Gateway always classifies, Orchestration always trusts"
Currently orchestration silently re-routes if intent/complexity are absent. This masks bugs (gateway failing to classify). Add an explicit check: if intent or complexity is None in OrchestrationRequest, log a warning with the session ID and re-route. In a follow-up, promote this to a validation error so the bug surface is visible.

6c. Pin EXPERIENCE__TRAINING_URL as required
Remove the "leave empty in API mode" escape hatch from the compose comment and experience settings. Training URL is always set.

6d. Regeneration script for gRPC stubs
shared/grpc_stubs/ contains generated protobuf code that must be kept in sync with shared/proto/inference.proto. Add scripts/gen_grpc.sh so contributors know how to regenerate rather than editing generated files by hand.

Risk: LOW across all 6b–6d.

Phase 7 — JSONL Compaction
The training service reads the entire /data/experiences.jsonl on every training trigger evaluation. With no cap, this file grows forever and makes each evaluation slower.

Add a configurable ring-buffer cap to the experience service:

When append_experience would push total count above MAX_RECORDS, evict the lowest-scoring record below MIN_QUALITY_KEEP. High-quality records accumulate; mediocre ones rotate out. This also improves training diversity over time (bias toward quality without losing recency).

Risk: LOW. Experience service is stateful, but the change only affects the append path, not the ChromaDB path.

Hardware Specs for Single-GPU Deployment
Minimum confirmed to run all services:

Resource	Minimum	Recommended
GPU	RTX 3090 (24 GB)	RTX 4090 (24 GB) or dual 3090
System RAM	32 GB	64 GB
Storage	100 GB NVMe	500 GB NVMe
CPU	8-core	12-core
VRAM breakdown (single-model profile after Phase 5):

Component	VRAM
DeepSeek-Coder-6.7B (fp8, single vLLM)	~8 GB
KV cache (4096 ctx, 8 concurrent req)	~4 GB
LoRA adapters in memory (×6)	~1 GB
Inference total	~13 GB
Available for training (Unsloth QLoRA)	~11 GB (sufficient for 7B 4-bit)
With the Phase 3 GPU coordinator this is safe: inference pauses, training gets the full budget, inference resumes.

Risk Summary
Phase	Risk	Breaking change	Effort
1 — Remove API mode	LOW	No (removes unused path)	~1 day
2 — Remove cloud ChromaDB	LOW	No	~2 hours
3 — GPU coordinator	MEDIUM	No (adds endpoints)	~3 days
4 — Node registry	MEDIUM	No (backward compat default)	~2 days
5 — Single-model mode	LOW	No (additive)	~1 day
6 — Module boundaries	LOW	No (internal moves)	~1 day
7 — JSONL compaction	LOW	No	~half day
Recommended execution order: 1 → 2 → 7 → 6 → 5 → 4 → 3 (hardest last, everything else de-risks it by simplifying the codebase first).

What NOT to refactor
LangGraph pipeline structure — well-designed, don't touch
gRPC for Orchestration ↔ Inference — correct choice for hot path
FastAPI + async throughout — correct
ChromaDB for RAG — appropriate for local portable use
Per-capability LoRA adapter naming — clean already