# IdlePods AI

A poor man's local (soon to be fully portable) coding/research assistant designed to be scalable and grow over time for full self-reliance.

The project currently uses rank-stabilized LoRA adapter training/retraining to steadily improve niche agent adapters over time 

More formally, this is a self-improving multi-agent LLM tool where a team of specialized agents collaborates to produce a solid response; every successful run feeds into training data that fine-tunes the models over time.

---

## Architecture

```
User submits prompt
    ↓
Gateway → Orchestration (LangGraph pipeline)
    ↓
Pipeline completes (converges or max_iterations)
    ↓
Orchestration publishes ExperienceEvent
    ↓ (fire-and-forget async, doesn't block user response)
Orchestration stores to JSONL + ChromaDB
    ↓ (async background task)
Training job evaluates thresholds
    ↓ Batch size: ≥ 50 total experiences accumulated (min_batch_size)
    ↓ Score spread: max_score - min_score ≥ 0.15 (ensures diverse quality labels, not all high or all low)
    ↓ Diversity ratio: ≥ 60% of records have unique prompt fingerprints (deduplication guard)
    ↓ Quality filter: Only records with final_score ≥ 0.65 are used (min_quality_score)
    ↓ IF thresholds met, launch LoRA subprocess
Unsloth fine-tunes new adapters
    ↓
Save to /data/lora_checkpoints/
    ↓
vLLM hot-reloads adapters
    ↓
Next request uses improved model
```

---

## Services

| Service | Port | Role |
| --- | --- | --- |
| [gateway](gateway/) | 8080 | Entry point — auth, routing |
| [orchestration](orchestration/) | 8001 | LangGraph agent pipeline + context (in-process) + experience (in-process) |
| [inference](inference/) | 8010 / 50051 | LLM backend — single Qwen/Qwen3-14B vLLM server with 6 LoRA adapters |
| [training](training/) | — | LoRA fine-tuning via Unsloth |
| [shared](shared/) | — | Pydantic contracts, QueryRouter, gRPC stubs |

Agent roles: `planner researcher coder debugger reviewer critic consensus`  
Intents: `CODING DEBUGGING RESEARCH ANALYSIS PLANNING QA GENERAL`  
Convergence threshold: score ≥ 0.85 (bands: <0.4 poor · 0.4–0.7 acceptable · 0.7–0.85 good)

---

## Quick start

Requires NVIDIA GPU + CUDA drivers and Docker Compose.

```bash
cp .env.example .env
docker compose -f docker/compose.yml up
```

All services expose `/health`. The system is ready when Docker Compose reports all containers healthy.

```bash
curl -X POST http://localhost:8080/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Write a Python function that debounces a callback"}'
```

If running services outside Docker, generate gRPC stubs first:

```bash
python scripts/generate_protos.py
```

---

## Demo

![alt text](image-1.png)

![alt text](image.png)

---

## Adapter training notes

All six LoRA adapters (`coding_lora`, `debugging_lora`, `review_lora`, `planning_lora`, `research_lora`, `criticism_lora`) are fine-tuned on **Qwen/Qwen3-14B** base.

Training uses **ChatML** format (`<|im_start|>role\ncontent<|im_end|>`). The masking boundary for `DataCollatorForCompletionOnlyLM` is `<|im_start|>assistant\n` — only assistant turns are training targets. Tool-result turns (`role="tool"`) are never trained on.

`coder` and `debugger` adapters receive tool-use SFT pairs (assistant tool-call emission + tool-result context) in addition to regular code-generation pairs. All adapters trained before the Qwen3-14B migration are invalid and require full retraining on the new base.

Thinking mode is disabled at inference time via `chat_template_kwargs={"enable_thinking": False}` — no `<think>` tokens appear in agent outputs.

---

## Architecture decisions

**Single Qwen/Qwen3-8B vLLM server** — all six agent roles (planner, researcher, coder, debugger, reviewer, critic) share one server. Per-role specialisation is handled by LoRA adapters selected per request.

**Native OpenAI function calling** — the coder and debugger agents use `--enable-auto-tool-choice --tool-call-parser hermes` on the vLLM server. Tool calls arrive as structured JSON (`response.tool_calls`); the `<<TOOL>><<END>>` regex-parse approach is retired. Supported tools: `read_file`, `write_file`, `list_files`, `run_command` (pytest/ruff/mypy allowlist).

**gRPC for Orchestration → Inference, HTTP elsewhere** — the orchestration-to-inference call is the hot path (one call per agent node per iteration). Everything else is low-frequency enough that HTTP simplicity wins.

**Fire-and-forget for experience recording and training notification** — decouples the user-facing response time from downstream storage and training evaluation latency.

**Context and Experience services are in-process** — no separate context or experience containers; both run as async tasks inside the orchestration process. ** Need to consider alternatives for better reliability.

## Limitations

Local inference requires a GPU (min 3090 or better for standard to optimal performance). No CPU fallback path.

Limited to domain specific tasks (using LoRA mainly for Coding, Critic, Debugger, Researcher). Need to consider alternatives like rsLoRA in the future for better stability.

Limited to local inference for full self training pipeline but scalable for self hosted vllm servers to serve baseline models.
