# Training Service

## What it does

The Training Service is the **self-improvement engine** of the system. Its job is to periodically take the experiences that have been gathered from successful pipeline runs and use them to fine-tune (improve) the local AI models — making future runs better at specific types of tasks.

This process is called **LoRA fine-tuning**. LoRA (Low-Rank Adaptation) is a technique for training a small, focused set of additional weights on top of a large pre-trained base model. Instead of retraining the entire model (which would take days and enormous resources), LoRA produces a small "adapter" file that adjusts the model's behaviour for specific tasks. The training process takes minutes to hours on a single GPU.

Think of it like this: the base model is a skilled generalist. The LoRA adapter is a specialised overlay of experience in a particular domain (coding, debugging, planning, etc.) that gets placed on top of the generalist to make it much better at that specific task.

Here is what happens step by step:

### Step 1 — Receive the trigger

The Experience Service sends a `POST /v1/training/trigger` request after recording each new experience. This does not mean training will start — it is just a signal to evaluate whether the conditions are right.

### Step 2 — Check if training is already running

Only one training job can run at a time (enforced by an asyncio lock). If a training job is already running, the trigger request immediately returns `triggered: false` and exits. The status endpoint can be polled to see when the job finishes.

### Step 3 — Load all experiences

The service reads all records from the JSONL experience log. This gives it a complete picture of the available training data.

### Step 4 — Evaluate data quality and diversity

Three criteria must ALL pass before training starts:

| Criterion | Default threshold | What it prevents |
|-----------|------------------|-----------------|
| **Minimum batch size** | ≥ 50 records total | Training on too little data, which produces an overfit (memorised) model |
| **Score spread** | max score − min score ≥ 0.15 | A dataset where all results have similar scores gives the model no signal to learn from — it cannot tell good from bad |
| **Diversity** | unique prompts / total records ≥ 0.60 | Training on near-duplicate prompts (same question asked slightly differently) teaches nothing new |

"Unique prompts" is determined by fingerprinting — each prompt's first 120 characters (lowercased, spaces collapsed) must be distinct for it to count as unique.

If any criterion fails, the endpoint returns `triggered: false` with a message explaining which criterion was not met.

### Step 5 — Filter to quality experiences only

Experiences with a `final_score` below 0.65 are excluded from the training dataset. The training data should only show the model examples of *good* solutions, not mediocre or bad ones.

### Step 6 — Launch training in a subprocess

To keep the HTTP service responsive during training (which can take hours), the actual fine-tuning is launched in a completely separate Python subprocess. The Training Service remains able to accept new trigger requests and status queries while training runs.

The subprocess:
1. Receives the filtered training records as a temporary JSONL file
2. Selects the right base model based on which capability is being trained:
   - `coder` or `debugger` → `deepseek-ai/deepseek-coder-6.7b-instruct` (better at code)
   - all other capabilities → `mistralai/Mistral-7B-Instruct-v0.1` (better at language tasks)
3. Runs the LoRA fine-tuning using HuggingFace's `transformers`, `peft`, and `accelerate` libraries
4. Saves the resulting adapter to `/data/lora_checkpoints/<capability>/`

### Step 7 — vLLM picks up the new adapter

When running in local mode, the vLLM servers are started with `--enable-lora`. vLLM can load new adapters on demand. The Inference Service caches the list of available adapters with a 2-minute TTL — after training completes and the TTL expires, the new adapter becomes active automatically without any service restart.

---

## API

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/v1/training/trigger` | Evaluate whether to start training and launch if conditions are met |
| `GET`  | `/v1/training/status` | Check whether a training job is currently running |
| `GET`  | `/health` | Liveness probe |

### Trigger request — `POST /v1/training/trigger`

```json
{
  "capability": "coder",
  "new_experience_count": 1,
  "session_id": "optional-uuid"
}
```

| Field | Description |
|-------|-------------|
| `capability` | Which agent role this experience is associated with (e.g. `coder`, `researcher`) |
| `new_experience_count` | Always `1` in the current implementation — informational only, not used for decisions |
| `session_id` | Optional ID for log tracing |

### Trigger response

```json
{
  "capability": "coder",
  "triggered": true,
  "reason": "criteria met: n=52, spread=0.28, diversity=0.71"
}
```

If training was not triggered:
```json
{
  "capability": "coder",
  "triggered": false,
  "reason": "training already running"
}
```
or
```json
{
  "capability": "coder",
  "triggered": false,
  "reason": "insufficient data: n=31 (need 50)"
}
```

### Status response — `GET /v1/training/status`

```json
{"running": false}
```

---

## Configuration

All configuration is supplied via environment variables with the `TRAINING__` prefix.

| Variable | Default | Description |
|----------|---------|-------------|
| `TRAINING__JSONL_PATH` | `/data/experiences.jsonl` | Path to the JSONL experience log (same file that Experience Service writes to) |
| `TRAINING__OUTPUT_DIR` | `/data/lora_checkpoints` | Where trained LoRA adapter files are saved |
| `TRAINING__DEEPSEEK_MODEL` | `deepseek-ai/deepseek-coder-6.7b-instruct` | Base model for coding and debugging capabilities |
| `TRAINING__MISTRAL_MODEL` | `mistralai/Mistral-7B-Instruct-v0.1` | Base model for all other capabilities |
| `TRAINING__MIN_BATCH_SIZE` | `50` | Minimum number of records before training can start |
| `TRAINING__MIN_SCORE_SPREAD` | `0.15` | Required difference between best and worst scores in the dataset |
| `TRAINING__MIN_DIVERSITY_RATIO` | `0.60` | Required fraction of unique prompts in the dataset |
| `TRAINING__MIN_QUALITY_SCORE` | `0.65` | Individual record minimum score to be included in training |
| `TRAINING__HF_TOKEN` | *(empty)* | HuggingFace authentication token (required for gated/private models) |
| `TRAINING__PORT` | `8013` | Port this service listens on |

---

## What it talks to

| Dependency | What it is | Why |
|---|---|---|
| JSONL file on disk | Experience log (shared with Experience Service) | Source of training data |
| `transformers` / `peft` / `accelerate` | HuggingFace training libraries | Run the actual LoRA fine-tuning |
| vLLM servers (indirectly) | GPU model servers | Consume the trained adapters after training completes |

The Training Service does not call any other microservice. It is a consumer of data, not a producer.

---

## Structure

```
app/
  config/settings.py            — all environment variable config
  utils/
    experience_reader.py        — load all records, check diversity, filter by quality
    trainer_launcher.py         — asyncio lock, subprocess launcher, status flag
  trainer_entry.py              — subprocess entry point: bridges experiences to LoRATrainer
  routes/trigger.py             — POST /v1/training/trigger, GET /v1/training/status, GET /health
  main.py                       — FastAPI application entry point
training/
  lora_trainer.py               — LoRATrainer class, dataset builder (existing training code)
```

---

## Known limitations

- Training is an expensive operation. Without a GPU, the subprocess will fail immediately. The service will log the failure, reset the running flag, and become available for future triggers.
- The training data records only contain `problem`, `context`, `solution`, `evaluation`, `improvement`, and `lessons_learned` fields. Currently `context`, `improvement`, and `lessons_learned` are always empty strings — only `problem` and `solution` are populated from experience data.
- The `min_batch_size` threshold is duplicated in both this service and the Experience Service (both default to 50). If you change the threshold in Docker Compose, you must update both `TRAINING__MIN_BATCH_SIZE` and `EXPERIENCE__MIN_BATCH_SIZE` to stay in sync.
- There is no mechanism to track which experiences have already been used for training. Every training run trains on ALL records above the quality threshold, including those used in previous runs. This means the model sees "old" data repeatedly. A proper implementation would track the last trained checkpoint and train on only new data (incremental fine-tuning).
