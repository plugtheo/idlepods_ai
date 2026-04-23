# Experience Service

## What it does

The Experience Service is the **memory layer** of the system. Every time the agent pipeline successfully completes a run, the result is sent here to be recorded. Over time this builds up a library of solved problems that the system can learn from — both to give better context to future runs and to train better AI models.

Think of it like a journal that captures every successful project: the original problem, the solution, how well it went, and which agents were involved. That journal is then used in two ways: the Context Service reads it to find similar past examples, and the Training Service reads it to periodically improve the AI models.

Here is what happens step by step when the Experience Service receives a new record:

### Step 1 — Append to the JSONL log

The full `ExperienceEvent` (every agent's output, the final answer, the quality score, the timestamp, everything) is immediately written to a JSONL file on disk. JSONL (JSON Lines) is a simple format where each line is one complete JSON record.

This file is the durable, authoritative record — it is never modified and never deleted. If anything else in the system breaks, this log always has the full history.

The write is protected by an async lock to ensure that if two runs finish at exactly the same time, their records don't get interleaved or corrupted.

### Step 2 — Upsert into ChromaDB

The user's original prompt is converted into a vector embedding (a mathematical representation of its meaning) using the same `all-MiniLM-L6-v2` model that the Context Service uses. This embedding, along with metadata about the run (quality score, agent chain, whether convergence was achieved, and the final output), is stored in ChromaDB.

This is what makes future few-shot retrieval work: when a new prompt arrives, the Context Service queries ChromaDB to find past prompts that are semantically similar to the new one.

**Note on metadata keys:** The final output is stored under a `"solution"` key in the ChromaDB metadata so that the Context Service can read it back correctly.

### Step 3 — Notify the Training Service (fire-and-forget)

After every new experience is recorded, the Training Service is notified so it can evaluate whether there is enough new data to start a LoRA fine-tuning run. This notification is sent asynchronously — the Experience Service does not wait for a response and the `/v1/experience` endpoint returns immediately.

As a pre-check, the Experience Service counts the total number of records in the JSONL file first. If there are fewer than `MIN_BATCH_SIZE` records total, it skips the notification entirely to avoid unnecessary calls to the Training Service.

---

## API

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/v1/experience` | Record one completed pipeline run |
| `GET`  | `/health` | Liveness probe — always returns `{"status": "ok"}` |

### Request — `POST /v1/experience`

```json
{
  "session_id": "uuid",
  "prompt": "Write a Python function that debounces a callback",
  "final_output": "Here is a debounce implementation...",
  "agent_chain": ["planner", "coder", "reviewer"],
  "contributions": [
    {"role": "planner", "output": "Plan: 1. ...", "quality_score": 0.0, "iteration": 1},
    {"role": "coder", "output": "def debounce...", "quality_score": 0.0, "iteration": 1},
    {"role": "reviewer", "output": "SCORE: 0.88 ...", "quality_score": 0.0, "iteration": 1}
  ],
  "final_score": 0.88,
  "iterations": 1,
  "converged": true,
  "timestamp": "2026-04-01T10:00:00Z"
}
```

| Field | Description |
|-------|-------------|
| `session_id` | Unique identifier for this run |
| `prompt` | The original user question or task |
| `final_output` | The best answer produced by the pipeline |
| `agent_chain` | The ordered list of agents that ran |
| `contributions` | Each agent's raw output and which iteration it came from. Note: `quality_score` per contribution is currently always `0.0` |
| `final_score` | The overall quality score (0–1) from the convergence scoring |
| `iterations` | How many full agent-chain loops ran |
| `converged` | Whether the quality threshold was reached (vs. hitting the max iteration cap) |
| `timestamp` | When the run completed |

### Response

```json
{"session_id": "uuid", "stored": true}
```

---

## Configuration

All configuration is supplied via environment variables with the `EXPERIENCE__` prefix.

| Variable | Default | Description |
|----------|---------|-------------|
| `EXPERIENCE__JSONL_PATH` | `/data/experiences.jsonl` | Path to the append-only log file |
| `EXPERIENCE__EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Embedding model for vectorising prompts |
| `EXPERIENCE__CHROMA_COLLECTION` | `experiences` | Name of the ChromaDB collection |
| `EXPERIENCE__TRAINING_URL` | `http://training:8013` | URL of the Training Service to notify |
| `EXPERIENCE__MIN_BATCH_SIZE` | `50` | Minimum total records before the Training Service is notified |
| `EXPERIENCE__PORT` | `8012` | Port this service listens on |

---

## What it talks to

| Downstream | How | Why |
|---|---|---|
| ChromaDB (local file) | Direct library call | Stores experience embeddings for later retrieval |
| Training Service | HTTP (fire-and-forget) | Notifies Training Service when enough data may be available |

The Experience Service has no external dependencies beyond local disk and the Training Service.

---

## Storage

Both the JSONL file and ChromaDB data live in Docker volumes:
- The **JSONL file** is stored in the `experiences` volume (mounted at `/data`)
- The **ChromaDB data** is stored in the `vector_store` sub-path of the same volume, which is also mounted by the Context Service at `/data/vector_store`

This sharing means that as soon as the Experience Service writes a new embedding, the Context Service can find it in future queries without any synchronisation needed.

---

## Structure

```
app/
  config/settings.py        — all environment variable config
  storage/
    jsonl_store.py          — thread-safe append / count / read from JSONL
    vector_store.py         — ChromaDB upsert with embedding generation
  clients/
    training.py             — fire-and-forget HTTP trigger to Training Service
  routes/record.py          — POST /v1/experience and GET /health
  main.py                   — FastAPI application entry point
```

---

## Known limitations

- The ChromaDB vector store grows indefinitely — there is no pruning or deduplication of low-quality experiences.
- The `contributions[].quality_score` field in the published event is always `0.0`. Individual agent scoring is computed internally by the Orchestration Service but is not yet passed through to the experience record. Only `final_score` (the overall run quality) is accurate.
