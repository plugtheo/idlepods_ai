# Context Service

## What it does

The Context Service **enriches every agent pipeline run with relevant background knowledge** before any agent starts working. Its goal is to make the agents smarter by giving them real examples of similar problems that were solved well in the past, plus relevant code files from the repository.

Think of it like a researcher who, before a meeting, pulls out three or four examples of similar projects and relevant documents so the team doesn't start from scratch.

The Orchestration Service calls the Context Service at the very start of each run. If the Context Service is slow or unavailable, the pipeline continues anyway with no context enrichment — it degrades gracefully rather than failing.

Here is what happens step by step when the Context Service receives a request:

### Step 1 — Semantic few-shot search

The service takes the user's prompt and converts it into a vector of numbers (a mathematical representation of its meaning) using a sentence embedding model (`all-MiniLM-L6-v2`). This is called an **embedding** — two sentences that mean similar things will have vectors that are close together in mathematical space, even if they use completely different words.

The service then queries a **ChromaDB** vector database (a database that is specifically designed to find items that are semantically similar) to find the most similar past experiences. It returns up to 4 examples (configurable) with a cosine similarity score above 0.68.

Each result is a `FewShotExample` containing:
- `problem` — the original user prompt from a past run
- `solution` — the final AI-generated answer from that run
- `score` — the quality score that was assigned to that solution (0–1)
- `category` — the capability type (e.g. `coding`, `research`)

These examples are included in the agents' context so they can see "here is what a good solution to a similar problem looked like."

**Important:** The vector database is populated by the Experience Service after each successful run. On a fresh installation, the database is empty and no few-shot examples will be returned until the system has processed some requests.

### Step 2 — Repository code snippet scan

For `coding`, `debugging`, and `analysis` intents only, the service also scans the local repository for relevant source files. It walks through up to 200 files, scores each one by how much keyword overlap it has with the user's prompt (comparing words in the filename and first line of the file against words in the prompt), and returns the top matches.

Each result is a `RepoSnippet` containing:
- `file` — the relative file path
- `snippet` — up to 300 characters of the most relevant portion of the file
- `relevance` — a score from 0 to 1

These snippets are included in the agents' context to help them understand the existing codebase when writing or debugging code.

### Step 3 — Build system hints

Based on the intent and complexity classification, the service also generates short guidance strings called `system_hints`. These are injected into every agent's system prompt for the duration of this request. For example, a coding request might inject a hint like "Focus on clean, tested, production-ready code." Currently hints are generated for `coding`, `debugging`, `research`, and `analysis` intents.

### Step 4 — Return everything together

Both results plus the hints are bundled into a `BuiltContext` object and returned to the Orchestration Service in a single response.

---

## API

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/v1/context/build` | Build enriched context for a prompt |
| `GET`  | `/health` | Liveness probe — always returns `{"status": "ok"}` |

### Request — `POST /v1/context/build`

```json
{
  "prompt": "Implement a rate-limiter in Python",
  "intent": "coding",
  "complexity": "moderate",
  "session_id": "optional-uuid"
}
```

| Field | Required | Description |
|-------|----------|-------------|
| `prompt` | Yes | The user's original question or task |
| `intent` | Yes | The classified intent (e.g. `coding`, `debugging`, `research`) |
| `complexity` | Yes | The classified complexity (`simple`, `moderate`, or `complex`) |
| `session_id` | No | Optional ID for log tracing |

### Response

```json
{
  "few_shots": [
    {
      "problem": "original user prompt from a past run",
      "solution": "the answer that was produced",
      "score": 0.91,
      "category": "coding"
    }
  ],
  "repo_snippets": [
    {
      "file": "app/utils/rate_limiter.py",
      "snippet": "class TokenBucket: ...",
      "relevance": 0.6
    }
  ],
  "system_hints": "Focus on clean, well-structured, production-ready code."
}
```

---

## Configuration

All configuration is supplied via environment variables with the `CONTEXT__` prefix.

| Variable | Default | Description |
|----------|---------|-------------|
| `CONTEXT__VECTOR_STORE_PATH` | `/data/vector_store` | Directory where ChromaDB stores its data |
| `CONTEXT__EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Name of the sentence-transformers embedding model to use |
| `CONTEXT__MAX_FEW_SHOTS` | `4` | Maximum number of few-shot examples to return |
| `CONTEXT__SIMILARITY_THRESHOLD` | `0.68` | Minimum cosine similarity score for a past example to be included |
| `CONTEXT__REPO_PATH` | `.` | Root directory to scan for repository code snippets |
| `CONTEXT__MAX_REPO_SNIPPETS` | `3` | Maximum number of code snippets to return |
| `CONTEXT__PORT` | `8011` | Port this service listens on |

---

## What it talks to

| Dependency | What it is | Why |
|---|---|---|
| ChromaDB (local file) | Vector database | Stores and searches past experience embeddings |
| `sentence-transformers` library | Embedding model | Converts text prompts into searchable vectors |
| Local filesystem | Repository files | Source for code snippets (only in coding/debugging mode) |

The Context Service does not call any other microservice.

---

## Storage

ChromaDB data is stored in the `vector_store` Docker volume (mounted at `/data/vector_store`). This volume is **shared with the Experience Service**, which writes new experience embeddings to the same location after every successful pipeline run. This means that as the system processes more requests, the few-shot retrieval automatically improves.

---

## Structure

```
app/
  config/settings.py        — all environment variable config
  utils/
    embedder.py             — lazy singleton that loads and caches the embedding model
  retrieval/
    few_shot.py             — ChromaDB vector similarity search
    repo.py                 — local file system keyword scan for code snippets
  routes/build.py           — POST /v1/context/build and GET /health
  main.py                   — FastAPI application entry point
```

---

## Known limitations

- The embedding model (`all-MiniLM-L6-v2`) runs synchronously on the CPU when processing each request. Under high concurrency, this can slow down other services sharing the same thread.
- The repository file scan is also synchronous — it blocks while walking the file system. Both operations run in the same concurrency primitive (`asyncio.gather`), but because neither actually yields control back to the event loop, they run sequentially rather than in parallel.
- System hints are only generated for `coding`, `debugging`, `research`, and `analysis` intents. Requests with `planning`, `qa`, or `general` intent receive an empty hints string.
- The vector database keeps growing indefinitely as experiences accumulate. There is no eviction, deduplication, or cleanup mechanism.
