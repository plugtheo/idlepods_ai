# Gateway Service

## What it does

The Gateway is the **front door** of the entire system. Every request from the outside world enters here first and only here. It does not run any AI models, store any data, or make any decisions about what the answer should be. Its only job is to prepare the request for the team of AI agents, hand it off, and relay the result back.

Here is what happens, step by step, every time a request arrives:

### Step 1 — Authentication check

If an API key has been configured (via the `GATEWAY__API_KEY` environment variable), the Gateway checks that the incoming request includes a matching `Authorization: Bearer <key>` header. Any request that fails this check is rejected with a 401 Unauthorized error before it goes any further. When the API key is left blank (the default for local development), this check is completely skipped.

The `/health` endpoint is always exempt from authentication.

### Step 2 — Intent classification

The Gateway reads the text of the user's prompt and figures out *what type of task it is* by scanning for keywords and patterns. This produces two labels:

- **Intent** — what the user is trying to do:
  - `coding` — writing new code
  - `debugging` — fixing broken code or errors
  - `research` — looking something up or explaining a concept
  - `analysis` — reviewing or evaluating something
  - `planning` — designing a system or architecture
  - `qa` — questions and answers about a topic
  - `general` — anything else

- **Complexity** — how hard the task is:
  - `simple` — short prompts (under ~25 words), no signal words indicating depth
  - `moderate` — medium-length prompts with average scope
  - `complex` — long prompts (over ~80 words) or prompts containing words like "architect", "microservice", "distributed", "authentication", etc.

The classification is done entirely by fast keyword matching — no AI model is called for this step. All patterns are scored and the intent with the most keyword matches wins.

### Step 3 — Agent chain selection

Based on the intent + complexity combination, the Gateway picks an ordered list of specialised AI agent roles that are best suited for the task. There are 21 pre-defined combinations. For example:

- A **complex coding** request → `["planner", "coder", "reviewer", "critic"]`
- A **simple debugging** request → `["debugger"]`
- A **moderate research** request → `["researcher", "critic"]`

This list tells the Orchestration Service which agents to run and in what order.

### Step 4 — Forward to Orchestration

The original prompt, the session ID, the detected intent and complexity, the agent chain, and any caller-supplied settings (max iterations, quality threshold) are packaged into a single request and sent to the Orchestration Service over HTTP. The Gateway waits up to 5 minutes (configurable) for a response.

### Step 5 — Relay the response

The response from the Orchestration Service — the final AI-generated answer, quality score, number of iterations, and per-agent trace — is forwarded directly back to the caller.

For the streaming endpoint, each agent update is relayed in real time as a Server-Sent Events (SSE) stream so the caller can display progress live.

---

## API

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/v1/chat` | Send a prompt, wait for the full completed response |
| `POST` | `/v1/chat/stream` | Send a prompt, receive live agent updates as an SSE stream |
| `GET`  | `/health` | Liveness probe — always returns `{"status": "ok"}` |

### Request — `POST /v1/chat` and `POST /v1/chat/stream`

```json
{
  "prompt": "Write a Python function that debounces a callback",
  "session_id": "optional-uuid-for-log-tracing",
  "max_iterations": 3,
  "convergence_threshold": 0.85
}
```

| Field | Required | Description |
|-------|----------|-------------|
| `prompt` | Yes | The user's question or task description |
| `session_id` | No | Optional ID you supply for log tracing. Auto-generated if omitted |
| `max_iterations` | No | How many times the agent chain can loop to improve the answer (default 5, max 20) |
| `convergence_threshold` | No | Quality score (0–1) at which the pipeline stops iterating and accepts the result (default 0.85) |

### Response — `POST /v1/chat`

```json
{
  "session_id": "uuid",
  "output": "Final answer text from the agent pipeline",
  "success": true,
  "confidence": 0.92,
  "iterations": 2,
  "converged": true
}
```

| Field | Description |
|-------|-------------|
| `session_id` | The session identifier (supplied or auto-generated) |
| `output` | The final answer produced by the AI pipeline |
| `success` | Whether the pipeline ran to completion without error |
| `confidence` | The quality score of the final answer, between 0 and 1 |
| `iterations` | How many full passes the agent chain made |
| `converged` | True if the quality threshold was met; false if the iteration cap was hit instead |

### Streaming — `POST /v1/chat/stream`

Returns a stream of Server-Sent Events. Each event has the format `data: {...}` followed by a blank line. Event types:

| `type` field | When it appears | Notable fields |
|---|---|---|
| `start` | Once, immediately | `session_id`, `agent_chain` |
| `agent_step` | After each agent responds | `role`, `output_summary`, `iteration` |
| `iteration_complete` | After each full chain pass | `iteration`, `score` |

---

## Configuration

All configuration is supplied via environment variables with the `GATEWAY__` prefix.

| Variable | Default | Description |
|----------|---------|-------------|
| `GATEWAY__ORCHESTRATION_URL` | `http://orchestration:8001` | Where the Orchestration Service is running |
| `GATEWAY__API_KEY` | *(empty)* | Bearer token required from callers. Leave empty to disable authentication |
| `GATEWAY__REQUEST_TIMEOUT` | `300.0` | Seconds to wait for the Orchestration Service before giving up |
| `GATEWAY__PORT` | `8080` | Port this service listens on |
| `GATEWAY__DEBUG` | `false` | Enable verbose debug logging |

---

## What it talks to

| Downstream service | How | Why |
|---|---|---|
| Orchestration Service | HTTP POST | Runs the actual agent pipeline |

The Gateway has no database and calls no AI model directly.

---

## Running locally

```bash
docker compose -f docker/compose.yml up gateway orchestration inference
```

The gateway is reachable at **http://localhost:8080**.

---

## Known limitations

- The SSE streaming proxy (`/v1/chat/stream`) relays events from the Orchestration Service but does not emit a final `done` event — the stream simply closes when the pipeline finishes. Clients should treat stream closure as the completion signal.
- Intent classification uses keyword matching only. It cannot understand nuanced prompts and may misclassify edge cases (e.g. a long research question about code might be classified as `coding`).
