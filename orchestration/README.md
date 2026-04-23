# Orchestration Service

## What it does

The Orchestration Service is the **brain of the system**. It receives a prepared request from the Gateway and runs a team of specialised AI agents in sequence, each building on the previous one's output, until the answer is good enough — or until a maximum number of attempts is reached.

Think of it like a software development team where a planner designs the approach, a developer writes the code, a reviewer checks it, and a critic pushes for improvements. The manager (the orchestrator) keeps the team looping until the work meets the quality bar.

Here is the full step-by-step journey:

### Step 1 — Fetch context (optional enrichment)

Before any agent runs, the Orchestration Service asks the Context Service for background knowledge relevant to this prompt. The Context Service returns:
- **Few-shot examples** — real examples of similar past problems and their solutions, to help agents understand the expected output format and quality
- **Repo snippets** — relevant code files from the local repository (for coding/debugging tasks)
- **Hints** — short guidance strings injected into every agent's instructions for this request

The Context Service call has a hard 2-second timeout. If it is slow or unavailable, the pipeline simply continues with no context enrichment — the agents still work, just without the extra examples.

### Step 2 — Build the pipeline state

The Orchestration Service prepares a shared state object (`AgentState`) that all agents will read from and write to. It contains:
- The original prompt
- The enriched context (few-shots, repo snippets, hints)
- The agent chain to run (e.g. `["planner", "coder", "reviewer", "critic"]`)
- Iteration tracking (current count, best score so far, best output so far)
- The entire conversation history (every agent's output from every iteration)

### Step 3 — Run the LangGraph pipeline

The agents are connected into a directed graph using **LangGraph**, a library for building stateful multi-step AI workflows. The graph structure is:

```
START
  │
  ▼
route_entry          — picks the first agent from the agent_chain
  │
  ▼
[agent runs in sequence: planner → coder → reviewer → critic → ...]
  │
  ▼
check_convergence    — scores the last iteration and decides what to do next
  │
  ├── score ≥ threshold or max iterations reached → finalize → [consensus] → END
  │
  └── score too low → go back to the first agent, start a new iteration
```

Every time an agent node runs:
1. It receives the current state (prompt + history + context)
2. It builds its own set of instructions (a system prompt describing its role, plus the conversation so far)
3. It calls the Inference Service to get the model's response
4. It appends its response to the shared history
5. The next agent picks up from there, seeing ALL previous agents' outputs in its context

### Step 4 — Convergence scoring

After every full pass through the agent chain (one "iteration"), the pipeline scores the quality of the work. Scoring works in two ways:
- **Explicit score extraction** — reviewer and critic agents are instructed to include a line like `SCORE: 0.87` in their output. The pipeline reads this to get a precise quality signal.
- **Heuristic scoring** — if no explicit score is found, the pipeline estimates quality by looking for positive signals (good answer indicators) and negative signals (blockers, warnings) in the output.

If the score reaches the convergence threshold (default 0.85 out of 1.0), the pipeline stops early. Otherwise it starts another iteration.

### Step 5 — Finalization

Once the pipeline exits (via convergence or hitting the max iteration cap), the best output from all iterations is selected. A final **consensus agent** may run to synthesize and clean up the output for short or mid-length agent chains. For very short chains (1–2 agents) that already achieved quality convergence, the consensus step is skipped.

### Step 6 — Publish the experience

After the pipeline completes, the full result is sent asynchronously (fire-and-forget — the response does not wait for this) to the Experience Service. This is how the system learns from its own outputs. The information published includes: the original prompt, the final answer, every agent's output, quality scores, and whether convergence was achieved.

---

## Agent roles

| Role | Model family | LoRA adapter | Purpose |
|------|-------------|-------------|---------|
| `planner` | Mistral | `planning_lora` | Breaks down the task into a structured plan |
| `researcher` | Mistral | `research_lora` | Gathers relevant concepts and background knowledge |
| `coder` | DeepSeek | `coding_lora` | Writes the actual code or solution |
| `debugger` | DeepSeek | `debugging_lora` | Identifies and fixes bugs or errors |
| `reviewer` | DeepSeek | `review_lora` | Reviews the work and provides a scored evaluation |
| `critic` | Mistral | `criticism_lora` | Provides harsh critical feedback and improvement suggestions |
| `review_critic` | Both | (runs reviewer then critic) | A combined role that runs reviewer and critic back-to-back |
| `consensus` | Mistral | *(none — uses base model)* | Synthesizes the final answer from the agent history |

**What is a LoRA adapter?** It is a small set of additional weights trained on top of a base language model to make it better at a specific task. The base model is like a generalist; the adapter is like specialised experience for that role. When running in local mode, each agent can optionally apply its adapter on the fly.

**What is a model family?** The system uses two open-source models: DeepSeek (better at writing and fixing code) and Mistral (better at reasoning, planning, and language tasks). The Inference Service routes each agent's request to the right model.

---

## Optimisations

To keep prompts from growing too large as the pipeline iterates:

- **History filtering** — each agent only sees the outputs from agents that are relevant to its role. For example, the reviewer only sees the coder's and debugger's outputs, not the planner's.
- **Structured extraction** — for the reviewer and critic, instead of including their full output in subsequent agents' context, only the structured fields (SCORE, ISSUES, SUGGESTIONS) are extracted and included.

Both optimisations are enabled by default and controlled via environment variables.

---

## API

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/v1/run` | Run the agent pipeline, wait for the full result |
| `POST` | `/v1/run/stream` | Run the agent pipeline, receive live updates as an SSE stream |
| `GET`  | `/health` | Liveness probe |

### Request — `POST /v1/run`

```json
{
  "prompt": "Implement a rate-limiter in Python",
  "agent_chain": ["planner", "coder", "reviewer", "critic"],
  "intent": "coding",
  "complexity": "moderate",
  "max_iterations": 5,
  "convergence_threshold": 0.85,
  "session_id": "optional-uuid"
}
```

| Field | Required | Description |
|-------|----------|-------------|
| `prompt` | Yes | The user's question or task |
| `agent_chain` | No | Which agents to run in order. Omit to let the service classify and decide |
| `intent` | No | Pre-classified intent from the Gateway. Omit to let the service classify |
| `complexity` | No | Pre-classified complexity from the Gateway |
| `max_iterations` | No | How many times the agent chain loops before stopping (default 5) |
| `convergence_threshold` | No | Quality score to stop iterating early (default 0.85) |
| `session_id` | No | Optional ID for log tracing |

### Response

```json
{
  "session_id": "uuid",
  "output": "Final synthesized answer",
  "success": true,
  "confidence": 0.91,
  "iterations": 3,
  "best_score": 0.91,
  "agent_steps": [
    {"role": "planner", "iteration": 1, "output_summary": "First 300 chars...", "score": 0.0},
    {"role": "coder", "iteration": 1, "output_summary": "First 300 chars...", "score": 0.0}
  ],
  "converged": true,
  "metadata": {"intent": "coding", "complexity": "moderate", "agent_chain": ["planner", "coder", "reviewer", "critic"]}
}
```

Note: `agent_steps[].score` is currently always `0.0` — per-step scores are computed internally but not yet surfaced in the response.

---

## Configuration

All configuration is supplied via environment variables with the `ORCHESTRATION__` prefix.

| Variable | Default | Description |
|----------|---------|-------------|
| `ORCHESTRATION__INFERENCE_URL` | `http://inference:8010` | Inference Service HTTP URL |
| `ORCHESTRATION__CONTEXT_URL` | `http://context:8011` | Context Service URL |
| `ORCHESTRATION__EXPERIENCE_URL` | `http://experience:8012` | Experience Service URL |
| `ORCHESTRATION__CONTEXT_TIMEOUT` | `2.0` | Seconds to wait for context enrichment before giving up |
| `ORCHESTRATION__REQUEST_TIMEOUT` | `180.0` | Seconds to wait for each agent inference call |
| `ORCHESTRATION__DEFAULT_MAX_ITERATIONS` | `5` | Default iteration cap (used only when `max_iterations` is not set in the request — currently the request field default of `5` takes precedence) |
| `ORCHESTRATION__CONVERGENCE_THRESHOLD` | `0.85` | Default quality threshold |
| `ORCHESTRATION__INFERENCE_USE_GRPC` | `false` | Set to `true` to call Inference via gRPC instead of HTTP (enabled in Docker Compose) |
| `ORCHESTRATION__INFERENCE_GRPC_HOST` | `inference` | Hostname of the Inference gRPC server |
| `ORCHESTRATION__INFERENCE_GRPC_PORT` | `50051` | Port of the Inference gRPC server |
| `ORCHESTRATION__OPTIMIZE_ROLE_HISTORY_FILTER` | `true` | Filter agent history by relevance per role |
| `ORCHESTRATION__OPTIMIZE_STRUCTURED_EXTRACTION` | `true` | Compress reviewer/critic output to key fields only |
| `ORCHESTRATION__PORT` | `8001` | Port this service listens on |

---

## What it talks to

| Downstream service | How | Why |
|---|---|---|
| Inference Service | gRPC (default in Docker) or HTTP | Get model responses for each agent |
| Context Service | HTTP | Fetch few-shot examples and repo snippets before the pipeline starts |
| Experience Service | HTTP (fire-and-forget) | Publish the completed run for memory and training |

---

## Structure

```
app/
  config/settings.py          — all environment variable config
  agents/prompts.py           — system prompts and role → model/adapter mappings
  graph/
    state.py                  — shared AgentState (all data that flows between agents)
    nodes.py                  — one function per agent role; all call the Inference Service
    edges.py                  — routing logic: which agent runs next, when to stop
    pipeline.py               — assembles the graph and compiles it for execution
  clients/
    inference.py              — HTTP client to the Inference Service
    inference_grpc.py         — gRPC client to the Inference Service (used in Docker)
    context.py                — HTTP client to Context Service (with timeout/fallback)
    experience.py             — fire-and-forget HTTP publisher to Experience Service
  routing/query_router.py     — fallback intent classifier (used for direct API calls)
  utils/
    scoring.py                — convergence scoring (explicit + heuristic)
    inference_optimizer.py    — history filtering and structured extraction
  routes/run.py               — POST /v1/run and POST /v1/run/stream handlers
  main.py                     — FastAPI application entry point
```

---

## Known limitations

- The convergence scoring is heuristic — it is not always accurate. A high score does not guarantee a correct or high-quality answer; it means the reviewer/critic noted improvement.
- Agent-to-agent communication happens through text (tokens), which means every agent must re-read and re-tokenize the full history on each call. For long agent chains with many iterations this creates significant computational overhead. An architecture that shares model state between agents at the vector level (before token decoding) would be dramatically more efficient, but is not currently implemented.
- The streaming endpoint (`POST /v1/run/stream`) emits `agent_step` and `iteration_complete` events during the run but does not emit a final `done` event — the stream simply closes. The final output is only available via the non-streaming endpoint.

