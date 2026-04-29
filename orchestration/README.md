# Orchestration Service

## Purpose

The Orchestration Service is the brain of the system. It receives a prompt and a list of agent roles (the agent chain), enriches the request with context (past examples, relevant code), then runs the agents in a loop until the answer is good enough. It implements a feedback loop: run each agent, score the output, decide whether to stop (converged) or loop again.

Think of it like a software development team: a planner designs an approach, a coder writes it, a reviewer checks it, a critic provides feedback. The orchestrator keeps the team looping until quality is acceptable.

## Core Components

- **LangGraph Pipeline** (`graph/pipeline.py`): A state machine that defines the execution flow. It routes from one agent to the next, checks convergence (quality good enough?), and loops if needed. Built using LangGraph, a library for multi-step AI workflows.

- **Agent State** (`graph/state.py`): A shared data structure (TypedDict) that flows through all nodes. Contains the original prompt, conversation history, context (code snippets, few-shot examples), iteration tracking, and all agent outputs.

- **Agent Nodes** (`graph/nodes.py`): One node per specialist role (planner, coder, reviewer, critic, etc.). Each node:
  - Receives the shared state
  - Builds a specialized system prompt for its role
  - Calls the Inference Service to generate output
  - Appends the output to the shared history
  - Passes control to the next agent

- **Context Builder** (`context/builder.py`): Runs once at request start, fetches and assembles:
  - **Few-shot examples** — past similar tasks and solutions, retrieved from a vector database (ChromaDB)
  - **Repo snippets** — relevant code files, discovered by scanning the repo and matching to the prompt
  - **Hints** — guidance text customized for intent and complexity

  All three run in parallel; failure in one doesn't block the others.

- **Convergence Scoring** (`utils/scoring.py`): Grades the quality of each iteration (0.0–1.0). Looks for explicit scores embedded in reviewer/critic output, or estimates quality by detecting improvement signals.

- **ChromaDB Client** (`db/chroma.py`): Singleton connection to ChromaDB, a vector database. Stores completed pipeline runs ("experiences") and retrieves few-shot examples.

## Data Flow

```
1. Request arrives (prompt, agent_chain, max_iterations)
   ↓
2. Context Building (parallel)
   ├─ Few-shot retrieval (ChromaDB vector search)
   ├─ Repo scanning (detect relevant code files)
   └─ Hint generation (based on intent/complexity)
   ↓
3. Pipeline Loop (Iteration 1)
   ├─ Agent 0 runs (receives state, calls Inference, appends output)
   ├─ Agent 1 runs (sees Agent 0's output)
   ├─ Agent 2 runs (sees Agent 0 and 1)
   └─ Score the iteration
   ↓
4. Convergence Check
   ├─ If score ≥ threshold → finalize and return
   └─ Else if iterations < max → loop back to Step 3
   ↓
5. Finalize and store experience (fire-and-forget)
   ↓
6. Return final answer to client
```

## Key Interactions

### With Inference Service
- **Outgoing**: GenerateRequest with messages, agent role name, LoRA adapter name, generation parameters
- **Incoming**: GenerateResponse with the generated text and token count
- Called once per agent per iteration

### With Context Builder
- Runs at request start, asynchronously
- Returns BuiltContext (few-shots, snippets, hints)
- Failures are graceful (returns empty context if timeout/error)

### With ChromaDB (Experience Storage)
- Stores completed runs (experience) for later retrieval as few-shot examples
- Caches file fingerprints per task_id to avoid re-scanning unchanged code

## Important Concepts

### The Iteration Loop

The pipeline doesn't just call each agent once. It loops:
1. Run the agent chain: agent 0 → agent 1 → agent 2 → ...
2. Score the final output
3. If good enough, stop. Otherwise, loop back to step 1.

This allows agents to refine each other's work across multiple passes.

### Agent State

A single shared data structure that all nodes read and write:
- **Identity**: session_id, task_id, user_prompt
- **Routing**: agent_chain (ordered list of role names), current position in chain
- **Context**: few_shots, repo_snippets, system_hints
- **Iteration**: current_iteration, max_iterations, convergence_threshold
- **History**: conversation_history (multi-turn state from prior requests), iteration_history (all agent outputs this request)

### Convergence Threshold

A configurable quality score (default 0.85) that determines when to stop. Higher thresholds require higher quality, potentially more iterations.

### LoRA Adapters

Fine-tuned neural network weights for specific roles (coding_lora, debugging_lora, etc.). The Inference Service applies these on-the-fly, making the same base model behave as different specialists.

### File Fingerprinting

To avoid re-scanning large repositories, the context builder caches file snippets by their first-line hash (fingerprint). If the fingerprint hasn't changed, the cached snippet is reused. This is stored per-task_id in ChromaDB.

### Experience Recording

After the pipeline completes, the full run (prompt, all agent outputs, scores, timing, agent roles) is saved as an "experience". These are used for two things:
- Retrieval as few-shot examples to help future agents

- Training data for the Training job to fine-tune adapters