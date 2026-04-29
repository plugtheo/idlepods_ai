# Gateway Service

## Purpose

The Gateway is the front door of the system. It validates who is asking (checks the API key), figures out what kind of work needs to be done (classifies the intent and complexity), and forwards the request to the Orchestration Service. It doesn't run any AI models itself — it just routes traffic.

## Core Components

- **APIKeyMiddleware**: Checks API keys on every incoming request. If authentication is configured, requests without a valid key are rejected immediately.

- **QueryRouter**: A fast keyword analyzer that reads the user's prompt and determines:
  - **Intent** — what the user is trying to do (code, debug, research, plan, etc.)
  - **Complexity** — how hard the task is (simple, moderate, or complex)
  - **Agent chain** — an ordered list of specialist agents suited for this task

- **Chat Route Handler**: Accepts the user's prompt, calls QueryRouter, builds an orchestration request, sends it to the Orchestration Service, and returns the final answer to the user.

## Data Flow

```
Client Request
  ↓
APIKeyMiddleware (validate credentials)
  ↓
QueryRouter (classify prompt)
  ├─ intent: coding | debugging | research | ...
  ├─ complexity: simple | moderate | complex
  └─ agent_chain: [coder, reviewer] | [debugger] | [planner, coder, reviewer, critic]
  ↓
Orchestration Service (HTTP POST)
  ↓
Final answer returned to client
```

## Key Interactions

- **Incoming**: HTTP requests with a prompt (and optional session ID for multi-turn conversations)
- **Outgoing**: Forwards an OrchestrationRequest to Orchestration Service
- **Response**: Returns the final output, confidence score, iteration count, and whether convergence was reached

## Important Concepts

### Intent Classification

The QueryRouter doesn't use an AI model — it uses fast pattern matching (keyword scoring). It looks for words like "code", "debug", "explain", "design", etc. and assigns an intent based on which words appear most. This is why it's so fast but can misclassify nuanced prompts.

### Agent Chains

Different tasks need different specialists:
- A complex coding task gets routed to `["planner", "coder", "reviewer", "critic"]` — one agent breaks down the problem, the next writes code, the next reviews it, and the last provides critical feedback.
- A simple debugging task gets just `["debugger"]` — single agent, no looping.

The Orchestration Service will run these agents in order, potentially looping until quality converges.

### Session IDs

If the client doesn't provide a session_id, the Gateway generates a UUID. This ties multi-turn conversations together so the Orchestration Service can maintain context across requests for the same user.

