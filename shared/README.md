# Shared Package

## What this is

The `shared/` package contains three things:
1. **Data contracts** — the Pydantic models that define exactly what data is sent between each pair of services
2. **Routing logic** — the `QueryRouter` classifier, shared between Gateway and Orchestration so routing behaviour is always identical
3. **gRPC stubs** — auto-generated Python code for the binary communication protocol used between Orchestration and Inference

This is the only place where inter-service message formats are defined. If a field needs to change, it changes here and every service that uses it picks up the change automatically.

---

## Why a shared package?

Each microservice runs in its own container, but they all need to agree on the shape of the data they exchange. Without a shared package, each service would have its own copy of these definitions — and they would inevitably drift out of sync as the system evolves. Centralising the contracts here means:
- A type error (wrong field name, wrong type) is caught at import time, not at runtime
- You can see in one place exactly what every service sends and expects to receive

---

## Data contracts

All contracts are in `shared/contracts/`. Each file is a set of **Pydantic models** — Python classes that enforce field types and required vs. optional fields automatically.

| File | Models defined | Which services use it |
|------|---------------|----------------------|
| `inference.py` | `Message`, `GenerateRequest`, `GenerateResponse` | Orchestration sends requests; Inference receives and responds |
| `context.py` | `ContextRequest`, `BuiltContext`, `FewShotExample`, `RepoSnippet` | Orchestration sends requests; Context receives and responds |
| `orchestration.py` | `OrchestrationRequest`, `OrchestrationResponse`, `AgentStep` | Gateway sends requests; Orchestration receives and responds; response flows back to the user |
| `experience.py` | `ExperienceEvent`, `AgentContribution` | Orchestration publishes; Experience receives |
| `training.py` | `TrainingTriggerRequest`, `TrainingTriggerResponse` | Experience publishes; Training receives |

### What is Pydantic?

Pydantic is a Python library for data validation. When you create a Pydantic model instance with a dictionary of data (e.g. from an HTTP request body), Pydantic automatically:
- Checks that all required fields are present
- Converts field values to the expected types (e.g. string `"true"` → bool `True`)
- Raises a clear, structured error if any field is wrong

This means every service boundary in the system has automatic input validation at no extra cost.

---

## Routing

`shared/routing/query_router.py` contains the `QueryRouter` — the stateless prompt classifier that determines intent and complexity and maps them to an agent chain.

Both the Gateway Service and the Orchestration Service import `QueryRouter` from here.  Keeping one canonical copy prevents the two services from silently diverging in their routing decisions.

```python
from shared.routing.query_router import QueryRouter, RouteDecision
```

---

## gRPC stubs

The `shared/grpc_stubs/` directory contains auto-generated Python bindings for the gRPC protocol used between the Orchestration Service and the Inference Service.

**What is gRPC?** It is a high-performance communication protocol that uses a compact binary format (Protocol Buffers) instead of JSON. It is faster and more efficient than HTTP/JSON for high-frequency calls between internal services.

The protocol definition lives in `shared/proto/inference.proto`, which defines:
- The `InferenceService` with a single `Generate` RPC call
- The `GenerateRequest` and `GenerateResponse` message formats
- The `MessageRole` enum (`SYSTEM`, `USER`, `ASSISTANT`)

The Python binding files (`inference_pb2.py` and `inference_pb2_grpc.py`) are **not committed to version control** — they are generated from the `.proto` file at build time.

**In Docker** — generation runs automatically as part of the Orchestration and Inference image builds (see the `RUN python -m grpc_tools.protoc …` step in `orchestration/Dockerfile`).

**Locally (before running tests or using gRPC outside Docker)** — run once:

```bash
python scripts/generate_protos.py
```

If you see `ModuleNotFoundError: No module named 'shared.grpc_stubs.inference_pb2'`, the stubs have not been generated yet. Run the script above and retry.

---

## Rules for this package

- **No service imports** — this package must not import anything from any service directory (gateway, inference, orchestration, etc.)
- **No I/O** — these are pure data definitions. No database calls, no HTTP calls, no file reads.
- **Breaking changes** — if you rename or remove a field, you must update every service that uses it in the same change. Adding optional fields (with a default value) is backwards-compatible.

---

## How services import from here

Each service's Docker image copies the `shared/` directory and sets `PYTHONPATH=/app`, so all services can import using:

```python
from shared.contracts.inference import GenerateRequest, GenerateResponse
from shared.contracts.orchestration import OrchestrationRequest
```
