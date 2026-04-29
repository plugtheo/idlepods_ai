# Shared Module

## Purpose

The Shared module is the contract layer — the single source of truth for how all services communicate. It defines the data models (Pydantic classes) that pass between services. By keeping definitions in one place, all services speak the same language and avoid reimplementing the same structures with drift.

## Core Components

- **Contracts** (`contracts/`): Pydantic models that define request and response shapes for each service-to-service interaction. Examples:
  - `orchestration.py` — OrchestrationRequest (what Gateway sends), OrchestrationResponse (what Orchestration returns)
  - `inference.py` — GenerateRequest (what Orchestration sends), GenerateResponse (what Inference returns)
  - `context.py` — BuiltContext (enriched prompt context)
  - `experience.py` — ExperienceEvent (completed pipeline run)

- **QueryRouter** (`routing/query_router.py`): A shared prompt classifier. Both Gateway and Orchestration import it, ensuring consistent routing behavior everywhere.

- **gRPC Stubs** (`grpc_stubs/`): Auto-generated Python bindings for the binary protocol between Orchestration and Inference. Generated from `.proto` files; not hand-edited.

## Why It Exists

Each service runs in its own container, but they must agree on data formats. Without a shared package, each would maintain its own copy of these models — and they would drift as the system evolves:

- Wrong field names go undetected until runtime
- Type mismatches (string vs. int) cause silent failures
- Adding a field requires coordinating changes across multiple services

Centralizing here means:
- Type errors are caught at import time
- Single source of truth for every contract
- Changes ripple automatically to all callers

## Important Concepts

### Pydantic Models

Pydantic enforces data validation automatically. When a service receives JSON and creates a Pydantic model from it:
- Required fields must be present
- Field types are validated (string "123" becomes int 123 if the field expects int)
- Invalid data raises a clear error

This means every service boundary has automatic input validation at no extra cost.

### Contract Rules

- **No service imports**: This module doesn't import from gateway, inference, orchestration, etc. It's purely data definitions.
- **No I/O**: No HTTP calls, database access, or file reads. Pure data models.
- **Breaking changes require coordination**: Renaming or removing a field means updating every service that uses it in the same change. Adding optional fields (with defaults) is safe.

### Communication Patterns

Services import and use these models for both HTTP and gRPC:
- **HTTP**: Pydantic automatically serializes to/from JSON
- **gRPC**: Protocol Buffers serialize to compact binary format

Example: `from shared.contracts.inference import GenerateRequest, GenerateResponse`

### gRPC Protocol

gRPC is a high-performance alternative to HTTP that uses binary serialization (Protocol Buffers). It's faster for frequent inter-service calls. The protocol definitions live in `.proto` files; Python bindings are auto-generated.

### Routing Centralization

QueryRouter is shared so Gateway and Orchestration always classify prompts the same way. If one service had its own classifier, they could silently diverge and produce inconsistent routing decisions.

### Experience Recording

The ExperienceEvent model captures everything learned from a completed pipeline run: the original prompt, final output, all agent outputs, scores, and metadata. This data is used both for few-shot learning (in future requests) and as training data for the Training Service.

