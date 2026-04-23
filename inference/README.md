# Inference Service

## What it does

The Inference Service is the **only part of the system that talks to AI language models**. No other service calls a model directly — they all go through here.

Think of it as a switchboard operator: it receives a conversation (a list of messages) from the Orchestration Service, figures out which model to send it to, calls that model, and returns the response. Callers do not need to know anything about *which* model is running underneath or *where* it lives.

Here is what happens every time the Inference Service receives a request:

### Step 1 — Receive the request

The Orchestration Service sends a request that includes:
- The full conversation so far (a sequence of messages between the "system", the "user", and the AI assistant)
- Which **model family** to use (`deepseek` for code-focused work, `mistral` for planning and analysis)
- Which **agent role** is making the call (e.g. `coder`, `planner`, `critic`)
- An optional **adapter name** — a fine-tuned specialisation to apply on top of the base model
- Generation settings: how many tokens to produce, how creative to be (`temperature`), etc.

### Step 2 — Dispatch to the right backend

The service is configured to run in one of two modes:

#### Local mode (`INFERENCE__MODE=local`)

The service calls one of two **vLLM** servers running locally on GPUs:

| Model family | vLLM server | Default port |
|---|---|---|
| `deepseek` | `vllm-deepseek` | 8000 |
| `mistral` | `vllm-mistral` | 8001 |

vLLM is a high-performance engine for running open-source language models. It exposes an API that looks like OpenAI's, so the Inference Service just sends an HTTP request to it.

**LoRA adapters:** If an adapter name is provided (e.g. `coding_lora`), the service first checks whether that adapter is currently loaded on the vLLM server (vLLM caches a list of loaded adapters with a 2-minute refresh). If it is loaded, the request is sent with the model identifier set to `base_model/adapter_name`, which tells vLLM to apply the specialised fine-tuning. If the adapter is not loaded, the request falls back to the base model with a warning in the logs.

#### API mode (`INFERENCE__MODE=api`, the default)

The service routes the request through **LiteLLM**, a library that provides a uniform interface across dozens of commercial AI providers (Anthropic, OpenAI, Google, etc.). By default this points at Anthropic's `claude-3-5-haiku` model. A single environment variable change swaps the provider with no code changes needed.

### Step 3 — Return the response

The model's reply is returned to the Orchestration Service as a `GenerateResponse` containing the generated text and token count.

---

## gRPC interface

In addition to the standard HTTP API, the Inference Service also runs a **gRPC server** on port 50051. gRPC is a high-performance alternative to HTTP that uses a binary message format instead of JSON, which reduces overhead for high-frequency calls.

The Docker Compose configuration enables this by default — the Orchestration Service is configured to use gRPC to talk to Inference, not HTTP. The HTTP endpoint still exists and can be used for testing or when gRPC is not needed.

---

## API

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/v1/generate` | Generate a response from the configured language model |
| `GET`  | `/health` | Liveness probe — also reports the current mode (`local` or `api`) |

### Request — `POST /v1/generate`

```json
{
  "model_family": "deepseek",
  "role": "coder",
  "messages": [
    {"role": "system", "content": "You are an expert Python developer."},
    {"role": "user", "content": "Write a thread-safe singleton."}
  ],
  "adapter_name": "coding_lora",
  "max_tokens": 2048,
  "temperature": 0.2,
  "top_p": 0.95,
  "session_id": "optional-uuid"
}
```

| Field | Required | Description |
|-------|----------|-------------|
| `model_family` | Yes | `deepseek` or `mistral` (only matters in local mode) |
| `role` | Yes | Agent role name, used for per-role model overrides in API mode |
| `messages` | Yes | The conversation history as a list of `{role, content}` objects |
| `adapter_name` | No | Name of a LoRA adapter to apply (local mode only). Omit to use the base model |
| `max_tokens` | No | Maximum number of tokens to generate (default 1024) |
| `temperature` | No | Randomness (0 = deterministic, 1 = very creative; default 0.2) |
| `top_p` | No | Nucleus sampling threshold (default 0.95) |
| `session_id` | No | Optional ID for log tracing |

### Response

```json
{
  "content": "Here is a thread-safe singleton implementation...",
  "model_family": "deepseek",
  "role": "coder",
  "tokens_generated": 412,
  "session_id": "optional-uuid"
}
```

---

## Configuration

All configuration is supplied via environment variables with the `INFERENCE__` prefix.

| Variable | Default | Description |
|----------|---------|-------------|
| `INFERENCE__MODE` | `api` | `local` to use vLLM servers; `api` to use a cloud AI provider |
| `INFERENCE__DEEPSEEK_URL` | `http://vllm-deepseek:8000` | URL of the DeepSeek vLLM server (local mode only) |
| `INFERENCE__MISTRAL_URL` | `http://vllm-mistral:8001` | URL of the Mistral vLLM server (local mode only) |
| `INFERENCE__DEEPSEEK_MODEL_ID` | `deepseek-ai/deepseek-coder-6.7b-instruct` | The model ID loaded on the DeepSeek vLLM server |
| `INFERENCE__MISTRAL_MODEL_ID` | `mistralai/Mistral-7B-Instruct-v0.1` | The model ID loaded on the Mistral vLLM server |
| `INFERENCE__API_PROVIDER` | `anthropic` | LiteLLM provider prefix (e.g. `openai`, `google`) |
| `INFERENCE__API_MODEL` | `claude-3-5-haiku-20241022` | Full LiteLLM model string |
| `INFERENCE__API_KEY` | *(empty)* | API key for the cloud provider (e.g. set via `ANTHROPIC_API_KEY`) |
| `INFERENCE__REQUEST_TIMEOUT_SECONDS` | `120.0` | Seconds to wait for a model response before giving up |
| `INFERENCE__PORT` | `8010` | Port this HTTP service listens on |
| `INFERENCE__GRPC_PORT` | `50051` | Port the gRPC server listens on |

Per-role model overrides can also be set via `INFERENCE__ROLE_MODEL_OVERRIDES` as a JSON object to send specific agent roles to different models in API mode.

---

## What it talks to

| Downstream | How | Why |
|---|---|---|
| vLLM servers (`vllm-deepseek`, `vllm-mistral`) | HTTP (OpenAI-compatible API) | Run open-source models in local mode |
| Cloud AI provider (Anthropic, etc.) | HTTP via LiteLLM | Generate responses in API mode |

The Inference Service has no database and stores no data.

---

## Running locally

```bash
# API mode (no GPU required — uses a cloud provider)
docker compose -f docker/compose.yml up inference

# Local mode (requires GPU and vLLM servers)
docker compose -f docker/compose.yml --profile local up inference vllm-deepseek vllm-mistral
```

---

## Structure

```
app/
  config/settings.py      — all environment variable config
  backends/
    base.py               — shared interface that both backends implement
    local_vllm.py         — calls vLLM via HTTP; manages adapter availability cache
    api.py                — calls any cloud provider via LiteLLM
    factory.py            — creates the right backend based on INFERENCE__MODE
  grpc/
    server.py             — gRPC server (runs alongside the HTTP server)
  routes/generate.py      — POST /v1/generate and GET /health
  main.py                 — FastAPI application entry point
```

## Local model setup

Models must be downloaded to your HuggingFace cache before running locally:

```bash
huggingface-cli download deepseek-ai/deepseek-coder-6.7b-instruct
huggingface-cli download mistralai/Mistral-7B-Instruct-v0.1
```

The vLLM containers mount `HF_CACHE_DIR` (default `~/.cache/huggingface`) so
no manual copying is needed.
