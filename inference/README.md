# Inference Service

## Purpose

The Inference Service is the only part of the system that calls language models. It's a switchboard operator: it receives a conversation from the Orchestration Service, routes it to the right model (based on agent role and task type), applies a fine-tuned adapter if needed, and returns the generated text.

The value of this service compounds when you have multiple model backends, remote vLLM instances, or need to A/B different backends per role thin setups don't benefit much.

## Core Components

- **InferenceBackend** (abstract base): Defines the interface every backend must implement. Callers don't need to know which concrete backend is active.

- **LocalVLLMBackend**: The main implementation. Calls vLLM (a fast inference server) over HTTP. Routes requests to either DeepSeek (better for code) or Mistral (better for language/reasoning) based on the agent role.

- **Adapter Registry** (`_AdapterRegistry`): Maintains a cache of available LoRA adapters. Periodically checks the vLLM server to discover newly trained adapters. Allows on-the-fly application of specialized fine-tuned weights.

- **Generate Route**: HTTP endpoint that accepts requests, forwards them to the backend, and returns the generated text.

- **gRPC Server**: A parallel high-performance server (binary protocol instead of JSON) for when Orchestration calls this service at high frequency.

## Key Interactions

### With Orchestration Service
- **Incoming**: GenerateRequest with:
  - Messages (conversation history)
  - Agent role (determines which model)
  - Adapter name (optional, for fine-tuned specialist behavior)
  - Generation parameters (max tokens, temperature, top_p)

- **Outgoing**: GenerateResponse with:
  - Generated text
  - Token count

### With vLLM Servers
- Calls vLLM's HTTP API endpoints to request text generation
- Queries `/v1/models` to check which adapters are loaded
- vLLM handles the actual model execution on GPU

## Important Concepts

### LoRA Adapters

Fine-tuned neural network weights layered on top of a frozen base model. Adapters are small (a few MB) and fast to train, yet teach the model new behaviors for specific tasks. Each agent role has an associated adapter:
- `coding_lora` — trained on coding examples
- `debugging_lora` — trained on debugging examples
- `planning_lora` — trained on planning examples

When the Training Service completes training, it writes an adapter file to disk. vLLM auto-discovers it, and this service can immediately apply it.

### Model Families

Two base models:
- **DeepSeek** (code-focused): Better at writing, understanding, and fixing code
- **Mistral** (language-focused): Better at reasoning, planning, analysis

The service routes each request to the appropriate model based on the agent role.

### Prompt Format for Adapters

Adapters were trained with a specific message format. If the prompt at inference doesn't match the training format exactly, the model produces poor output. The format is:
```
[SYSTEM]
<system message>

[USER]
<user message>

[RESPONSE]
<model output>
```

This must be byte-for-byte identical at training and inference time.

### Dual Interface (HTTP and gRPC)

The service listens on both HTTP and gRPC. HTTP is easier for testing and calling from other languages. gRPC is faster for high-frequency calls between services (smaller payload, binary protocol). Orchestration can use either based on configuration.

### Stop Tokens

When using adapters, the service sets stop tokens to prevent the model from generating spurious delimiters like `[SYSTEM]` or `[RESPONSE]` mid-answer. These are training artifacts that should never appear in the actual output.

### Temperature and Sampling

Generation parameters control randomness:
- **temperature**: Higher = more creative (0.2 is precise, 1.0 is very creative)
- **top_p**: Nucleus sampling; only consider the most likely tokens

### Benefits of the inference intermediary

  - **Protocol abstraction** — orchestration can switch between HTTP and gRPC (ORCHESTRATION__INFERENCE_USE_GRPC=true) without knowing which vLLM instance it's talking to. That toggle lives in one place.
  - **Adapter management** — the AdapterRegistry tracks which LoRA adapters are loaded on which vLLM server and refreshes them on a schedule. Without it, orchestration would need to know adapter-to-model mappings itself.
  - **Backend swappability** — factory.py dispatches to local_vllm, remote_vllm, etc. via the INFERENCE__*_BACKEND env vars. You can point a model family at a remote server without touching orchestration at all.
  - **Single healthcheck surface** — orchestration only needs to depend on inference being healthy, not on both vllm services individually

