# Training Service

## Purpose

The Training Service is the self-improvement engine. It watches the stream of completed pipeline runs ("experiences") and decides when to fine-tune the specialist adapters. When enough diverse, high-quality examples have been gathered, it launches a training job to improve the model at specific tasks.

Fine-tuning in this system means LoRA adaptation: adding a small layer of specialized weights on top of a frozen base model. This is fast (minutes to hours on GPU) and doesn't require retraining the whole model.

## Core Components

- **Trigger Endpoint**: HTTP endpoint that receives new experience metadata from the Orchestration Service. This is the signal to evaluate whether training conditions are met.

- **Batch Evaluator**: Checks if the accumulated batch of experiences meets three quality criteria:
  - **Size**: At least N examples (default 50)
  - **Diversity**: At least P% unique prompts (prevents training on near-duplicates; default 60%)
  - **Score spread**: Difference between best and worst scores is large enough (gives the model a signal; default 0.15)
  
  All three must pass before training starts.

- **Trainer Subprocess**: Launches the actual fine-tuning in a separate Python process. Keeps the HTTP service responsive while training runs (potentially hours). Uses HuggingFace libraries (transformers, peft) to train LoRA adapters.

- **Adapter Writer**: Saves trained adapter weights to disk. vLLM auto-discovers them within a few seconds.

## Data Flow

```
Orchestration Service (each completed pipeline run)
  ↓
Experience Event (role, prompt, output, score, intent, complexity)
  ↓
Training Service Trigger
  ├─ Append to experience log (JSONL file)
  ├─ Evaluate batch readiness:
  │  ├─ Count examples
  │  ├─ Check diversity (unique prompt ratio)
  │  └─ Check score spread
  │
  ├─ If ready to train:
  │  ├─ Filter low-quality examples (score < 0.65)
  │  ├─ Format as training data (system/user/response format)
  │  └─ Launch trainer subprocess
  │     ├─ Load base model
  │     ├─ Load existing adapter (if any)
  │     ├─ Fine-tune on new examples (LoRA)
  │     └─ Save adapter to disk
  │
  └─ Return status (queued, training, or skipped)
  ↓
vLLM adapter registry (auto-discovers new adapter)
  ↓
Next Inference call can use updated adapter
```

## Key Interactions

### With Orchestration Service
- **Incoming**: Experience events (role, prompt, output, quality score, metadata)
- **Pattern**: Fire-and-forget; Training Service responds immediately, launches training in background if criteria are met

### With Data Storage
- **Input**: Reads experience batch from JSONL file
- **Output**: Formatted training data (system/user/response triplets) passed to trainer subprocess
- **Persistence**: Experience log accumulates across all requests

### With Inference Service (Indirect)
- Once training completes, adapter is saved to disk
- vLLM's adapter registry auto-discovers it
- Next request to Inference Service can load and use the updated adapter

## Important Concepts

### LoRA Adapters

A small set of additional weights (typically 1-2% of base model size) trained on top of a frozen base model. Benefits:
- Fast to train (hours vs. days for full model)
- Can have multiple adapters on same base model
- Low inference overhead

Each agent role gets its own adapter (coding_lora, debugging_lora, planning_lora, etc.).

### Experience Batches

Experiences are grouped by agent role. Training is per-role, not global. Example:
- Batch for coding_lora: all prompts where an agent's role was "coder"
- Batch for planning_lora: all prompts where an agent's role was "planner"

This allows each specialist to improve independently.

### Diversity Evaluation

Not all data is equally valuable for training. The evaluator prevents overfitting by checking:
- **Minimum size**: Enough examples to learn patterns (default 50)
- **Diversity ratio**: Enough unique prompts (default 60%); prevents training on near-duplicates
- **Score spread**: Large enough difference between best/worst scores (default 0.15); ensures the model sees both good and mediocre examples to learn the difference

### Quality Filtering

Before training, examples with a score below 0.65 are excluded. The model should learn from successful outputs, not failures.

### Subprocess Isolation

Training runs in a separate Python process so:
- The HTTP service remains responsive
- Resource usage is isolated
- Training can fail without crashing the service
- Multiple services can be deployed without contention

### Prompt Fingerprinting

To detect unique prompts, the first 120 characters (lowercased, spaces collapsed) are hashed. If the hash is distinct, it counts as a unique prompt. This prevents training on trivial variations of the same question.

### Training Data Format

Training data for LoRA adapters must match the exact format used at inference:
```
[SYSTEM]
<system message>

[USER]
<user message>

[RESPONSE]
<expected output>
```

This format is enforced during both training and inference; any mismatch causes poor output.

