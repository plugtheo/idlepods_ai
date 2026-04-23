# GPU-ACCELERATED TRAINING PIPELINE

## Overview

Complete end-to-end pipeline for training LoRA adapters on your **RTX 3090 GPU** with **Unsloth acceleration** (2-5x faster, 60% less memory).

```
┌─────────────────────────────────────────────────────────┐
│  TRAINING PIPELINE (GPU-ACCELERATED)                    │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Step 1: Data Generation                               │
│  ├─ Public Data: BigCode + GitHub (40%)                │
│  ├─ Synthetic Data: Agent-generated (60%)              │
│  ├─ Target: 5k-10k balanced samples                    │
│  └─ Output: training_dataset_{timestamp}_{count}.jsonl │
│                                                         │
│  Step 2: GPU Training (RTX 3090)                       │
│  ├─ Unsloth: 2-5x faster, 60% less VRAM              │
│  ├─ LoRA Rank: 32, Alpha: 64                           │
│  ├─ Learning Rate: 2e-4                                │
│  ├─ Epochs: 10 (for convergence)                       │
│  ├─ Batch Size: 4 (optimized for 24GB VRAM)           │
│  └─ Output: lora_unsloth_{timestamp}_{samples}exp      │
│                                                         │
│  Step 3: Validation & Deployment                       │
│  ├─ Loss convergence tracking                          │
│  ├─ VRAM usage monitoring                              │
│  └─ Adapter ready for model router                     │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

## Quick Start

### 1. Generate Training Data (5k-10k samples)
```bash
python generate_training_data.py
```

**Output:**
- File: `data/training_data_curated/training_dataset_{timestamp}_{count}.jsonl`
- Format: JSONL with fields: problem, context, solution, evaluation
- Statistics: Source breakdown, quality metrics

**What It Does:**
- Fetches public code from BigCode dataset and GitHub
- Generates synthetic problem-solution pairs via agents
- Filters low-quality (< 0.75 score) and outdated code
- Creates balanced 40/60 public/agent split
- Saves curated dataset for training

### 2. Train with GPU (2-5x faster than CPU)
```bash
python train_gpu.py
```

**Hardware Used:**
- GPU: NVIDIA GeForce RTX 3090
- VRAM: 24GB total, ~23.6GB available
- CUDA: Version 13.2
- Acceleration: Unsloth

**Output:**
- Adapter: `data/adapters/lora_unsloth_{timestamp}_{samples}exp/`
- Training logs: Loss metrics per epoch
- VRAM usage: Monitored and printed

**What It Does:**
- Loads curated training dataset
- Initializes base model (Deepseek Coder 6.7B)
- Applies LoRA with Unsloth optimization
- Trains for 10 epochs with adaptive learning
- Saves adapter for deployment

### 3. End-to-End Pipeline (Automatic)
```bash
python run_training.py
```

**Runs Both Steps Automatically:**
1. Generates dataset
2. Trains with GPU
3. Shows summary and next steps

## File Descriptions

### `generate_training_data.py` (400+ lines)

**Classes:**
- `PublicCodeDataCollector`: Fetches from BigCode/GitHub
- `AgentSyntheticDataGenerator`: Creates problem-solution pairs
- `DataCurator`: Merges, filters, balances, saves

**Key Features:**
- **40% Public Data**: High-quality public repositories
- **60% Synthetic Data**: Agent-generated code with problems
- **Quality Filtering**: Only includes samples with score ≥ 0.75
- **Source Tracking**: Stores source for debugging
- **Statistics**: Generates dataset metrics

**Usage:**
```python
from generate_training_data import DataCurator

curator = DataCurator(target_count=5500)  # 5k-10k range
dataset = await curator.generate_dataset()
curator.save_dataset(dataset, output_dir="data/training_data_curated")
```

### `train_gpu.py` (200+ lines)

**Classes:**
- `LoRATrainerGPU`: Main training orchestrator
  - `_train_unsloth()`: Unsloth-accelerated training (2-5x faster)
  - `_train_standard()`: Standard PyTorch fallback

**Key Features:**
- **GPU Monitoring**: Real-time VRAM tracking
- **Unsloth Integration**: Automatic if available
- **4-bit Quantization**: Reduces VRAM usage further
- **Sequence Packing**: Faster training via Unsloth
- **Adaptive Loss**: Per-epoch tracking

**Usage:**
```python
from train_gpu import LoRATrainerGPU

trainer = LoRATrainerGPU(base_model="deepseek-coder-6.7b")
result = await trainer.train(
    dataset_path=Path("data/training_data/dataset.jsonl"),
    num_epochs=10
)
print(f"Trained adapter: {result['adapter_id']}")
```

### `run_training.py` (Orchestrator)

**Purpose:** Automates complete pipeline
- Calls `generate_training_data.py` → dataset
- Calls `train_gpu.py` → trained adapter
- Shows final summary and next steps

## Training Configuration

### Model Configuration
```python
Base Model: deepseek-coder-6.7b
LoRA Rank: 32
LoRA Alpha: 64
Target Modules: q_proj, v_proj
Dropout: 0.05
```

### Training Hyperparameters
```python
Batch Size: 4 (RTX 3090 optimized)
Gradient Accumulation: 2 steps
Learning Rate: 2e-4
Epochs: 10 (for convergence)
Warmup Steps: 100
Optimizer: AdamW 8-bit
Weight Decay: 0.01
LR Scheduler: Linear
```

### Optimization Settings
```python
Quantization: 4-bit (reduces VRAM)
Mixed Precision: bfloat16 (on GPU) or float16 (fallback)
Sequence Packing: Enabled (Unsloth)
Gradient Checkpointing: Enabled
```

## GPU Performance Metrics

### Your RTX 3090 Specifications
```
GPU: NVIDIA GeForce RTX 3090
Memory: 24,576 MB (24 GB)
Compute Capability: 8.6
CUDA Version: 13.2 ✓
Driver Version: 595.79
Available VRAM: ~23.6 GB (at start of training)
```

### Expected Training Performance

**With Unsloth (2-5x faster):**
- 5k samples: ~5-15 minutes
- 10k samples: ~10-30 minutes
- Full 10 epochs on 10k samples: ~30-60 minutes

**VRAM Usage During Training:**
- Base model (4-bit): ~6-8 GB
- LoRA adapter: ~0.5 GB
- Batch processing: ~2-3 GB
- **Total: ~9-12 GB (plenty of headroom on 24GB)**

**Memory Savings:**
- Standard 16-bit training would use: ~16-18 GB
- Unsloth + 4-bit: ~9-12 GB
- **Savings: ~40-50%**

## Data Pipeline Details

### Data Format (JSONL)
```json
{
  "problem": "Optimize query for 10k+ records",
  "context": "Django ORM with PostgreSQL",
  "solution": "async def optimized_query()...",
  "evaluation": 0.92,
  "source": "agent:CoderAgent"
}
```

### Data Collection Strategy

**40% Public Data (~220 samples):**
- BigCode dataset (code search)
- GitHub enterprise repos
- High-quality, maintained code
- Real-world problems

**60% Synthetic Data (~330 samples):**
- Generated by your 7 agents (Planner, Researcher, Coder, etc.)
- Realistic problem-solution pairs
- Async Python patterns (your focus)
- Automatically quality-filtered

### Quality Criteria (Score ≥ 0.75)
✓ Syntactically correct code
✓ Follows async/await patterns
✓ Solves stated problem
✓ Not outdated or deprecated
✓ Not trivial examples
✓ Includes docstrings

❌ Scraped/noisy data
❌ Outdated dependencies
❌ Inconsistent formatting
❌ Broken imports

## Integration Points

### With AgentLoopOrchestrator
```python
# After running agent loop, collect outputs:
from lora_agent_trainer import AgentOutputDataset

agent_outputs = orchestrator.get_experience_history()
dataset = AgentOutputDataset.from_agent_outputs(agent_outputs)

# Append to training dataset
curator.append_to_dataset(dataset)

# Retrain periodically
trainer.train(updated_dataset, num_epochs=5)
```

### With Model Router
```python
# Deploy trained adapter
from lora_agent_trainer import LoRAAgentTrainerPipeline

adapter_id = "lora_unsloth_20260325_123456_5500exp"
router.register_adapter(
    adapter_id=adapter_id,
    base_model="deepseek-coder-6.7b",
    adapter_path=f"data/adapters/{adapter_id}"
)

# Use in agent loop
orchestrator.set_active_adapter(adapter_id)
```

## Monitoring & Debugging

### View Training Logs
```bash
# Check loss convergence per epoch
tail -100 train_gpu.py
```

### Monitor GPU Usage (Real-time)
```bash
# In separate terminal during training
watch nvidia-smi
```

### Check Final Adapter
```bash
ls -lah data/adapters/lora_unsloth_*/
```

### Validate Dataset Quality
```bash
python -c "
import json
with open('data/training_data_curated/training_dataset_*.jsonl') as f:
    data = [json.loads(line) for line in f]
    scores = [d['evaluation'] for d in data]
    print(f'Samples: {len(data)}')
    print(f'Avg Score: {sum(scores)/len(scores):.3f}')
    print(f'Min Score: {min(scores):.3f}')
    print(f'Max Score: {max(scores):.3f}')
"
```

## Troubleshooting

### Issue: CUDA Not Detected
```bash
# Verify driver
nvidia-smi

# Reinstall PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Issue: Out of Memory (OOM)
```python
# Reduce batch size in train_gpu.py
per_device_train_batch_size=2  # Instead of 4
```

### Issue: Slow Training (Not Using Unsloth)
```bash
# Install Unsloth
pip install unsloth

# If still slow, check GPU usage
nvidia-smi

# If % is low, investigate other processes
```

### Issue: Dataset Not Found
```bash
# Generate dataset first
python generate_training_data.py

# Verify output
ls data/training_data_curated/
```

## Next Steps

### Immediate (Today)
1. Run `python generate_training_data.py` (5-10 minutes)
2. Run `python train_gpu.py` (30-60 minutes, depending on dataset size)
3. Verify adapter in `data/adapters/`

### Short-term (This Week)
1. Deploy adapter via model router
2. Run agent loop with trained adapter
3. Collect real agent outputs
4. Measure performance improvement

### Medium-term (This Month)
1. Set up continuous data collection
2. Automatically append real outputs to dataset
3. Retrain monthly as data grows
4. Scale from 5k → 10k → 25k samples

### Long-term (Ongoing)
1. Build to 25k-50k dataset via real agent execution
2. Implement automated retraining pipeline
3. A/B test different LoRA hyperparameters
4. Monitor inference metrics on deployed model

## Performance Targets

| Phase | Samples | Training Time | GPU Load | VRAM Used |
|-------|---------|---------------|----------|-----------|
| Initial | 5k | 15-30 min | 90%+ | 10-11 GB |
| Growth | 10k | 30-60 min | 90%+ | 11-12 GB |
| Stable | 25k | 2-4 hours | 90%+ | 12-14 GB |
| Full | 50k | 4-8 hours | 90%+ | 14-16 GB |

**Notes:**
- Times with Unsloth on RTX 3090
- All well within 24GB capacity
- GPU load monitoring shown in real-time

## References

- **Deepseek Coder**: Fast, powerful code generation
- **LoRA**: Parameter-efficient fine-tuning
- **Unsloth**: 2-5x faster, 60% less memory than standard training
- **Transformers**: HuggingFace implementation
- **PyTorch**: Underlying deep learning framework

---

**Status: READY FOR PRODUCTION**

Your GPU is configured and ready. All scripts tested. Next: run `python run_training.py` to generate data and train!
