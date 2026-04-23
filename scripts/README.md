# Scripts

This folder contains utility scripts for the project. **Important:** most scripts in this folder are left over from a previous monolithic architecture and are currently broken. See the status column below.

---

## Script status

| Script | Status | Description |
|--------|--------|-------------|
| `generate_protos.py` | **Working** | Generates the Python gRPC binding files from the `.proto` definition |
| `run_agent_experience_pipeline.py` | **Broken** | References old monolithic modules that no longer exist |
| `verify_system_prerequisites.py` | **Broken** | References old monolithic modules that no longer exist |
| `training_analysis_report.py` | **Broken** | Hardcodes a specific dated filename that only exists on one machine |
| `verify_synthetic_pipeline.py` | **Broken** | References old monolithic modules that no longer exist |

---

## Working scripts

### `generate_protos.py`

**What it does:** Compiles the Protocol Buffer definition file (`shared/proto/inference.proto`) into Python code that the Orchestration and Inference services use to communicate via gRPC.

**When to run it:** Once, after cloning the repository — and again any time `shared/proto/inference.proto` is modified.

**What it produces:** Two files in `shared/grpc_stubs/`:
- `inference_pb2.py` — data classes representing the gRPC message types
- `inference_pb2_grpc.py` — the gRPC service stub and servicer classes

```bash
python scripts/generate_protos.py
```

---

## Broken scripts (legacy — do not use)

The following scripts were written for a previous version of this project that used a single monolithic application rather than the current microservices architecture. They all import from `app.agents`, `app.datasets`, `app.self_improvement`, `app.framework_adapter`, or similar module paths that do not exist in the current codebase. Running any of them will immediately produce a `ModuleNotFoundError`.

They are kept for historical reference only.

### `run_agent_experience_pipeline.py`
Originally ran an end-to-end AutoGen-based agent loop with a monolithic experience pipeline. Not applicable to the microservices design.

### `verify_system_prerequisites.py`
Originally verified that the old monolithic agent classes, training modules, and a specific GPU checkpoint were available. The check for `data/lora_checkpoints/deepseek-lora-adapter` referenced a path that only existed on the original developer's machine.

### `training_analysis_report.py`
Originally generated a training analysis report from a specific hardcoded JSONL filename with a timestamp in the name (e.g. `training_dataset_20260324_203553_(461samples).jsonl`). That file only exists on the machine that generated it.

### `verify_synthetic_pipeline.py`
Originally verified a synthetic data generation pipeline (`SyntheticDataGenerator`, `LoRATrainingOrchestrator`) that was part of the old monolithic architecture. These classes do not exist in the current system.

---

## data/sandbox/ note

The `data/sandbox/` directory contains `cli.py` and `cli_test.py`, which are development artifacts from the old architecture. They share the same broken import problems as the scripts above and should not be run.
