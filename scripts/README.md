# Scripts

This folder contains utility scripts for the project.

---

### `generate_protos.py`

**What it does:** Compiles the Protocol Buffer definition file (`shared/proto/inference.proto`) into Python code that the Orchestration and Inference services use to communicate via gRPC.

**When to run it:** Once, after cloning the repository — and again any time `shared/proto/inference.proto` is modified.

**What it produces:** Two files in `shared/grpc_stubs/`:
- `inference_pb2.py` — data classes representing the gRPC message types
- `inference_pb2_grpc.py` — the gRPC service stub and servicer classes

```bash
python scripts/generate_protos.py
```
