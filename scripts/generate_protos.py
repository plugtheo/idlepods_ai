#!/usr/bin/env python3
"""
Generate gRPC Python stubs from shared/proto/inference.proto.

Run from the project root:
    python scripts/generate_protos.py

Requires grpcio-tools to be installed:
    pip install grpcio-tools
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
PROTO_DIR = ROOT / "shared" / "proto"
OUT_DIR = ROOT / "shared" / "grpc_stubs"

OUT_DIR.mkdir(parents=True, exist_ok=True)

proto_files = list(PROTO_DIR.glob("*.proto"))
if not proto_files:
    print("No .proto files found in", PROTO_DIR)
    sys.exit(1)

cmd = [
    sys.executable, "-m", "grpc_tools.protoc",
    f"-I{PROTO_DIR}",
    f"--python_out={OUT_DIR}",
    f"--grpc_python_out={OUT_DIR}",
] + [str(p) for p in proto_files]

print("Running:", " ".join(cmd))
result = subprocess.run(cmd)
if result.returncode != 0:
    print("Proto generation failed.  Install grpcio-tools and retry.")
    sys.exit(result.returncode)

print(f"Stubs written to {OUT_DIR}")
