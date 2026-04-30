#!/usr/bin/env python3
"""
Generate gRPC Python stubs from shared/proto/inference.proto.

Run from the project root:
    python scripts/generate_protos.py

Requires grpcio-tools to be installed:
    pip install grpcio-tools
"""

import hashlib
import re
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

# Strip non-deterministic Protobuf Python version comment so generated files
# are byte-identical across grpcio-tools versions (same proto source → same hash).
for _stub_file in OUT_DIR.glob("inference_pb2*.py"):
    _text = _stub_file.read_text(encoding="utf-8")
    _text = re.sub(r"# Protobuf Python Version: [^\n]+\n", "", _text)
    _stub_file.write_text(_text, encoding="utf-8")

# Write a version file containing the sha256 hash of the proto source.
# Both the inference server and orchestration client import this to verify
# schema compatibility at startup.
_proto_source = (PROTO_DIR / "inference.proto").read_bytes()
_schema_hash = hashlib.sha256(_proto_source).hexdigest()
(OUT_DIR / "_version.py").write_text(f'PROTO_SCHEMA_HASH = "{_schema_hash}"\n', encoding="utf-8")
print(f"Proto schema hash written to {OUT_DIR / '_version.py'}: {_schema_hash[:12]}…")
