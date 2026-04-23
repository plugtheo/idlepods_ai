#!/bin/bash
set -e

# WSL2 + Docker: triton needs an unversioned libcuda.so for its JIT CUDA driver compilation,
# but the NVIDIA WSL2 driver only exposes libcuda.so.1.
# Create a versioned→unversioned symlink at startup in a tmpfs location.
CUDA_LIB=$(find /usr/lib/wsl/drivers/ -name 'libcuda.so.1' 2>/dev/null | head -1)
if [ -n "$CUDA_LIB" ]; then
    mkdir -p /tmp/cuda_stubs
    ln -sf "$CUDA_LIB" /tmp/cuda_stubs/libcuda.so
    export LIBRARY_PATH="/tmp/cuda_stubs:${LIBRARY_PATH:-}"
fi

exec "$@"
