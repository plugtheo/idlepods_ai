# Use a PyTorch image with CUDA so the training subprocess has GPU access.
# The CUDA 12.1 runtime matches the GPU drivers expected on most modern hosts.
# If your host has a different CUDA version, replace cuda12.1 accordingly.
FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime

WORKDIR /app

# gcc/g++ are needed by triton (bundled with torch 2.7+) to JIT-compile CUDA driver utils.
RUN apt-get update && apt-get install -y --no-install-recommends gcc g++ && rm -rf /var/lib/apt/lists/*

# Install training dependencies (torch is already in base image; pip will skip it)
COPY training/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Unsloth 2026.x needs torchao>=0.17 which uses torch.utils._pytree.register_constant
# (added in torch 2.7.0).  PyTorch cu124 maxes out at 2.6.0, so we use cu126.
# The host GPU driver (595.79 / CUDA 13.2) is forward-compatible with cu126.
RUN pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cu126 \
    "torch==2.7.0" "torchvision==0.22.0" "torchaudio==2.7.0" \
    && pip install --no-cache-dir "unsloth[cu126-torch270]"

# Create a GNU ld linker script stub so triton can compile its CUDA driver shim on
# WSL2 (where the real libcuda.so.1 is visible to the runtime linker but the build-time
# linker needs an unversioned libcuda.so in one of the -L directories).
RUN echo 'GROUP ( AS_NEEDED ( libcuda.so.1 ) )' > \
    /opt/conda/lib/python3.10/site-packages/triton/backends/nvidia/lib/libcuda.so

COPY shared/ /app/shared/
COPY training/app/ /app/services/training/app/
# Include the monolith trainer so trainer_entry.py can import it
COPY training/training/ /app/training/
COPY training/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENV PYTHONPATH=/app

VOLUME ["/data"]
EXPOSE 8013

ENTRYPOINT ["/entrypoint.sh"]
CMD ["uvicorn", "services.training.app.main:app", \
     "--host", "0.0.0.0", "--port", "8013", "--workers", "1"]
