FROM python:3.11-slim

WORKDIR /app

# Install dependencies (includes grpcio-tools for proto compilation)
COPY inference/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Copy shared contracts and proto definitions
COPY shared/ /app/shared/

# Generate gRPC Python stubs from the proto definition
RUN mkdir -p /app/shared/grpc_stubs && \
    python -m grpc_tools.protoc \
      -I /app/shared/proto \
      --python_out=/app/shared/grpc_stubs \
      --grpc_python_out=/app/shared/grpc_stubs \
      /app/shared/proto/inference.proto

# Copy service code
COPY inference/app/ /app/services/inference/app/

ENV PYTHONPATH=/app

EXPOSE 8010
EXPOSE 50051

# Single worker: inference is GPU-bound so multiple Python workers
# don't add throughput, and --workers 1 avoids two processes racing
# to bind the gRPC port (50051) on startup.
CMD ["uvicorn", "services.inference.app.main:app", \
     "--host", "0.0.0.0", "--port", "8010", "--workers", "1"]
