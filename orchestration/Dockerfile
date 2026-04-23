FROM python:3.11-slim

WORKDIR /app

COPY orchestration/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY shared/ /app/shared/

# Generate gRPC Python stubs so the optional gRPC inference client works
RUN mkdir -p /app/shared/grpc_stubs && \
    python -m grpc_tools.protoc \
      -I /app/shared/proto \
      --python_out=/app/shared/grpc_stubs \
      --grpc_python_out=/app/shared/grpc_stubs \
      /app/shared/proto/inference.proto

COPY orchestration/app/ /app/services/orchestration/app/

ENV PYTHONPATH=/app

EXPOSE 8001

CMD ["uvicorn", "services.orchestration.app.main:app", \
     "--host", "0.0.0.0", "--port", "8001", "--workers", "1"]
