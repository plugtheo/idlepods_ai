FROM python:3.11-slim

WORKDIR /app

COPY gateway/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY shared/ /app/shared/
COPY gateway/app/ /app/services/gateway/app/

ENV PYTHONPATH=/app

EXPOSE 8080

CMD ["uvicorn", "services.gateway.app.main:app", \
     "--host", "0.0.0.0", "--port", "8080", "--workers", "2"]
