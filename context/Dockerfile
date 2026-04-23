FROM python:3.11-slim

WORKDIR /app

COPY context/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY shared/ /app/shared/
COPY context/app/ /app/services/context/app/

ENV PYTHONPATH=/app

EXPOSE 8011

CMD ["uvicorn", "services.context.app.main:app", \
     "--host", "0.0.0.0", "--port", "8011", "--workers", "1"]
