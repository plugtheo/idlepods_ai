FROM python:3.11-slim

WORKDIR /app

COPY experience/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY shared/ /app/shared/
COPY experience/app/ /app/services/experience/app/

ENV PYTHONPATH=/app

VOLUME ["/data"]
EXPOSE 8012

CMD ["uvicorn", "services.experience.app.main:app", \
     "--host", "0.0.0.0", "--port", "8012", "--workers", "1"]
