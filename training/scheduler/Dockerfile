FROM python:3.12-slim

# Docker CLI + compose plugin so the scheduler can `docker compose run` the training service.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl gnupg \
 && install -m 0755 -d /etc/apt/keyrings \
 && curl -fsSL https://download.docker.com/linux/debian/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg \
 && chmod a+r /etc/apt/keyrings/docker.gpg \
 && echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/debian bookworm stable" > /etc/apt/sources.list.d/docker.list \
 && apt-get update && apt-get install -y --no-install-recommends docker-ce-cli docker-compose-plugin \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY training/scheduler/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Reuse threshold logic + settings from the training service tree.
COPY training/app/__init__.py            /app/app/__init__.py
COPY training/app/config/__init__.py     /app/app/config/__init__.py
COPY training/app/config/settings.py     /app/app/config/settings.py
COPY training/app/utils/__init__.py      /app/app/utils/__init__.py
COPY training/app/utils/experience_reader.py /app/app/utils/experience_reader.py
COPY shared/                             /app/shared/
COPY training/scheduler/scheduler.py     /app/scheduler.py

ENV PYTHONPATH=/app

CMD ["python", "/app/scheduler.py"]
