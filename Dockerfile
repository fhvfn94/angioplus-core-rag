
FROM python:3.12-slim-bookworm

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt
# Development image.
# Project source is mounted by docker-compose via:
# volumes:
#   - .:/app
#
# Production deployments may instead COPY the project into the image.
