# Build from the repository root:  docker build -t abacus-bot .
# The widget directory sits beside backend/, so the build context must be the repo root.
# Only used for a Render/Fly/self-hosted deploy — Vercel doesn't use this Dockerfile,
# it builds api/index.py directly via its Python runtime.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

COPY backend/ backend/
COPY widget/ widget/

WORKDIR /app/backend

EXPOSE 8000

HEALTHCHECK --interval=60s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -fsS http://localhost:8000/ || exit 1

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
