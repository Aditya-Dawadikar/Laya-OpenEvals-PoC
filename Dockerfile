# Experiment runner: Laya (CPU) + the OpenEvals adapter under test, installed from uv.lock.
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.11.8 /uv /usr/local/bin/uv

RUN apt-get update \
 && apt-get install -y --no-install-recommends git ca-certificates \
 && rm -rf /var/lib/apt/lists/*

ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_CACHE_DIR=/tmp/uv-cache \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    USE_TF=0 \
    HF_HOME=/cache/hf

WORKDIR /app

# Dependencies first (cached layer), then the project itself
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project && rm -rf /tmp/uv-cache
COPY README.md ./
COPY src ./src
RUN uv sync --frozen --no-dev && rm -rf /tmp/uv-cache

CMD ["laya-poc", "run"]
