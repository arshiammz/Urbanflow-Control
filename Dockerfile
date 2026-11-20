FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY scripts ./scripts
COPY dashboards ./dashboards
COPY configs ./configs
COPY docs ./docs

ARG INSTALL_RL=0
RUN pip install --upgrade pip && pip install ".[api,dashboard]" \
    && if [ "$INSTALL_RL" = "1" ]; then pip install ".[rl]"; fi

EXPOSE 8000 8501

CMD ["uvicorn", "urbanflow.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
