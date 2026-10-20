FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN groupadd -r appgroup && useradd -r -g appgroup -d /app -s /sbin/nologin appuser

COPY pyproject.toml README.md ./
COPY src/ ./src/
COPY config/ ./config/

RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-build-isolation . && \
    python -m src.models.trainer && \
    chown -R appuser:appgroup /app

USER appuser

EXPOSE 8000

CMD ["uvicorn", "src.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]