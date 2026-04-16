FROM python:3.14-slim as builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


FROM python:3.14-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /install /usr/local

COPY api/ ./api/
COPY src/ ./src/
COPY models/ ./models/
COPY templates/ ./api/templates/

ENV PYTHONUNBUFFERED=1 \
    MODEL_DIR=/app/models \
    PYTHONPATH=/app

RUN mkdir -p $MODEL_DIR

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["python", "-m", "api.app"]