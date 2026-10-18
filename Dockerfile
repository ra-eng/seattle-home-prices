FROM python:3.10-slim

WORKDIR /app

# Install curl for health checks
RUN apt-get update && apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

RUN useradd --create-home --uid 1000 app
COPY --chown=app:app src .
USER app

# Single-row predictions are slower with multithreaded BLAS/OpenMP (thread oversubscription);
# parallelism comes from uvicorn workers instead.
ENV WEB_CONCURRENCY=2 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1

HEALTHCHECK --interval=10s --timeout=3s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]