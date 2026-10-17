FROM python:3.10-slim

WORKDIR /app

# Install curl for health checks
RUN apt-get update && apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY src .

# Single-row predictions are slower with multithreaded BLAS/OpenMP (thread oversubscription);
# parallelism comes from uvicorn workers instead.
ENV WEB_CONCURRENCY=2 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]