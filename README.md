# FastAPI Machine Learning Model Deployment

This project implements a RESTful API using FastAPI to deploy a machine learning model for predicting home prices based on various features. The model is trained on real estate data and can provide predictions based on user input.

## Project Structure

```
mle-project-challenge-2026
├── create_model.py            # Trains the model and writes the artifacts
├── src
│   ├── main.py                # App setup: startup loading, logging, error handling, CORS
│   ├── api
│   │   ├── endpoints.py       # /health and /predict routes
│   │   ├── schemas.py         # Request/response models and input validation
│   │   └── resources.py       # Model artifacts loaded once at startup
│   ├── model
│   │   ├── Dockerfile         # Image that runs create_model.py
│   │   ├── model.pkl          # Serialized machine learning model
│   │   └── model_features.json # Features required for predictions
│   └── data
│       ├── kc_house_data.csv  # Training data for the model
│       ├── zipcode_demographics.csv # Demographic data for predictions
│       └── future_unseen_examples.csv # Examples for testing the API
├── scripts
│   └── benchmark.py           # Latency and load benchmarks
├── test
│   ├── conftest.py            # Shared fixtures
│   ├── unit/                  # In-process tests with TestClient
│   └── integration/           # Tests against the running container
├── requirements.txt           # Runtime dependencies
├── requirements-test.txt      # Test dependencies
├── Dockerfile                 # API image
├── docker-compose.test.yml    # API + integration test runner
└── Makefile                   # Test commands
```

## Setup Instructions

### Prerequisites

- Docker installed on your system
- Git (for cloning the repository)

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd mle-project-challenge-2026
```

### Step 2: Generate Model Artifacts (First Time Only)

Before running the API, you need to generate the model files. This only needs to be done once, or when you want to retrain the model.

**Build the model creation Docker image:**
```bash
docker build -f src/model/Dockerfile -t create-model .
```

**Run the container to generate model artifacts:**
```bash
docker run --rm -v "$(pwd)/src/model:/app/model" create-model
```

This will create `model.pkl` and `model_features.json` in the `src/model/` directory.

### Step 3: Build and Run the API

**Build the API Docker image:**
```bash
docker build -t mle-project-challenge-2026 .
```

**Run the API container:**
```bash
docker run -d -p 8000:8000 --name housing-api mle-project-challenge-2026
```

### Step 4: Access the API

Open your browser and go to `http://127.0.0.1:8000/docs` to view the interactive API documentation.

### Managing the Container

**Stop the container:**
```bash
docker stop housing-api
```

**Start the container again:**
```bash
docker start housing-api
```

**Remove the container:**
```bash
docker rm housing-api
```

**View container logs:**
```bash
docker logs housing-api
```

## Usage

To get predictions from the model, send a POST request to the `/predict` endpoint with the home features in JSON format. The interactive contract is available at `/docs`.

### API contract

**`POST /predict`**

| Field | Type | Rules |
|---|---|---|
| `zipcode` | string | Required, exactly 5 digits, must exist in the demographics data |
| `bedrooms` | integer | Optional, `>= 0` |
| `bathrooms`, `sqft_living`, `sqft_lot`, `floors`, `sqft_above`, `sqft_basement` | number | Optional, `>= 0` |

Unknown fields are ignored, so a full row of `future_unseen_examples.csv` can be sent as is.

```json
{"predicted_price": 450000.0}
```

| Status | When | Body |
|---|---|---|
| `200` | Prediction made | `{"predicted_price": float}` |
| `422` | Invalid input (missing or malformed zipcode, negative values, wrong types) | Pydantic validation details |
| `422` | Zipcode not in the demographics data | `{"detail": "Unknown zipcode: 00000"}` |
| `500` | Unexpected error (logged with stack trace on the server) | `{"detail": "Internal server error"}` |

**`GET /health`** returns `200 {"status": "healthy"}` once the model is loaded and `503 {"status": "loading"}` before that. The Docker image uses it as its `HEALTHCHECK`, and an orchestrator can use it as a readiness probe.

### Configuration

| Variable | Default | Purpose |
|---|---|---|
| `MODEL_DIR` | `model` | Folder with `model.pkl` and `model_features.json` |
| `DATA_DIR` | `data` | Folder with `zipcode_demographics.csv` |
| `LOG_LEVEL` | `INFO` | Python logging level |
| `CORS_ORIGINS` | `*` | Comma-separated list of allowed origins |
| `WEB_CONCURRENCY` | `2` (Docker) | Number of uvicorn worker processes |
| `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS` | `1` (Docker) | Native threads per prediction (see [Performance](#performance)) |

### Logging and monitoring

Logs go to stdout (`docker logs housing-api`), one line per event:

```
2026-10-18 10:00:00,123 INFO main resources loaded in 35 ms
2026-10-18 10:00:01,456 INFO api.endpoints prediction zipcode=98125 imputed=['bathrooms'] price=555500
2026-10-18 10:00:01,457 INFO main request method=POST path=/predict status=200 duration_ms=7.9
```

- **Request log**: method, path, status and latency of every request.
- **Prediction log**: which fields were imputed, to track how often clients send incomplete data.
- **Error log**: unexpected exceptions with full stack trace; the client only gets a generic 500.
- **Container health**: `docker ps` shows `healthy` / `unhealthy` from the `HEALTHCHECK`.

### Missing value imputation

Only `zipcode` is required. Any other field can be omitted or sent as `null`:

```bash
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" \
  -d '{"zipcode": "98042", "bedrooms": 3}'
```

An unknown `zipcode` returns `422`, since its demographics can't be looked up.

**Approach.** Missing values are filled by a `KNNImputer(n_neighbors=5, weights="distance")`, the first step of the model pipeline (`KNNImputer → RobustScaler → KNeighborsRegressor`). Because it's part of the pipeline, it is fitted on the training split only (no data leakage) and ships inside `model.pkl`. Neighbors are found using every feature that is present, including the zipcode demographics, so a missing value is estimated from similar homes in similar areas.

**Why KNN instead of mean/median** (`notebooks/imputation_experiment.ipynb`, 15% MCAR): KNN is more accurate on the features that drive price, because it uses the correlations between them:

| Feature | Mean RMSE | KNN RMSE | Change |
|---|---|---|---|
| sqft_above | 812 | 672 | -17% |
| bathrooms | 0.77 | 0.65 | -16% |
| floors | 0.54 | 0.45 | -15% |
| sqft_living | 924 | 896 | -3% |

Mean/median do better on `sqft_lot`, which has heavy outliers and little correlation with the other features.

**Impact on predicted price** (test split, missing values injected into the house features):

| Scenario | RMSE | MAE |
|---|---|---|
| Complete data | 201,659 | 102,045 |
| 15% missing | 208,395 | 104,078 |
| 50% missing | 229,813 | 119,054 |
| Only zipcode | 332,381 | 185,150 |

We also tried scaling *before* imputing. It was marginally better with partial data (-1%) but 10% worse when only the zipcode is sent, so we kept the notebook's order.

**Limitations.** `k=5` is the scikit-learn default and was not tuned. The model is trained on complete data only, so it never sees imputed values during training.

## Performance

### How it was measured

- `time.perf_counter()`: latency of each step of `/predict`, in process, over the 100 rows of `future_unseen_examples.csv`.
- `cProfile`: which functions the time is spent in (`python -m cProfile -s cumtime scripts/benchmark.py`).
- Load test: 400 requests from 20 concurrent `httpx` clients against a running API.

```bash
uv run python scripts/benchmark.py                                   # in-process latency
uv run python scripts/benchmark.py --load http://localhost:8000      # load test a running API
```

### Bottlenecks and fixes

| Bottleneck | Fix |
|---|---|
| Model (9 MB pickle), feature list and demographics CSV reloaded on every request | Loaded once at startup in the FastAPI `lifespan` (`src/api/resources.py`) |
| Demographics filtered with pandas and merged with `pd.concat` per request | Pre-indexed `dict` by zipcode; the input row is built in one step |
| OpenBLAS and OpenMP each spawn 16 threads for a single-row prediction; they compete for the CPU and make imputation ~9x slower | `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`; parallelism comes from workers |
| `async def` endpoint running CPU-bound code blocks the event loop, so requests are served one at a time | Sync `def` endpoint, run by FastAPI in its threadpool |
| Container started with `--reload` and a single worker | No `--reload`; `WEB_CONCURRENCY=2` uvicorn workers |
| `/health` reported healthy before the model was loaded | Returns `503` until resources are loaded |

### Results

In-process latency per prediction:

| Scenario | Before p50 / p95 | After p50 / p95 |
|---|---|---|
| Complete data | 4.4 / 5.5 ms | 1.3 / 1.9 ms |
| 30% missing fields | 49 / 101 ms | 6.8 / 7.9 ms |

Load test (20 concurrent clients, 30% missing fields):

| Version | Throughput | p50 | p95 |
|---|---|---|---|
| Before | 15 req/s | 1,315 ms | 1,760 ms |
| After, 1 worker | 160 req/s | 124 ms | 207 ms |
| After, 2 workers | 213 req/s | 89 ms | 132 ms |

Predictions are unchanged: `test_predictions_match_previous_implementation` checks every example against the previous code.

### Scaling further

Each worker is a separate process holding its own copy of the model (~9 MB), so `WEB_CONCURRENCY` can grow with the CPU count. The API is stateless and its artifacts are read-only, so it scales horizontally behind a load balancer, with `/health` as the readiness probe.

## Code quality

| Area | Change |
|---|---|
| Input validation | Zipcode must be 5 digits; house features must be `>= 0`; response contract declared with `PredictionResponse` |
| Error handling | Unexpected errors are logged with stack trace and returned as a generic `500`; missing artifacts fail the startup instead of the first request |
| Logging and monitoring | Request log with latency, prediction log with imputed fields, startup load time, Docker `HEALTHCHECK` on a readiness-aware `/health` |
| Security | Container runs as a non-root user; CORS origins configurable and no longer combined with credentials |
| Organization | Schemas, routes, resources and app setup in separate modules; dead code (`utils/loader.py`, duplicated `create_model.py`) removed; `create_model.py` uses its `demographics_path` argument and closes its files |
| Testing | Validation, error, logging, OpenAPI and parity tests; every row of `future_unseen_examples.csv` is sent to the API; integration tests cover missing data and invalid input; `pytest.ini` sets the import path |
| Dependencies | Unused `hypothesis` removed; Makefile uses Docker Compose v2; test compose file no longer binds host port 8000 |

## Testing

This project uses Docker-based testing to ensure environment consistency between testing and production. All tests run inside Docker containers, eliminating "works on my machine" issues.

### Quick Start

Run unit tests (fast, recommended for development):
```bash
make test-unit
```

Run integration tests (full environment):
```bash
make test-integration
```

Run unit tests locally without Docker (`pytest.ini` already puts `src` on the path):
```bash
pip install -r requirements.txt -r requirements-test.txt
pytest test/unit
```

Run all tests:
```bash
make test-all
```

### Test Types

**Unit Tests** (`test/unit/`)
- Use FastAPI TestClient for in-process testing
- No external dependencies or containers required
- Fast execution (typically under 30 seconds)
- Ideal for rapid development iteration

**Integration Tests** (`test/integration/`)
- Test against a real running API container
- Verify end-to-end functionality via HTTP requests
- Ensure Docker networking and orchestration work correctly
- More comprehensive but slower execution

### Running Tests

#### Unit Tests Only

```bash
make test-unit
```

This command:
- Builds the test Docker image
- Runs only tests in `test/unit/` directory
- Generates coverage reports
- Completes quickly without starting the full API container

#### Integration Tests Only

```bash
make test-integration
```

This command:
- Builds both API and test containers using Docker Compose
- Starts the API container and waits for health check
- Runs tests in `test/integration/` directory
- Automatically stops and removes containers when complete

#### All Tests

```bash
make test-all
```

Runs both integration and unit tests for comprehensive validation.

### Viewing Coverage Reports

After running tests, coverage reports are generated in the `test-results/` directory:

**HTML Coverage Report:**
```bash
open test-results/coverage/index.html
```

**Terminal Coverage Summary:**
Coverage is automatically displayed in the terminal after test execution.

### Running Specific Tests

Run a specific test file:
```bash
docker run --rm ml-api-test pytest test/unit/test_api_unit.py -v
```

Run a specific test function:
```bash
docker run --rm ml-api-test pytest test/unit/test_api_unit.py::test_predict_endpoint -v
```

Run tests matching a pattern:
```bash
docker run --rm ml-api-test pytest -k "predict" -v
```

### Troubleshooting

**Issue: "Cannot connect to the Docker daemon"**

Solution: Ensure Docker is running on your system.
```bash
docker ps  # Should list running containers without error
```

**Issue: Integration tests fail with connection errors**

Solution: Check if the API container is healthy.
```bash
docker compose -f docker-compose.test.yml up
# In another terminal:
docker compose -f docker-compose.test.yml ps
docker compose -f docker-compose.test.yml logs api
```

**Issue: Tests pass locally but fail in Docker**

Solution: This usually indicates environment differences. Check:
- Model artifacts exist in `model/` directory
- Data files exist in `data/` directory
- All dependencies are listed in `requirements.txt`

**Issue: "Port 8000 already in use"**

Solution: Stop any running containers or services using port 8000.
```bash
docker compose -f docker-compose.test.yml down
docker stop housing-api  # If the main API is running
```

**Issue: Test results not appearing in `test-results/` directory**

Solution: Ensure the directory exists and has proper permissions.
```bash
mkdir -p test-results
chmod 755 test-results
```

**Issue: Tests are very slow**

Solution: Run unit tests only for faster feedback during development.
```bash
make test-unit  # Much faster than integration tests
```

**Issue: "Image not found" errors**

Solution: Build the test image explicitly.
```bash
make test-build
```

### Cleaning Up

Remove test containers and artifacts:
```bash
make clean
```

This removes:
- All Docker containers created by docker-compose.test.yml
- All test result files and coverage reports

### Development Workflow

For rapid development iteration:

1. Make code changes in `src/` or test changes in `test/`
2. Run unit tests: `make test-unit`
3. Fix any issues and repeat
4. Before committing, run full suite: `make test-all`

The Docker setup mounts source code as volumes, so you don't need to rebuild containers for every change during integration testing.

## Feedback

We welcome any feedback regarding the project or the interview process. Your insights are valuable to us as we strive to improve the experience for future candidates.