import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.endpoints import router as api_router
from api.resources import load_resources

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model artifacts once at startup instead of on every request."""
    start = time.perf_counter()
    app.state.resources = load_resources(
        Path(os.getenv("MODEL_DIR", "model")),
        Path(os.getenv("DATA_DIR", "data")),
    )
    logger.info("resources loaded in %.0f ms", (time.perf_counter() - start) * 1000)
    yield


app = FastAPI(title="Sound Realty Home Price API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log every request with its latency and turn unexpected errors into a generic 500."""
    start = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        logger.exception("unhandled error on %s %s", request.method, request.url.path)
        response = JSONResponse(status_code=500, content={"detail": "Internal server error"})
    duration_ms = (time.perf_counter() - start) * 1000
    logger.info("request method=%s path=%s status=%d duration_ms=%.1f",
                request.method, request.url.path, response.status_code, duration_ms)
    return response


app.include_router(api_router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
