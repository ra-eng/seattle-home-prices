import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.endpoints import router as api_router
from api.resources import load_resources


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model artifacts once at startup instead of on every request."""
    app.state.resources = load_resources(
        Path(os.getenv("MODEL_DIR", "model")),
        Path(os.getenv("DATA_DIR", "data")),
    )
    yield


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
