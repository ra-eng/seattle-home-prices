from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

router = APIRouter()


class HomeFeatures(BaseModel):
    bedrooms: int | None = None
    bathrooms: float | None = None
    sqft_living: float | None = None
    sqft_lot: float | None = None
    floors: float | None = None
    sqft_above: float | None = None
    sqft_basement: float | None = None
    zipcode: str


@router.get("/health")
def health_check(request: Request):
    """
    Health check endpoint for container orchestration.
    Returns 200 once the model is loaded and the API is ready to accept requests.
    """
    if getattr(request.app.state, "resources", None) is None:
        return JSONResponse(status_code=503, content={"status": "loading"})
    return {"status": "healthy"}


@router.post("/predict")
def predict(home_features: HomeFeatures, request: Request):
    """Predict a home price, imputing missing house features with the model's KNNImputer.

    Declared as sync so FastAPI runs the CPU-bound prediction in its threadpool
    instead of blocking the event loop.
    """
    resources = request.app.state.resources
    input_data = resources.build_row(
        home_features.zipcode, home_features.model_dump(exclude={"zipcode"})
    )
    if input_data is None:
        raise HTTPException(status_code=422, detail=f"Unknown zipcode: {home_features.zipcode}")

    prediction = resources.model.predict(input_data)

    return {"predicted_price": float(prediction[0])}
