import logging

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from api.schemas import HomeFeatures, PredictionResponse

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/health")
def health_check(request: Request):
    """
    Health check endpoint for container orchestration.
    Returns 200 once the model is loaded and the API is ready to accept requests.
    """
    if getattr(request.app.state, "resources", None) is None:
        return JSONResponse(status_code=503, content={"status": "loading"})
    return {"status": "healthy"}


@router.post("/predict", response_model=PredictionResponse)
def predict(home_features: HomeFeatures, request: Request) -> PredictionResponse:
    """Predict a home price, imputing missing house features with the model's KNNImputer.

    Declared as sync so FastAPI runs the CPU-bound prediction in its threadpool
    instead of blocking the event loop.
    """
    resources = request.app.state.resources
    house = home_features.model_dump(exclude={"zipcode"})
    input_data = resources.build_row(home_features.zipcode, house)
    if input_data is None:
        raise HTTPException(status_code=422, detail=f"Unknown zipcode: {home_features.zipcode}")

    prediction = float(resources.model.predict(input_data)[0])

    imputed = [name for name, value in house.items() if value is None]
    logger.info("prediction zipcode=%s imputed=%s price=%.0f",
                home_features.zipcode, imputed, prediction)
    return PredictionResponse(predicted_price=prediction)
