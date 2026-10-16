from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import json
import pickle
import pandas as pd

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
async def health_check():
    """
    Health check endpoint for container orchestration.
    Returns 200 if API is ready to accept requests.
    """
    return {"status": "healthy"}


@router.post("/predict")
async def predict(home_features: HomeFeatures):
    # Load the model and features
    with open("model/model.pkl", "rb") as model_file:
        model = pickle.load(model_file)

    with open("model/model_features.json") as features_file:
        model_features = json.load(features_file)

    # Load demographic data
    demographics = pd.read_csv("data/zipcode_demographics.csv", dtype={"zipcode": str})
    demographic_info = demographics[demographics["zipcode"] == home_features.zipcode]
    if demographic_info.empty:
        raise HTTPException(status_code=422, detail=f"Unknown zipcode: {home_features.zipcode}")

    # Combine input data with demographic data; missing fields become NaN for the KNN imputer
    input_data = pd.concat(
        [pd.DataFrame([home_features.model_dump()]), demographic_info.reset_index(drop=True)],
        axis=1,
    )
    input_data = input_data[model_features].astype(float)

    # Make prediction
    prediction = model.predict(input_data)

    return {"predicted_price": float(prediction[0])}
