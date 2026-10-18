from pydantic import BaseModel, Field


class HomeFeatures(BaseModel):
    """Prediction request. Only zipcode is required; missing house features are imputed."""

    bedrooms: int | None = Field(default=None, ge=0)
    bathrooms: float | None = Field(default=None, ge=0)
    sqft_living: float | None = Field(default=None, ge=0)
    sqft_lot: float | None = Field(default=None, ge=0)
    floors: float | None = Field(default=None, ge=0)
    sqft_above: float | None = Field(default=None, ge=0)
    sqft_basement: float | None = Field(default=None, ge=0)
    zipcode: str = Field(pattern=r"^\d{5}$", examples=["98125"])


class PredictionResponse(BaseModel):
    """Predicted sale price in US dollars."""

    predicted_price: float
