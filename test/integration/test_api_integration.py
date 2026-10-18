import pytest


@pytest.mark.integration
def test_predict_endpoint_integration(http_client, sample_home_features):
    """Test the /predict endpoint via HTTP using httpx client."""
    response = http_client.post("/predict", json=sample_home_features)
    assert response.status_code == 200
    response_data = response.json()
    assert "predicted_price" in response_data
    assert isinstance(response_data["predicted_price"], float)
    assert response_data["predicted_price"] > 0


@pytest.mark.integration
def test_health_endpoint_integration(http_client):
    """Test the /health endpoint via HTTP using httpx client."""
    response = http_client.get("/health")
    assert response.status_code == 200
    response_data = response.json()
    assert "status" in response_data
    assert response_data["status"] == "healthy"


@pytest.mark.integration
def test_predict_imputes_missing_fields_integration(http_client):
    """Test the running API imputes null and omitted fields."""
    response = http_client.post(
        "/predict", json={"zipcode": "98125", "bedrooms": 3, "bathrooms": None}
    )
    assert response.status_code == 200
    assert response.json()["predicted_price"] > 0


@pytest.mark.integration
@pytest.mark.parametrize("payload", [
    {"zipcode": "00000"},
    {"zipcode": "abc"},
    {"zipcode": "98125", "sqft_living": -100},
    {"bedrooms": 3},
])
def test_predict_rejects_invalid_requests_integration(http_client, payload):
    """Test the running API returns 422 for unknown zipcodes and invalid input."""
    response = http_client.post("/predict", json=payload)
    assert response.status_code == 422
