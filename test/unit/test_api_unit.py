import pytest


def test_health_endpoint(test_client):
    """Test the /health endpoint returns correct status."""
    response = test_client.get("/health")
    assert response.status_code == 200
    response_data = response.json()
    assert "status" in response_data
    assert response_data["status"] == "healthy"


def test_predict_endpoint_valid_input(test_client, sample_home_features):
    """Test the /predict endpoint with valid input."""
    response = test_client.post("/predict", json=sample_home_features)
    assert response.status_code == 200
    response_data = response.json()
    assert "predicted_price" in response_data
    assert isinstance(response_data["predicted_price"], float)


@pytest.mark.parametrize("missing", [["bedrooms"], ["sqft_living", "sqft_above", "floors"]])
def test_predict_imputes_missing_fields(test_client, sample_home_features, missing):
    """Test the /predict endpoint imputes fields that are omitted."""
    for field in missing:
        del sample_home_features[field]

    response = test_client.post("/predict", json=sample_home_features)

    assert response.status_code == 200
    assert isinstance(response.json()["predicted_price"], float)


def test_predict_imputes_explicit_nulls(test_client, sample_home_features):
    """Test the /predict endpoint treats null values as missing."""
    sample_home_features["bathrooms"] = None

    response = test_client.post("/predict", json=sample_home_features)

    assert response.status_code == 200


def test_predict_with_only_zipcode(test_client):
    """Test the /predict endpoint works when only zipcode is sent."""
    response = test_client.post("/predict", json={"zipcode": "98042"})

    assert response.status_code == 200
    assert isinstance(response.json()["predicted_price"], float)


def test_predict_requires_zipcode(test_client, sample_home_features):
    """Test the /predict endpoint rejects requests without zipcode."""
    del sample_home_features["zipcode"]

    response = test_client.post("/predict", json=sample_home_features)

    assert response.status_code == 422


def test_predict_rejects_unknown_zipcode(test_client, sample_home_features):
    """Test the /predict endpoint rejects zipcodes without demographics."""
    sample_home_features["zipcode"] = "00000"

    response = test_client.post("/predict", json=sample_home_features)

    assert response.status_code == 422
