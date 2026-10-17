import json
import pickle

import pandas as pd
import pytest
from conftest import SRC_DIR
from fastapi.testclient import TestClient

HOUSE_FEATURES = ["bedrooms", "bathrooms", "sqft_living", "sqft_lot", "floors",
                  "sqft_above", "sqft_basement"]


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


def test_health_returns_503_until_model_is_loaded(monkeypatch):
    """Test /health reports not ready while resources are not loaded."""
    from src.main import app
    monkeypatch.setattr(app.state, "resources", None, raising=False)

    response = TestClient(app).get("/health")

    assert response.status_code == 503


def test_model_is_loaded_once_across_requests(monkeypatch, sample_home_features):
    """Test the model is loaded at startup and not on every request."""
    from src.main import app
    calls = []
    original_load = pickle.load
    monkeypatch.setattr(pickle, "load", lambda file: calls.append(file) or original_load(file))
    monkeypatch.setenv("MODEL_DIR", str(SRC_DIR / "model"))
    monkeypatch.setenv("DATA_DIR", str(SRC_DIR / "data"))

    with TestClient(app) as client:
        for _ in range(5):
            assert client.post("/predict", json=sample_home_features).status_code == 200

    assert len(calls) == 1


def test_predictions_match_previous_implementation(test_client):
    """Test the optimized path predicts exactly what the per-request loading code did."""
    with open(SRC_DIR / "model" / "model.pkl", "rb") as model_file:
        model = pickle.load(model_file)
    with open(SRC_DIR / "model" / "model_features.json") as features_file:
        features = json.load(features_file)
    demographics = pd.read_csv(SRC_DIR / "data" / "zipcode_demographics.csv", dtype={"zipcode": str})
    examples = pd.read_csv(SRC_DIR / "data" / "future_unseen_examples.csv", dtype={"zipcode": str})

    for example in examples[HOUSE_FEATURES + ["zipcode"]].to_dict(orient="records"):
        demographic_info = demographics[demographics["zipcode"] == example["zipcode"]]
        expected_input = pd.concat(
            [pd.DataFrame([example]), demographic_info.reset_index(drop=True)], axis=1
        )[features].astype(float)

        response = test_client.post("/predict", json=example)

        assert response.json()["predicted_price"] == pytest.approx(model.predict(expected_input)[0])
