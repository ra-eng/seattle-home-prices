from pathlib import Path

import pytest

from api.resources import load_resources

SRC_DIR = Path(__file__).resolve().parents[2] / "src"


@pytest.fixture(scope="module")
def resources():
    return load_resources(SRC_DIR / "model", SRC_DIR / "data")


def test_load_resources_matches_model_features(resources):
    """Test the feature list matches what the model was trained on."""
    assert len(resources.features) == resources.model.n_features_in_


def test_load_resources_indexes_demographics_by_string_zipcode(resources):
    """Test demographics are keyed by zipcode strings, as sent in requests."""
    assert "98042" in resources.demographics
    assert all(isinstance(zipcode, str) for zipcode in resources.demographics)


def test_build_row_returns_none_for_unknown_zipcode(resources):
    """Test build_row signals a zipcode without demographics."""
    assert resources.build_row("00000", {"bedrooms": 3}) is None


def test_build_row_follows_model_feature_order(resources):
    """Test the row has a single record with columns in the model's order."""
    row = resources.build_row("98042", {"bedrooms": 3})

    assert row.shape == (1, len(resources.features))
    assert list(row.columns) == resources.features


def test_build_row_merges_house_and_demographic_values(resources):
    """Test house values and zipcode demographics land in their columns."""
    row = resources.build_row("98042", {"bedrooms": 3, "sqft_living": 1500.0})

    assert row.at[0, "bedrooms"] == 3
    assert row.at[0, "sqft_living"] == 1500.0
    assert row.at[0, "ppltn_qty"] == resources.demographics["98042"]["ppltn_qty"]


def test_build_row_marks_missing_house_features_as_nan(resources):
    """Test omitted and null house features become NaN for the imputer."""
    row = resources.build_row("98042", {"bedrooms": 3, "bathrooms": None})

    missing = ["bathrooms", "sqft_living", "sqft_lot", "floors", "sqft_above", "sqft_basement"]
    assert row[missing].isna().all(axis=None)
    assert row.drop(columns=missing).notna().all(axis=None)
