import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.pipeline import Pipeline


@dataclass(frozen=True)
class Resources:
    """Read-only artifacts loaded once at startup and shared across requests."""

    model: Pipeline
    features: list[str]
    demographics: dict[str, dict[str, float]]

    def build_row(self, zipcode: str, house: dict[str, float | None]) -> pd.DataFrame | None:
        """Build a single model input row by merging house features with zipcode demographics.

        Missing house features become NaN so the pipeline's KNNImputer can fill them.
        Returns None when the zipcode has no demographic data.
        """
        demographic_info = self.demographics.get(zipcode)
        if demographic_info is None:
            return None
        return pd.DataFrame([{**demographic_info, **house}], columns=self.features, dtype=float)


def load_resources(model_dir: Path, data_dir: Path) -> Resources:
    """Load the model, its feature order and the demographics indexed by zipcode."""
    with open(model_dir / "model.pkl", "rb") as model_file:
        model = pickle.load(model_file)

    with open(model_dir / "model_features.json") as features_file:
        features = json.load(features_file)

    demographics = pd.read_csv(data_dir / "zipcode_demographics.csv", dtype={"zipcode": str})

    return Resources(
        model=model,
        features=features,
        demographics=demographics.set_index("zipcode").to_dict(orient="index"),
    )
