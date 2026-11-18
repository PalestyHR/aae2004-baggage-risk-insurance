"""LightGBM forecasting model with train-window preprocessing and asymmetric loss."""

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

FORECAST_PARAMETERS = {
    "n_estimators": 200,
    "learning_rate": 0.03,
    "num_leaves": 7,
    "max_depth": 3,
    "min_child_samples": 20,
    "reg_lambda": 1,
    "verbosity": -1,
    "n_jobs": 2,
    "random_state": 42,
}


def asymmetric_squared_loss(targets, predictions):
    """Return LightGBM gradients and Hessians for under:over penalties of 1:2."""

    errors = predictions - targets
    penalties = np.where(errors < 0, 2 / 3, 4 / 3)

    return penalties * errors, penalties


class ForecastingModel:
    """Predict monthly mishandling rates in bags per 1,000 loaded bags.

    Each input row describes one carrier-month using the configured lagged
    features. fit() estimates preprocessing and trees on that training window.
    """

    def __init__(self, feature_columns):
        self.feature_columns = list(feature_columns)
        self.numeric_columns = []

        for name in self.feature_columns:
            if name != "airline_id":
                self.numeric_columns.append(name)

        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        scaler = StandardScaler()
        transformers = [
            ("id", encoder, ["airline_id"]),
            ("num", scaler, self.numeric_columns),
        ]
        self.preprocessor = ColumnTransformer(transformers)

        self.regressor = lgb.LGBMRegressor(
            **FORECAST_PARAMETERS,
            objective=asymmetric_squared_loss,
        )

    def fit(self, frame):
        """Fit on historical carrier-month rows containing the target rate."""

        features = self.preprocessor.fit_transform(frame[self.feature_columns])
        targets = frame["target_rate_per_1000"].to_numpy()
        self.base_score = float(targets.mean())
        self.training_last_month = frame["target_month"].max()

        initial_scores = np.full(len(targets), self.base_score)
        self.regressor.fit(features, targets, init_score=initial_scores)

        return self

    def predict(self, frame):
        """Transform forecast features and return nonnegative risk estimates."""

        features = self.preprocessor.transform(frame[self.feature_columns])
        predictions = self.regressor.predict(features) + self.base_score

        return np.maximum(0, predictions)

    def save(self, model_directory, forecast_month="2026-08-01"):
        """Export native LightGBM weights and the fitted preprocessing values."""

        path = Path(model_directory)
        path.mkdir(parents=True, exist_ok=True)
        model_file = str(path / "risk_2026_08_lightgbm.txt")
        self.regressor.booster_.save_model(model_file)
        scaler = self.preprocessor.named_transformers_["num"]
        encoder = self.preprocessor.named_transformers_["id"]
        categories = []

        for value in encoder.categories_[0]:
            category = int(value)
            categories.append(category)

        schema = {
            "base_score": self.base_score,
            "categorical_column": "airline_id",
            "categories": categories,
            "numeric_columns": self.numeric_columns,
            "mean": scaler.mean_.tolist(),
            "scale": scaler.scale_.tolist(),
            "training_last_month": self.training_last_month,
            "forecast_month": forecast_month,
            "prediction_clip_min": 0,
            "feature_columns": self.feature_columns,
            "loss_under_over": [1, 2],
        }
        preprocessing_file = path / "risk_preprocessing.json"
        preprocessing_text = json.dumps(schema, indent=2)
        preprocessing_file.write_text(preprocessing_text, encoding="utf-8")


class FrozenRiskPredictor:
    def __init__(self, model_directory):
        path = Path(model_directory)
        preprocessing_file = path / "risk_preprocessing.json"
        preprocessing_text = preprocessing_file.read_text()
        model_file = str(path / "risk_2026_08_lightgbm.txt")

        self.schema = json.loads(preprocessing_text)
        self.booster = lgb.Booster(model_file=model_file)

    def predict(self, frame):
        """Predict mishandled bags per 1,000 for each carrier-month row."""

        schema = self.schema
        missing = set(schema["feature_columns"]) - set(frame.columns)

        if missing:
            raise ValueError(f"Missing feature columns: {sorted(missing)}")
        # Same order as ColumnTransformer: categorical one-hot then standardized
        # numeric fields. Unknown categories receive all-zero one-hot encoding.

        carrier_ids = frame[schema["categorical_column"]].to_numpy()
        categories = np.asarray(schema["categories"])
        category_matches = carrier_ids[:, None] == categories[None, :]
        categorical = category_matches.astype(float)

        numeric = frame[schema["numeric_columns"]].to_numpy(dtype=float)
        mean = np.asarray(schema["mean"])
        scale = np.asarray(schema["scale"])
        numeric = (numeric - mean) / scale

        if not np.isfinite(numeric).all():
            raise ValueError("Core inputs must be finite")

        matrix = np.concatenate([categorical, numeric], axis=1)
        predictions = self.booster.predict(matrix) + schema["base_score"]
        minimum = schema["prediction_clip_min"]

        return np.maximum(minimum, predictions)
