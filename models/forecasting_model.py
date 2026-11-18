"""LightGBM forecasting model with train-window preprocessing and asymmetric loss."""

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CARRIER_COLUMN = "airline_id"
TARGET_COLUMN = "target_rate_per_1000"
MONTH_COLUMN = "target_month"

MODEL_FILENAME = "risk_2026_08_lightgbm.txt"
PREPROCESSING_FILENAME = "risk_preprocessing.json"
FORECAST_MONTH = "2026-08-01"

MINIMUM_RATE = 0
UNDER_PENALTY = 2 / 3
OVER_PENALTY = 4 / 3

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
    penalties = np.where(errors < 0, UNDER_PENALTY, OVER_PENALTY)

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
            if name != CARRIER_COLUMN:
                self.numeric_columns.append(name)

        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        scaler = StandardScaler()
        transformers = [
            ("id", encoder, [CARRIER_COLUMN]),
            ("num", scaler, self.numeric_columns),
        ]
        self.preprocessor = ColumnTransformer(transformers)

        self.regressor = lgb.LGBMRegressor(
            **FORECAST_PARAMETERS,
            objective=asymmetric_squared_loss,
        )

    def fit(self, frame):
        """Fit on historical carrier-month rows containing the target rate."""

        feature_frame = frame[self.feature_columns]
        target_column = frame[TARGET_COLUMN]
        month_column = frame[MONTH_COLUMN]

        features = self.preprocessor.fit_transform(feature_frame)
        targets = target_column.to_numpy()
        target_mean = targets.mean()
        self.base_score = float(target_mean)
        self.training_last_month = month_column.max()

        fitted_transformers = self.preprocessor.named_transformers_
        self.encoder = fitted_transformers["id"]
        self.scaler = fitted_transformers["num"]

        initial_scores = np.full(len(targets), self.base_score)
        self.regressor.fit(features, targets, init_score=initial_scores)

        return self

    def predict(self, frame):
        """Transform forecast features and return nonnegative risk estimates."""

        feature_frame = frame[self.feature_columns]
        features = self.preprocessor.transform(feature_frame)
        raw_predictions = self.regressor.predict(features)
        predictions = raw_predictions + self.base_score

        return np.maximum(MINIMUM_RATE, predictions)

    def save(self, model_directory, forecast_month=FORECAST_MONTH):
        """Export native LightGBM weights and the fitted preprocessing values."""

        path = Path(model_directory)
        path.mkdir(parents=True, exist_ok=True)
        model_path = path / MODEL_FILENAME
        model_file = str(model_path)
        booster = self.regressor.booster_
        booster.save_model(model_file)

        fitted_categories = self.encoder.categories_
        carrier_categories = fitted_categories[0]
        fitted_mean = self.scaler.mean_
        fitted_scale = self.scaler.scale_
        mean_values = fitted_mean.tolist()
        scale_values = fitted_scale.tolist()
        categories = []

        for value in carrier_categories:
            category = int(value)
            categories.append(category)

        schema = {
            "base_score": self.base_score,
            "categorical_column": CARRIER_COLUMN,
            "categories": categories,
            "numeric_columns": self.numeric_columns,
            "mean": mean_values,
            "scale": scale_values,
            "training_last_month": self.training_last_month,
            "forecast_month": forecast_month,
            "prediction_clip_min": MINIMUM_RATE,
            "feature_columns": self.feature_columns,
            "loss_under_over": [1, 2],
        }
        preprocessing_file = path / PREPROCESSING_FILENAME
        preprocessing_text = json.dumps(schema, indent=2)
        preprocessing_file.write_text(preprocessing_text, encoding="utf-8")


class FrozenRiskPredictor:
    """Load fitted preprocessing values and LightGBM weights for prediction."""

    def __init__(self, model_directory):
        path = Path(model_directory)
        preprocessing_file = path / PREPROCESSING_FILENAME
        model_path = path / MODEL_FILENAME
        model_file = str(model_path)

        preprocessing_text = preprocessing_file.read_text()
        schema = json.loads(preprocessing_text)
        self.schema = schema
        self.booster = lgb.Booster(model_file=model_file)

        self.categorical_column = schema["categorical_column"]
        self.numeric_columns = schema["numeric_columns"]
        self.base_score = schema["base_score"]
        self.minimum_rate = schema["prediction_clip_min"]

        categories = schema["categories"]
        mean = schema["mean"]
        scale = schema["scale"]

        category_values = np.asarray(categories)
        self.category_row = category_values.reshape(1, -1)
        self.mean = np.asarray(mean)
        self.scale = np.asarray(scale)

    def predict(self, frame):
        """Predict monthly mishandling rates with the saved preprocessing."""

        carrier_column = frame[self.categorical_column]
        numeric_frame = frame[self.numeric_columns]

        carrier_ids = carrier_column.to_numpy()
        carrier_rows = carrier_ids.reshape(-1, 1)
        category_matches = carrier_rows == self.category_row
        categorical_features = category_matches.astype(float)

        numeric_features = numeric_frame.to_numpy(dtype=float)
        numeric_features = (numeric_features - self.mean) / self.scale
        feature_groups = [categorical_features, numeric_features]
        features = np.concatenate(feature_groups, axis=1)

        raw_predictions = self.booster.predict(features)
        predictions = raw_predictions + self.base_score

        return np.maximum(self.minimum_rate, predictions)
