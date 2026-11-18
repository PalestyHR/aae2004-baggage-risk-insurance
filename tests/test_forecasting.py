"""Regression checks for forecast serialization and report import behavior."""

import importlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from models.forecasting_model import ForecastingModel, FrozenRiskPredictor
from scripts.compare_loss_ratios import objective_checks

ROOT = Path(__file__).resolve().parents[1]


class ForecastingTests(unittest.TestCase):
    def test_saved_predictor_matches_fitted_model(self):
        feature_file = ROOT / "configs/feature_spec.json"
        features = json.loads(feature_file.read_text(encoding="utf-8"))["core_features"]
        data = pd.read_csv(ROOT / "data/processed/lag1/all_real.csv")
        training = data[data.target_month < "2026-08-01"]
        forecast = data[data.target_month == "2026-08-01"].copy()
        unknown_carrier = forecast.iloc[[0]].copy()
        unknown_carrier["airline_id"] = -1
        forecast = pd.concat([forecast, unknown_carrier], ignore_index=True)
        model = ForecastingModel(features).fit(training)
        expected = model.predict(forecast)

        with tempfile.TemporaryDirectory() as directory:
            model.save(directory)
            actual = FrozenRiskPredictor(directory).predict(forecast)

        np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-7)
        self.assertTrue(np.isfinite(actual).all())
        self.assertTrue((actual >= 0).all())

    def test_loss_derivatives(self):
        objective_checks()

    def test_report_import_has_no_io_or_argument_parsing(self):
        with (
            patch("pandas.read_csv", side_effect=AssertionError("Unexpected read")),
            patch.object(
                Path, "write_text", side_effect=AssertionError("Unexpected write")
            ),
            patch(
                "argparse.ArgumentParser.parse_args",
                side_effect=AssertionError("Unexpected CLI parsing"),
            ),
        ):
            report = importlib.import_module("scripts.export_baseline_comparison")
            importlib.reload(report)


if __name__ == "__main__":
    unittest.main()
