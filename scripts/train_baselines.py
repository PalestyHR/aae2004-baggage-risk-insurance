from pathlib import Path
import sys
import json
import warnings
import time

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import ElasticNet, PoissonRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
)
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[1]


def estimator(name):
    if name == "ElasticNet":
        return ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=20000)
    if name == "Poisson_GLM":
        return PoissonRegressor(alpha=0.1, max_iter=2000)
    if name == "RandomForest":
        return RandomForestRegressor(
            n_estimators=300, min_samples_leaf=8, n_jobs=2, random_state=42
        )
    if name == "ExtraTrees":
        return ExtraTreesRegressor(
            n_estimators=300, min_samples_leaf=8, n_jobs=2, random_state=42
        )
    if name == "HistGBR":
        return HistGradientBoostingRegressor(
            max_iter=200,
            learning_rate=0.03,
            max_leaf_nodes=7,
            min_samples_leaf=20,
            l2_regularization=1,
            early_stopping=False,
            random_state=42,
        )
    if name == "LightGBM":
        return LGBMRegressor(
            n_estimators=200,
            learning_rate=0.03,
            num_leaves=7,
            max_depth=3,
            min_child_samples=20,
            reg_lambda=1,
            verbosity=-1,
            n_jobs=2,
            random_state=42,
        )
    if name == "XGBoost":
        return XGBRegressor(
            n_estimators=200,
            learning_rate=0.03,
            max_depth=3,
            min_child_weight=20,
            reg_lambda=1,
            objective="reg:squarederror",
            tree_method="hist",
            n_jobs=2,
            random_state=42,
        )


def evaluate(records):
    errors = records.prediction - records.target_rate_per_1000
    weights = records.target_enplaned_bags
    return {
        "n": len(records),
        "weighted_MAE": float(np.average(abs(errors), weights=weights)),
        "MAE": float(abs(errors).mean()),
        "RMSE": float(np.sqrt((errors**2).mean())),
        "weighted_bias": float(np.average(errors, weights=weights)),
    }


def main():
    DATA = ROOT / "data/processed"
    OUT = ROOT / "results/reproduced_baselines"
    OUT.mkdir(exist_ok=True)
    spec = json.loads((DATA / "feature_spec.json").read_text())
    feature_columns = spec["core_features"]
    data = pd.read_csv(DATA / "lag1/all_real.csv")
    data["date"] = pd.to_datetime(data.target_month)
    assert not data.is_synthetic.any()
    names = [
        "blend50",
        "seasonal_naive",
        "ElasticNet",
        "Poisson_GLM",
        "RandomForest",
        "ExtraTrees",
        "HistGBR",
        "LightGBM",
        "XGBoost",
    ]
    rows = []
    scores = []
    training_warnings = []
    for year in [2025, 2026]:
        for name in names:
            parts = []
            for date in sorted(data.loc[data.date.dt.year == year, "date"].unique()):
                train = data[data.date < date]
                test = data[data.date == date]
                assert train.date.max() < test.date.min()
                if name == "blend50":
                    pred = 0.5 * (test.recent_rate + test.rate_year_ago)
                elif name == "seasonal_naive":
                    pred = test.rate_year_ago
                else:
                    numeric_columns = []
                    for c in feature_columns:
                        if c != "airline_id":
                            numeric_columns.append(c)
                    preprocessor = ColumnTransformer(
                        [
                            (
                                "id",
                                OneHotEncoder(
                                    handle_unknown="ignore", sparse_output=False
                                ),
                                ["airline_id"],
                            ),
                            ("num", StandardScaler(), numeric_columns),
                        ]
                    )
                    train_features = preprocessor.fit_transform(train[feature_columns])
                    test_features = preprocessor.transform(test[feature_columns])
                    model = estimator(name)
                    fit_options = (
                        {
                            "sample_weight": train.target_enplaned_bags
                            / train.target_enplaned_bags.mean()
                        }
                        if name == "Poisson_GLM"
                        else {}
                    )
                    with warnings.catch_warnings(record=True) as captured_warnings:
                        warnings.simplefilter("always")
                        model.fit(
                            train_features, train.target_rate_per_1000, **fit_options
                        )
                        pred = np.maximum(0, model.predict(test_features))
                        warning_records = []
                        for weights in captured_warnings:
                            warning_records.append(
                                {
                                    "year": year,
                                    "model": name,
                                    "date": str(date),
                                    "warning": str(weights.message),
                                }
                            )
                        training_warnings.extend(warning_records)
                fold_predictions = test[
                    [
                        "row_id",
                        "target_month",
                        "carrier",
                        "target_rate_per_1000",
                        "target_enplaned_bags",
                    ]
                ].copy()
                fold_predictions["prediction"] = np.asarray(pred)
                fold_predictions["model"] = name
                fold_predictions["year"] = year
                assert np.isfinite(fold_predictions.prediction).all()
                parts.append(fold_predictions)
            predictions_frame = pd.concat(parts)
            rows.append(predictions_frame)
            s = {"year": year, "model": name, **evaluate(predictions_frame)}
            scores.append(s)
            print(s, flush=True)
        if year == 2025:
            choice = min(scores, key=lambda x: x["weighted_MAE"])["model"]
            print("PRE-TEST CHOICE", choice, flush=True)
    predictions_frame = pd.concat(rows, ignore_index=True)
    predictions_frame.to_csv(OUT / "predictions.csv", index=False)
    pd.DataFrame(scores).to_csv(OUT / "metrics.csv", index=False)
    monthly = []
    for (year, model, month), records in predictions_frame.groupby(
        ["year", "model", "target_month"]
    ):
        monthly.append(
            {"year": int(year), "model": model, "month": month, **evaluate(records)}
        )
    pd.DataFrame(monthly).to_csv(OUT / "monthly.csv", index=False)
    (OUT / "audit.json").write_text(
        json.dumps(
            {
                "choice_on_2025": choice,
                "features": feature_columns,
                "warnings": training_warnings,
                "test_has_been_previously_inspected": True,
                "metrics": scores,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
