"""Controlled asymmetric squared-loss experiment, not profit optimization."""

from pathlib import Path
import sys
import json
import hashlib
import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from lightgbm import LGBMRegressor
import lightgbm

ROOT = Path(__file__).resolve().parents[1]

RATIOS = [
    (1, 4),
    (1, 2),
    (1, 1.5),
    (1, 1),
    (1.5, 1),
    (2, 1),
    (4, 1),
]
PARAMS = dict(
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


def coeffs(under, over):
    return 2 * under / (under + over), 2 * over / (under + over)


class AsymmetricSquaredLoss:
    """LightGBM objective with configurable under/over coefficients."""

    def __init__(self, under, over):
        self.under, self.over = coeffs(under, over)

    def __call__(self, targets, predictions, weight):
        errors = predictions - targets
        penalties = np.where(errors < 0, self.under, self.over)
        weights = np.ones_like(errors) if weight is None else weight
        return weights * penalties * errors, weights * penalties


def objective_checks():
    y = np.array([1.0, 3.0, 5.0, 2.0])
    predictions_frame = np.array([0.2, 4.0, 3.5, 3.0])
    weights = np.array([0.2, 0.5, 1.0, 2.0])
    eps = 1e-5
    for under, over in RATIOS:
        a, b = coeffs(under, over)
        fn = AsymmetricSquaredLoss(under, over)
        loss = lambda z: 0.5 * weights * np.where(z - y < 0, a, b) * (z - y) ** 2
        grad, hess = fn(y, predictions_frame, weights)
        assert np.allclose(
            grad,
            (loss(predictions_frame + eps) - loss(predictions_frame - eps)) / (2 * eps),
            atol=1e-8,
        )
        assert np.allclose(
            hess,
            (
                fn(y, predictions_frame + eps, weights)[0]
                - fn(y, predictions_frame - eps, weights)[0]
            )
            / (2 * eps),
            atol=1e-8,
        )
        assert np.all(hess > 0)


def metrics(predictions_frame, threshold):
    errors = predictions_frame.prediction - predictions_frame.target_rate_per_1000
    weights = predictions_frame.target_enplaned_bags
    out = {
        "n": len(predictions_frame),
        "weighted_MAE": float(np.average(abs(errors), weights=weights)),
        "RMSE": float(np.sqrt(np.mean(errors**2))),
        "weighted_RMSE": float(np.sqrt(np.average(errors**2, weights=weights))),
        "weighted_bias": float(np.average(errors, weights=weights)),
        "underprediction_fraction": float((errors < 0).mean()),
        "weighted_undergap": float(np.average(np.maximum(-errors, 0), weights=weights)),
        "weighted_overgap": float(np.average(np.maximum(errors, 0), weights=weights)),
    }
    high = predictions_frame.target_rate_per_1000 >= threshold
    out["high_risk_n"] = int(high.sum())
    out["high_risk_weighted_MAE"] = (
        float(np.average(abs(errors[high]), weights=weights[high]))
        if high.any()
        else None
    )
    return out


def main():
    DATA = ROOT / "data/processed"
    OUT = ROOT / "results/reproduced_loss_ratios"
    OUT.mkdir(exist_ok=True)
    feature_columns = json.loads((DATA / "feature_spec.json").read_text())[
        "core_features"
    ]
    data = pd.read_csv(DATA / "lag1/all_real.csv")
    data["date"] = pd.to_datetime(data.target_month)
    assert not data.is_synthetic.any()
    threshold = float(
        data.loc[data.date.dt.year <= 2024, "target_rate_per_1000"].quantile(0.8)
    )
    protocol = {
        "penalty_order": "under:over",
        "ratios": RATIOS,
        "coefficient_normalization": "a+b=2",
        "loss": "0.5*w*c*(prediction-target)^2; c=a if prediction<target else b",
        "training_weights": ["uniform", "volume_normalized_to_training_mean"],
        "init": "same training weighted mean in every ratio within each fold/weighting arm",
        "selection": "2025 weighted MAE, include symmetric 1:1; select per weighting arm and overall before 2026",
        "params": PARAMS,
        "high_risk_threshold_train_only": threshold,
        "features": feature_columns,
        "synthetic_rows_used": False,
        "realized_test_exposure": "evaluation weights only, never predictor",
        "2026_status": "previously inspected exploratory test",
        "source_sha256": hashlib.sha256(
            (DATA / "lag1/all_real.csv").read_bytes()
        ).hexdigest(),
        "lightgbm_version": lightgbm.__version__,
    }
    (OUT / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    objective_checks()
    predictions = []
    summary = []
    checks = []
    training_warnings = []
    selected = {}
    for year in [2025, 2026]:
        year_predictions = []
        for date in sorted(data.loc[data.date.dt.year == year, "date"].unique()):
            train = data[data.date < date]
            test = data[data.date == date]
            assert train.date.max() < test.date.min()
            preprocessor = ColumnTransformer(
                [
                    (
                        "id",
                        OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ["airline_id"],
                    ),
                    (
                        "num",
                        StandardScaler(),
                        [c for c in feature_columns if c != "airline_id"],
                    ),
                ]
            )
            train_features = preprocessor.fit_transform(train[feature_columns])
            test_features = preprocessor.transform(test[feature_columns])
            y = train.target_rate_per_1000.to_numpy()
            for weighting in ["uniform", "volume"]:
                weights = (
                    np.ones(len(train))
                    if weighting == "uniform"
                    else (
                        train.target_enplaned_bags / train.target_enplaned_bags.mean()
                    ).to_numpy()
                )
                base_score = float(np.average(y, weights=weights))
                with warnings.catch_warnings(record=True) as captured_warnings:
                    warnings.simplefilter("always")
                    builtin = LGBMRegressor(**PARAMS, objective="regression").fit(
                        train_features, y, sample_weight=weights
                    )
                    builtin_predictions = np.maximum(0, builtin.predict(test_features))
                    for under, over in RATIOS:
                        name = f"{under:g}:{over:g}"
                        model = LGBMRegressor(
                            **PARAMS, objective=AsymmetricSquaredLoss(under, over)
                        )
                        model.fit(
                            train_features,
                            y,
                            sample_weight=weights,
                            init_score=np.full(len(train), base_score),
                        )
                        pred = np.maximum(0, model.predict(test_features) + base_score)
                        if under == over:
                            diff = float(np.max(abs(pred - builtin_predictions)))
                            checks.append(
                                {
                                    "month": str(date),
                                    "weighting": weighting,
                                    "symmetric_builtin_max_diff": diff,
                                }
                            )
                            assert diff < 1e-5, (date, weighting, diff)
                        fold_predictions = test[
                            [
                                "row_id",
                                "target_month",
                                "carrier",
                                "target_rate_per_1000",
                                "target_enplaned_bags",
                            ]
                        ].copy()
                        fold_predictions["prediction"] = pred
                        fold_predictions["year"] = year
                        fold_predictions["weighting"] = weighting
                        fold_predictions["under_over_ratio"] = name
                        fold_predictions["training_max_month"] = (
                            train.target_month.max()
                        )
                        assert np.isfinite(pred).all()
                        year_predictions.append(fold_predictions)
                    training_warnings.extend(str(x.message) for x in captured_warnings)
        predictions_frame = pd.concat(year_predictions, ignore_index=True)
        predictions.append(predictions_frame)
        for (weighting, ratio), records in predictions_frame.groupby(
            ["weighting", "under_over_ratio"]
        ):
            summary.append(
                {
                    "year": year,
                    "weighting": weighting,
                    "under_over_ratio": ratio,
                    **metrics(records, threshold),
                }
            )
        if year == 2025:
            candidates = [x for x in summary if x["year"] == 2025]
            for weighting in ["uniform", "volume"]:
                selected[weighting] = min(
                    [x for x in candidates if x["weighting"] == weighting],
                    key=lambda x: x["weighted_MAE"],
                )
            selected["overall"] = min(candidates, key=lambda x: x["weighted_MAE"])
            (OUT / "selection_before_2026.json").write_text(
                json.dumps(selected, indent=2), encoding="utf-8"
            )
            print(
                "2025 selection",
                [
                    (k, v["weighting"], v["under_over_ratio"], v["weighted_MAE"])
                    for k, v in selected.items()
                ],
                flush=True,
            )
        print(
            pd.DataFrame([x for x in summary if x["year"] == year])[
                [
                    "year",
                    "weighting",
                    "under_over_ratio",
                    "weighted_MAE",
                    "weighted_RMSE",
                    "weighted_bias",
                ]
            ].to_string(index=False),
            flush=True,
        )
    all_predictions = pd.concat(predictions, ignore_index=True)
    all_predictions.to_csv(OUT / "predictions.csv", index=False)
    pd.DataFrame(summary).to_csv(OUT / "metrics.csv", index=False)
    monthly = []
    for keys, records in all_predictions.groupby(
        ["year", "weighting", "under_over_ratio", "target_month"]
    ):
        monthly.append(
            dict(
                zip(["year", "weighting", "under_over_ratio", "month"], keys),
                **metrics(records, threshold),
            )
        )
    pd.DataFrame(monthly).to_csv(OUT / "monthly.csv", index=False)
    (OUT / "implementation_checks.json").write_text(
        json.dumps(
            {
                "gradient_hessian_finite_difference": "passed",
                "builtin_symmetry_checks": checks,
                "warnings": training_warnings,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
