"""Train and evaluate monthly baggage-risk and volume forecasts."""

from pathlib import Path
import sys
import json
import warnings

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.forecasting_model import FORECAST_PARAMETERS, ForecastingModel


def count_metrics(targets, predictions):
    target_values = np.asarray(targets)
    predicted_values = np.asarray(predictions)
    valid_targets = np.isfinite(target_values)
    valid_predictions = np.isfinite(predicted_values)

    valid = valid_targets & valid_predictions

    targets = target_values[valid]
    predictions = predicted_values[valid]
    errors = predictions - targets
    absolute_errors = abs(errors)

    squared_errors = errors**2

    mean_error = absolute_errors.mean()
    mean_squared_error = squared_errors.mean()
    root_mean_squared_error = np.sqrt(mean_squared_error)
    total_absolute_error = absolute_errors.sum()

    observed_total = targets.sum()
    predicted_total = predictions.sum()
    error_total = errors.sum()
    percentage_error = 100 * total_absolute_error / observed_total

    percentage_bias = 100 * error_total / observed_total

    return {
        "n": len(targets),
        "MAE": float(mean_error),
        "RMSE": float(root_mean_squared_error),
        "WAPE_percent": float(percentage_error),
        "observed_total": float(observed_total),
        "predicted_total": float(predicted_total),
        "aggregate_bias_percent": float(percentage_bias),
    }


def main():
    data_directory = ROOT / "data/processed"
    output_directory = ROOT / "results/reproduced_risk"
    output_directory.mkdir(exist_ok=True)
    feature_columns_file = ROOT / "configs/feature_spec.json"
    feature_columns_config = json.loads(
        feature_columns_file.read_text(encoding="utf-8")
    )
    feature_columns = feature_columns_config["core_features"]
    # Align observations and lagged exposure features before monthly fitting.

    data = pd.read_csv(data_directory / "lag1/all_real.csv")
    data["date"] = pd.to_datetime(data.target_month)
    observations = pd.read_csv(
        data_directory / "carrier_month_observed_NOT_model_input.csv"
    )

    observations["target_month"] = observations.date
    data = data.merge(
        observations[["airline_id", "target_month", "passenger_segments"]],
        on=["airline_id", "target_month"],
        validate="one_to_one",
        how="left",
    )
    observed_index = observations.set_index(["airline_id", "t"])

    for feature_name, source in [
        ("bags_year_ago", "enplaned_bags"),
        ("passengers_year_ago", "passenger_segments"),
    ]:
        historical_volumes = []

        for _, record in data.iterrows():
            historical_volumes.append(
                observed_index.loc[
                    (
                        record.airline_id,
                        12 * record.date.year + record.date.month - 1 - 12,
                    ),
                    source,
                ]
            )

        data[feature_name] = historical_volumes

    assert not data.is_synthetic.any() and data[feature_columns].notna().all().all()
    model_parameters = FORECAST_PARAMETERS.copy()
    protocol = {
        "date": "2026-11-06",
        "population": "BTS domestic operating-carrier monthly records",
        "application_context_only": "Cathay",
        "features": feature_columns,
        "risk": (
            "LightGBM asymmetric squared loss under:over=1:2, equal row "
            "weights, a+b=2, mean initialization; selected previously on "
            "2025"
        ),
        "volume_candidates": [
            "LightGBM direct squared loss",
            "last year same month",
            "equal blend latest month and last year",
        ],
        "volume_selection": "2025 WAPE separately for baggage and passenger-segment volume; freeze before 2026",
        "risk_selection": "no change based on 2026",
        "training": "expanding monthly refit, all earlier target months; timely lag1 availability assumed",
        "observed_exposure_at_forecast_time": False,
        "abnormal_count_formula": "predicted_rate_per_1000 * predicted_baggage_volume / 1000",
        "unavailable": [
            "permanent-loss counts",
            "Cathay test labels",
            "eligible customer counts",
            "purchase rate",
            "expected covered claims",
            "costs",
            "compensation amount",
        ],
        "actuals_use": "target labels in past training; future actuals evaluation only",
        "test_status": "exploratory: 2026 already inspected",
        "parameters": model_parameters,
    }
    (output_directory / "protocol.json").write_text(
        json.dumps(protocol, indent=2), encoding="utf-8"
    )
    records = []
    metrics = []
    choices = {}
    training_warnings = []
    # Select exposure forecasts on 2025, then evaluate rolling 2026 predictions.

    for year in [2025, 2026]:
        monthly_predictions = []

        for date in sorted(data.loc[data.date.dt.year == year, "date"].unique()):
            train = data[data.date < date]
            test = data[data.date == date]
            assert train.date.max() < date
            fold_predictions = test[
                [
                    "row_id",
                    "carrier",
                    "airline_id",
                    "target_month",
                    "target_rate_per_1000",
                    "target_mishandled_bags",
                    "target_enplaned_bags",
                    "passenger_segments",
                ]
            ].copy()
            fold_predictions["training_last_month"] = train.target_month.max()
            fold_predictions["year"] = year

            with warnings.catch_warnings(record=True) as captured_warnings:
                warnings.simplefilter("always")
                targets = train.target_rate_per_1000.to_numpy()
                risk_model = ForecastingModel(feature_columns).fit(train)
                fold_predictions["rate_ml"] = risk_model.predict(test)
                preprocessor = risk_model.preprocessor

                train_features = preprocessor.transform(train[feature_columns])
                test_features = preprocessor.transform(test[feature_columns])

                if pd.Timestamp(date) == pd.Timestamp("2026-08-01"):
                    risk_model.save(ROOT / "checkpoints")

                symmetric_model = LGBMRegressor(**model_parameters).fit(
                    train_features, targets
                )
                fold_predictions["rate_symmetric"] = np.maximum(
                    0, symmetric_model.predict(test_features)
                )

                for name, target_column in [
                    ("bags", "target_enplaned_bags"),
                    ("passengers", "passenger_segments"),
                ]:
                    valid = train[target_column].notna().to_numpy()
                    model = LGBMRegressor(**model_parameters).fit(
                        train_features[valid], train.loc[valid, target_column]
                    )
                    fold_predictions[name + "_lgb"] = np.maximum(
                        0, model.predict(test_features)
                    )

                warning_records = []

                for warning in captured_warnings:
                    warning_records.append(str(warning.message))

                training_warnings.extend(warning_records)

            fold_predictions["rate_simple"] = 0.5 * (
                test.recent_rate.to_numpy() + test.rate_year_ago.to_numpy()
            )
            fold_predictions["bags_seasonal"] = test.bags_year_ago.to_numpy()
            fold_predictions["bags_blend"] = 0.5 * (
                test.bags_year_ago.to_numpy() + test.recent_bag_volume.to_numpy()
            )
            fold_predictions["passengers_seasonal"] = (
                test.passengers_year_ago.to_numpy()
            )

            fold_predictions["passengers_blend"] = 0.5 * (
                test.passengers_year_ago.to_numpy()
                + test.lag_passenger_segments.to_numpy()
            )
            monthly_predictions.append(fold_predictions)

        predictions = pd.concat(monthly_predictions, ignore_index=True)

        for name, target in [
            ("bags", "target_enplaned_bags"),
            ("passengers", "passenger_segments"),
        ]:
            for method in ["lgb", "seasonal", "blend"]:
                metrics.append(
                    {
                        "year": year,
                        "target": name,
                        "method": method,
                        **count_metrics(
                            predictions[target].to_numpy(),
                            predictions[name + "_" + method].to_numpy(),
                        ),
                    }
                )

        if year == 2025:
            for name in ["bags", "passengers"]:
                selected_metrics = []

                for x in metrics:
                    if x["target"] == name and x["year"] == 2025:
                        selected_metrics.append(x)

                choices[name] = min(selected_metrics, key=lambda x: x["WAPE_percent"])[
                    "method"
                ]

            (output_directory / "volume_selection_before_2026.json").write_text(
                json.dumps(choices, indent=2), encoding="utf-8"
            )
            print("2025 volume selection", choices, flush=True)

        predictions["forecast_baggage_volume"] = predictions["bags_" + choices["bags"]]
        predictions["forecast_passenger_segments"] = predictions[
            "passengers_" + choices["passengers"]
        ]

        for rate in ["ml", "symmetric", "simple"]:
            predictions["anomaly_count_" + rate] = (
                predictions["rate_" + rate] * predictions.forecast_baggage_volume / 1000
            )
            metrics.append(
                {
                    "year": year,
                    "target": "anomaly_count",
                    "method": rate + "_same_selected_exposure",
                    **count_metrics(
                        predictions.target_mishandled_bags.to_numpy(),
                        predictions["anomaly_count_" + rate].to_numpy(),
                    ),
                }
            )

        predictions["anomaly_count_simple_full"] = (
            predictions.rate_simple * predictions.bags_blend / 1000
        )
        metrics.append(
            {
                "year": year,
                "target": "anomaly_count",
                "method": "simple_full_blend",
                **count_metrics(
                    predictions.target_mishandled_bags.to_numpy(),
                    predictions.anomaly_count_simple_full.to_numpy(),
                ),
            }
        )

        for rate in ["ml", "symmetric", "simple"]:
            errors = predictions["rate_" + rate] - predictions.target_rate_per_1000
            weights = predictions.target_enplaned_bags
            metrics.append(
                {
                    "year": year,
                    "target": "rate_per_1000",
                    "method": rate,
                    "n": len(predictions),
                    "weighted_MAE": float(np.average(abs(errors), weights=weights)),
                    "weighted_RMSE": float(
                        np.sqrt(np.average(errors**2, weights=weights))
                    ),
                    "weighted_bias": float(np.average(errors, weights=weights)),
                }
            )

        records.append(predictions)

    predictions = pd.concat(records, ignore_index=True)
    predictions.to_csv(output_directory / "predictions.csv", index=False)
    pd.DataFrame(metrics).to_csv(output_directory / "metrics.csv", index=False)
    monthly = []

    for month, month_predictions in predictions[predictions.year == 2026].groupby(
        "target_month"
    ):
        monthly.append(
            {
                "month": month,
                "cases": len(month_predictions),
                "predicted_passenger_segments": month_predictions.forecast_passenger_segments.sum(),
                "actual_passenger_segments": month_predictions.passenger_segments.sum(
                    min_count=1
                ),
                "predicted_bags": month_predictions.forecast_baggage_volume.sum(),
                "actual_bags": month_predictions.target_enplaned_bags.sum(),
                "predicted_mishandled_bags": month_predictions.anomaly_count_ml.sum(),
                "actual_mishandled_bags": month_predictions.target_mishandled_bags.sum(),
                "passenger_label_coverage": month_predictions.passenger_segments.notna().mean(),
            }
        )

    pd.DataFrame(monthly).to_csv(
        output_directory / "monthly_totals_2026.csv", index=False
    )
    frontier = predictions[predictions.year == 2026][
        [
            "row_id",
            "carrier",
            "target_month",
            "forecast_passenger_segments",
            "forecast_baggage_volume",
            "rate_ml",
            "anomaly_count_ml",
        ]
    ].copy()
    frontier["insured_claim_count"] = "UNKNOWN"
    frontier["permanent_loss_count"] = "NOT_IDENTIFIED"
    frontier["payout_budget_formula"] = (
        "forecast_passenger_segments * eligible_itinerary_factor * "
        "purchase_rate * premium - nonclaim_costs - target_profit"
    )

    frontier["average_benefit_ceiling_formula"] = (
        "payout_budget / expected_covered_claims; only when budget>=0 and claims>0"
    )
    frontier.to_csv(
        output_directory / "profitability_inputs_no_payout_assumption.csv", index=False
    )
    (output_directory / "audit.json").write_text(
        json.dumps(
            {
                "selected_volume_methods": choices,
                "warnings": training_warnings,
                "risk_retrained": True,
                "no_synthetic_rows": True,
                "no_business_parameters_filled": True,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    selected_metrics = []

    for x in metrics:
        if x["year"] == 2026:
            selected_metrics.append(x)

    print(pd.DataFrame(selected_metrics).to_string(index=False))
    interface = predictions[predictions.year == 2026].copy()
    interface = interface[interface.target_month == "2026-08-01"]
    interface = interface.rename(
        columns={
            "target_month": "period",
            "rate_ml": "risk_rate_per_1000",
            "forecast_passenger_segments": "passenger_segments_forecast",
            "forecast_baggage_volume": "bag_loads",
            "anomaly_count_ml": "predicted_mishandled_bags",
        }
    )
    interface = interface[
        [
            "period",
            "carrier",
            "risk_rate_per_1000",
            "passenger_segments_forecast",
            "bag_loads",
            "predicted_mishandled_bags",
            "training_last_month",
        ]
    ].rename(
        columns={
            "passenger_segments_forecast": "passenger_segments",
            "training_last_month": "training_cutoff",
        }
    )

    interface["population"] = "BTS US domestic, not Cathay"
    interface.to_csv(output_directory / "forecast_interface_2026_08.csv", index=False)


if __name__ == "__main__":
    main()
