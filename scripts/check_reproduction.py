from pathlib import Path
import argparse
import sys
import json

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from models.forecasting_model import FrozenRiskPredictor


def compare(left, right, keys, values, tolerance=1e-6):
    a = pd.read_csv(left)
    b = pd.read_csv(right)
    merged = a.merge(b, on=keys, suffixes=("_new", "_ref"), validate="one_to_one")
    assert len(merged) == len(a) == len(b)

    for c in values:
        assert np.allclose(
            merged[c + "_new"],
            merged[c + "_ref"],
            rtol=1e-8,
            atol=tolerance,
            equal_nan=True,
        ), c

    return len(merged)


def main():
    parser = argparse.ArgumentParser(description="Verify reproduced numerical outputs.")
    parser.add_argument(
        "--full",
        action="store_true",
        help="Also verify baselines, loss ratios and simulations.",
    )
    args = parser.parse_args()
    count = compare(
        ROOT / "results/reproduced_risk/predictions.csv",
        ROOT / "results/reference/model1/predictions.csv",
        ["row_id"],
        [
            "rate_ml",
            "forecast_baggage_volume",
            "forecast_passenger_segments",
            "anomaly_count_ml",
        ],
    )
    df = pd.read_csv(ROOT / "data/processed/lag1/test_real.csv")
    df = df[df.target_month == "2026-08-01"]

    loaded = FrozenRiskPredictor(ROOT / "checkpoints").predict(df)
    saved = (
        pd.read_csv(ROOT / "results/reproduced_risk/predictions.csv")
        .set_index("row_id")
        .loc[df.row_id, "rate_ml"]
        .to_numpy()
    )
    assert np.allclose(loaded, saved, rtol=1e-9, atol=1e-7)
    new = json.loads(
        (ROOT / "results/reproduced_insurance/selected_menu.json").read_text()
    )
    ref = json.loads((ROOT / "configs/selected_menu_v2.json").read_text())
    assert new["selection"]["design_id"] == ref["selection"]["design_id"] == "M0775"

    for key in ["robust_expected_profit", "robust_margin"]:
        assert np.isclose(
            new["selection"][key], ref["selection"][key], rtol=1e-8, atol=1e-6
        )

    assert new["plans"] == ref["plans"]
    compare(
        ROOT / "results/reproduced_insurance/selected_scenarios.csv",
        ROOT / "results/reference/insurance/selected_scenarios.csv",
        ["scenario"],
        ["total_expected_profit", "pooled_margin", "total_expected_claims"],
    )
    result = {
        "risk_records_reproduced": count,
        "frozen_booster_roundtrip_cases": len(df),
        "insurance_selected_id": "M0775",
        "status": "PASS",
        "note": "Numerical reproduction verifies implementation, not empirical demand/claim assumptions.",
    }

    if args.full:
        result["baseline_records_reproduced"] = compare(
            ROOT / "results/reproduced_baselines/predictions.csv",
            ROOT / "results/reference/integrated_model_comparison/predictions.csv",
            ["row_id", "year", "model"],
            ["prediction"],
        )
        result["loss_ratio_records_reproduced"] = compare(
            ROOT / "results/reproduced_loss_ratios/predictions.csv",
            ROOT / "results/reference/asymmetric_loss_comparison/predictions.csv",
            ["row_id", "year", "weighting", "under_over_ratio"],
            ["prediction"],
        )
        selection = json.loads(
            (
                ROOT / "results/reproduced_loss_ratios/selection_before_2026.json"
            ).read_text()
        )
        assert selection["overall"]["under_over_ratio"] == "1:2"
        result["selected_loss_ratio"] = "1:2"
        simulations = pd.read_csv(
            ROOT / "results/selected_menu_monte_carlo/results.csv"
        )
        assert len(simulations) == 72
        assert (simulations.draws == 30000).all()
        assert (
            abs(
                simulations.simulated_mean_profit - simulations.expected_profit_analytic
            )
            <= 6 * simulations.mean_standard_error
        ).all()
        result["monte_carlo_cases"] = len(simulations)
        result["draws_per_case"] = 30000

    (ROOT / "results/reproduction_check.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
