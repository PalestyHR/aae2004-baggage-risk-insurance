"""Simulate monthly profit for the selected baggage protection menu."""

from pathlib import Path
import sys
import json
import copy
import argparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pandas as pd
from models.insurance import simulate_cohort


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--forecast",
        type=Path,
        default=ROOT / "results/reference/model1/forecast_interface_2026_08.csv",
        help="Forecast CSV; defaults to the published reference.",
    )
    parser.add_argument(
        "--menu",
        type=Path,
        default=ROOT / "configs/selected_menu_v2.json",
        help="Menu JSON; defaults to the published selected menu.",
    )
    args = parser.parse_args()
    output_directory = ROOT / "results/selected_menu_monte_carlo"
    output_directory.mkdir(parents=True, exist_ok=True)
    forecasts = pd.read_csv(args.forecast)
    plans_file = args.menu
    plans_config = json.loads(plans_file.read_text(encoding="utf-8"))

    plans = plans_config["plans"]
    assumptions_file = ROOT / "configs/insurance_assumptions.json"
    assumptions_config = json.loads(assumptions_file.read_text(encoding="utf-8"))

    base_assumptions = assumptions_config["base_assumptions"]
    forecast_records = forecasts.to_dict("records")
    rows = []

    for price_sensitivity in [1.0, 2.0, 3.0]:
        for risk, risk_multiplier, loss_share in [
            ("base", 1.0, 0.05),
            ("higher", 2.0, 0.1),
        ]:
            assumptions = copy.deepcopy(base_assumptions)
            demand = assumptions["demand"]
            demand["log_price_sensitivity"] = price_sensitivity
            assumptions["conditional_loss_share"] = loss_share

            assumptions["tier_hazard_multipliers"] = {
                plan["name"]: risk_multiplier for plan in plans
            }

            for case_index, row in enumerate(forecast_records):
                risk_seed_offset = 1000 if risk == "higher" else 0
                sensitivity_seed_offset = int(price_sensitivity) * 100
                seed = (
                    20261106 + sensitivity_seed_offset + case_index + risk_seed_offset
                )
                simulation = simulate_cohort(
                    row, plans, assumptions, draws=30000, seed=seed
                )
                assert (
                    abs(
                        simulation["simulated_mean_profit"]
                        - simulation["expected_profit_analytic"]
                    )
                    <= 6 * simulation["mean_standard_error"]
                )
                rows.append(
                    {
                        "scenario": f"beta{price_sensitivity}_{risk}",
                        "carrier": row["carrier"],
                        **simulation,
                    }
                )

    results = pd.DataFrame(rows)
    results.to_csv(output_directory / "results.csv", index=False)
    summary = (
        results.groupby("scenario")
        .agg(
            cases=("carrier", "size"),
            minimum_profit_probability=("probability_profit_positive", "min"),
            maximum_profit_probability=("probability_profit_positive", "max"),
            minimum_05_profit_quantile=("profit_quantile_05", "min"),
        )
        .reset_index()
    )
    summary.to_csv(output_directory / "summary.csv", index=False)
    (output_directory / "notes.txt").write_text(
        (
            "Selected input menu; 72 single-operator single-cohort "
            "scenarios, 30000 draws each. Same uncalibrated "
            "risk/demand/cost assumptions as optimization. All means "
            "reconcile within six Monte Carlo standard errors. This is not "
            "annual empirical profitability or a certification of 95% "
            "profit probability. No parameter/forecast uncertainty has been"
            " calibrated.\n"
        ),
        encoding="utf-8",
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
