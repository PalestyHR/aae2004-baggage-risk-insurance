import json
import unittest
from pathlib import Path

import pandas as pd
from models.insurance import (
    claims_from_forecast,
    expected_portfolio,
    simulate_cohort,
    tier_probabilities,
)

ROOT = Path(__file__).resolve().parents[1]


class InsuranceTests(unittest.TestCase):
    def setUp(self):
        plans_file = ROOT / "configs/selected_menu_v2.json"
        plans_config = json.loads(plans_file.read_text(encoding="utf-8"))
        self.plans = plans_config["plans"]

        assumptions_file = ROOT / "configs/insurance_assumptions.json"
        assumptions_config = json.loads(assumptions_file.read_text(encoding="utf-8"))
        self.assumptions = assumptions_config["base_assumptions"]

        forecast_file = ROOT / "results/reference/model1/forecast_interface_2026_08.csv"
        forecasts = pd.read_csv(forecast_file)
        first_forecast = forecasts.iloc[0]
        self.forecast = first_forecast.to_dict()

    def test_claim_probabilities(self):
        risk = claims_from_forecast(20, 1, 1, 1, 0.5, 0.02)
        self.assertAlmostEqual(risk["delay_claim_probability"], 0.01)
        self.assertAlmostEqual(risk["loss_claim_probability"], 0.0004)
        self.assertAlmostEqual(risk["no_paid_claim_probability"], 0.9896)

    def test_reference_purchase_probabilities(self):
        for plan in self.plans:
            plan.update(plan["reference"])

        choices = tier_probabilities(self.plans, self.assumptions["demand"])

        demand = self.assumptions["demand"]
        reference_probabilities = demand["reference_purchase_probabilities"]

        for name, value in reference_probabilities.items():
            self.assertAlmostEqual(choices[name], value)

        self.assertAlmostEqual(sum(choices.values()), 1)

    def test_expected_profit(self):
        result = expected_portfolio(self.forecast, self.plans, self.assumptions)
        ref = pd.read_csv(
            ROOT / "results/reference/insurance/selected_operator_results.csv"
        )
        expected = ref[
            (ref.carrier == self.forecast["carrier"]) & (ref.scenario == "beta1.0_base")
        ].iloc[0]
        self.assertAlmostEqual(
            result["expected_profit"], expected.expected_profit, places=6
        )

    def test_simulated_mean(self):
        result = simulate_cohort(
            self.forecast, self.plans, self.assumptions, draws=5000
        )
        error = abs(
            result["simulated_mean_profit"] - result["expected_profit_analytic"]
        )
        self.assertLessEqual(error, 6 * result["mean_standard_error"])


if __name__ == "__main__":
    unittest.main()
