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
        self.plans = json.loads((ROOT / "configs/selected_menu_v2.json").read_text())[
            "plans"
        ]
        self.assumptions = json.loads(
            (ROOT / "configs/insurance_assumptions.json").read_text()
        )["base_assumptions"]
        self.forecast = (
            pd.read_csv(
                ROOT / "results/reference/model1/forecast_interface_2026_08.csv"
            )
            .iloc[0]
            .to_dict()
        )

    def test_claim_probabilities(self):
        risk = claims_from_forecast(20, 1, 1, 1, 0.5, 0.02)
        self.assertAlmostEqual(risk["delay_claim_probability"], 0.01)
        self.assertAlmostEqual(risk["loss_claim_probability"], 0.0004)
        self.assertAlmostEqual(risk["no_paid_claim_probability"], 0.9896)

    def test_reference_purchase_probabilities(self):
        for plan in self.plans:
            plan.update(plan["reference"])
        choices = tier_probabilities(self.plans, self.assumptions["demand"])
        for name, value in self.assumptions["demand"][
            "reference_purchase_probabilities"
        ].items():
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
