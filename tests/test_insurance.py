import unittest
import math
from baggage_project.insurance import (
    claims_from_forecast,
    tier_probabilities,
    benefit_frontier,
    pareto_frontier,
)


class InsuranceTests(unittest.TestCase):
    def test_event_partition(self):
        q = claims_from_forecast(20, 1, 1, 1, 0.5, 0.02)
        self.assertAlmostEqual(q["delay_claim_probability"], 0.01)
        self.assertAlmostEqual(q["loss_claim_probability"], 0.0004)
        self.assertAlmostEqual(q["no_paid_claim_probability"], 0.9896)
        with self.assertRaises(ValueError):
            claims_from_forecast(5, 1, 1, 1, 0.9, 0.2)

    def test_no_silent_calibration(self):
        with self.assertRaises(ValueError):
            claims_from_forecast(5, 1, 1, None, 0.5, 0.05)

    def test_budget_boundary(self):
        r = benefit_frontier(
            policies=100,
            price=10,
            expense_rate=0.1,
            variable_cost=1,
            allocated_fixed_cost=100,
            target_profit=100,
            qd=0.01,
            ql=0.001,
            delay_benefit=100,
        )
        self.assertAlmostEqual(r["loss_benefit_ceiling"], 5000)
        # Substitute the ceiling into the accounting equation.
        self.assertAlmostEqual(
            100 * (10 * 0.9 - 1 - 0.01 * 100 - 0.001 * r["loss_benefit_ceiling"]) - 100,
            100,
        )

    def test_pareto_tradeoff(self):
        rows = [
            {
                "name": "A",
                "expected_profit": 30,
                "expected_profit_margin": 0.3,
                "feasible": True,
            },
            {
                "name": "B",
                "expected_profit": 20,
                "expected_profit_margin": 0.5,
                "feasible": True,
            },
            {
                "name": "C",
                "expected_profit": 10,
                "expected_profit_margin": 0.2,
                "feasible": True,
            },
        ]
        self.assertEqual({r["name"] for r in pareto_frontier(rows)}, {"A", "B"})


if __name__ == "__main__":
    unittest.main()
