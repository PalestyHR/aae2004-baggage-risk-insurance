"""Original import path for the four insurance calculations."""

from models.insurance import (
    claims_from_forecast,
    tier_probabilities,
    expected_portfolio,
    simulate_cohort,
)

__all__ = [
    "claims_from_forecast",
    "tier_probabilities",
    "expected_portfolio",
    "simulate_cohort",
]
