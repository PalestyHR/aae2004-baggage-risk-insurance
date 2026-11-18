"""Compatibility imports for the original package path."""

from models.insurance import (
    benefit_frontier,
    bounded,
    claims_from_forecast,
    expected_portfolio,
    nonnegative,
    pareto_frontier,
    positive,
    simulate_cohort,
    tier_probabilities,
)

__all__ = [
    "benefit_frontier",
    "bounded",
    "claims_from_forecast",
    "expected_portfolio",
    "nonnegative",
    "pareto_frontier",
    "positive",
    "simulate_cohort",
    "tier_probabilities",
]
