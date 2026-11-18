"""Baggage protection pricing and profitability for one sales cohort.

The calculation follows claim risk, passenger choice, portfolio economics
and random claims. All monetary inputs use the configured currency.
"""

import math

import numpy as np


def claims_from_forecast(
    rate_per_1000,
    bag_count,
    segment_count,
    portfolio_hazard_multiplier,
    delay_share,
    loss_share,
):
    """Convert a baggage forecast into delay and final-loss probabilities."""
    bag_probability = rate_per_1000 / 1000
    exposures = bag_count * segment_count
    if portfolio_hazard_multiplier == 0 or bag_probability == 0:
        mishandling_probability = 0.0
    elif bag_probability == 1:
        mishandling_probability = 1.0
    else:
        mishandling_probability = -math.expm1(
            exposures * portfolio_hazard_multiplier * math.log1p(-bag_probability)
        )
    delay_probability = mishandling_probability * delay_share
    loss_probability = mishandling_probability * loss_share
    return {
        "policy_mishandling_probability": mishandling_probability,
        "delay_claim_probability": delay_probability,
        "loss_claim_probability": loss_probability,
        "no_paid_claim_probability": max(0.0, 1 - delay_probability - loss_probability),
        "status": "scenario_conversion_not_calibrated",
    }


def tier_probabilities(plans, demand):
    """Compute multinomial-logit probabilities, including no purchase."""
    shares = demand["reference_purchase_probabilities"]
    names = []
    total = 0.0

    for plan in plans:
        name = plan["name"]
        purchase_probability = shares[name]
        names.append(name)
        total += purchase_probability

    no_purchase_probability = 1 - total
    price_sensitivity = demand["log_price_sensitivity"]
    delay_sensitivity = demand["log_delay_benefit_sensitivity"]
    loss_sensitivity = demand["log_loss_benefit_sensitivity"]
    logits = [0.0]

    for plan in plans:
        name = plan["name"]
        purchase_probability = shares[name]
        reference = plan["reference"]
        price_ratio = plan["price"] / reference["price"]
        delay_ratio = plan["delay_benefit"] / reference["delay_benefit"]
        loss_ratio = plan["loss_benefit"] / reference["loss_benefit"]

        if purchase_probability == 0:
            utility = -math.inf
        else:
            utility = math.log(purchase_probability / no_purchase_probability)

        utility -= price_sensitivity * math.log(price_ratio)
        utility += delay_sensitivity * math.log(delay_ratio)
        utility += loss_sensitivity * math.log(loss_ratio)
        logits.append(utility)

    max_logit = max(logits)
    exp_utilities = []
    for utility in logits:
        exp_utility = math.exp(utility - max_logit)
        exp_utilities.append(exp_utility)

    normalizer = sum(exp_utilities)
    choice_names = ["no_purchase"]
    choice_names.extend(names)
    probabilities = {}
    for index, name in enumerate(choice_names):
        probabilities[name] = exp_utilities[index] / normalizer

    return probabilities


def expected_portfolio(forecast, plans, assumptions):
    """Calculate expected sales, payouts, expenses and profit for one cohort."""
    passenger_volume = forecast["passenger_segments"]
    eligible_fraction = assumptions["eligible_opportunity_fraction"]
    eligible_count = math.floor(passenger_volume * eligible_fraction)
    covered_bags = assumptions["covered_bags"]
    covered_segments = assumptions["covered_segments"]
    choice_probabilities = tier_probabilities(plans, assumptions["demand"])
    states = assumptions["risk_states"]
    delay_share = assumptions["conditional_delay_share"]
    loss_share = assumptions["conditional_loss_share"]
    system_cost = assumptions["period_system_cost"]
    target_profit = assumptions["target_period_profit"]
    tier_rows = []
    state_claim_probabilities = {}
    for plan in plans:
        name = plan["name"]
        delay_probability = loss_probability = 0
        state_claim_probabilities[name] = {}
        for state in states:
            hazard_multiplier = (
                assumptions["tier_hazard_multipliers"][name]
                * state["hazard_multiplier"]
            )
            risk = claims_from_forecast(
                forecast["risk_rate_per_1000"],
                covered_bags,
                covered_segments,
                hazard_multiplier,
                delay_share,
                loss_share,
            )
            state_claim_probabilities[name][state["name"]] = risk
            delay_probability += state["probability"] * risk["delay_claim_probability"]
            loss_probability += state["probability"] * risk["loss_claim_probability"]
        policy_count = eligible_count * choice_probabilities[name]
        claim_per_policy = (
            delay_probability * plan["delay_benefit"]
            + loss_probability * plan["loss_benefit"]
        )
        variable_cost = assumptions["variable_cost_per_policy"][name]
        expense_rate = assumptions["external_expense_rates"][name]
        revenue = policy_count * plan["price"]
        claims = policy_count * claim_per_policy
        expense = policy_count * variable_cost + expense_rate * revenue
        tier_rows.append(
            {
                "name": name,
                "purchase_probability": choice_probabilities[name],
                "expected_policies": policy_count,
                "delay_claim_probability": delay_probability,
                "loss_claim_probability": loss_probability,
                "expected_paid_policies": policy_count
                * (delay_probability + loss_probability),
                "expected_claim_per_policy": claim_per_policy,
                "revenue": revenue,
                "claims": claims,
                "nonclaim_variable_expense": expense,
                "contribution_before_shared_system_cost": revenue - claims - expense,
            }
        )
    revenue = sum((r["revenue"] for r in tier_rows))
    claims = sum((r["claims"] for r in tier_rows))
    expense = sum((r["nonclaim_variable_expense"] for r in tier_rows))
    sales = sum((r["expected_policies"] for r in tier_rows))
    profit = revenue - claims - expense - system_cost
    budget = revenue - expense - system_cost - target_profit
    return {
        "currency": assumptions["currency"],
        "forecast_period": forecast["period"],
        "eligible_opportunities": eligible_count,
        "tiers": tier_rows,
        "choice_probabilities": choice_probabilities,
        "risk_states": states,
        "state_claim_probabilities": state_claim_probabilities,
        "expected_sales": sales,
        "expected_revenue": revenue,
        "expected_claims": claims,
        "nonclaim_variable_expense": expense,
        "period_system_cost": system_cost,
        "expected_profit": profit,
        "expected_profit_margin": profit / revenue if revenue else None,
        "expected_loss_ratio": claims / revenue if revenue else None,
        "expected_paid_policy_fraction": (
            sum((r["expected_paid_policies"] for r in tier_rows)) / sales
            if sales
            else None
        ),
        "allowable_total_payout_at_target_profit": budget,
        "payout_budget_slack": budget - claims,
        "meets_target_expected_profit": profit >= target_profit,
        "evidence_status": "scenario_economics_not_validated_sales_or_claims",
    }


def simulate_cohort(forecast, plans, assumptions, draws=20000, seed=42):
    """Simulate purchases and mutually exclusive claims under shared risk states."""

    expected = expected_portfolio(forecast, plans, assumptions)
    rng = np.random.default_rng(seed)
    names = [plan["name"] for plan in plans]
    choice = expected["choice_probabilities"]
    counts = rng.multinomial(
        expected["eligible_opportunities"],
        [choice[n] for n in ["no_purchase"] + names],
        size=draws,
    )
    states = assumptions["risk_states"]
    sampled_states = rng.choice(
        len(states), size=draws, p=[x["probability"] for x in states]
    )
    profits = np.full(draws, -float(assumptions["period_system_cost"]))
    for plan_index, plan in enumerate(plans, start=1):
        policy_counts = counts[:, plan_index]
        name = plan["name"]
        price = plan["price"]
        profits += policy_counts * (
            price * (1 - assumptions["external_expense_rates"][name])
            - assumptions["variable_cost_per_policy"][name]
        )
        for state_index, state in enumerate(states):
            state_indices = np.where(sampled_states == state_index)[0]
            risk = expected["state_claim_probabilities"][name][state["name"]]
            delay_probability = risk["delay_claim_probability"]
            loss_probability = risk["loss_claim_probability"]
            lost = rng.binomial(policy_counts[state_indices], loss_probability)
            delayed = rng.binomial(
                policy_counts[state_indices] - lost,
                (
                    delay_probability / (1 - loss_probability)
                    if loss_probability < 1
                    else 0
                ),
            )
            profits[state_indices] -= (
                lost * plan["loss_benefit"] + delayed * plan["delay_benefit"]
            )
    return {
        "expected_profit_analytic": expected["expected_profit"],
        "simulated_mean_profit": float(profits.mean()),
        "mean_standard_error": float(profits.std(ddof=1) / math.sqrt(draws)),
        "probability_profit_positive": float((profits > 0).mean()),
        "probability_target_met": float(
            (profits >= assumptions["target_period_profit"]).mean()
        ),
        "profit_quantile_05": float(np.quantile(profits, 0.05)),
        "profit_quantile_50": float(np.quantile(profits, 0.5)),
        "profit_quantile_95": float(np.quantile(profits, 0.95)),
        "draws": draws,
        "seed": seed,
        "scope": "single cohort under explicit assumptions; not calibrated annual profitability",
    }
