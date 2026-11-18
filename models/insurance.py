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
        log_survival = math.log1p(-bag_probability)
        scaled_log_survival = exposures * portfolio_hazard_multiplier * log_survival
        mishandling_probability = -math.expm1(scaled_log_survival)

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

    demand = assumptions["demand"]
    tier_hazards = assumptions["tier_hazard_multipliers"]
    processing_costs = assumptions["variable_cost_per_policy"]
    expense_rates = assumptions["external_expense_rates"]

    forecast_rate = forecast["risk_rate_per_1000"]
    choice_probabilities = tier_probabilities(plans, demand)

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
        plan_risks = {}
        tier_hazard = tier_hazards[name]

        delay_benefit = plan["delay_benefit"]
        loss_benefit = plan["loss_benefit"]
        price = plan["price"]
        purchase_probability = choice_probabilities[name]

        for state in states:
            state_name = state["name"]
            state_probability = state["probability"]
            state_hazard = state["hazard_multiplier"]
            hazard_multiplier = tier_hazard * state_hazard

            risk = claims_from_forecast(
                forecast_rate,
                covered_bags,
                covered_segments,
                hazard_multiplier,
                delay_share,
                loss_share,
            )
            state_delay_probability = risk["delay_claim_probability"]
            state_loss_probability = risk["loss_claim_probability"]
            plan_risks[state_name] = risk

            delay_probability += state_probability * state_delay_probability
            loss_probability += state_probability * state_loss_probability

        state_claim_probabilities[name] = plan_risks
        policy_count = eligible_count * purchase_probability
        claim_per_policy = (
            delay_probability * delay_benefit + loss_probability * loss_benefit
        )
        variable_cost = processing_costs[name]

        expense_rate = expense_rates[name]
        revenue = policy_count * price
        claims = policy_count * claim_per_policy
        expense = policy_count * variable_cost + expense_rate * revenue

        paid_probability = delay_probability + loss_probability
        paid_policies = policy_count * paid_probability
        contribution = revenue - claims - expense
        tier_rows.append(
            {
                "name": name,
                "purchase_probability": choice_probabilities[name],
                "expected_policies": policy_count,
                "delay_claim_probability": delay_probability,
                "loss_claim_probability": loss_probability,
                "expected_paid_policies": paid_policies,
                "expected_claim_per_policy": claim_per_policy,
                "revenue": revenue,
                "claims": claims,
                "nonclaim_variable_expense": expense,
                "contribution_before_shared_system_cost": contribution,
            }
        )

    revenue = 0
    claims = 0
    expense = 0
    sales = 0

    paid_policies = 0

    for tier in tier_rows:
        revenue += tier["revenue"]
        claims += tier["claims"]
        expense += tier["nonclaim_variable_expense"]
        sales += tier["expected_policies"]

        paid_policies += tier["expected_paid_policies"]

    profit = revenue - claims - expense - system_cost
    budget = revenue - expense - system_cost - target_profit
    profit_margin = None
    loss_ratio = None

    paid_policy_fraction = None

    if revenue:
        profit_margin = profit / revenue
        loss_ratio = claims / revenue

    if sales:
        paid_policy_fraction = paid_policies / sales

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
        "expected_profit_margin": profit_margin,
        "expected_loss_ratio": loss_ratio,
        "expected_paid_policy_fraction": (paid_policies / sales if sales else None),
        "allowable_total_payout_at_target_profit": budget,
        "payout_budget_slack": budget - claims,
        "meets_target_expected_profit": profit >= target_profit,
        "evidence_status": "scenario_economics_not_validated_sales_or_claims",
    }


def simulate_cohort(forecast, plans, assumptions, draws=20000, seed=42):
    """Simulate purchases and mutually exclusive claims under shared risk states."""

    expected = expected_portfolio(forecast, plans, assumptions)
    rng = np.random.default_rng(seed)
    names = ["no_purchase"]

    for plan in plans:
        name = plan["name"]
        names.append(name)

    choice = expected["choice_probabilities"]
    eligible_count = expected["eligible_opportunities"]
    claim_probabilities = expected["state_claim_probabilities"]
    processing_costs = assumptions["variable_cost_per_policy"]

    expense_rates = assumptions["external_expense_rates"]
    system_cost = float(assumptions["period_system_cost"])
    states = assumptions["risk_states"]

    purchase_probabilities = []

    for name in names:
        probability = choice[name]
        purchase_probabilities.append(probability)

    state_probabilities = []

    for state in states:
        probability = state["probability"]
        state_probabilities.append(probability)

    state_count = len(states)
    counts = rng.multinomial(eligible_count, purchase_probabilities, size=draws)
    sampled_states = rng.choice(state_count, size=draws, p=state_probabilities)
    profits = np.full(draws, -system_cost)

    for plan_index, plan in enumerate(plans, start=1):
        policy_counts = counts[:, plan_index]
        name = plan["name"]
        price = plan["price"]
        profits += policy_counts * (
            price * (1 - expense_rates[name]) - processing_costs[name]
        )

        plan_risks = claim_probabilities[name]

        for state_index, state in enumerate(states):
            state_name = state["name"]
            state_mask = sampled_states == state_index
            state_indices = np.where(state_mask)[0]
            risk = plan_risks[state_name]

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

    mean_profit = float(profits.mean())
    standard_deviation = profits.std(ddof=1)
    standard_error = float(standard_deviation / math.sqrt(draws))
    profitable_trials = profits > 0

    target_profit = assumptions["target_period_profit"]
    target_met_trials = profits >= target_profit
    profit_probability = float(profitable_trials.mean())
    target_probability = float(target_met_trials.mean())

    return {
        "expected_profit_analytic": expected["expected_profit"],
        "simulated_mean_profit": mean_profit,
        "mean_standard_error": standard_error,
        "probability_profit_positive": profit_probability,
        "probability_target_met": target_probability,
        "profit_quantile_05": float(np.quantile(profits, 0.05)),
        "profit_quantile_50": float(np.quantile(profits, 0.5)),
        "profit_quantile_95": float(np.quantile(profits, 0.95)),
        "draws": draws,
        "seed": seed,
        "scope": "single cohort under explicit assumptions; not calibrated annual profitability",
    }
