"""Insurance design layer on the frozen BTS forecasting pipeline.

All monetary values use one declared currency. No probabilities, customer
preferences or system costs are silently calibrated from BTS mishandling data.
The portfolio unit here is one sales cohort, e.g. one month's departures.
"""

import math


def bounded(x, name, lo=0.0, hi=1.0):
    if x is None or not math.isfinite(float(x)) or not lo <= x <= hi:
        raise ValueError(f"{name} must be supplied in [{lo}, {hi}]")
    return float(x)


def nonnegative(x, name):
    if x is None or not math.isfinite(float(x)) or x < 0:
        raise ValueError(f"{name} must be supplied and nonnegative")
    return float(x)


def positive(x, name):
    x = nonnegative(x, name)
    if x == 0:
        raise ValueError(f"{name} must be positive")
    return x


def claims_from_forecast(
    rate_per_1000,
    bag_count,
    segment_count,
    portfolio_hazard_multiplier,
    delay_share,
    loss_share,
):
    """Assumed unit mapping, NOT empirical calibration.
    One bag/one segment reduces to the scaled baggage rate at multiplier=1.
    Additional exposures use an independence-style survival approximation.
    Shares are conditional FINAL policy outcomes, not bag-level industry shares.
    """
    bag_probability = bounded(rate_per_1000 / 1000, "predicted rate / 1000")
    for name, x in [("bag_count", bag_count), ("segment_count", segment_count)]:
        if not isinstance(x, int) or isinstance(x, bool) or x < 1:
            raise ValueError(name + " must be a positive integer")
    hazard_multiplier = nonnegative(
        portfolio_hazard_multiplier, "portfolio_hazard_multiplier"
    )
    delay_fraction = bounded(delay_share, "conditional delay share")
    loss_fraction = bounded(loss_share, "conditional loss share")
    if delay_fraction + loss_fraction > 1 + 1e-12:
        raise ValueError("Final delay and loss shares must sum to <=1")
    exposures = bag_count * segment_count
    mishandling_probability = (
        0.0
        if hazard_multiplier == 0 or bag_probability == 0
        else (
            1.0
            if bag_probability == 1
            else -math.expm1(
                exposures * hazard_multiplier * math.log1p(-bag_probability)
            )
        )
    )
    delay_probability, loss_probability = (
        mishandling_probability * delay_fraction,
        mishandling_probability * loss_fraction,
    )
    return {
        "policy_mishandling_probability": mishandling_probability,
        "delay_claim_probability": delay_probability,
        "loss_claim_probability": loss_probability,
        "no_paid_claim_probability": max(0.0, 1 - delay_probability - loss_probability),
        "status": "scenario_conversion_not_calibrated",
    }


def tier_probabilities(plans, demand):
    """Multinomial logit with a no-purchase option; a customer buys <=1 tier.
    Reference shares and preference coefficients are required scenario inputs.
    Coverage changes affect utility explicitly, not just cost.
    """
    names = [plan["name"] for plan in plans]
    if len(set(names)) != len(names):
        raise ValueError("Duplicate plan names")
    shares = demand["reference_purchase_probabilities"]
    if set(shares) != set(names):
        raise ValueError("Reference shares must match plan names")
    total = sum(bounded(shares[n], n + " reference share") for n in names)
    if total >= 1:
        raise ValueError("Positive reference no-purchase share required")
    no_purchase_probability = 1 - total
    logits = [0.0]
    price_sensitivity = nonnegative(
        demand["log_price_sensitivity"], "log_price_sensitivity"
    )
    delay_sensitivity = nonnegative(
        demand["log_delay_benefit_sensitivity"], "log_delay_benefit_sensitivity"
    )
    loss_sensitivity = nonnegative(
        demand["log_loss_benefit_sensitivity"], "log_loss_benefit_sensitivity"
    )
    for plan in plans:
        reference = plan["reference"]
        price = positive(plan["price"], "price")
        delay_benefit = positive(plan["delay_benefit"], "delay_benefit")
        loss_benefit = positive(plan["loss_benefit"], "loss_benefit")
        if loss_benefit < delay_benefit:
            raise ValueError("Loss benefit must cover previously paid delay benefit")
        utility = (
            -math.inf
            if shares[plan["name"]] == 0
            else math.log(shares[plan["name"]] / no_purchase_probability)
        )
        utility -= price_sensitivity * math.log(
            price / positive(reference["price"], "reference price")
        )
        utility += delay_sensitivity * math.log(
            delay_benefit
            / positive(reference["delay_benefit"], "reference delay benefit")
        )
        utility += loss_sensitivity * math.log(
            loss_benefit / positive(reference["loss_benefit"], "reference loss benefit")
        )
        logits.append(utility)
    max_logit = max(logits)
    exp_utilities = [math.exp(x - max_logit) for x in logits]
    normalizer = sum(exp_utilities)
    return dict(zip(["no_purchase"] + names, [x / normalizer for x in exp_utilities]))


def expected_portfolio(forecast, plans, assumptions):
    """Returns expected cohort economics, after specified allocated system cost.
    No forecasting test labels are accepted or needed by this interface.
    """
    names = [plan["name"] for plan in plans]
    passenger_volume = nonnegative(
        forecast["passenger_segments"], "forecast passenger segments"
    )
    bag_volume = nonnegative(forecast["bag_loads"], "forecast bag loads")
    eligible_fraction = bounded(
        assumptions["eligible_opportunity_fraction"], "eligible_opportunity_fraction"
    )
    eligible_count = math.floor(passenger_volume * eligible_fraction)
    covered_bags = assumptions["covered_bags"]
    covered_segments = assumptions["covered_segments"]
    if eligible_count * covered_bags * covered_segments > bag_volume + 1e-6:
        raise ValueError(
            "Eligible policy exposures exceed forecast bag loads; revise unit mapping"
        )
    choice_probabilities = tier_probabilities(plans, assumptions["demand"])
    states = assumptions["risk_states"]
    if not states or not math.isclose(
        sum(bounded(t["probability"], "state probability") for t in states),
        1,
        abs_tol=1e-9,
    ):
        raise ValueError("Risk-state probabilities must sum to one")
    if len({t["name"] for t in states}) != len(states):
        raise ValueError("Duplicate risk-state names")
    delay_share = assumptions["conditional_delay_share"]
    loss_share = assumptions["conditional_loss_share"]
    system_cost = nonnegative(assumptions["period_system_cost"], "period_system_cost")
    target_profit = nonnegative(
        assumptions["target_period_profit"], "target_period_profit"
    )
    if set(assumptions["tier_hazard_multipliers"]) != set(names):
        raise ValueError("Provide each tier hazard multiplier")
    if set(assumptions["variable_cost_per_policy"]) != set(names):
        raise ValueError("Provide each tier variable cost")
    if set(assumptions["external_expense_rates"]) != set(names):
        raise ValueError("Provide each tier expense rate")
    tier_rows = []
    state_claim_probabilities = {}
    for plan in plans:
        name = plan["name"]
        delay_probability = loss_probability = 0
        state_claim_probabilities[name] = {}
        for state in states:
            hazard_multiplier = nonnegative(
                assumptions["tier_hazard_multipliers"][name], "tier multiplier"
            ) * nonnegative(state["hazard_multiplier"], "state multiplier")
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
        variable_cost = nonnegative(
            assumptions["variable_cost_per_policy"][name], "variable cost"
        )
        expense_rate = bounded(
            assumptions["external_expense_rates"][name],
            "external expense rate",
            0,
            0.999999999,
        )
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
    revenue = sum(r["revenue"] for r in tier_rows)
    claims = sum(r["claims"] for r in tier_rows)
    expense = sum(r["nonclaim_variable_expense"] for r in tier_rows)
    sales = sum(r["expected_policies"] for r in tier_rows)
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
            sum(r["expected_paid_policies"] for r in tier_rows) / sales
            if sales
            else None
        ),
        "allowable_total_payout_at_target_profit": budget,
        "payout_budget_slack": budget - claims,
        "meets_target_expected_profit": profit >= target_profit,
        "evidence_status": "scenario_economics_not_validated_sales_or_claims",
    }


def benefit_frontier(
    *,
    policies,
    price,
    expense_rate,
    variable_cost,
    allocated_fixed_cost,
    target_profit,
    qd,
    ql,
    delay_benefit,
):
    """Conditional loss-benefit ceiling. If price/benefit changes uptake, rerun demand.
    Returns no automatic infinite benefit when loss probability is zero.
    """
    policy_count = positive(policies, "policies")
    premium = positive(price, "price")
    proportional_expense = bounded(expense_rate, "expense rate", 0, 0.999999999)
    processing_cost = nonnegative(variable_cost, "variable cost")
    fixed_cost = nonnegative(allocated_fixed_cost, "fixed cost")
    profit_target = nonnegative(target_profit, "target profit")
    qd = bounded(qd, "qd")
    ql = bounded(ql, "ql")
    delay_amount = nonnegative(delay_benefit, "delay benefit")
    if qd + ql > 1:
        raise ValueError("Invalid mutually exclusive probabilities")
    allowance = (
        premium * (1 - proportional_expense)
        - processing_cost
        - (fixed_cost + profit_target) / policy_count
    )
    residual = allowance - qd * delay_amount
    return {
        "payout_allowance_per_policy": allowance,
        "remaining_after_delay": residual,
        "loss_benefit_ceiling": residual / ql if ql > 0 and residual >= 0 else None,
        "status": (
            "conditional_ceiling"
            if ql > 0 and residual >= 0
            else "infeasible_or_unidentified"
        ),
    }


def simulate_cohort(forecast, plans, assumptions, draws=20000, seed=42):
    """Scenario-only cohort profit distribution. A shared risk state creates
    cross-policy dependence; conditional claims use mutually exclusive multinomial
    outcomes. Not an annual probability from one observed month.
    """
    import numpy as np

    if not isinstance(draws, int) or draws < 100:
        raise ValueError("Use >=100 draws")
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
            # Vectorized sequential binomials are equivalent to a multinomial:
            # lost first, then delayed among policies not classed as lost.
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


def pareto_frontier(candidates):
    """Maximize expected profit and margin, among externally constrained designs.
    Customer coverage/demand/risk requirements must be checked before passing in.
    """
    feasible = [
        c
        for c in candidates
        if c.get("feasible", False) and c.get("expected_profit_margin") is not None
    ]
    result = []
    for candidate in feasible:
        if not all(
            math.isfinite(candidate[x])
            for x in ["expected_profit", "expected_profit_margin"]
        ):
            raise ValueError("Nonfinite objective")
        dominated = any(
            other["expected_profit"] >= candidate["expected_profit"]
            and other["expected_profit_margin"] >= candidate["expected_profit_margin"]
            and (
                other["expected_profit"] > candidate["expected_profit"]
                or other["expected_profit_margin"] > candidate["expected_profit_margin"]
            )
            for other in feasible
        )
        if not dominated:
            result.append(candidate)
    return result
