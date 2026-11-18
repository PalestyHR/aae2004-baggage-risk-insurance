"""Joint numerical menu design before marketing labels.
Scenario-robust finite-grid frontier, not an empirically calibrated optimum.
"""

from pathlib import Path
import sys
import json
import itertools
import copy
import hashlib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from models.insurance import expected_portfolio


def main():
    HERE = ROOT
    OUT = ROOT / "results/reproduced_insurance"
    OUT.mkdir(exist_ok=True)
    FORECAST = ROOT / "results/reproduced_risk/forecast_interface_2026_08.csv"
    if not FORECAST.exists():
        FORECAST = ROOT / "results/reference/model1/forecast_interface_2026_08.csv"
    forecasts = pd.read_csv(FORECAST)
    assumptions_file = ROOT / "configs/insurance_assumptions.json"
    assumptions_text = assumptions_file.read_text()
    assumptions_config = json.loads(assumptions_text)
    base_assumptions = assumptions_config["base_assumptions"]

    plans_file = ROOT / "configs/insurance_design_template.json"
    plans_text = plans_file.read_text()
    plans_config = json.loads(plans_text)
    reference_plans = plans_config["plans"]
    reference_prices = np.array([3.0, 6.0, 11.0])
    reference_delay_benefits = np.array([50.0, 100.0, 200.0])
    reference_loss_benefits = np.array([300.0, 600.0, 1200.0])
    factors = [
        0.75,
        1,
        1.25,
        1.5,
        2,
        2.5,
        3,
    ]
    benefits = [1, 1.25, 1.5]
    eligible_counts = np.floor(forecasts.passenger_segments.to_numpy() * 0.2).astype(
        int
    )
    mishandling_rates = forecasts.risk_rate_per_1000.to_numpy() / 1000
    assert (eligible_counts <= forecasts.bag_loads).all()
    scenario_settings = []
    for beta in [1.0, 2.0, 3.0]:
        for risk, rho, sl in [("base", 1.0, 0.05), ("higher", 2.0, 0.1)]:
            scenario_settings.append(
                {
                    "name": f"beta{beta}_{risk}",
                    "beta": beta,
                    "rho": rho,
                    "sd": 0.5,
                    "sl": sl,
                }
            )
    scenarios = scenario_settings
    protocol = {
        "decision_variables": "three independent price multipliers; common delay multiplier; common loss multiplier",
        "currency": "USD",
        "price_reference": reference_prices.tolist(),
        "delay_reference": reference_delay_benefits.tolist(),
        "loss_reference": reference_loss_benefits.tolist(),
        "price_multiplier_grid": factors,
        "benefit_multiplier_grid": benefits,
        "objectives": [
            "min_scenario sum_operator expected_profit",
            "min_scenario pooled_expected_profit_margin",
        ],
        "constraints": [
            "P1<P2<P3",
            "D1<D2<D3",
            "L1<L2<L3",
            "each L>=D",
            "benefits at least reference amounts",
            "every operator expected profit >=0 in all six scenarios",
        ],
        "scenarios": scenarios,
        "shared_assumptions": base_assumptions,
        "selection": "among nondominated points maximize min(normalized robust_profit, normalized robust_margin); equal-priority preference, not unique optimum",
        "risk_forecast": "frozen August2026 forecasts; no model refit and no realized August outcomes used",
        "business_status": "hypothetical demand, eligible pool and cost assumptions, not Cathay evidence",
        "caution": "robust only to the six listed scenarios; no guaranteed cash-flow solvency or 95% profit-probability constraint",
    }
    (OUT / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    designs = []
    outcomes = []
    for price_multipliers in itertools.product(factors, repeat=3):
        price = reference_prices * np.array(price_multipliers)
        if not np.all(np.diff(price) > 0):
            continue
        for delay_multiplier, loss_multiplier in itertools.product(benefits, repeat=2):
            delay_benefits = reference_delay_benefits * delay_multiplier
            loss_benefits = reference_loss_benefits * loss_multiplier
            if np.any(loss_benefits < delay_benefits):
                continue
            design_id = f"M{len(designs)+1:04d}"
            values = []
            minimum_operator_profit = float("inf")
            for scenario in scenarios:
                price_sensitivity = scenario["beta"]
                risk_multiplier = scenario["rho"]
                delay_share = scenario["sd"]
                loss_share = scenario["sl"]
                reference_shares = np.array([0.05, 0.03, 0.02])
                reference_odds = reference_shares / 0.9
                price_ratio = price / reference_prices
                delay_ratio = delay_benefits / reference_delay_benefits
                loss_ratio = loss_benefits / reference_loss_benefits

                utilities = np.log(reference_odds)
                utilities = utilities - price_sensitivity * np.log(price_ratio)
                utilities = utilities + 0.1 * np.log(delay_ratio)
                utilities = utilities + 0.1 * np.log(loss_ratio)
                exp_utilities = np.exp(utilities)
                purchase_probabilities = exp_utilities / (1 + exp_utilities.sum())
                # Same common-state mixture as the financial module, now repeated per scenario.
                log_survival = np.log1p(-mishandling_rates)
                normal_log_survival = risk_multiplier * log_survival
                shock_log_survival = 6 * risk_multiplier * log_survival
                normal_risk = -np.expm1(normal_log_survival)
                shock_risk = -np.expm1(shock_log_survival)
                policy_risk = 0.98 * normal_risk + 0.02 * shock_risk
                delay_probabilities = policy_risk * delay_share
                loss_probabilities = policy_risk * loss_share
                sales = eligible_counts[:, None] * purchase_probabilities[None, :]
                revenue = (sales * price).sum(axis=1)
                claim = (
                    sales
                    * (
                        delay_probabilities[:, None] * delay_benefits
                        + loss_probabilities[:, None] * loss_benefits
                    )
                ).sum(axis=1)
                variable_costs = 0.25 * revenue + 0.5 * sales.sum(axis=1)
                profit = revenue - claim - variable_costs - 50000
                margin = profit.sum() / revenue.sum()
                minimum_operator_profit = min(minimum_operator_profit, profit.min())
                values.append((float(profit.sum()), float(margin)))
                outcomes.append(
                    {
                        "design_id": design_id,
                        "scenario": scenario["name"],
                        "total_revenue": float(revenue.sum()),
                        "total_expected_claims": float(claim.sum()),
                        "total_variable_cost": float(variable_costs.sum()),
                        "total_system_cost": 600000.0,
                        "total_expected_profit": float(profit.sum()),
                        "pooled_margin": float(margin),
                        "minimum_operator_profit": float(profit.min()),
                        "expected_policies": float(sales.sum()),
                    }
                )
            scenario_profits = []
            scenario_margins = []
            for scenario_profit, scenario_margin in values:
                scenario_profits.append(scenario_profit)
                scenario_margins.append(scenario_margin)
            designs.append(
                {
                    "design_id": design_id,
                    "price_A": price[0],
                    "price_B": price[1],
                    "price_C": price[2],
                    "delay_A": delay_benefits[0],
                    "delay_B": delay_benefits[1],
                    "delay_C": delay_benefits[2],
                    "loss_A": loss_benefits[0],
                    "loss_B": loss_benefits[1],
                    "loss_C": loss_benefits[2],
                    "robust_expected_profit": min(scenario_profits),
                    "robust_margin": min(scenario_margins),
                    "minimum_operator_profit_all_scenarios": minimum_operator_profit,
                    "feasible": minimum_operator_profit >= 0,
                }
            )
    design_results = pd.DataFrame(designs)
    scenario_results = pd.DataFrame(outcomes)
    feasible = design_results[design_results.feasible].copy()
    assert len(feasible)
    objective = feasible[["robust_expected_profit", "robust_margin"]].to_numpy()
    keep = []
    for x in objective:
        no_worse = np.all(objective >= x, axis=1)
        strictly_better = np.any(objective > x, axis=1)
        dominates = no_worse & strictly_better
        dominated = np.any(dominates)
        keep.append(not dominated)
    frontier = feasible.loc[keep].copy()
    for col in ["robust_expected_profit", "robust_margin"]:
        minimum = frontier[col].min()
        maximum = frontier[col].max()
        frontier["norm_" + col] = (
            (frontier[col] - minimum) / (maximum - minimum)
            if maximum > minimum
            else 1.0
        )
    frontier["balance_score"] = frontier[
        ["norm_robust_expected_profit", "norm_robust_margin"]
    ].min(axis=1)
    frontier = frontier.sort_values(
        ["balance_score", "robust_expected_profit"], ascending=False
    )
    selected = frontier.iloc[0].to_dict()
    design_id = selected["design_id"]
    design_results.to_csv(OUT / "all_designs.csv", index=False)
    scenario_results.to_csv(OUT / "scenario_results.csv", index=False)
    frontier.to_csv(OUT / "pareto_frontier.csv", index=False)
    selected_out = scenario_results[scenario_results.design_id == design_id]
    selected_out.to_csv(OUT / "selected_scenarios.csv", index=False)
    chosen = []
    plan_names = ["Basic", "Standard", "Premium"]
    for i, label in enumerate(["A", "B", "C"]):
        reference_plan = reference_plans[i]
        plan_reference = reference_plan["reference"]
        chosen.append(
            {
                "name": plan_names[i],
                "numeric_tier": label,
                "price": float(selected["price_" + label]),
                "delay_benefit": float(selected["delay_" + label]),
                "loss_benefit": float(selected["loss_" + label]),
                "reference": plan_reference,
            }
        )
    (OUT / "selected_menu.json").write_text(
        json.dumps(
            {
                "selection": selected,
                "plans": chosen,
                "currency": "USD",
                "is_final_quote": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    plan_rows = []
    for plan in chosen:
        plan_fields = {}
        for field, value in plan.items():
            if field != "reference":
                plan_fields[field] = value
        plan_rows.append(plan_fields)
    plan_table = pd.DataFrame(plan_rows)
    plan_table.to_csv(OUT / "selected_menu.csv", index=False)
    individual = []
    for scenario in scenarios:
        assumptions = copy.deepcopy(base_assumptions)
        demand = assumptions["demand"]
        demand["log_price_sensitivity"] = scenario["beta"]
        assumptions["conditional_delay_share"] = scenario["sd"]
        assumptions["conditional_loss_share"] = scenario["sl"]
        plan_values = {}
        risk_multiplier = scenario["rho"]
        for plan in chosen:
            plan_name = plan["name"]
            plan_values[plan_name] = risk_multiplier
        assumptions["tier_hazard_multipliers"] = plan_values
        for row in forecasts.to_dict("records"):
            portfolio = expected_portfolio(row, chosen, assumptions)
            individual.append(
                {
                    "scenario": scenario["name"],
                    "carrier": row["carrier"],
                    "expected_profit": portfolio["expected_profit"],
                    "margin": portfolio["expected_profit_margin"],
                    "revenue": portfolio["expected_revenue"],
                    "claims": portfolio["expected_claims"],
                    "sales": portfolio["expected_sales"],
                }
            )
    operator_results = pd.DataFrame(individual)
    for scenario, group_results in operator_results.groupby("scenario"):
        expected = selected_out[selected_out.scenario == scenario].iloc[0]
        assert (
            abs(group_results.expected_profit.sum() - expected.total_expected_profit)
            < 1e-6
        )
        assert abs(group_results.revenue.sum() - expected.total_revenue) < 1e-6
    operator_results.to_csv(OUT / "selected_operator_results.csv", index=False)
    extra = []
    for purchase_probability, system_cost in itertools.product(
        [0.02, 0.05, 0.1], [10000.0, 50000.0, 200000.0]
    ):
        assumptions = copy.deepcopy(base_assumptions)
        demand = assumptions["demand"]
        demand["reference_purchase_probabilities"] = {
            "Basic": purchase_probability * 0.5,
            "Standard": purchase_probability * 0.3,
            "Premium": purchase_probability * 0.2,
        }
        assumptions["period_system_cost"] = system_cost
        for row in forecasts.to_dict("records"):
            portfolio = expected_portfolio(row, chosen, assumptions)
            extra.append(
                {
                    "carrier": row["carrier"],
                    "reference_total_purchase_probability": purchase_probability,
                    "system_cost": system_cost,
                    "expected_profit": portfolio["expected_profit"],
                    "margin": portfolio["expected_profit_margin"],
                }
            )
    pd.DataFrame(extra).to_csv(OUT / "selected_extra_sensitivity.csv", index=False)
    assert np.allclose(
        scenario_results.total_expected_profit,
        scenario_results.total_revenue
        - scenario_results.total_expected_claims
        - scenario_results.total_variable_cost
        - scenario_results.total_system_cost,
    )
    loss_flags = []
    for x in extra:
        loss_flags.append(x["expected_profit"] < 0)
    summary = {
        "candidate_count": len(design_results),
        "feasible_count": len(feasible),
        "pareto_count": len(frontier),
        "selected_id": design_id,
        "checks": "all scenario accounting identities; selected independently reconciled via insurance_model; nondominance checked against all feasible points",
        "extra_sensitivity_negative_cases": int(sum(loss_flags)),
        "extra_sensitivity_count": len(extra),
        "source_forecast_sha256": hashlib.sha256(FORECAST.read_bytes()).hexdigest(),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(json.dumps(chosen, indent=2))
    print(selected_out.to_string(index=False))


if __name__ == "__main__":
    main()
