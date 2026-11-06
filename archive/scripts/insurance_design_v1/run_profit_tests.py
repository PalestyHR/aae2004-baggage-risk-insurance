"""Explicit scenario tests of the proposed design, not calibrated profitability.
Each BTS carrier is a separate illustrative operator, each paying its own system
cost. All results are one monthly cohort; never multiply August by 12 implicitly.
"""
from pathlib import Path
import sys,json,csv,copy,itertools
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'tmp/baggage_ml_dependencies'))
import numpy as np
import pandas as pd
from insurance_model import expected_portfolio,simulate_cohort,pareto_frontier
HERE=Path(__file__).parent;OUT=HERE/'profit_tests';OUT.mkdir(exist_ok=True)
plans=json.loads((HERE/'design_config_template.json').read_text())['plans']
forecasts=list(csv.DictReader((HERE/'forecast_interface_2026_08.csv').open(encoding='utf-8-sig')))
for f in forecasts:
 for key in ['risk_rate_per_1000','passenger_segments','bag_loads']: f[key]=float(f[key])
base={'currency':'USD','covered_bags':1,'covered_segments':1,'eligible_opportunity_fraction':.2,
 'conditional_delay_share':.5,'conditional_loss_share':.05,
 'tier_hazard_multipliers':{p['name']:1. for p in plans},
 'risk_states':[{'name':'normal','probability':.98,'hazard_multiplier':1.},
                {'name':'common_disruption','probability':.02,'hazard_multiplier':6.}],
 'demand':{'reference_purchase_probabilities':{'Basic':.05,'Standard':.03,'Premium':.02},
           'log_price_sensitivity':1.,'log_delay_benefit_sensitivity':.1,'log_loss_benefit_sensitivity':.1},
 'variable_cost_per_policy':{p['name']:.5 for p in plans},
 'external_expense_rates':{p['name']:.25 for p in plans},
 'period_system_cost':50000.,'target_period_profit':0.}
spec={'all_financial_and_conversion_inputs':'ASSUMPTIONS, not empirically estimated',
 'scope':'each of the 12 US BTS carrier forecast cohorts separately; not Cathay actual business',
 'forecast':'frozen forecast for August2026 based on data through July2026',
 'prices_and_benefits':'user-provided reference candidates in USD, not approved prices',
 'base_assumptions':base,'why_eligibility_fraction':'.2 is illustrative eligible direct single-bag opportunity fraction; not obtained from passenger-segment data',
 'cost_interpretation':'USD50000 per operator per monthly cohort for system development allocation+operations. Not supplier quote. External expenses25% exclude internal transfers in chosen product scope; per-policy0.5 is handling cost; no further tax/capital cost unless embedded explicitly.',
 'sensitivity_grid':{'purchase_total':[.02,.05,.1],'tier_hazard_multiplier':[1,2,4],'system_cost':[10000,50000,200000]},
 'claim_condition_stress_grid':{'conditional_delay_share':[.25,.5,.75],'conditional_loss_share':[.02,.05,.15]},
 'common_shock_assumption':'2% cohort-wide state, sixfold hazard; means and simulation both include it. Not measured event frequency.',
 'pareto_search':'exploratory price factors[1,1.25,1.5,2] and benefit factors[1,1.25,1.5], common scaling across tiers; minimum benefits at reference levels',
 'profit_margin_definition':'expected profit / expected premium revenue',
 'system_cost_coverage_definition':'contribution before system cost / system cost; NOT investment ROI',
 'not_annual_or_guaranteed':True}
(OUT/'explicit_assumptions.json').write_text(json.dumps(spec,indent=2),encoding='utf-8')
def flatten(f,o):
 rev=o['expected_revenue'];profit=o['expected_profit'];K=o['period_system_cost']
 return {'carrier':f['carrier'],'period':f['period'],'forecast_passenger_segments':f['passenger_segments'],
 'forecast_bag_loads':f['bag_loads'],'forecast_mishandled_per_1000':f['risk_rate_per_1000'],
 'eligible_opportunities':o['eligible_opportunities'],'expected_sales':o['expected_sales'],
 'revenue_usd':rev,'expected_payout_usd':o['expected_claims'],'variable_expense_usd':o['nonclaim_variable_expense'],
 'system_cost_usd':K,'expected_profit_usd':profit,'net_margin':o['expected_profit_margin'],
 'loss_ratio':o['expected_loss_ratio'],'paid_policy_fraction':o['expected_paid_policy_fraction'],
 'system_cost_coverage':(profit+K)/K if K else None,
 'system_cost_break_even_usd':profit+K}
base_rows=[];tier_rows=[];sim_rows=[]
for i,f in enumerate(forecasts):
 o=expected_portfolio(f,plans,base);base_rows.append(flatten(f,o))
 for t in o['tiers']:tier_rows.append({'carrier':f['carrier'],**t})
 sim=simulate_cohort(f,plans,base,draws=30000,seed=42+i)
 sim_rows.append({'carrier':f['carrier'],**sim})
 print(f['carrier'],'margin',round(o['expected_profit_margin'],4),'profit',round(o['expected_profit'],2),'sim_profit_probability',round(sim['probability_profit_positive'],4),flush=True)
 assert abs(sim['simulated_mean_profit']-sim['expected_profit_analytic'])<=6*sim['mean_standard_error']
grid=[]
for q,rho,K in itertools.product([.02,.05,.1],[1,2,4],[10000,50000,200000]):
 a=copy.deepcopy(base);a['period_system_cost']=K
 a['demand']['reference_purchase_probabilities']={'Basic':.5*q,'Standard':.3*q,'Premium':.2*q}
 a['tier_hazard_multipliers']={p['name']:rho for p in plans}
 for f in forecasts:
  o=expected_portfolio(f,plans,a)
  grid.append({'purchase_probability_at_reference':q,'portfolio_hazard_multiplier':rho,'period_system_cost':K,**flatten(f,o)})
claim_grid=[]
for sd,sl in itertools.product([.25,.5,.75],[.02,.05,.15]):
 a=copy.deepcopy(base);a['conditional_delay_share']=sd;a['conditional_loss_share']=sl
 for f in forecasts:claim_grid.append({'conditional_delay_share':sd,'conditional_loss_share':sl,**flatten(f,expected_portfolio(f,plans,a))})

# Price/benefit trade-off is checked over multiple demand-sensitivity assumptions.
candidates=[];frontiers=[]
for beta in [.5,1,2]:
 a=copy.deepcopy(base);a['demand']['log_price_sensitivity']=beta
 for f in forecasts:
  one=[]
  for pf,bf in itertools.product([1,1.25,1.5,2],[1,1.25,1.5]):
   ps=copy.deepcopy(plans)
   for p in ps:
    p['price']=p['reference']['price']*pf;p['delay_benefit']=p['reference']['delay_benefit']*bf;p['loss_benefit']=p['reference']['loss_benefit']*bf
   o=expected_portfolio(f,ps,a)
   row={'carrier':f['carrier'],'beta':beta,'price_factor':pf,'benefit_factor':bf,
        'expected_profit':o['expected_profit'],'expected_profit_margin':o['expected_profit_margin'],
        'expected_sales':o['expected_sales'],'expected_claims':o['expected_claims'],
        'feasible':o['meets_target_expected_profit'],'constraint_scope':'minimum reference benefits and nonnegative expected profit ONLY; no certified downside constraint'}
   one.append(row);candidates.append(row)
  frontiers.extend(pareto_frontier(one))
for name,rows in [('base_case',base_rows),('tier_details_before_shared_system_cost',tier_rows),('monte_carlo',sim_rows),('sensitivity',grid),('claim_share_sensitivity',claim_grid),('price_benefit_candidates',candidates),('conditional_pareto_frontier',frontiers)]:pd.DataFrame(rows).to_csv(OUT/(name+'.csv'),index=False)
g=pd.DataFrame(grid);b=pd.DataFrame(base_rows)
minimum_purchase=[]
# Under the fixed reference menu and fixed mix, contribution scales linearly in
# uptake; this is conditional on the stated costs/risk/selection assumptions.
for r in base_rows:
 qbreak=.1*r['system_cost_usd']/r['system_cost_break_even_usd']
 minimum_purchase.append({'carrier':r['carrier'],'break_even_purchase_fraction_of_eligible':qbreak,'scope':'reference menu, base risk and cost assumptions; excludes target-profit uplift'})
pd.DataFrame(minimum_purchase).to_csv(OUT/'break_even_purchase_rate.csv',index=False)
summary={'base_case_operators':len(b),'base_case_profit_positive_operators':int(b.expected_profit_usd.gt(0).sum()),
 'base_case_net_margin_min':float(b.net_margin.min()),'base_case_net_margin_max':float(b.net_margin.max()),
 'base_case_net_margin_pooled_across_separate_cases':float(b.expected_profit_usd.sum()/b.revenue_usd.sum()),
 'sensitivity_case_count':len(g),'sensitivity_positive_cases':int(g.expected_profit_usd.gt(0).sum()),
 'sensitivity_negative_cases':int(g.expected_profit_usd.lt(0).sum()),
 'sensitivity_min_margin':float(g.net_margin.min()),'sensitivity_max_margin':float(g.net_margin.max()),
 'sensitivity_cases_are_not_probabilities':True,
 'monte_carlo_profit_prob_min':min(x['probability_profit_positive'] for x in sim_rows),
 'monte_carlo_profit_prob_max':max(x['probability_profit_positive'] for x in sim_rows),
 'interpretation':'all profit probabilities conditional on assumed sales, claim composition, hazard states and costs; no empirical guarantee'}
(OUT/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
print(json.dumps(summary,indent=2))
