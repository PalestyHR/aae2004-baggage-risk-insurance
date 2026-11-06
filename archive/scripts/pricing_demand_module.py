"""Separate price-demand scenario module. No empirically identified elasticity yet.
UK/US market aggregates are context, not Cathay conversion observations.
"""
from pathlib import Path
import json,math
ROOT=Path(__file__).resolve().parents[3]
PARAMS=ROOT/'Assignments/Group_Project/Material/insurance_market/market_parameters.json'
def load_market_context():
 return json.loads(PARAMS.read_text(encoding='utf-8'))
def decision_vector(*,predicted_baggage_rate_per_1000,price,reference_price,expected_payout_per_policy,variable_cost_per_policy):
 """All monetary inputs must use the same currency; no implicit FX conversion.
 Expected payout must come from a separately justified policy-eligibility model.
 Market values are metadata only and not joined as causal baggage-risk inputs.
 """
 if reference_price<=0 or price<=0: raise ValueError('Prices must be positive')
 if min(predicted_baggage_rate_per_1000,expected_payout_per_policy,variable_cost_per_policy)<0:raise ValueError('Negative risk/cost')
 return {'predicted_baggage_rate_per_1000':predicted_baggage_rate_per_1000,'candidate_price':price,'price_ratio':price/reference_price,'log_price_ratio':math.log(price/reference_price),'expected_payout_per_policy':expected_payout_per_policy,'variable_cost_per_policy':variable_cost_per_policy,'unit_expected_contribution':price-expected_payout_per_policy-variable_cost_per_policy}
def demand_scenario(*,price,reference_price,eligible_customers,reference_purchase_probability,price_sensitivity):
 """Logit-demand SCENARIO. Sensitivity is an explicit assumption, not fitted.
 logit(q)=logit(q0)-sensitivity*log(P/P0). Positive sensitivity lowers purchase odds.
 This parameter is NOT a constant quantity elasticity.
 """
 if not (price>0 and reference_price>0 and eligible_customers>=0 and 0<reference_purchase_probability<1 and price_sensitivity>=0):raise ValueError('Invalid scenario inputs')
 logodds=math.log(reference_purchase_probability/(1-reference_purchase_probability))-price_sensitivity*math.log(price/reference_price)
 prob=1/(1+math.exp(-max(-700,min(700,logodds))))
 return {'purchase_probability':prob,'expected_policies':eligible_customers*prob,'parameter_status':'assumed_not_estimated'}
def evaluate_product(*,risk_vector,reference_price,eligible_customers,reference_purchase_probability,price_sensitivity,fixed_cost):
 if fixed_cost<0:raise ValueError('Negative fixed cost')
 demand=demand_scenario(price=risk_vector['candidate_price'],reference_price=reference_price,eligible_customers=eligible_customers,reference_purchase_probability=reference_purchase_probability,price_sensitivity=price_sensitivity)
 return {**risk_vector,**demand,'expected_product_contribution_after_fixed_cost':demand['expected_policies']*risk_vector['unit_expected_contribution']-fixed_cost,'excludes':'capital, tax and contractual allocation unless included explicitly in cost inputs'}
if __name__=='__main__':
 context=load_market_context()
 assert context['estimated_demand_parameters']['price_sensitivity'] is None
 low=demand_scenario(price=5,reference_price=5,eligible_customers=10000,reference_purchase_probability=.1,price_sensitivity=1)
 high=demand_scenario(price=8,reference_price=5,eligible_customers=10000,reference_purchase_probability=.1,price_sensitivity=1)
 assert abs(low['purchase_probability']-.1)<1e-12
 assert 0<high['purchase_probability']<low['purchase_probability']<1
 print('PASS: context loads; unknown elasticity stays null; explicit scenario is bounded and decreases with price.')
