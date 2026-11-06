"""Synthetic arithmetic fixtures only. No claim of empirical financial validity."""
from pathlib import Path
import sys,json,copy,math
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'tmp/baggage_ml_dependencies'))
from insurance_model import claims_from_forecast,tier_probabilities,expected_portfolio,benefit_frontier,simulate_cohort,pareto_frontier
HERE=Path(__file__).parent
config=json.loads((HERE/'design_config_template.json').read_text());plans=config['plans']
a={'currency':'USD','covered_bags':1,'covered_segments':1,'eligible_opportunity_fraction':.2,
   'conditional_delay_share':.5,'conditional_loss_share':.02,
   'tier_hazard_multipliers':{p['name']:1 for p in plans},
   'risk_states':[{'name':'normal','probability':1.,'hazard_multiplier':1.}],
   'demand':{'reference_purchase_probabilities':{'Basic':.05,'Standard':.03,'Premium':.02},'log_price_sensitivity':1.,'log_delay_benefit_sensitivity':.1,'log_loss_benefit_sensitivity':.1},
   'variable_cost_per_policy':{p['name']:.5 for p in plans},'external_expense_rates':{p['name']:.25 for p in plans},'period_system_cost':100.,'target_period_profit':20.}
fixture={'period':'SYNTHETIC_TEST_ONLY','risk_rate_per_1000':20.,'passenger_segments':10000.,'bag_loads':5000.}
checks=[]
def check(name,condition):
 assert condition,name
 checks.append(name)
def rejects(name,fn):
 try:fn()
 except (ValueError,TypeError):checks.append(name);return
 raise AssertionError(name)
risk=claims_from_forecast(20,1,1,1,.5,.02)
check('conditional probabilities and mutual exclusivity',math.isclose(risk['delay_claim_probability'],.01) and math.isclose(risk['loss_claim_probability'],.0004) and math.isclose(sum(risk[k] for k in ['delay_claim_probability','loss_claim_probability','no_paid_claim_probability']),1))
rejects('invalid shares rejected',lambda:claims_from_forecast(5,1,1,1,.9,.2))
rejects('unknown multiplier not silently set to one',lambda:claims_from_forecast(5,1,1,None,.5,.05))
check('multiple exposures increase scenario risk',claims_from_forecast(5,2,1,1,.5,.05)['policy_mishandling_probability']>claims_from_forecast(5,1,1,1,.5,.05)['policy_mishandling_probability'])
q=tier_probabilities(plans,a['demand'])
check('reference shares reproduced with no purchase',math.isclose(q['no_purchase'],.9) and all(math.isclose(q[n],x) for n,x in a['demand']['reference_purchase_probabilities'].items()))
expensive=copy.deepcopy(plans);expensive[0]['price']*=2
q2=tier_probabilities(expensive,a['demand'])
check('own price reduces own uptake; shares remain bounded',q2['Basic']<q['Basic'] and math.isclose(sum(q2.values()),1))
out=expected_portfolio(fixture,plans,a)
check('whole product accounting and fixed cost counted once',math.isclose(out['expected_profit'],389.4) and math.isclose(out['expected_sales'],200))
check('reference expected payouts correct',[round(x['expected_claim_per_policy'],2) for x in out['tiers']]==[.62,1.24,2.48])
aa=copy.deepcopy(a);aa['period_system_cost']+=50
check('fixed system cost flows through profit',math.isclose(expected_portfolio(fixture,plans,aa)['expected_profit'],out['expected_profit']-50))
aa=copy.deepcopy(a);aa['eligible_opportunity_fraction']=1
rejects('exposure inconsistency rejected',lambda:expected_portfolio(fixture,plans,aa))
rejects('unset design config cannot produce fake profit',lambda:expected_portfolio(fixture,plans,config['assumptions']))
cap=benefit_frontier(policies=100,price=10,expense_rate=.1,variable_cost=1,allocated_fixed_cost=100,target_profit=100,qd=.01,ql=.001,delay_benefit=100)
check('conditional benefit ceiling',math.isclose(cap['loss_benefit_ceiling'],5000))
zero=benefit_frontier(policies=100,price=10,expense_rate=.1,variable_cost=1,allocated_fixed_cost=100,target_profit=100,qd=.01,ql=0,delay_benefit=100)
check('zero loss probability not unlimited benefit',zero['loss_benefit_ceiling'] is None)
samples=[{'name':'A','expected_profit':30,'expected_profit_margin':.3,'feasible':True},{'name':'B','expected_profit':20,'expected_profit_margin':.5,'feasible':True},{'name':'C','expected_profit':10,'expected_profit_margin':.2,'feasible':True}]
check('Pareto removes dominated candidate',{x['name'] for x in pareto_frontier(samples)}=={'A','B'})
aa=copy.deepcopy(a);aa['risk_states']=[{'name':'normal','probability':.9,'hazard_multiplier':1},{'name':'shared_disruption','probability':.1,'hazard_multiplier':3}]
simulation=simulate_cohort(fixture,plans,aa,draws=30000,seed=42)
check('Monte Carlo reconciles analytic mean with shared state',abs(simulation['simulated_mean_profit']-simulation['expected_profit_analytic'])<=5*simulation['mean_standard_error'])
(HERE/'implementation_tests.json').write_text(json.dumps({'status':'PASS','checks':checks,'n_checks':len(checks),'fixture_status':'artificial arithmetic fixtures, NOT proposed product calibration','monte_carlo_mean_reconciliation_only':{'analytic':simulation['expected_profit_analytic'],'simulated':simulation['simulated_mean_profit'],'standard_error':simulation['mean_standard_error']}},indent=2),encoding='utf-8')
print(f'PASS {len(checks)} checks. No empirical profitability asserted.')
