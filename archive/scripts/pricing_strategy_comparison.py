"""Historical-risk pricing SCENARIO comparison, not observed insurance profits."""
from pathlib import Path
import sys,json,itertools,hashlib
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'tmp/baggage_ml_dependencies'))
import numpy as np
import pandas as pd
OUT=Path(__file__).with_suffix('');OUT.mkdir(exist_ok=True)
DATA=ROOT/'Assignments/Group_Project/Material/integrated_dataset/dataset'
obs=pd.read_csv(DATA/'carrier_month_observed_NOT_model_input.csv')
hist=obs[obs.year<2026].groupby('airline_id')[['mishandled_bags','enplaned_bags']].sum()
base=1000*hist.mishandled_bags/hist.enplaned_bags
te=pd.read_csv(DATA/'lag1/test_real.csv')
mlpath=Path(__file__).parent/'asymmetric_loss_comparison/predictions.csv'
ml=pd.read_csv(mlpath)
ml=ml[(ml.year==2026)&(ml.weighting=='uniform')&(ml.under_over_ratio=='1:2')]
te=te.merge(ml[['row_id','prediction']],on='row_id',validate='one_to_one')
assert len(te)==101
te['fixed_risk']=te.airline_id.map(base)
te['simple_risk']=.5*(te.recent_rate+te.rate_year_ago)
te['ml_risk']=te.prediction
config={'run_date':'2026-11-06','real_inputs':'2026 observed baggage rates and prior predictions; monthly cases are US carrier-months, NOT Cathay claims','ml_choice':'LightGBM under:over 1:2 selected by 2025 weighted MAE, not by scenario profit','fixed_risk':'per-carrier pooled observed rate in 2019-2025; frozen across 2026','simple_risk':'50/50 previous month and previous-year same-month','eligible_customers_per_case':10000,'interpretation':'one policy / one bag / one covered trip proxy; not real customer count','reference_price_hkd':5,'reference_purchase_probability':.1,'unit_variable_cost_hkd':1,'target_expected_margin':.3,'pricing_rule':'round(clip((expected payout + variable cost)/(1-margin),3,12),1)','price_bounds_hkd':[3,12],'claim_conversion_grid':[.25,.5,1.0],'benefit_hkd_grid':[300,1000,3000],'logit_price_sensitivity_grid':[.5,1,2],'base_case':{'claim_conversion':.5,'benefit_hkd':1000,'beta':1},'fixed_costs':'excluded; report contribution before fixed overhead/capital/tax. Difference is available budget for incremental implementation cost.','claim_selection':'buyers assumed same risk as eligible group; no adverse selection','payouts':'one hypothetical fixed benefit for qualifying mishandling, not actual insurance claims','no_causal_demand_estimate':True,'no_guaranteed_profit':True}
(OUT/'scenario_assumptions.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
def uptake(price,beta):
 return 1/(1+np.exp(-(np.log(.1/.9)-beta*np.log(price/5))))
def price_from(risk,k,benefit):return np.round(np.clip((risk/1000*k*benefit+1)/.7,3,12),1)
rows=[]
for k,benefit,beta in itertools.product([.25,.5,1.],[300,1000,3000],[.5,1.,2.]):
 scenario=f'k{k:g}_b{benefit}_s{beta:g}'
 for strategy,rcol in [('fixed','fixed_risk'),('simple_dynamic','simple_risk'),('ml_dynamic','ml_risk')]:
  o=te[['row_id','target_month','carrier','target_rate_per_1000']].copy()
  o['scenario']=scenario;o['claim_conversion']=k;o['benefit_hkd']=benefit;o['beta']=beta;o['strategy']=strategy;o['risk_estimate']=te[rcol].to_numpy()
  o['price_hkd']=price_from(o.risk_estimate,k,benefit)
  o['expected_policies']=10000*uptake(o.price_hkd,beta)
  o['assumed_actual_claim_probability']=o.target_rate_per_1000/1000*k
  assert o.assumed_actual_claim_probability.between(0,1).all()
  o['expected_claims']=o.expected_policies*o.assumed_actual_claim_probability
  o['revenue_hkd']=o.expected_policies*o.price_hkd
  o['expected_payout_hkd']=o.expected_claims*benefit
  o['variable_cost_hkd']=o.expected_policies
  o['expected_contribution_hkd']=o.revenue_hkd-o.expected_payout_hkd-o.variable_cost_hkd
  o['floor_binding']=o.price_hkd.eq(3);o['cap_binding']=o.price_hkd.eq(12)
  rows.append(o)
p=pd.concat(rows,ignore_index=True)
def summarize(g):
 return {'cases':len(g),'avg_offered_price_hkd':g.price_hkd.mean(),'avg_paid_price_hkd':g.revenue_hkd.sum()/g.expected_policies.sum(),'policies_per_10000':g.expected_policies.mean(),'revenue_per_10000':g.revenue_hkd.mean(),'payout_per_10000':g.expected_payout_hkd.mean(),'contribution_per_10000':g.expected_contribution_hkd.mean(),'negative_expected_contribution_cases':int((g.expected_contribution_hkd<0).sum()),'floor_case_fraction':g.floor_binding.mean(),'cap_case_fraction':g.cap_binding.mean()}
summary=[]
for keys,g in p.groupby(['scenario','claim_conversion','benefit_hkd','beta','strategy']):summary.append(dict(zip(['scenario','claim_conversion','benefit_hkd','beta','strategy'],keys),**summarize(g)))
s=pd.DataFrame(summary)
basecase=s[(s.claim_conversion==.5)&(s.benefit_hkd==1000)&(s.beta==1)]
wide=s.pivot(index='scenario',columns='strategy',values='contribution_per_10000')
wide['ml_minus_fixed']=wide.ml_dynamic-wide.fixed;wide['ml_minus_simple']=wide.ml_dynamic-wide.simple_dynamic
wide['simple_minus_fixed']=wide.simple_dynamic-wide.fixed
wide.to_csv(OUT/'incremental_contribution.csv')
p.to_csv(OUT/'case_details.csv',index=False);s.to_csv(OUT/'scenario_summary.csv',index=False);basecase.to_csv(OUT/'base_case.csv',index=False)
monthly=[]
bp=p[(p.claim_conversion==.5)&(p.benefit_hkd==1000)&(p.beta==1)]
for keys,g in bp.groupby(['target_month','strategy']):monthly.append(dict(zip(['month','strategy'],keys),**summarize(g)))
pd.DataFrame(monthly).to_csv(OUT/'base_case_monthly.csv',index=False)
checks={'three_strategies_same_cases':all(g.row_id.nunique()==101 for _,g in p.groupby(['scenario','strategy'])),'fixed_price_constant_within_carrier':p[p.strategy=='fixed'].groupby(['scenario','carrier']).price_hkd.nunique().max()==1,'nonnegative_finite_sales':bool((np.isfinite(p.expected_policies)&p.expected_policies.between(0,10000)).all()),'accounting_reconciles':bool(np.allclose(p.expected_contribution_hkd,p.revenue_hkd-p.expected_payout_hkd-p.variable_cost_hkd)),'price_independent_of_realized_target':'verified formula uses risk_estimate only; fixed pool excludes 2026','test_predictions_source_sha256':hashlib.sha256(mlpath.read_bytes()).hexdigest(),'ml_beats_fixed_scenarios':int((wide.ml_minus_fixed>1e-8).sum()),'ml_beats_simple_scenarios':int((wide.ml_minus_simple>1e-8).sum()),'total_scenarios':len(wide)}
assert all(checks[x] for x in ['three_strategies_same_cases','fixed_price_constant_within_carrier','nonnegative_finite_sales','accounting_reconciles'])
(OUT/'audit.json').write_text(json.dumps(checks,indent=2,default=lambda x:bool(x)),encoding='utf-8')
print(basecase.to_string(index=False));print(json.dumps(checks,indent=2,default=lambda x:bool(x)))

# Secondary control: all strategies maximize the SAME assumed demand-profit
# function over a common price grid, using only their own predicted risk.
grid=np.round(np.arange(3,12.00001,.1),1)
optrows=[]
for k,benefit,beta in itertools.product([.25,.5,1.],[300,1000,3000],[.5,1.,2.]):
 scenario=f'k{k:g}_b{benefit}_s{beta:g}'
 for strategy,rcol in [('fixed','fixed_risk'),('simple_dynamic','simple_risk'),('ml_dynamic','ml_risk')]:
  o=te[['row_id','target_month','carrier','target_rate_per_1000']].copy()
  estimated_cost=te[rcol].to_numpy()/1000*k*benefit+1
  predicted_contribution=10000*uptake(grid,beta)[None,:]*(grid[None,:]-estimated_cost[:,None])
  chosen=grid[np.argmax(predicted_contribution,axis=1)]
  o['scenario']=scenario;o['claim_conversion']=k;o['benefit_hkd']=benefit;o['beta']=beta;o['strategy']=strategy
  o['price_hkd']=chosen;o['expected_policies']=10000*uptake(chosen,beta)
  o['revenue_hkd']=o.expected_policies*chosen
  o['expected_payout_hkd']=o.expected_policies*o.target_rate_per_1000/1000*k*benefit
  o['variable_cost_hkd']=o.expected_policies
  o['expected_contribution_hkd']=o.revenue_hkd-o.expected_payout_hkd-o.variable_cost_hkd
  o['floor_binding']=o.price_hkd.eq(3);o['cap_binding']=o.price_hkd.eq(12)
  optrows.append(o)
op=pd.concat(optrows,ignore_index=True)
os=[]
for keys,g in op.groupby(['scenario','claim_conversion','benefit_hkd','beta','strategy']):os.append(dict(zip(['scenario','claim_conversion','benefit_hkd','beta','strategy'],keys),**summarize(g)))
os=pd.DataFrame(os);os.to_csv(OUT/'profit_search_scenario_summary.csv',index=False);op.to_csv(OUT/'profit_search_case_details.csv',index=False)
ow=os.pivot(index='scenario',columns='strategy',values='contribution_per_10000')
ow['ml_minus_fixed']=ow.ml_dynamic-ow.fixed;ow['ml_minus_simple']=ow.ml_dynamic-ow.simple_dynamic
ow.to_csv(OUT/'profit_search_incremental_contribution.csv')
oa={'price_grid':'HKD3 to12 step0.1, illustrative not fitted','ml_beats_fixed':int((ow.ml_minus_fixed>1e-8).sum()),'ml_ties_fixed':int((abs(ow.ml_minus_fixed)<=1e-8).sum()),'ml_loses_fixed':int((ow.ml_minus_fixed< -1e-8).sum()),'ml_beats_simple':int((ow.ml_minus_simple>1e-8).sum()),'ml_ties_simple':int((abs(ow.ml_minus_simple)<=1e-8).sum()),'ml_loses_simple':int((ow.ml_minus_simple< -1e-8).sum())}
(OUT/'profit_search_audit.json').write_text(json.dumps(oa,indent=2),encoding='utf-8')
print('PROFIT SEARCH',json.dumps(oa));print(os[(os.claim_conversion==.5)&(os.benefit_hkd==1000)].to_string(index=False))
