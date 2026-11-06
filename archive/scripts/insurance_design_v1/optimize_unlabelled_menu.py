"""Joint numerical menu design before marketing labels.
Scenario-robust finite-grid frontier, not an empirically calibrated optimum.
"""
from pathlib import Path
import sys,json,itertools,copy,hashlib
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'tmp/baggage_ml_dependencies'))
import numpy as np
import pandas as pd
from insurance_model import expected_portfolio
HERE=Path(__file__).parent;OUT=HERE/'optimized_menu_v2';OUT.mkdir(exist_ok=True)
f=pd.read_csv(HERE/'forecast_interface_2026_08.csv')
prior=json.loads((HERE/'profit_tests/explicit_assumptions.json').read_text())['base_assumptions']
priorplans=json.loads((HERE/'design_config_template.json').read_text())['plans']
pref=np.array([3.,6.,11.]);dref=np.array([50.,100.,200.]);lref=np.array([300.,600.,1200.])
factors=[.75,1,1.25,1.5,2,2.5,3];benefits=[1,1.25,1.5]
N=np.floor(f.passenger_segments.to_numpy()*.2).astype(int);p=f.risk_rate_per_1000.to_numpy()/1000
assert (N<=f.bag_loads).all()
scenarios=[{'name':f'beta{beta}_{risk}','beta':beta,'rho':rho,'sd':.5,'sl':sl}
           for beta in [1.,2.,3.] for risk,rho,sl in [('base',1.,.05),('higher',2.,.10)]]
protocol={'decision_variables':'three independent price multipliers; common delay multiplier; common loss multiplier',
 'currency':'USD','price_reference':pref.tolist(),'delay_reference':dref.tolist(),'loss_reference':lref.tolist(),
 'price_multiplier_grid':factors,'benefit_multiplier_grid':benefits,
 'objectives':['min_scenario sum_operator expected_profit','min_scenario pooled_expected_profit_margin'],
 'constraints':['P1<P2<P3','D1<D2<D3','L1<L2<L3','each L>=D','benefits at least reference amounts','every operator expected profit >=0 in all six scenarios'],
 'scenarios':scenarios,'shared_assumptions':prior,
 'selection':'among nondominated points maximize min(normalized robust_profit, normalized robust_margin); equal-priority preference, not unique optimum',
 'risk_forecast':'frozen August2026 forecasts; no model refit and no realized August outcomes used',
 'business_status':'hypothetical demand, eligible pool and cost assumptions, not Cathay evidence',
 'caution':'robust only to the six listed scenarios; no guaranteed cash-flow solvency or 95% profit-probability constraint'}
(OUT/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')

designs=[];outcomes=[]
for pf in itertools.product(factors,repeat=3):
 price=pref*np.array(pf)
 if not np.all(np.diff(price)>0):continue
 for df,lf in itertools.product(benefits,repeat=2):
  cd=dref*df;cl=lref*lf
  if np.any(cl<cd):continue
  ident=f'M{len(designs)+1:04d}'
  values=[];minoperator=float('inf')
  for s in scenarios:
   u=np.log(np.array([.05,.03,.02])/.9)-s['beta']*np.log(price/pref)+.1*np.log(cd/dref)+.1*np.log(cl/lref)
   expu=np.exp(u);q=expu/(1+expu.sum())
   # Same common-state mixture as the financial module, now repeated per scenario.
   qm=.98*(-np.expm1(s['rho']*np.log1p(-p)))+.02*(-np.expm1(6*s['rho']*np.log1p(-p)))
   qd=qm*s['sd'];ql=qm*s['sl']
   sales=N[:,None]*q[None,:]
   revenue=(sales*price).sum(axis=1)
   claim=(sales*(qd[:,None]*cd+ql[:,None]*cl)).sum(axis=1)
   var=.25*revenue+.5*sales.sum(axis=1)
   profit=revenue-claim-var-50000
   margin=profit.sum()/revenue.sum()
   minoperator=min(minoperator,profit.min())
   values.append((float(profit.sum()),float(margin)))
   outcomes.append({'design_id':ident,'scenario':s['name'],'total_revenue':float(revenue.sum()),'total_expected_claims':float(claim.sum()),'total_variable_cost':float(var.sum()),'total_system_cost':600000.,'total_expected_profit':float(profit.sum()),'pooled_margin':float(margin),'minimum_operator_profit':float(profit.min()),'expected_policies':float(sales.sum())})
  designs.append({'design_id':ident,'price_A':price[0],'price_B':price[1],'price_C':price[2],
                  'delay_A':cd[0],'delay_B':cd[1],'delay_C':cd[2],'loss_A':cl[0],'loss_B':cl[1],'loss_C':cl[2],
                  'robust_expected_profit':min(v[0] for v in values),'robust_margin':min(v[1] for v in values),
                  'minimum_operator_profit_all_scenarios':minoperator,'feasible':minoperator>=0})
d=pd.DataFrame(designs);o=pd.DataFrame(outcomes)
feasible=d[d.feasible].copy();assert len(feasible)
objective=feasible[['robust_expected_profit','robust_margin']].to_numpy()
keep=[]
for x in objective:
 dominated=np.any(np.all(objective>=x,axis=1)&np.any(objective>x,axis=1))
 keep.append(not dominated)
front=feasible.loc[keep].copy()
for col in ['robust_expected_profit','robust_margin']:
 lo=front[col].min();hi=front[col].max();front['norm_'+col]=(front[col]-lo)/(hi-lo) if hi>lo else 1.
front['balance_score']=front[['norm_robust_expected_profit','norm_robust_margin']].min(axis=1)
front=front.sort_values(['balance_score','robust_expected_profit'],ascending=False)
selected=front.iloc[0].to_dict();ident=selected['design_id']
d.to_csv(OUT/'all_designs.csv',index=False);o.to_csv(OUT/'scenario_results.csv',index=False);front.to_csv(OUT/'pareto_frontier.csv',index=False)
selected_out=o[o.design_id==ident];selected_out.to_csv(OUT/'selected_scenarios.csv',index=False)
chosen=[]
for i,label in enumerate(['A','B','C']):
 chosen.append({'name':['Basic','Standard','Premium'][i],'numeric_tier':label,'price':float(selected['price_'+label]),'delay_benefit':float(selected['delay_'+label]),'loss_benefit':float(selected['loss_'+label]),'reference':priorplans[i]['reference']})
(OUT/'selected_menu.json').write_text(json.dumps({'selection':selected,'plans':chosen,'currency':'USD','is_final_quote':False},indent=2),encoding='utf-8')
pd.DataFrame([{k:v for k,v in x.items() if k!='reference'} for x in chosen]).to_csv(OUT/'selected_menu.csv',index=False)

# Independently reconcile selected vectorized results through the reviewed module.
individual=[]
for s in scenarios:
 a=copy.deepcopy(prior);a['demand']['log_price_sensitivity']=s['beta']
 a['conditional_delay_share']=s['sd'];a['conditional_loss_share']=s['sl']
 a['tier_hazard_multipliers']={x['name']:s['rho'] for x in chosen}
 for row in f.to_dict('records'):
  z=expected_portfolio(row,chosen,a)
  individual.append({'scenario':s['name'],'carrier':row['carrier'],'expected_profit':z['expected_profit'],'margin':z['expected_profit_margin'],'revenue':z['expected_revenue'],'claims':z['expected_claims'],'sales':z['expected_sales']})
ind=pd.DataFrame(individual)
for s,g in ind.groupby('scenario'):
 expected=selected_out[selected_out.scenario==s].iloc[0]
 assert abs(g.expected_profit.sum()-expected.total_expected_profit)<1e-6
 assert abs(g.revenue.sum()-expected.total_revenue)<1e-6
ind.to_csv(OUT/'selected_operator_results.csv',index=False)

# Additional sensitivity outside the optimization set: test, don't claim robustness.
extra=[]
for qtotal,K in itertools.product([.02,.05,.1],[10000.,50000.,200000.]):
 a=copy.deepcopy(prior);a['demand']['reference_purchase_probabilities']={'Basic':qtotal*.5,'Standard':qtotal*.3,'Premium':qtotal*.2};a['period_system_cost']=K
 for row in f.to_dict('records'):
  z=expected_portfolio(row,chosen,a)
  extra.append({'carrier':row['carrier'],'reference_total_purchase_probability':qtotal,'system_cost':K,'expected_profit':z['expected_profit'],'margin':z['expected_profit_margin']})
pd.DataFrame(extra).to_csv(OUT/'selected_extra_sensitivity.csv',index=False)
assert np.allclose(o.total_expected_profit,o.total_revenue-o.total_expected_claims-o.total_variable_cost-o.total_system_cost)
summary={'candidate_count':len(d),'feasible_count':len(feasible),'pareto_count':len(front),'selected_id':ident,
         'checks':'all scenario accounting identities; selected independently reconciled via insurance_model; nondominance checked against all feasible points',
         'extra_sensitivity_negative_cases':int(sum(x['expected_profit']<0 for x in extra)),'extra_sensitivity_count':len(extra),
         'source_forecast_sha256':hashlib.sha256((HERE/'forecast_interface_2026_08.csv').read_bytes()).hexdigest()}
(OUT/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
print(json.dumps(summary,indent=2));print(json.dumps(chosen,indent=2));print(selected_out.to_string(index=False))
