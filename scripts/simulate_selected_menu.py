"""Additional implementation check for the adopted M0775 menu, scenario only."""
from pathlib import Path
import sys,json,copy
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import pandas as pd
from baggage_project.insurance import simulate_cohort
OUT=ROOT/'results/selected_menu_monte_carlo';OUT.mkdir(exist_ok=True)
f=pd.read_csv(ROOT/'results/reference/model1/forecast_interface_2026_08.csv')
plans=json.loads((ROOT/'configs/selected_menu_v2.json').read_text())['plans']
base=json.loads((ROOT/'configs/insurance_assumptions.json').read_text())['base_assumptions']
rows=[]
for b in [1.,2.,3.]:
 for risk,rho,sl in [('base',1.,.05),('higher',2.,.1)]:
  a=copy.deepcopy(base);a['demand']['log_price_sensitivity']=b
  a['conditional_loss_share']=sl;a['tier_hazard_multipliers']={p['name']:rho for p in plans}
  for i,row in enumerate(f.to_dict('records')):
   out=simulate_cohort(row,plans,a,draws=30000,seed=20261106+int(b)*100+i+(1000 if risk=='higher' else 0))
   assert abs(out['simulated_mean_profit']-out['expected_profit_analytic'])<=6*out['mean_standard_error']
   rows.append({'scenario':f'beta{b}_{risk}','carrier':row['carrier'],**out})
r=pd.DataFrame(rows);r.to_csv(OUT/'results.csv',index=False)
summary=r.groupby('scenario').agg(cases=('carrier','size'),minimum_profit_probability=('probability_profit_positive','min'),maximum_profit_probability=('probability_profit_positive','max'),minimum_05_profit_quantile=('profit_quantile_05','min')).reset_index()
summary.to_csv(OUT/'summary.csv',index=False)
(OUT/'notes.txt').write_text('Adopted M0775 menu; 72 single-operator single-cohort scenarios, 30000 draws each. Same uncalibrated risk/demand/cost assumptions as optimization. All means reconcile within six Monte Carlo standard errors. This is not annual empirical profitability or a certification of 95% profit probability. No parameter/forecast uncertainty has been calibrated.\n',encoding='utf-8')
print(summary.to_string(index=False))
