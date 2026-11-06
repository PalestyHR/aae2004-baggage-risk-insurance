from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import pandas as pd
from baggage_project.risk import FrozenRiskPredictor

def compare(left,right,keys,values,tolerance=1e-6):
    a=pd.read_csv(left);b=pd.read_csv(right)
    merged=a.merge(b,on=keys,suffixes=('_new','_ref'),validate='one_to_one')
    assert len(merged)==len(a)==len(b)
    for c in values:
        assert np.allclose(merged[c+'_new'],merged[c+'_ref'],rtol=1e-8,atol=tolerance,equal_nan=True),c
    return len(merged)

count=compare(ROOT/'results/reproduced_risk/predictions.csv',ROOT/'results/reference/model1/predictions.csv',
              ['row_id'],['rate_ml','forecast_baggage_volume','forecast_passenger_segments','anomaly_count_ml'])
df=pd.read_csv(ROOT/'data/processed/lag1/test_real.csv')
df=df[df.target_month=='2026-08-01']
loaded=FrozenRiskPredictor(ROOT/'models').predict(df)
saved=pd.read_csv(ROOT/'results/reproduced_risk/predictions.csv').set_index('row_id').loc[df.row_id,'rate_ml'].to_numpy()
assert np.allclose(loaded,saved,rtol=1e-9,atol=1e-7)
new=json.loads((ROOT/'results/reproduced_insurance/selected_menu.json').read_text())
ref=json.loads((ROOT/'configs/selected_menu_v2.json').read_text())
assert new['selection']['design_id']==ref['selection']['design_id']=='M0775'
for key in ['robust_expected_profit','robust_margin']:
    assert np.isclose(new['selection'][key],ref['selection'][key],rtol=1e-8,atol=1e-6)
assert new['plans']==ref['plans']
compare(ROOT/'results/reproduced_insurance/selected_scenarios.csv',
        ROOT/'results/reference/insurance_design_history/optimized_menu_v2/selected_scenarios.csv',
        ['scenario'],['total_expected_profit','pooled_margin','total_expected_claims'])
result={'risk_records_reproduced':count,'frozen_booster_roundtrip_cases':len(df),'insurance_selected_id':'M0775',
        'status':'PASS','note':'Numerical reproduction verifies implementation, not empirical demand/claim assumptions.'}
(ROOT/'results/reproduction_check.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
