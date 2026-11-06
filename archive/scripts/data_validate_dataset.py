from pathlib import Path
import sys,json,hashlib
ROOT=Path(__file__).parent;sys.path.insert(0,str(ROOT.parents[3]/'tmp/baggage_ml_dependencies'))
import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
OUT=ROOT/'dataset';spec=json.loads((OUT/'feature_spec.json').read_text())
report={}
for lag in [1,3]:
 folder=OUT/f'lag{lag}'
 parts={s:pd.read_csv(folder/f'{s}_real.csv') for s in ['train','validation','test']}
 tr,va,te=[parts[s] for s in ['train','validation','test']]
 for s,d in parts.items():
  assert not d.is_synthetic.any()
  assert len(d)==d.row_id.nunique()
  assert d[spec['core_features']].notna().all().all()
  assert np.allclose(d.target_rate_per_1000,1000*d.target_mishandled_bags/d.target_enplaned_bags)
  delta=pd.to_datetime(d.target_month).dt.to_period('M').astype(int)-pd.to_datetime(d.latest_feature_month).dt.to_period('M').astype(int)
  assert delta.eq(lag).all()
 assert set(tr.row_id).isdisjoint(set(va.row_id)|set(te.row_id))
 assert set(va.row_id).isdisjoint(set(te.row_id))
 syn=pd.read_csv(folder/'synthetic_train_optional.csv')
 assert syn.is_synthetic.all() and set(syn.parent_row_id).issubset(set(tr.row_id))
 assert np.allclose(syn.target_rate_per_1000,tr.set_index('row_id').loc[syn.parent_row_id].target_rate_per_1000)
 # A small fit is only a schema/readiness check, not a performance experiment.
 X=tr[spec['core_features']].copy();V=va[spec['core_features']].copy()
 cats=sorted(X.airline_id.unique());X.airline_id=pd.Categorical(X.airline_id,categories=cats);V.airline_id=pd.Categorical(V.airline_id,categories=cats)
 m=LGBMRegressor(n_estimators=10,num_leaves=7,n_jobs=2,verbosity=-1,random_state=42)
 m.fit(X,tr.target_rate_per_1000,categorical_feature=['airline_id'])
 assert np.isfinite(m.predict(V)).all()
 report[f'lag{lag}']={'real_split_rows':{s:len(d) for s,d in parts.items()},'synthetic_train_copies':len(syn),'core_features':len(spec['core_features']),'checks':'PASS: keys, splits, lag dates, target arithmetic, core missingness, synthetic provenance, LightGBM input smoke test'}
manifest=[]
for p in sorted(OUT.rglob('*.csv')): manifest.append({'file':str(p.relative_to(OUT)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size})
(OUT/'validation_checks.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
(OUT/'file_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2))
