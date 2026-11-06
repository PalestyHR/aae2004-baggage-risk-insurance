"""Controlled asymmetric squared-loss experiment, not profit optimization."""
from pathlib import Path
import sys,json,hashlib,warnings
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'tmp/baggage_ml_dependencies'))
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from lightgbm import LGBMRegressor
import lightgbm
DATA=ROOT/'Assignments/Group_Project/Material/integrated_dataset/dataset'
OUT=Path(__file__).with_suffix('');OUT.mkdir(exist_ok=True)
cols=json.loads((DATA/'feature_spec.json').read_text())['core_features']
d=pd.read_csv(DATA/'lag1/all_real.csv');d['date']=pd.to_datetime(d.target_month)
assert not d.is_synthetic.any()
RATIOS=[(1,4),(1,2),(1,1.5),(1,1),(1.5,1),(2,1),(4,1)]
PARAMS=dict(n_estimators=200,learning_rate=.03,num_leaves=7,max_depth=3,min_child_samples=20,reg_lambda=1,verbosity=-1,n_jobs=2,random_state=42)
threshold=float(d.loc[d.date.dt.year<=2024,'target_rate_per_1000'].quantile(.8))
protocol={'penalty_order':'under:over','ratios':RATIOS,'coefficient_normalization':'a+b=2','loss':'0.5*w*c*(prediction-target)^2; c=a if prediction<target else b','training_weights':['uniform','volume_normalized_to_training_mean'],'init':'same training weighted mean in every ratio within each fold/weighting arm','selection':'2025 weighted MAE, include symmetric 1:1; select per weighting arm and overall before 2026','params':PARAMS,'high_risk_threshold_train_only':threshold,'features':cols,'synthetic_rows_used':False,'realized_test_exposure':'evaluation weights only, never predictor','2026_status':'previously inspected exploratory test','source_sha256':hashlib.sha256((DATA/'lag1/all_real.csv').read_bytes()).hexdigest(),'lightgbm_version':lightgbm.__version__}
(OUT/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
def coeffs(under,over):return 2*under/(under+over),2*over/(under+over)
def obj_factory(under,over):
 a,b=coeffs(under,over)
 def objective(y_true,y_pred,weight):
  e=y_pred-y_true;c=np.where(e<0,a,b)
  w=np.ones_like(e) if weight is None else weight
  return w*c*e,w*c
 return objective
def objective_checks():
 y=np.array([1.,3.,5.,2.]);p=np.array([.2,4.,3.5,3.]);w=np.array([.2,.5,1.,2.])
 eps=1e-5
 for under,over in RATIOS:
  a,b=coeffs(under,over);fn=obj_factory(under,over)
  loss=lambda z:.5*w*np.where(z-y<0,a,b)*(z-y)**2
  grad,hess=fn(y,p,w)
  assert np.allclose(grad,(loss(p+eps)-loss(p-eps))/(2*eps),atol=1e-8)
  assert np.allclose(hess,(fn(y,p+eps,w)[0]-fn(y,p-eps,w)[0])/(2*eps),atol=1e-8)
  assert np.all(hess>0)
objective_checks()
def metrics(p):
 e=p.prediction-p.target_rate_per_1000;w=p.target_enplaned_bags
 out={'n':len(p),'weighted_MAE':float(np.average(abs(e),weights=w)),'RMSE':float(np.sqrt(np.mean(e**2))),'weighted_RMSE':float(np.sqrt(np.average(e**2,weights=w))),'weighted_bias':float(np.average(e,weights=w)),'underprediction_fraction':float((e<0).mean()),'weighted_undergap':float(np.average(np.maximum(-e,0),weights=w)),'weighted_overgap':float(np.average(np.maximum(e,0),weights=w))}
 high=p.target_rate_per_1000>=threshold
 out['high_risk_n']=int(high.sum());out['high_risk_weighted_MAE']=float(np.average(abs(e[high]),weights=w[high])) if high.any() else None
 return out
predictions=[];summary=[];checks=[];warn=[];selected={}
for year in [2025,2026]:
 yp=[]
 for date in sorted(d.loc[d.date.dt.year==year,'date'].unique()):
  tr=d[d.date<date];te=d[d.date==date];assert tr.date.max()<te.date.min()
  prep=ColumnTransformer([('id',OneHotEncoder(handle_unknown='ignore',sparse_output=False),['airline_id']),('num',StandardScaler(),[c for c in cols if c!='airline_id'])])
  X=prep.fit_transform(tr[cols]);V=prep.transform(te[cols]);y=tr.target_rate_per_1000.to_numpy()
  for weighting in ['uniform','volume']:
   w=np.ones(len(tr)) if weighting=='uniform' else (tr.target_enplaned_bags/tr.target_enplaned_bags.mean()).to_numpy()
   base=float(np.average(y,weights=w))
   with warnings.catch_warnings(record=True) as ws:
    warnings.simplefilter('always')
    builtin=LGBMRegressor(**PARAMS,objective='regression').fit(X,y,sample_weight=w)
    bp=np.maximum(0,builtin.predict(V))
    for under,over in RATIOS:
     name=f'{under:g}:{over:g}'
     model=LGBMRegressor(**PARAMS,objective=obj_factory(under,over))
     model.fit(X,y,sample_weight=w,init_score=np.full(len(tr),base))
     pred=np.maximum(0,model.predict(V)+base)
     if under==over:
      diff=float(np.max(abs(pred-bp)));checks.append({'month':str(date),'weighting':weighting,'symmetric_builtin_max_diff':diff})
      assert diff<1e-5,(date,weighting,diff)
     o=te[['row_id','target_month','carrier','target_rate_per_1000','target_enplaned_bags']].copy()
     o['prediction']=pred;o['year']=year;o['weighting']=weighting;o['under_over_ratio']=name;o['training_max_month']=tr.target_month.max()
     assert np.isfinite(pred).all();yp.append(o)
    warn.extend(str(x.message) for x in ws)
 p=pd.concat(yp,ignore_index=True);predictions.append(p)
 for (weighting,ratio),g in p.groupby(['weighting','under_over_ratio']):summary.append({'year':year,'weighting':weighting,'under_over_ratio':ratio,**metrics(g)})
 if year==2025:
  candidates=[x for x in summary if x['year']==2025]
  for weighting in ['uniform','volume']:
   selected[weighting]=min([x for x in candidates if x['weighting']==weighting],key=lambda x:x['weighted_MAE'])
  selected['overall']=min(candidates,key=lambda x:x['weighted_MAE'])
  (OUT/'selection_before_2026.json').write_text(json.dumps(selected,indent=2),encoding='utf-8')
  print('2025 selection',[(k,v['weighting'],v['under_over_ratio'],v['weighted_MAE']) for k,v in selected.items()],flush=True)
 print(pd.DataFrame([x for x in summary if x['year']==year])[['year','weighting','under_over_ratio','weighted_MAE','weighted_RMSE','weighted_bias']].to_string(index=False),flush=True)
allp=pd.concat(predictions,ignore_index=True)
allp.to_csv(OUT/'predictions.csv',index=False);pd.DataFrame(summary).to_csv(OUT/'metrics.csv',index=False)
monthly=[]
for keys,g in allp.groupby(['year','weighting','under_over_ratio','target_month']):monthly.append(dict(zip(['year','weighting','under_over_ratio','month'],keys),**metrics(g)))
pd.DataFrame(monthly).to_csv(OUT/'monthly.csv',index=False)
(OUT/'implementation_checks.json').write_text(json.dumps({'gradient_hessian_finite_difference':'passed','builtin_symmetry_checks':checks,'warnings':warn},indent=2),encoding='utf-8')
