from pathlib import Path
import sys,json,warnings,time
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'tmp/baggage_ml_dependencies'))
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.linear_model import ElasticNet,PoissonRegressor
from sklearn.ensemble import RandomForestRegressor,ExtraTreesRegressor,HistGradientBoostingRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
DATA=ROOT/'Assignments/Group_Project/Material/integrated_dataset/dataset'
OUT=Path(__file__).with_suffix('');OUT.mkdir(exist_ok=True)
spec=json.loads((DATA/'feature_spec.json').read_text());cols=spec['core_features']
d=pd.read_csv(DATA/'lag1/all_real.csv');d['date']=pd.to_datetime(d.target_month)
assert not d.is_synthetic.any()
def estimator(name):
 if name=='ElasticNet': return ElasticNet(alpha=.01,l1_ratio=.5,max_iter=20000)
 if name=='Poisson_GLM':return PoissonRegressor(alpha=.1,max_iter=2000)
 if name=='RandomForest':return RandomForestRegressor(n_estimators=300,min_samples_leaf=8,n_jobs=2,random_state=42)
 if name=='ExtraTrees':return ExtraTreesRegressor(n_estimators=300,min_samples_leaf=8,n_jobs=2,random_state=42)
 if name=='HistGBR':return HistGradientBoostingRegressor(max_iter=200,learning_rate=.03,max_leaf_nodes=7,min_samples_leaf=20,l2_regularization=1,early_stopping=False,random_state=42)
 if name=='LightGBM':return LGBMRegressor(n_estimators=200,learning_rate=.03,num_leaves=7,max_depth=3,min_child_samples=20,reg_lambda=1,verbosity=-1,n_jobs=2,random_state=42)
 if name=='XGBoost':return XGBRegressor(n_estimators=200,learning_rate=.03,max_depth=3,min_child_weight=20,reg_lambda=1,objective='reg:squarederror',tree_method='hist',n_jobs=2,random_state=42)
def evaluate(g):
 e=g.prediction-g.target_rate_per_1000;w=g.target_enplaned_bags
 return {'n':len(g),'weighted_MAE':float(np.average(abs(e),weights=w)),'MAE':float(abs(e).mean()),'RMSE':float(np.sqrt((e**2).mean())),'weighted_bias':float(np.average(e,weights=w))}
names=['blend50','seasonal_naive','ElasticNet','Poisson_GLM','RandomForest','ExtraTrees','HistGBR','LightGBM','XGBoost']
rows=[];scores=[];warn=[]
for year in [2025,2026]:
 for name in names:
  parts=[]
  for date in sorted(d.loc[d.date.dt.year==year,'date'].unique()):
   tr=d[d.date<date];te=d[d.date==date]
   assert tr.date.max()<te.date.min()
   if name=='blend50': pred=.5*(te.recent_rate+te.rate_year_ago)
   elif name=='seasonal_naive':pred=te.rate_year_ago
   else:
    prep=ColumnTransformer([('id',OneHotEncoder(handle_unknown='ignore',sparse_output=False),['airline_id']),('num',StandardScaler(),[c for c in cols if c!='airline_id'])])
    X=prep.fit_transform(tr[cols]);V=prep.transform(te[cols]);m=estimator(name)
    kw={'sample_weight':tr.target_enplaned_bags/tr.target_enplaned_bags.mean()} if name=='Poisson_GLM' else {}
    with warnings.catch_warnings(record=True) as ws:
     warnings.simplefilter('always');m.fit(X,tr.target_rate_per_1000,**kw);pred=np.maximum(0,m.predict(V))
     warn.extend([{'year':year,'model':name,'date':str(date),'warning':str(w.message)} for w in ws])
   o=te[['row_id','target_month','carrier','target_rate_per_1000','target_enplaned_bags']].copy();o['prediction']=np.asarray(pred);o['model']=name;o['year']=year
   assert np.isfinite(o.prediction).all();parts.append(o)
  p=pd.concat(parts);rows.append(p);s={'year':year,'model':name,**evaluate(p)};scores.append(s);print(s,flush=True)
 if year==2025:
  choice=min(scores,key=lambda x:x['weighted_MAE'])['model'];print('PRE-TEST CHOICE',choice,flush=True)
p=pd.concat(rows,ignore_index=True)
p.to_csv(OUT/'predictions.csv',index=False);pd.DataFrame(scores).to_csv(OUT/'metrics.csv',index=False)
monthly=[]
for (year,model,month),g in p.groupby(['year','model','target_month']):monthly.append({'year':int(year),'model':model,'month':month,**evaluate(g)})
pd.DataFrame(monthly).to_csv(OUT/'monthly.csv',index=False)
(OUT/'audit.json').write_text(json.dumps({'choice_on_2025':choice,'features':cols,'warnings':warn,'test_has_been_previously_inspected':True,'metrics':scores},indent=2),encoding='utf-8')
