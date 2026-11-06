"""Pre-specified periodic learning comparison; 2025 selects window, 2026 evaluates.
Delay d means target month t can access observations only through t-d.
Fixed RF holds fitted parameters fixed, but receives the same refreshed features
as updating RF, isolating retraining from simply receiving recent information.
"""
from pathlib import Path
import sys,json,hashlib
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'tmp/baggage_ml_dependencies'))
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor
OUT=Path(__file__).parent/'periodic_learning'
OUT.mkdir(exist_ok=True)
SRC=ROOT/'Assignments/Group_Project/Material/T_MISHANDLED_BAGGAGE_20261105_225724/T_MISHANDLED_BAGGAGE.csv'
raw=pd.read_csv(SRC)
d=raw[(raw.YEAR>=2019)&(raw.FORM_TYPE=='FORM-1')].copy()
d['t']=d.YEAR*12+d.MONTH-1
d['rate']=1000*d.MISHANDLED_BAGGAGE/d.ENPLANED_BAGGAGE
assert not d.duplicated(['AIRLINE_ID','t']).any()
assert d.ENPLANED_BAGGAGE.gt(0).all()
lookup=d.set_index(['AIRLINE_ID','t'])
features=['AIRLINE_ID','month_sin','month_cos','trend','recent_rate','mean3','std3','rate_change','year_ago','recent_volume','volume_change']
def build(delay):
 rows=[]
 for _,r in d.iterrows():
  try:
   hist=[lookup.loc[(r.AIRLINE_ID,r.t-delay-j)] for j in range(4)]
   old=lookup.loc[(r.AIRLINE_ID,r.t-12)]
  except KeyError: continue
  rates=np.array([x.rate for x in hist])
  rows.append(dict(r,month_sin=np.sin(2*np.pi*r.MONTH/12),month_cos=np.cos(2*np.pi*r.MONTH/12),trend=r.t-2019*12,recent_rate=rates[0],mean3=rates[:3].mean(),std3=rates[:3].std(),rate_change=rates[0]-rates[1:].mean(),year_ago=old.rate,recent_volume=hist[0].ENPLANED_BAGGAGE,volume_change=hist[0].ENPLANED_BAGGAGE/hist[1].ENPLANED_BAGGAGE-1))
 return pd.DataFrame(rows).sort_values(['t','AIRLINE_ID'])
def fit(frame):
 prep=ColumnTransformer([('id',OneHotEncoder(handle_unknown='ignore',sparse_output=False),['AIRLINE_ID']),('numeric','passthrough',features[1:])])
 rf=RandomForestRegressor(n_estimators=300,min_samples_leaf=8,random_state=42,n_jobs=2)
 return make_pipeline(prep,rf).fit(frame[features],frame.rate)
def training(data,cutoff,window):
 train=data[data.t<=cutoff]
 if window==36: train=train[train.t>cutoff-36]
 assert train.t.max()<=cutoff
 return train
def forecast(data,year,delay,window,mode):
 result=[]; fitted=None; initial_cutoff=year*12-delay
 if mode=='fixed_rf': fitted=fit(training(data,initial_cutoff,window))
 for t in sorted(data.loc[data.YEAR==year,'t'].unique()):
  test=data[data.t==t]; cutoff=t-delay
  if mode=='updating_rf': fitted=fit(training(data,cutoff,window))
  if mode in ['fixed_rf','updating_rf']: pred=fitted.predict(test[features])
  else: pred=test['year_ago' if mode=='seasonal_naive' else 'mean3'].to_numpy()
  o=test[['YEAR','MONTH','t','AIRLINE_ID','CARRIER','rate','ENPLANED_BAGGAGE']].copy()
  o['prediction']=pred;o['mode']=mode;o['delay']=delay;o['window']=str(window)
  o['fit_cutoff']=initial_cutoff if mode=='fixed_rf' else cutoff
  result.append(o)
 return pd.concat(result,ignore_index=True)
def metrics(p):
 e=p.prediction-p.rate;w=p.ENPLANED_BAGGAGE
 return {'n':len(p),'MAE':float(abs(e).mean()),'RMSE':float(np.sqrt(np.mean(e**2))),'weighted_MAE':float(np.average(abs(e),weights=w)),'weighted_bias':float(np.average(e,weights=w))}
predictions=[];validation=[];summary=[];audit=[]
for delay in [1,3]:
 data=build(delay)
 candidates=[]
 for window in ['expanding',36]:
  p=forecast(data,2025,delay,window,'updating_rf')
  # Select only using validation outcomes available at the first test origin.
  p=p[p.t<=2026*12-delay]
  m={'delay':delay,'window':str(window),**metrics(p)};validation.append(m);candidates.append((m['weighted_MAE'],window))
 selected=min(candidates,key=lambda x:x[0])[1]
 for mode in ['seasonal_naive','recent_mean3','fixed_rf','updating_rf']:
  p=forecast(data,2026,delay,selected,mode);predictions.append(p)
  summary.append({'delay':delay,'selected_window':str(selected),'mode':mode,**metrics(p)})
 test=data[data.YEAR==2026]
 audit.append({'delay':delay,'selected_window':str(selected),'test_rows':len(test),'excluded_test_keys':d[(d.YEAR==2026)&(~d.set_index(['AIRLINE_ID','t']).index.isin(test.set_index(['AIRLINE_ID','t']).index))][['CARRIER','YEAR','MONTH']].to_dict('records')})
allp=pd.concat(predictions,ignore_index=True)
monthly=[];carrier=[]
for keys,g in allp.groupby(['delay','mode','MONTH']): monthly.append(dict(zip(['delay','mode','MONTH'],keys),**metrics(g)))
for keys,g in allp.groupby(['delay','mode','CARRIER']): carrier.append(dict(zip(['delay','mode','CARRIER'],keys),**metrics(g)))
for name,rows in [('validation',validation),('summary',summary),('monthly',monthly),('carrier',carrier)]: pd.DataFrame(rows).to_csv(OUT/(name+'.csv'),index=False)
allp.to_csv(OUT/'predictions.csv',index=False)
payload={'source_sha256':hashlib.sha256(SRC.read_bytes()).hexdigest(),'audit':audit,'validation':validation,'test':summary}
(OUT/'audit.json').write_text(json.dumps(payload,indent=2),encoding='utf-8')
print(json.dumps(payload,indent=2))
