"""Frozen-origin annual forecasts. No 2026 observations enter features or fitting."""
from pathlib import Path
import sys, json, hashlib
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'tmp/baggage_ml_dependencies'))
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor

OUT = Path(__file__).resolve().parent
SRC = ROOT / 'Assignments/Group_Project/Material/T_MISHANDLED_BAGGAGE_20261105_225724/T_MISHANDLED_BAGGAGE.csv'
raw = pd.read_csv(SRC)
d = raw[(raw.YEAR >= 2019) & (raw.FORM_TYPE == 'FORM-1')].copy()
assert not d.duplicated(['AIRLINE_ID','YEAR','MONTH']).any()
assert d[['MISHANDLED_BAGGAGE','ENPLANED_BAGGAGE']].notna().all().all()
assert ((d.ENPLANED_BAGGAGE > 0) & (d.MISHANDLED_BAGGAGE >= 0) & (d.MISHANDLED_BAGGAGE <= d.ENPLANED_BAGGAGE)).all()
d['rate'] = d.MISHANDLED_BAGGAGE / d.ENPLANED_BAGGAGE * 1000
lag = d[['AIRLINE_ID','YEAR','MONTH','rate','MISHANDLED_BAGGAGE','ENPLANED_BAGGAGE']].copy()
lag.YEAR += 1
lag = lag.rename(columns={'rate':'lag_rate','MISHANDLED_BAGGAGE':'lag_count','ENPLANED_BAGGAGE':'lag_exposure'})
data = d.merge(lag,on=['AIRLINE_ID','YEAR','MONTH'],how='left',validate='one_to_one')
excluded = data[data.lag_rate.isna()]
data = data.dropna(subset=['lag_rate']).sort_values(['YEAR','MONTH','AIRLINE_ID'])
data['trend'] = data.YEAR - 2019
data['sin_month'] = np.sin(2*np.pi*data.MONTH/12)
data['cos_month'] = np.cos(2*np.pi*data.MONTH/12)
features=['AIRLINE_ID','trend','sin_month','cos_month','lag_rate','lag_count','lag_exposure']
numeric=features[1:]
def model(name):
    prep=ColumnTransformer([('carrier',OneHotEncoder(handle_unknown='ignore',sparse_output=False),['AIRLINE_ID']),('numeric',StandardScaler(),numeric)])
    reg=Ridge(alpha=10) if name=='ridge' else RandomForestRegressor(n_estimators=300,min_samples_leaf=8,max_features=1.0,random_state=42,n_jobs=2)
    return make_pipeline(prep,reg)
def evaluate(frame,pred,target):
    y=frame[target].to_numpy(); err=pred-y
    out={'n':len(y),'MAE':float(np.abs(err).mean()),'RMSE':float(np.sqrt(np.mean(err**2))),'WAPE_percent':float(100*np.abs(err).sum()/np.abs(y).sum()),'bias_percent':float(100*err.sum()/y.sum())}
    if target=='rate': out['exposure_weighted_MAE_per_1000']=float(np.average(np.abs(err),weights=frame.ENPLANED_BAGGAGE))
    return out
metrics=[]; predictions=[]; selected={}
for target,lagcol in [('MISHANDLED_BAGGAGE','lag_count'),('rate','lag_rate')]:
    for year in [2025,2026]:
        train=data[data.YEAR<year]; test=data[data.YEAR==year]
        candidates=['seasonal_naive','ridge','random_forest']
        for name in candidates:
            if name=='seasonal_naive': pred=test[lagcol].to_numpy()
            else:
                fitted=model(name).fit(train[features],np.log1p(train[target]))
                pred=np.maximum(np.expm1(fitted.predict(test[features])),0)
            if target=='rate': pred=np.minimum(pred,1000)
            metrics.append({'target':target,'year':year,'model':name,**evaluate(test,pred,target)})
            o=test[['YEAR','MONTH','AIRLINE_ID','CARRIER','MISHANDLED_BAGGAGE','ENPLANED_BAGGAGE','rate']].copy()
            o['target']=target;o['model']=name;o['prediction']=pred
            predictions.append(o)
        if year==2025:
            key='WAPE_percent' if target=='MISHANDLED_BAGGAGE' else 'exposure_weighted_MAE_per_1000'
            selected[target]=min([m for m in metrics if m['year']==2025 and m['target']==target],key=lambda m:m[key])['model']
audit={'source_sha256':hashlib.sha256(SRC.read_bytes()).hexdigest(),'raw_rows':len(raw),'raw_year_min':int(raw.YEAR.min()),'raw_year_max':int(raw.YEAR.max()),'eligible_form1_2019plus':len(d),'supervised_rows_by_year':{str(k):int(v) for k,v in data.groupby('YEAR').size().items()},'excluded_no_same_month_prior_year':len(excluded),'excluded_test_rows':int((excluded.YEAR==2026).sum()),'2026_month_counts':{str(k):int(v) for k,v in d[d.YEAR==2026].groupby('MONTH').size().items()},'selected_on_2025_only':selected,'metrics':metrics}
pd.DataFrame(metrics).to_csv(OUT/'metrics.csv',index=False)
pd.concat(predictions).to_csv(OUT/'predictions.csv',index=False)
excluded.to_csv(OUT/'excluded_rows.csv',index=False)
(OUT/'audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
print(json.dumps(audit,indent=2))
