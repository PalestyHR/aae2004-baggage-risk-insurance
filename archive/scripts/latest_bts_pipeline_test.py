"""BTS risk + volume pipeline. Cathay is application context, not tested population.
No compensation amount, purchase rate or eligibility conversion is assumed.
"""
from pathlib import Path
import sys,json,hashlib,warnings
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'tmp/baggage_ml_dependencies'))
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from lightgbm import LGBMRegressor
DATA=ROOT/'Assignments/Group_Project/Material/integrated_dataset/dataset'
OUT=Path(__file__).with_suffix('');OUT.mkdir(exist_ok=True)
cols=json.loads((DATA/'feature_spec.json').read_text())['core_features']
d=pd.read_csv(DATA/'lag1/all_real.csv');d['date']=pd.to_datetime(d.target_month)
obs=pd.read_csv(DATA/'carrier_month_observed_NOT_model_input.csv')
obs['target_month']=obs.date
d=d.merge(obs[['airline_id','target_month','passenger_segments']],on=['airline_id','target_month'],validate='one_to_one',how='left')
idx=obs.set_index(['airline_id','t'])
for c,source in [('bags_year_ago','enplaned_bags'),('passengers_year_ago','passenger_segments')]:
 d[c]=[idx.loc[(r.airline_id,12*r.date.year+r.date.month-1-12),source] for _,r in d.iterrows()]
assert not d.is_synthetic.any() and d[cols].notna().all().all()
params=dict(n_estimators=200,learning_rate=.03,num_leaves=7,max_depth=3,min_child_samples=20,reg_lambda=1,verbosity=-1,n_jobs=2,random_state=42)
protocol={'date':'2026-11-06','population':'BTS domestic operating-carrier monthly records','application_context_only':'Cathay','features':cols,'risk':'LightGBM asymmetric squared loss under:over=1:2, equal row weights, a+b=2, mean initialization; selected previously on 2025','volume_candidates':['LightGBM direct squared loss','last year same month','equal blend latest month and last year'],'volume_selection':'2025 WAPE separately for baggage and passenger-segment volume; freeze before 2026','risk_selection':'no change based on 2026','training':'expanding monthly refit, all earlier target months; timely lag1 availability assumed','observed_exposure_at_forecast_time':False,'abnormal_count_formula':'predicted_rate_per_1000 * predicted_baggage_volume / 1000','unavailable':['permanent-loss counts','Cathay test labels','eligible customer counts','purchase rate','expected covered claims','costs','compensation amount'],'actuals_use':'target labels in past training; future actuals evaluation only','test_status':'exploratory: 2026 already inspected','parameters':params}
(OUT/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
def objective(y,p):
 e=p-y;c=np.where(e<0,2/3,4/3)
 return c*e,c
def count_metrics(y,p):
 mask=np.isfinite(y)&np.isfinite(p);y=np.asarray(y)[mask];p=np.asarray(p)[mask]
 e=p-y
 return {'n':len(y),'MAE':float(abs(e).mean()),'RMSE':float(np.sqrt((e**2).mean())),'WAPE_percent':float(100*abs(e).sum()/y.sum()),'observed_total':float(y.sum()),'predicted_total':float(p.sum()),'aggregate_bias_percent':float(100*e.sum()/y.sum())}
records=[];metrics=[];choices={};warn=[]
for year in [2025,2026]:
 foldrows=[]
 for date in sorted(d.loc[d.date.dt.year==year,'date'].unique()):
  tr=d[d.date<date];te=d[d.date==date]
  assert tr.date.max()<date
  prep=ColumnTransformer([('id',OneHotEncoder(handle_unknown='ignore',sparse_output=False),['airline_id']),('num',StandardScaler(),[c for c in cols if c!='airline_id'])])
  X=prep.fit_transform(tr[cols]);V=prep.transform(te[cols])
  o=te[['row_id','carrier','airline_id','target_month','target_rate_per_1000','target_mishandled_bags','target_enplaned_bags','passenger_segments']].copy()
  o['training_last_month']=tr.target_month.max();o['year']=year
  with warnings.catch_warnings(record=True) as ws:
   warnings.simplefilter('always')
   y=tr.target_rate_per_1000.to_numpy();base=float(y.mean())
   m=LGBMRegressor(**params,objective=objective).fit(X,y,init_score=np.full(len(y),base))
   o['rate_ml']=np.maximum(0,m.predict(V)+base)
   sm=LGBMRegressor(**params).fit(X,y);o['rate_symmetric']=np.maximum(0,sm.predict(V))
   for name,col in [('bags','target_enplaned_bags'),('passengers','passenger_segments')]:
    valid=tr[col].notna().to_numpy();model=LGBMRegressor(**params).fit(X[valid],tr.loc[valid,col])
    o[name+'_lgb']=np.maximum(0,model.predict(V))
   warn.extend(str(w.message) for w in ws)
  o['rate_simple']=.5*(te.recent_rate.to_numpy()+te.rate_year_ago.to_numpy())
  o['bags_seasonal']=te.bags_year_ago.to_numpy();o['bags_blend']=.5*(te.bags_year_ago.to_numpy()+te.recent_bag_volume.to_numpy())
  o['passengers_seasonal']=te.passengers_year_ago.to_numpy();o['passengers_blend']=.5*(te.passengers_year_ago.to_numpy()+te.lag_passenger_segments.to_numpy())
  foldrows.append(o)
 p=pd.concat(foldrows,ignore_index=True)
 for name,target in [('bags','target_enplaned_bags'),('passengers','passenger_segments')]:
  for method in ['lgb','seasonal','blend']:
   metrics.append({'year':year,'target':name,'method':method,**count_metrics(p[target].to_numpy(),p[name+'_'+method].to_numpy())})
 if year==2025:
  for name in ['bags','passengers']:
   choices[name]=min([x for x in metrics if x['target']==name and x['year']==2025],key=lambda x:x['WAPE_percent'])['method']
  (OUT/'volume_selection_before_2026.json').write_text(json.dumps(choices,indent=2),encoding='utf-8')
  print('2025 volume selection',choices,flush=True)
 p['forecast_baggage_volume']=p['bags_'+choices['bags']]
 p['forecast_passenger_segments']=p['passengers_'+choices['passengers']]
 for rate in ['ml','symmetric','simple']:
  p['anomaly_count_'+rate]=p['rate_'+rate]*p.forecast_baggage_volume/1000
  metrics.append({'year':year,'target':'anomaly_count','method':rate+'_same_selected_exposure',**count_metrics(p.target_mishandled_bags.to_numpy(),p['anomaly_count_'+rate].to_numpy())})
 p['anomaly_count_simple_full']=p.rate_simple*p.bags_blend/1000
 metrics.append({'year':year,'target':'anomaly_count','method':'simple_full_blend',**count_metrics(p.target_mishandled_bags.to_numpy(),p.anomaly_count_simple_full.to_numpy())})
 for rate in ['ml','symmetric','simple']:
  e=p['rate_'+rate]-p.target_rate_per_1000;w=p.target_enplaned_bags
  metrics.append({'year':year,'target':'rate_per_1000','method':rate,'n':len(p),'weighted_MAE':float(np.average(abs(e),weights=w)),'weighted_RMSE':float(np.sqrt(np.average(e**2,weights=w))),'weighted_bias':float(np.average(e,weights=w))})
 records.append(p)
p=pd.concat(records,ignore_index=True);p.to_csv(OUT/'predictions.csv',index=False)
pd.DataFrame(metrics).to_csv(OUT/'metrics.csv',index=False)
monthly=[]
for month,g in p[p.year==2026].groupby('target_month'):
 monthly.append({'month':month,'cases':len(g),'predicted_passenger_segments':g.forecast_passenger_segments.sum(),'actual_passenger_segments':g.passenger_segments.sum(min_count=1),'predicted_bags':g.forecast_baggage_volume.sum(),'actual_bags':g.target_enplaned_bags.sum(),'predicted_mishandled_bags':g.anomaly_count_ml.sum(),'actual_mishandled_bags':g.target_mishandled_bags.sum(),'passenger_label_coverage':g.passenger_segments.notna().mean()})
pd.DataFrame(monthly).to_csv(OUT/'monthly_totals_2026.csv',index=False)
frontier=p[p.year==2026][['row_id','carrier','target_month','forecast_passenger_segments','forecast_baggage_volume','rate_ml','anomaly_count_ml']].copy()
frontier['insured_claim_count']='UNKNOWN';frontier['permanent_loss_count']='NOT_IDENTIFIED'
frontier['payout_budget_formula']='forecast_passenger_segments * eligible_itinerary_factor * purchase_rate * premium - nonclaim_costs - target_profit'
frontier['average_benefit_ceiling_formula']='payout_budget / expected_covered_claims; only when budget>=0 and claims>0'
frontier.to_csv(OUT/'profitability_inputs_no_payout_assumption.csv',index=False)
(OUT/'audit.json').write_text(json.dumps({'selected_volume_methods':choices,'warnings':warn,'data_sha256':hashlib.sha256((DATA/'lag1/all_real.csv').read_bytes()).hexdigest(),'risk_retrained':True,'no_synthetic_rows':True,'no_business_parameters_filled':True},indent=2),encoding='utf-8')
print(pd.DataFrame([x for x in metrics if x['year']==2026]).to_string(index=False))
