"""Build observed monthly panel and strictly lagged training tables.
Synthetic robustness copies inherit labels; no fabricated outcome observations.
"""
from pathlib import Path
import sys,json,zipfile,hashlib,calendar
ROOT=Path(__file__).parent
WORK=ROOT.parents[3]
sys.path.insert(0,str(WORK/'tmp/baggage_ml_dependencies'))
import numpy as np
import pandas as pd
OUT=ROOT/'dataset';OUT.mkdir(exist_ok=True)
audit={}
bag_path=ROOT.parent/'T_MISHANDLED_BAGGAGE_20261105_225724/T_MISHANDLED_BAGGAGE.csv'
b=pd.read_csv(bag_path)
b=b[(b.YEAR>=2019)&(b.FORM_TYPE=='FORM-1')].copy()
b=b.rename(columns={'YEAR':'year','MONTH':'month','AIRLINE_ID':'airline_id','CARRIER':'carrier','MISHANDLED_BAGGAGE':'mishandled_bags','ENPLANED_BAGGAGE':'enplaned_bags'})
b=b[['year','month','airline_id','carrier','mishandled_bags','enplaned_bags']]
assert not b.duplicated(['year','month','airline_id']).any()
assert b.enplaned_bags.gt(0).all() and b.mishandled_bags.ge(0).all()
b['rate_per_1000']=1000*b.mishandled_bags/b.enplaned_bags
b['t']=12*b.year+b.month-1
b['date']=pd.to_datetime(dict(year=b.year,month=b.month,day=1)).dt.strftime('%Y-%m-%d')

# T100: select scheduled passenger/cargo service, aggregate operating DOT IDs.
ts=[]
for path in sorted((ROOT/'raw_t100').glob('t100_*.zip')):
 with zipfile.ZipFile(path) as z: ts.append(pd.read_csv(z.open(z.namelist()[0])))
t=pd.concat(ts,ignore_index=True)
t=t[t.CLASS=='F'].rename(columns={'YEAR':'year','MONTH':'month','AIRLINE_ID':'airline_id'})
audit['t100_scheduled_segment_rows']=len(t)
audit['t100_latest_month']=f'{int(t.year.max())}-{int(t[t.year==t.year.max()].month.max()):02d}'
nums={'PASSENGERS':'passenger_segments','SEATS':'seats','DEPARTURES_PERFORMED':'departures','DEPARTURES_SCHEDULED':'scheduled_departures'}
c=t.groupby(['year','month','airline_id'])[list(nums)].sum(min_count=1).rename(columns=nums).reset_index()
c['seat_occupancy']=c.passenger_segments/c.seats.replace(0,np.nan)
orig=t.groupby(['year','month','ORIGIN'])[['PASSENGERS','DEPARTURES_PERFORMED']].sum().rename_axis(['year','month','airport']).reset_index()
dest=t.groupby(['year','month','DEST'])[['PASSENGERS','DEPARTURES_PERFORMED']].sum().rename_axis(['year','month','airport']).reset_index()
ap=pd.concat([orig,dest]).groupby(['year','month','airport']).sum().rename(columns={'PASSENGERS':'airport_passenger_movements','DEPARTURES_PERFORMED':'airport_aircraft_movements'}).reset_index()
ap.to_csv(OUT/'airport_month_observed.csv',index=False)

# Delay-source carrier codes are operating-carrier codes for these matches.
delay=pd.read_csv(ROOT/'raw_delay/Airline_Delay_Cause.csv')
assert not delay.duplicated(['year','month','carrier','airport']).any()
audit['delay_airport_carrier_month_rows']=len(delay)
delay=delay.merge(ap,on=['year','month','airport'],how='left',validate='many_to_one')
total=delay.groupby(['year','month','carrier'])[['arr_flights','arr_del15','arr_cancelled','arr_diverted','arr_delay','weather_ct','weather_delay','nas_ct','late_aircraft_ct']].sum(min_count=1).reset_index()
total=total.rename(columns={'arr_flights':'arrival_flights'})
for raw,name in [('arr_del15','arrival_delay_rate'),('arr_cancelled','cancel_rate'),('arr_diverted','diversion_rate'),('weather_ct','weather_delay_fraction'),('nas_ct','nas_delay_fraction'),('late_aircraft_ct','late_aircraft_delay_fraction'),('arr_delay','delay_minutes_per_arrival'),('weather_delay','weather_delay_minutes_per_arrival')]:
 total[name]=total[raw]/total.arrival_flights.replace(0,np.nan)
total=total.drop(columns=['arr_del15','arr_cancelled','arr_diverted','arr_delay','weather_ct','weather_delay','nas_ct','late_aircraft_ct'])

# Genuine GSOD daily weather. Sentinels are missing, not zero.
w=[]
for path in sorted((ROOT/'raw_weather').glob('*.csv')):
 x=pd.read_csv(path,dtype={'FRSHTT':str})
 x['airport']=path.stem.split('_')[0]
 x['day']=pd.to_datetime(x.DATE);x['year']=x.day.dt.year;x['month']=x.day.dt.month
 for col,invalid in [('TEMP',9999.9),('VISIB',999.9),('WDSP',999.9),('MXSPD',999.9),('PRCP',99.99)]:
  x[col]=pd.to_numeric(x[col],errors='coerce').mask(lambda v:v>=invalid)
 x['temp_c']=(x.TEMP-32)*5/9
 x['wind_knots']=x.WDSP
 x['low_visibility_day']=(x.VISIB<3).astype(float).where(x.VISIB.notna())
 x['rain_day']=(x.PRCP>0).astype(float).where(x.PRCP.notna())
 w.append(x)
w=pd.concat(w,ignore_index=True)
assert not w.duplicated(['airport','DATE']).any()
weather_cols=['temp_c','wind_knots','low_visibility_day','rain_day']
wm=w.groupby(['year','month','airport'])[weather_cols].mean().reset_index()
days=w.groupby(['year','month','airport']).size().rename('observed_days').reset_index()
wm=wm.merge(days,on=['year','month','airport'],validate='one_to_one')
wm['calendar_days']=[calendar.monthrange(y,m)[1] for y,m in zip(wm.year,wm.month)]
wm['day_coverage']=wm.observed_days/wm.calendar_days
# Exclude clearly partial monthly means; retain raw coverage in weather audit table.
wm.loc[wm.day_coverage<.8,weather_cols]=np.nan
wm.to_csv(OUT/'weather_airport_month_observed.csv',index=False)
audit['weather_daily_rows']=len(w);audit['weather_date_min']=str(w.day.min().date());audit['weather_date_max']=str(w.day.max().date())
delay=delay.merge(wm,on=['year','month','airport'],how='left',validate='many_to_one')

def weighted_feature(frame,col):
 valid=frame[col].notna()&frame.arr_flights.gt(0)
 denom=frame.loc[valid,'arr_flights'].sum();alln=frame.arr_flights.sum()
 return (float(np.average(frame.loc[valid,col],weights=frame.loc[valid,'arr_flights'])) if denom else np.nan,float(denom/alln) if alln else np.nan)
weighted=[]
for keys,g in delay.groupby(['year','month','carrier']):
 row=dict(zip(['year','month','carrier'],keys))
 for col in ['airport_passenger_movements','airport_aircraft_movements']+weather_cols:
  val,coverage=weighted_feature(g,col)
  row['network_'+col]=val;row['coverage_'+col]=coverage
 weighted.append(row)
total=total.merge(pd.DataFrame(weighted),on=['year','month','carrier'],validate='one_to_one')
panel=b.merge(c,on=['year','month','airline_id'],how='left',validate='one_to_one').merge(total,on=['year','month','carrier'],how='left',validate='one_to_one')
panel.to_csv(OUT/'carrier_month_observed_NOT_model_input.csv',index=False)
audit['observed_baggage_rows']=len(panel)
audit['missing_t100_rows']=panel[panel.passenger_segments.isna()][['date','carrier']].to_dict('records')
audit['missing_delay_rows']=panel[panel.arrival_flights.isna()][['date','carrier']].to_dict('records')

# Every dynamic predictor is calendar-lagged, never target-month observed data.
idx=panel.set_index(['airline_id','t'])
extra=[x for x in c.columns if x not in ['year','month','airline_id']]+[x for x in total.columns if x not in ['year','month','carrier']]
def training_table(lag):
 rows=[];excluded=[]
 for _,r in panel.iterrows():
  try:
   hist=[idx.loc[(r.airline_id,r.t-lag-j)] for j in range(4)];old=idx.loc[(r.airline_id,r.t-12)]
  except KeyError: excluded.append({'carrier':r.carrier,'date':r.date,'reason':'required calendar baggage history absent'});continue
  h=hist[0];rates=np.array([x.rate_per_1000 for x in hist])
  row={'row_id':f'{r.airline_id}_{r.date}','carrier':r.carrier,'airline_id':r.airline_id,'target_month':r.date,'latest_feature_month':h.date,'assumed_availability_lag_months':lag,'split':'train' if r.year<=2024 else ('validation' if r.year==2025 else 'test'),'is_synthetic':False,'sample_weight':1.0,'target_rate_per_1000':r.rate_per_1000,'target_mishandled_bags':r.mishandled_bags,'target_enplaned_bags':r.enplaned_bags,'month_sin':np.sin(2*np.pi*r.month/12),'month_cos':np.cos(2*np.pi*r.month/12),'month_trend':r.t-2019*12,'recent_rate':rates[0],'rate_mean3':rates[:3].mean(),'rate_std3':rates[:3].std(),'rate_change':rates[0]-rates[1:].mean(),'rate_year_ago':old.rate_per_1000,'recent_bag_volume':h.enplaned_bags,'bag_volume_growth':h.enplaned_bags/hist[1].enplaned_bags-1}
  for col in extra: row['lag_'+col]=h[col]
  row['lag_passenger_growth']=h.passenger_segments/hist[1].passenger_segments-1 if pd.notna(h.passenger_segments) and hist[1].passenger_segments>0 else np.nan
  rows.append(row)
 table=pd.DataFrame(rows)
 assert (pd.to_datetime(table.latest_feature_month)<pd.to_datetime(table.target_month)).all()
 assert not table.row_id.duplicated().any()
 return table,pd.DataFrame(excluded)

for lag in [1,3]:
 table,excluded=training_table(lag)
 folder=OUT/f'lag{lag}';folder.mkdir(exist_ok=True)
 table.to_csv(folder/'all_real.csv',index=False)
 for split in ['train','validation','test']: table[table.split==split].to_csv(folder/f'{split}_real.csv',index=False)
 excluded.to_csv(folder/'excluded_rows.csv',index=False)
 audit[f'lag{lag}_split_counts']=table.groupby('split').size().to_dict()
 audit[f'lag{lag}_missing_by_feature']=table.isna().sum().loc[lambda x:x>0].to_dict()

 # Synthetic feature-missingness augmentation: preserves real targets, no new
 # invented outcomes. Training-only sources, 20% copies, 10% external fields masked.
 tr=table[table.split=='train'];rng=np.random.default_rng(42)
 syn=tr.sample(frac=.2,random_state=42).copy()
 syn['parent_row_id']=syn.row_id;syn['row_id']='SYN_MASK_'+syn.row_id
 syn['is_synthetic']=True;syn['sample_weight']=.25
 syn['synthetic_method']='external_feature_dropout_10pct'
 syn['label_origin']='inherited_from_parent_real_training_row'
 synthetic_fields=[x for x in table if x.startswith('lag_') and not x.startswith('lag_coverage_')]
 # Keep coverage paired with masked network fields.
 for i in syn.index:
  chosen=rng.choice(synthetic_fields,size=max(1,round(.1*len(synthetic_fields))),replace=False)
  for col in chosen:
   syn.loc[i,col]=np.nan
   if col.startswith('lag_network_'):
    cov=col.replace('lag_network_','lag_coverage_')
    if cov in syn: syn.loc[i,cov]=0
 syn.to_csv(folder/'synthetic_train_optional.csv',index=False)
 pd.concat([tr,syn],ignore_index=True).to_csv(folder/'train_with_synthetic_OPTIONAL.csv',index=False)
 assert not set(syn.parent_row_id)&set(table[table.split!='train'].row_id)
 audit[f'lag{lag}_synthetic_copies']=len(syn)

metadata=['row_id','carrier','target_month','latest_feature_month','assumed_availability_lag_months','split','is_synthetic','sample_weight']
target=['target_rate_per_1000','target_mishandled_bags','target_enplaned_bags']
features=[x for x in table if x not in metadata+target]
optional=[x for x in features if any(y in x for y in weather_cols)]
spec={'target':'target_rate_per_1000','categorical_features':['airline_id'],'core_features':[x for x in features if x not in optional],'optional_observed_weather_features':optional,'never_use_as_features':metadata+target+['parent_row_id','label_origin','synthetic_method'],'primary_split':'train <=2024, validation 2025, test 2026','note':'lag1 assumes timely internal data; lag3 assumes availability through t-3; neither reconstructs historical publication revisions.'}
(OUT/'feature_spec.json').write_text(json.dumps(spec,indent=2),encoding='utf-8')
pd.DataFrame([{'column':x,'role':'target' if x in target else ('feature' if x in features else 'metadata'),'missing_rows':int(table[x].isna().sum())} for x in table]).to_csv(OUT/'column_inventory.csv',index=False)
(OUT/'audit.json').write_text(json.dumps(audit,indent=2,default=str),encoding='utf-8')
print(json.dumps({k:v for k,v in audit.items() if 'missing_by_feature' not in k},indent=2,default=str))
