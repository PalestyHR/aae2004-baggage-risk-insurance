"""Expanding-history simple forecasts, observations available through t-1."""
from pathlib import Path
source=Path(__file__).with_name('periodic_learning.py').read_text(encoding='utf-8')
exec(compile(source.split('predictions=[];validation=[];summary=[];audit=[]')[0],str(Path(__file__).with_name('periodic_learning.py')),'exec'))
OUT=Path(__file__).parent/'simple_baselines'
OUT.mkdir(exist_ok=True)
data=build(1)
rows=[]
for _,r in data[data.YEAR.isin([2025,2026])].iterrows():
 h=d[(d.AIRLINE_ID==r.AIRLINE_ID)&(d.t<r.t)].sort_values('t')
 assert len(h) and h.t.max()<r.t
 last=lookup.loc[(r.AIRLINE_ID,r.t-1)].rate
 seasonal=lookup.loc[(r.AIRLINE_ID,r.t-12)].rate
 values={'last_month':last,'last_year_same_month':seasonal,'history_pooled_rate':1000*h.MISHANDLED_BAGGAGE.sum()/h.ENPLANED_BAGGAGE.sum(),'historical_same_month_mean':h[h.MONTH==r.MONTH].rate.mean(),'ewma_alpha_0.3':h.rate.ewm(alpha=.3,adjust=False).mean().iloc[-1],'equal_blend_last_month_last_year':.5*(last+seasonal)}
 for n in [3,6,12]:
  window=h[h.t>=r.t-n]
  assert len(window)==n
  values[f'mean_{n}_months']=window.rate.mean()
 for name,value in values.items():
  rows.append({'YEAR':r.YEAR,'MONTH':r.MONTH,'t':r.t,'AIRLINE_ID':r.AIRLINE_ID,'CARRIER':r.CARRIER,'rate':r.rate,'ENPLANED_BAGGAGE':r.ENPLANED_BAGGAGE,'model':name,'prediction':value})
p=pd.DataFrame(rows);summaries=[]
for (year,name),g in p.groupby(['YEAR','model']): summaries.append({'year':int(year),'model':name,**metrics(g)})
s=pd.DataFrame(summaries)
selected=s[s.year==2025].sort_values('weighted_MAE').iloc[0].model
expected=set(map(tuple,data[data.YEAR==2026][['AIRLINE_ID','t']].to_numpy()))
for name,g in p[p.YEAR==2026].groupby('model'):
 assert set(map(tuple,g[['AIRLINE_ID','t']].to_numpy()))==expected
 assert len(g)==101 and np.isfinite(g.prediction).all()
p.to_csv(OUT/'predictions.csv',index=False);s.to_csv(OUT/'metrics.csv',index=False)
monthly=[]
for (name,month),g in p[p.YEAR==2026].groupby(['model','MONTH']): monthly.append({'model':name,'month':int(month),**metrics(g)})
pd.DataFrame(monthly).to_csv(OUT/'monthly.csv',index=False)
print('Selected on 2025:',selected)
print(s.sort_values(['year','weighted_MAE']).to_string(index=False))
(OUT/'Method.txt').write_text('Simple baseline comparison, 2026-11-06\nSame source and target as periodic_learning. FORM-1, 2019 onward.\nAssume all preceding-month outcomes available; refresh forecasts monthly.\nPer-carrier forecasts. Historical pooled rate = sum(count)/sum(exposure)*1000.\nMoving means and historical same-calendar-month mean average monthly rates equally.\nEWMA alpha=0.3, adjust=False, fixed in advance; equal blend weights 0.5 each.\nNo 2026 tuning. 2025 selects best simple method: '+selected+'\n2026: identical 101 records, Jan-Aug. 2026 has already been inspected in previous experiments; exploratory comparison, not untouched final test.\nNo intervals or significance claims. US domestic baggage mishandling is not Cathay loss/claim frequency.\nMetrics in bags per 1000; test exposures weight evaluation only. Original CSV unchanged.\n',encoding='utf-8')
