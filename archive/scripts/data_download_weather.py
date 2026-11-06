from pathlib import Path
import sys,urllib.request,concurrent.futures,json,hashlib,time
ROOT=Path(__file__).parent
sys.path.insert(0,str(ROOT.parents[3]/'tmp/baggage_ml_dependencies'))
import pandas as pd
d=pd.read_csv(ROOT/'raw_delay/Airline_Delay_Cause.csv')
top=d[d.year==2019].groupby('airport').arr_flights.sum().nlargest(20).index
s=pd.read_csv(ROOT/'isd-history.csv',dtype=str)
stations=[]
for a in top:
 m=s[(s.ICAO=='K'+a)&(s.BEGIN<='20190101')].sort_values('END',ascending=False)
 if len(m):
  r=m.iloc[0];stations.append({'airport':a,'station':r.USAF+r.WBAN,'name':r['STATION NAME'],'inventory_end':r.END})
pd.DataFrame(stations).to_csv(ROOT/'weather_station_mapping.csv',index=False)
def fetch(task):
 r,y=task;name=f'{r["airport"]}_{y}.csv';p=ROOT/'raw_weather'/name;p.parent.mkdir(exist_ok=True)
 u=f'https://www.ncei.noaa.gov/data/global-summary-of-the-day/access/{y}/{r["station"]}.csv'
 try:
  if not p.exists(): p.write_bytes(urllib.request.urlopen(u,timeout=40).read())
  return {'airport':r['airport'],'year':y,'url':u,'file':str(p.relative_to(ROOT)),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'status':'downloaded'}
 except Exception as e: return {'airport':r['airport'],'year':y,'url':u,'status':'unavailable','error':str(e)}
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex: results=list(ex.map(fetch,[(r,y) for r in stations for y in range(2019,2027)]))
(ROOT/'weather_sources.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
print(pd.DataFrame(results).groupby(['year','status']).size().to_string())
