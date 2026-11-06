from pathlib import Path
import urllib.request,urllib.parse,http.cookiejar,html.parser,concurrent.futures,zipfile,json,hashlib,time
ROOT=Path(__file__).parent
URL='https://www.transtats.bts.gov/DL_SelectFields.aspx?gnoyr_VQ=GEE&QO_fu146_anzr=Nv4%20Pn44vr45'
class Parser(html.parser.HTMLParser):
 def __init__(self): super().__init__();self.fields={}
 def handle_starttag(self,tag,attrs):
  a=dict(attrs)
  if tag=='input' and a.get('type')=='hidden' and a.get('name','').startswith('__'): self.fields[a['name']]=a.get('value','')
def fetch(year):
 p=ROOT/'raw_t100'/f't100_{year}.zip';p.parent.mkdir(exist_ok=True)
 if not p.exists():
  op=urllib.request.build_opener(urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()))
  s=op.open(URL,timeout=60).read().decode();parser=Parser();parser.feed(s)
  form=parser.fields|{'cboGeography':'All','cboYear':str(year),'cboPeriod':'All','btnDownload':'Download'}
  cols='DEPARTURES_SCHEDULED DEPARTURES_PERFORMED SEATS PASSENGERS AIRLINE_ID UNIQUE_CARRIER ORIGIN DEST YEAR MONTH CLASS'.split()
  form.update({c:'on' for c in cols})
  b=op.open(urllib.request.Request(URL,data=urllib.parse.urlencode(form).encode()),timeout=180).read()
  if not b.startswith(b'PK'):
   (ROOT/f't100_error_{year}.html').write_bytes(b);return {'year':year,'error':'not a ZIP','bytes':len(b)}
  p.write_bytes(b)
 with zipfile.ZipFile(p) as z: members=z.namelist()
 result={'year':year,'url':URL,'form':{'year':year,'period':'All'},'path':str(p.relative_to(ROOT)),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'members':members,'retrieved_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
 print(year,result['bytes'],members,flush=True);return result
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex: results=list(ex.map(fetch,range(2019,2027)))
(ROOT/'t100_sources.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
print('Done',flush=True)
