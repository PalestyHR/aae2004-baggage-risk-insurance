from pathlib import Path
import urllib.request,concurrent.futures,json,hashlib,zipfile,time
ROOT=Path(__file__).parent
base='https://www.bts.gov/sites/bts.dot.gov/files/docs/airline-data/domestic-segments/'
names=['db28seg.dd.wac.2011.2020.zip','db28seg.dd.wac.2021.2022.zip','DB28SEG.DD.WAC.202301.202312.REL01.04MAR2024.zip','DB28SEG.DD.WAC.202401.202412.REL01.04MAR2025.zip','DB28SEG.DD.WAC.202501.202512.REL01.03MAR2026.zip','DB28SEG.DD.WAC.202508.202607.REL01.02OCT2026.zip']
def fetch(name):
 p=ROOT/'raw_t100'/name;p.parent.mkdir(exist_ok=True)
 if not p.exists():
  with urllib.request.urlopen(base+name,timeout=180) as resp: p.write_bytes(resp.read())
 with zipfile.ZipFile(p) as z: members=z.namelist()
 o={'file':str(p.relative_to(ROOT)),'url':base+name,'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'members':members,'retrieved_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
 print(name,o['bytes'],members[:4],flush=True);return o
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex: results=list(ex.map(fetch,names))
(ROOT/'t100_sources.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
