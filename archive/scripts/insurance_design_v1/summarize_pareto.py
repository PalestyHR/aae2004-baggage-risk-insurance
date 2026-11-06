"""Report conditional grid frontiers; no claim of global/empirical optimality."""
from pathlib import Path
import csv,json
HERE=Path(__file__).parent;OUT=HERE/'profit_tests'
rows=list(csv.DictReader((OUT/'price_benefit_candidates.csv').open()))
aggregate=[]
for beta in [.5,1.,2.]:
 groups={}
 for r in rows:
  if float(r['beta'])!=beta:continue
  key=(float(r['price_factor']),float(r['benefit_factor']))
  g=groups.setdefault(key,{'profit':0.,'revenue':0.,'sales':0.,'positive_operators':0,'cases':0})
  profit=float(r['expected_profit']);margin=float(r['expected_profit_margin'])
  # In this experiment all three prices and benefits use common scaling and
  # share common sensitivities. Conditional tier mix remains 50/30/20.
  revenue=float(r['expected_sales'])*5.5*key[0]
  assert abs(profit-revenue*margin)<1e-6
  g['profit']+=profit;g['revenue']+=revenue;g['sales']+=float(r['expected_sales'])
  g['positive_operators']+=int(profit>0);g['cases']+=1
 for key,g in groups.items():
  g['margin']=g['profit']/g['revenue']
 for (pf,bf),g in groups.items():
  dominated=any(h['profit']>=g['profit'] and h['margin']>=g['margin'] and (h['profit']>g['profit'] or h['margin']>g['margin']) for h in groups.values())
  aggregate.append({'beta':beta,'price_factor':pf,'benefit_factor':bf,'pooled_expected_profit_usd':g['profit'],'pooled_margin':g['margin'],'operators':g['cases'],'profitable_operators':g['positive_operators'],'nondominated_on_pooled_objectives':not dominated,'all_operators_profitable':g['positive_operators']==g['cases']})
with (OUT/'pooled_conditional_frontiers.csv').open('w',newline='',encoding='utf-8-sig') as f:
 w=csv.DictWriter(f,fieldnames=aggregate[0]);w.writeheader();w.writerows(aggregate)
menus=[]
for label,bf in [('A_margin_focus',1.),('B_compromise_candidate',1.25),('C_profit_focus',1.5)]:
 for name,p,cd,cl in [('Basic',3,50,300),('Standard',6,100,600),('Premium',11,200,1200)]:
  menus.append({'candidate':label,'plan':name,'currency':'USD','premium':2*p,'delay_benefit':bf*cd,'total_loss_benefit':bf*cl,'condition':'beta=1 baseline assumptions; grid frontier only; not approved quote'})
with (OUT/'pareto_candidate_menus.csv').open('w',newline='',encoding='utf-8-sig') as f:
 w=csv.DictWriter(f,fieldnames=menus[0]);w.writeheader();w.writerows(menus)
print(json.dumps([r for r in aggregate if r['beta']==1 and r['nondominated_on_pooled_objectives']],indent=2))
