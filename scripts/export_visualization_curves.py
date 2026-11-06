"""Recover diagnostics from frozen tree prefixes; no fitting or model selection.
Exports PNG/SVG and CSV only. August curves are held-out test diagnostics.
"""
from pathlib import Path
import sys,json,hashlib
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from baggage_project.risk import FrozenRiskPredictor
OUT=ROOT/'visualizations';OUT.mkdir(exist_ok=True)
PLOT=OUT/'figures';PLOT.mkdir(exist_ok=True)
SOURCE=OUT/'source_data';SOURCE.mkdir(exist_ok=True)
contract={'conclusion':'Show convergence and held-out behavior of the frozen August model, plus already computed product trade-offs.',
 'backend':'Python matplotlib','archetype':'quantitative grid','exports':['PNG 450dpi','editable-text SVG','CSV'],
 'model1_provenance':'recovered by predicting with prefixes of the saved 200-tree GBDT; original fit did not save per-round evaluation history',
 'model1_sets':'training targets before Aug2026; Aug2026 held-out diagnostic. No validation set, early stopping or test-based re-selection added.',
 'model2_provenance':'finite-grid Pareto/scenario outputs; not a trained loss curve',
 'fixed_iteration':200,'no_fitting':True,'no_new_efficacy_claim':True}
(OUT/'provenance.json').write_text(json.dumps(contract,indent=2),encoding='utf-8')
frozen=FrozenRiskPredictor(ROOT/'models');s=frozen.schema
d=pd.read_csv(ROOT/'data/processed/lag1/all_real.csv')
train=d[d.target_month<'2026-08-01'].copy();test=d[d.target_month=='2026-08-01'].copy()
def matrix(frame):
 ids=frame.airline_id.to_numpy();cat=(ids[:,None]==np.asarray(s['categories'])[None,:]).astype(float)
 x=(frame[s['numeric_columns']].to_numpy()-np.asarray(s['mean']))/np.asarray(s['scale'])
 return np.concatenate([cat,x],axis=1)
sets={'train':train,'test_august':test};Xs={name:matrix(frame) for name,frame in sets.items()}
assert frozen.booster.num_trees()==200 and len(test)==12
rows=[]
for iteration in range(201):
 for name,frame in sets.items():
  raw=np.full(len(frame),s['base_score']) if iteration==0 else frozen.booster.predict(Xs[name],num_iteration=iteration,num_threads=2)+s['base_score']
  y=frame.target_rate_per_1000.to_numpy();err=raw-y;deployed=np.maximum(raw,0);e=deployed-y;w=frame.target_enplaned_bags.to_numpy()
  c=np.where(err<0,2/3,4/3)
  rows.append({'iteration':iteration,'set':name,'n':len(frame),'asymmetric_objective_mean_raw':float(np.mean(.5*c*err**2)),
               'MSE_raw':float(np.mean(err**2)),'MAE_deployed':float(abs(e).mean()),'RMSE_deployed':float(np.sqrt((e**2).mean())),
               'weighted_MAE_deployed':float(np.average(abs(e),weights=w)),
               'weighted_RMSE_deployed':float(np.sqrt(np.average(e**2,weights=w))),
               'weighted_bias_deployed':float(np.average(e,weights=w))})
curves=pd.DataFrame(rows);curves.to_csv(SOURCE/'model1_tree_prefix_metrics.csv',index=False)
last=curves[(curves.iteration==200)&(curves['set']=='test_august')].iloc[0]
assert np.isclose(last.weighted_MAE_deployed,.3476354269167175,atol=1e-8)
assert np.allclose(np.maximum(frozen.booster.predict(Xs['test_august'],num_threads=2)+s['base_score'],0),frozen.predict(test),atol=1e-8)

plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],'font.size':10,
                     'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.8,
                     'legend.frameon':False,'svg.fonttype':'none','axes.titlesize':11,'figure.facecolor':'white'})
colors={'train':'#537A99','test_august':'#C67443'}
def export(fig,name):
 fig.savefig(PLOT/(name+'.png'),dpi=450,bbox_inches='tight')
 fig.savefig(PLOT/(name+'.svg'),bbox_inches='tight');plt.close(fig)
fig,ax=plt.subplots(figsize=(7,4.2),layout='constrained')
for name,label in [('train','Training'),('test_august','August test (diagnostic)')]:
 g=curves[curves['set']==name];ax.plot(g.iteration,g.asymmetric_objective_mean_raw,label=label,color=colors[name],lw=1.8)
ax.set(xlabel='Boosting iterations (tree prefix)',ylabel='Mean asymmetric squared loss',title='Frozen risk model: reconstructed loss history')
ax.legend();ax.grid(alpha=.16);export(fig,'01_model1_loss')
fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
for ax,key,title in zip(axes,['weighted_MAE_deployed','weighted_RMSE_deployed'],['Weighted MAE','Weighted RMSE']):
 for name,label in [('train','Training'),('test_august','August test')]:
  g=curves[curves['set']==name];ax.plot(g.iteration,g[key],color=colors[name],label=label,lw=1.7)
 ax.set(xlabel='Boosting iterations',ylabel='Bags per 1,000 enplaned bags',title=title);ax.grid(alpha=.16)
axes[0].legend();export(fig,'02_model1_metrics')

pred=pd.read_csv(ROOT/'results/reference/model1/predictions.csv');pred=pred[pred.year==2026]
monthly=[]
for month,g in pred.groupby('target_month'):
 for method in ['ml','symmetric','simple']:
  e=g['rate_'+method]-g.target_rate_per_1000
  monthly.append({'month':month,'method':method,'n':len(g),'weighted_MAE':np.average(abs(e),weights=g.target_enplaned_bags)})
m=pd.DataFrame(monthly);m.to_csv(SOURCE/'model1_rolling_monthly_metrics.csv',index=False)
fig,ax=plt.subplots(figsize=(7,4.2),layout='constrained')
for name,label,color in [('ml','Asymmetric risk model','#537A99'),('symmetric','Symmetric loss','#769F75'),('simple','Simple blend','#A19A90')]:
 g=m[m.method==name];ax.plot(pd.to_datetime(g.month),g.weighted_MAE,label=label,marker='o',ms=4,lw=1.5,color=color)
import matplotlib.dates as mdates
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'));ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.set(ylabel='Weighted MAE (bags per 1,000)',xlabel='2026 target month',title='Rolling one-step backtests (separate monthly fits)');ax.legend();ax.grid(alpha=.16)
export(fig,'03_model1_monthly_backtests')

source=ROOT/'results/reference/insurance_design_history/optimized_menu_v2'
designs=pd.read_csv(source/'all_designs.csv');front=pd.read_csv(source/'pareto_frontier.csv');sc=pd.read_csv(source/'selected_scenarios.csv')
designs.to_csv(SOURCE/'model2_designs.csv',index=False);front.to_csv(SOURCE/'model2_pareto_frontier.csv',index=False);sc.to_csv(SOURCE/'model2_selected_scenarios.csv',index=False)
fig,ax=plt.subplots(figsize=(7,4.4),layout='constrained')
g=designs[designs.feasible];ax.scatter(g.robust_expected_profit/1e6,g.robust_margin*100,s=19,color='#B7C1C7',label='Feasible menus (98)',alpha=.65)
g=front.sort_values('robust_expected_profit');ax.plot(g.robust_expected_profit/1e6,g.robust_margin*100,'o-',ms=5,lw=1.4,color='#537A99',label='Finite-grid frontier (7)')
g=front[front.design_id=='M0775'];ax.scatter(g.robust_expected_profit/1e6,g.robust_margin*100,s=110,marker='*',color='#C67443',label='Adopted M0775',zorder=4)
ax.set(xlabel='Worst-scenario expected profit (million USD)',ylabel='Worst-scenario expected margin (%)',title='Product model: conditional Pareto frontier')
ax.legend();ax.grid(alpha=.16);export(fig,'04_model2_pareto')
sc['beta']=sc.scenario.str.extract(r'beta([\d.]+)').astype(float);sc['risk']=np.where(sc.scenario.str.endswith('_higher'),'higher','base')
fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
for risk,label,color in [('base','Baseline risk','#537A99'),('higher','Higher risk','#C67443')]:
 g=sc[sc.risk==risk].sort_values('beta')
 axes[0].plot(g.beta,g.total_expected_profit/1e6,'o-',color=color,label=label)
 axes[1].plot(g.beta,g.pooled_margin*100,'o-',color=color,label=label)
for ax in axes:ax.set_xlabel('Assumed price sensitivity (beta)');ax.set_xticks([1,2,3]);ax.grid(alpha=.16)
axes[0].set(ylabel='Expected profit (million USD)',title='M0775 scenario profit');axes[1].set(ylabel='Expected margin (%)',title='M0775 scenario margin')
axes[0].legend();export(fig,'05_model2_scenario_returns')
summary={'iteration_count':200,'training_cases':len(train),'test_cases':len(test),
         'last_iteration_test_metrics':last.to_dict(),'recovery_check':'PASS: final predictions reproduce frozen model',
         'plots':5,'no_test_based_iteration_selection':True,'model_sha256':hashlib.sha256((ROOT/'models/risk_2026_08_lightgbm.txt').read_bytes()).hexdigest(),
         'matplotlib_version':matplotlib.__version__}
(OUT/'checks.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
print(json.dumps(summary,indent=2))
