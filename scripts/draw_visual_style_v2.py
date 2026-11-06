"""Diverse reference-inspired styling of existing results; no model/text edits."""
from pathlib import Path
import json,math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'visualizations/style_v2'
for x in ['figures','source_data']:(OUT/x).mkdir(parents=True,exist_ok=True)
metrics=pd.read_csv(ROOT/'results/baseline_comparison_2026_01_08.csv').set_index('model_id')
pred=pd.read_csv(ROOT/'results/baseline_comparison_case_predictions.csv')
ids=metrics.index.tolist()
names={'seasonal_naive':'Seasonal naive','blend50':'50/50 blend','ElasticNet':'ElasticNet','Poisson_GLM':'Poisson GLM','RandomForest':'Random forest','ExtraTrees':'Extra trees','HistGBR':'Hist. boosting','LightGBM':'Sym. LightGBM','XGBoost':'XGBoost','adopted':'Adopted model'}
colors={'adopted':'#DD2DAA','HistGBR':'#0453A0','XGBoost':'#7B3864','LightGBM':'#5D9FDB','RandomForest':'#66C2A5','ExtraTrees':'#ABDDA4','ElasticNet':'#F46D43','Poisson_GLM':'#FDAE61','seasonal_naive':'#9198A3','blend50':'#5E4FA2'}
markers=dict(zip(ids,['o','s','D','v','^','P','h','<','>','*']))
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],'font.size':10,
 'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.8,'legend.frameon':False,
 'svg.fonttype':'none','axes.titleweight':'semibold','axes.labelcolor':'#303540','text.color':'#252932',
 'figure.facecolor':'white','savefig.facecolor':'white'})
def save(fig,name):
 fig.savefig(OUT/'figures'/(name+'.png'),dpi=450,bbox_inches='tight')
 fig.savefig(OUT/'figures'/(name+'.svg'),bbox_inches='tight');plt.close(fig)
def panel(ax,label):ax.text(-.10,1.08,label,transform=ax.transAxes,fontweight='bold',fontsize=13,va='bottom')
keys=['rate_weighted_MAE','rate_weighted_RMSE','rate_MAE','rate_RMSE','count_WAPE_percent_same_forecast_exposure']
labels=['Weighted\nMAE','Weighted\nRMSE','MAE','RMSE','Count\nWAPE']
score=100*metrics[keys].min()/metrics[keys]
score.to_csv(OUT/'source_data/radar_metric_scores.csv');metrics.to_csv(OUT/'source_data/metrics_raw.csv')
angles=np.linspace(0,2*np.pi,len(keys),endpoint=False);closed=np.r_[angles,angles[0]]
fig,axs=plt.subplots(1,2,figsize=(12,7.2),subplot_kw={'projection':'polar'})
groups=[['seasonal_naive','blend50','ElasticNet','Poisson_GLM','adopted'],['RandomForest','ExtraTrees','HistGBR','LightGBM','XGBoost','adopted']]
for i,(ax,group,title) in enumerate(zip(axs,groups,['Statistical and linear baselines','Tree-based alternatives'])):
 ax.set_theta_offset(np.pi/2);ax.set_theta_direction(-1);ax.set_ylim(0,106)
 ax.set_xticks(angles,labels);ax.tick_params(axis='x',pad=13);ax.set_yticks([20,40,60,80,100]);ax.set_yticklabels(['20','40','60','80','100'],fontsize=7,color='#7B8290');ax.set_rlabel_position(12)
 ax.spines['polar'].set_visible(False);ax.grid(color='#DFE3E8',lw=.65)
 ax.plot(np.linspace(0,2*np.pi,361),np.full(361,100),color='#555E6A',lw=.9)
 for name in group:
  y=score.loc[name].to_numpy();y=np.r_[y,y[0]]
  ax.plot(closed,y,color=colors[name],lw=2.5 if name=='adopted' else 1.25,marker=markers[name],ms=6 if name=='adopted' else 3.8,label=names[name],zorder=5 if name=='adopted' else 2)
  if name=='adopted':ax.fill(closed,y,color=colors[name],alpha=.07)
 fig.text(.265 if i==0 else .735,.86,title,ha='center',fontsize=11,fontweight='bold')
 ax.legend(loc='upper center',bbox_to_anchor=(.5,-.15),ncol=2,fontsize=8.5)
 ax.text(-.15,1.12,chr(97+i),transform=ax.transAxes,fontweight='bold',fontsize=14)
fig.subplots_adjust(left=.07,right=.93,top=.75,bottom=.23,wspace=.37)
fig.suptitle('Model performance profiles',y=.99,fontsize=16,fontweight='bold')
fig.text(.5,.94,'Outer ring = best observed value per metric (100); larger is better',ha='center',fontsize=10,color='#616975')
save(fig,'01_model_comparison_radar')

# Outer reference is exact agreement with observed data, not raw risk or accuracy.
aug=pred[pred.target_month=='2026-08-01'];carriers=aug[aug.model_id=='adopted'].sort_values('target_rate_per_1000').carrier.tolist()
ac=np.linspace(0,2*np.pi,len(carriers),endpoint=False);aclose=np.r_[ac,ac[0]]
agreement=[];fig=plt.figure(figsize=(8,7));ax=fig.add_subplot(111,projection='polar')
ax.set_theta_offset(np.pi/2);ax.set_theta_direction(-1);ax.set_ylim(0,105)
ax.set_xticks(ac,carriers);ax.set_yticks([25,50,75,100]);ax.set_yticklabels(['25','50','75','100'],color='#808895',fontsize=8);ax.spines['polar'].set_visible(False);ax.grid(color='#DFE3E8',lw=.65)
ax.plot(np.linspace(0,2*np.pi,361),np.full(361,100),ls='--',color='#333942',lw=1.1,label='Observed reference (100)')
for name in ['blend50','HistGBR','XGBoost','adopted']:
 g=aug[aug.model_id==name].set_index('carrier').loc[carriers]
 y=100*np.maximum(0,1-abs(g.predicted_rate/g.target_rate_per_1000-1)).to_numpy()
 for carrier,v in zip(carriers,y):agreement.append({'model_id':name,'carrier':carrier,'relative_agreement_score':v})
 ax.plot(aclose,np.r_[y,y[0]],color=colors[name],lw=2.3 if name=='adopted' else 1.3,marker=markers[name],ms=5,label=names[name])
 if name=='adopted':ax.fill(aclose,np.r_[y,y[0]],alpha=.06,color=colors[name])
ax.tick_params(axis='x',pad=8);ax.set_title('August forecast agreement by carrier',pad=30,fontsize=14)
ax.legend(loc='upper center',bbox_to_anchor=(.5,-.10),ncol=2,fontsize=9)
fig.subplots_adjust(top=.84,bottom=.22);save(fig,'02_august_observed_reference_radar')
pd.DataFrame(agreement).to_csv(OUT/'source_data/august_radar_agreement.csv',index=False)

monthly=[]
for (month,name),g in pred.groupby(['target_month','model_id']):
 e=g.predicted_rate-g.target_rate_per_1000
 monthly.append({'month':month,'model_id':name,'n':len(g),'observed_count':g.target_mishandled_bags.sum(),'forecast_count':g.predicted_count_same_exposure.sum(),'weighted_MAE':np.average(abs(e),weights=g.target_enplaned_bags)})
monthly=pd.DataFrame(monthly);monthly.to_csv(OUT/'source_data/monthly_all_models.csv',index=False)
chosen=['seasonal_naive','blend50','HistGBR','XGBoost','LightGBM','adopted']
fig,axs=plt.subplots(2,1,figsize=(10.5,8),sharex=True)
truth=monthly[monthly.model_id=='adopted'];x=np.arange(8)
axs[0].plot(x,truth.observed_count/1000,'o-',color='#242B35',lw=2.6,ms=5,label='Observed')
for name in chosen:
 g=monthly[monthly.model_id==name].sort_values('month')
 for ax,key,scale in [(axs[0],'forecast_count',1000),(axs[1],'weighted_MAE',1)]:
  ax.plot(x,g[key]/scale,color=colors[name],marker=markers[name],lw=2.5 if name=='adopted' else 1.3,ms=5,label=names[name],alpha=1 if name=='adopted' else .83)
for ax in axs:ax.grid(alpha=.15);ax.set_axisbelow(True)
axs[0].set(ylabel='Abnormal bags (thousands)',title='Observed and forecast monthly totals');axs[1].set(ylabel='Weighted MAE (bags per 1,000)',title='Monthly risk-forecast errors',xlabel='2026 target month')
axs[1].set_xticks(x,['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug']);axs[0].legend(ncol=3,fontsize=8.5,loc='upper center',bbox_to_anchor=(.5,1.32))
panel(axs[0],'a');panel(axs[1],'b');fig.subplots_adjust(top=.83,left=.11,right=.97,bottom=.08,hspace=.35)
save(fig,'03_multi_model_monthly_lines')

order=metrics.sort_values('rate_weighted_MAE').index.tolist()
heat=monthly.pivot(index='model_id',columns='month',values='weighted_MAE').loc[order]
heat.to_csv(OUT/'source_data/monthly_error_heatmap.csv')
cmap=LinearSegmentedColormap.from_list('reference_pastel',['#FFF5D7','#D9ECA3','#8ACCB6','#56A0CC','#5E4FA2'])
fig,ax=plt.subplots(figsize=(10,6));im=ax.imshow(heat.to_numpy(),cmap=cmap,aspect='auto',vmin=0,vmax=heat.to_numpy().max())
ax.set_xticks(np.arange(8),['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug']);ax.set_yticks(np.arange(10),[names[n] for n in order]);ax.tick_params(length=0,pad=9)
for i in range(10):
 for j in range(8):
  v=heat.iloc[i,j];ax.text(j,i,f'{v:.3f}',ha='center',va='center',fontsize=9,color='white' if v>1 else '#27313A')
from matplotlib.patches import Rectangle
i=order.index('adopted');ax.add_patch(Rectangle((-.5,i-.5),8,1,fill=False,edgecolor=colors['adopted'],lw=2.2))
ax.get_yticklabels()[i].set_color(colors['adopted']);ax.get_yticklabels()[i].set_weight('bold')
ax.set_title('Monthly error landscape across all models',pad=16,fontsize=14)
cb=fig.colorbar(im,ax=ax,pad=.025,shrink=.8);cb.set_label('Weighted MAE (bags per 1,000)')
fig.subplots_adjust(left=.20,right=.94,bottom=.10,top=.87);save(fig,'04_monthly_error_heatmap')

error=[]
for name,g in pred.groupby('model_id'):
 for v in abs(g.predicted_rate-g.target_rate_per_1000):error.append({'model_id':name,'absolute_rate_error':v})
error=pd.DataFrame(error);error.to_csv(OUT/'source_data/case_absolute_errors.csv',index=False)
fig,ax=plt.subplots(figsize=(12,5));datasets=[error[error.model_id==n].absolute_rate_error.to_numpy() for n in order]
vi=ax.violinplot(datasets,positions=np.arange(10),widths=.78,showextrema=False)
for body,name in zip(vi['bodies'],order):body.set_facecolor(colors[name]);body.set_edgecolor(colors[name]);body.set_alpha(.24)
ax.boxplot(datasets,positions=np.arange(10),widths=.22,showfliers=False,patch_artist=True,
 boxprops={'facecolor':'white','edgecolor':'#59616B','linewidth':.8},medianprops={'color':'#252B33','linewidth':1.3},whiskerprops={'color':'#59616B','linewidth':.8},capprops={'color':'#59616B','linewidth':.8})
rng=np.random.default_rng(42)
for i,(name,values) in enumerate(zip(order,datasets)):
 ax.scatter(i+rng.uniform(-.18,.18,len(values)),values,s=9,color=colors[name],alpha=.35,zorder=3)
ax.set_xticks(np.arange(10),[names[n] for n in order],rotation=28,ha='right');ax.set(ylabel='Absolute rate error (bags per 1,000)',title='Error distributions over 101 carrier-month cases')
ax.grid(axis='y',alpha=.16);fig.subplots_adjust(left=.08,right=.98,top=.88,bottom=.26);save(fig,'05_error_violin_boxplot')

fig,ax=plt.subplots(figsize=(11,5))
for name in ids:
 ax.plot(np.arange(5),score.loc[name],color=colors[name],lw=2.8 if name=='adopted' else 1.15,marker=markers[name],ms=6 if name=='adopted' else 4,label=names[name],alpha=1 if name=='adopted' else .80)
ax.axhline(100,color='#555E6A',lw=.8,ls='--');ax.set_xticks(np.arange(5),['Weighted MAE','Weighted RMSE','MAE','RMSE','Count WAPE'])
ax.set(ylabel='Relative score (best per metric = 100)',title='Multi-metric performance profiles',ylim=(50,103));ax.grid(axis='y',alpha=.15)
ax.legend(loc='upper center',bbox_to_anchor=(.5,-.12),ncol=5,fontsize=8.5);fig.subplots_adjust(top=.87,bottom=.27,left=.09,right=.98);save(fig,'06_parallel_metric_profiles')

source=ROOT/'results/reference/insurance_design_history/optimized_menu_v2'
des=pd.read_csv(source/'all_designs.csv');front=pd.read_csv(source/'pareto_frontier.csv');sc=pd.read_csv(source/'selected_scenarios.csv')
fig,axs=plt.subplots(1,2,figsize=(12,4.8))
g=des[des.feasible];axs[0].scatter(g.robust_expected_profit/1e6,g.robust_margin*100,c=g.robust_margin,cmap='PuBu',s=30,alpha=.45,edgecolor='white',lw=.4)
g=front.sort_values('robust_expected_profit');axs[0].plot(g.robust_expected_profit/1e6,g.robust_margin*100,'o-',lw=1.5,color=colors['HistGBR'],ms=5)
g=front[front.design_id=='M0775'];axs[0].scatter(g.robust_expected_profit/1e6,g.robust_margin*100,marker='*',s=160,color=colors['adopted'],zorder=5)
axs[0].annotate('M0775',xy=(g.robust_expected_profit.iloc[0]/1e6,g.robust_margin.iloc[0]*100),xytext=(-45,28),textcoords='offset points',color=colors['adopted'],weight='bold',arrowprops={'arrowstyle':'-','color':colors['adopted']})
axs[0].set(xlabel='Worst-scenario profit (million USD)',ylabel='Worst-scenario margin (%)',title='Profit-margin trade-off')
for higher,label,color in [(False,'Baseline risk',colors['HistGBR']),(True,'Higher risk',colors['adopted'])]:
 g=sc[sc.scenario.str.endswith('_higher')==higher].copy();g['beta']=g.scenario.str.extract(r'beta([\d.]+)').astype(float);g=g.sort_values('beta')
 axs[1].plot(g.beta,g.pooled_margin*100,marker='o' if not higher else 'D',color=color,lw=2,label=label)
 for x,y in zip(g.beta,g.pooled_margin*100):axs[1].annotate(f'{y:.2f}%',(x,y),xytext=(0,9),textcoords='offset points',ha='center',fontsize=8.5,color=color)
axs[1].set(xlabel='Price sensitivity (beta)',ylabel='Expected margin (%)',title='Adopted menu across six scenarios',xticks=[1,2,3],ylim=(36,55));axs[1].legend(loc='center right',fontsize=9)
for ax in axs:ax.grid(alpha=.13);ax.set_axisbelow(True)
panel(axs[0],'a');panel(axs[1],'b');fig.subplots_adjust(left=.07,right=.98,bottom=.16,top=.84,wspace=.30);save(fig,'07_insurance_composite')

notes='''# 新版可视化素材

参考所给图片的蓝色、品红、紫色、青绿和暖橙配色，以不同线型和点形配合颜色。
所有图片为450dpi PNG及可编辑SVG，源数据在source_data。未改动正文和模型。

1. **01 雷达图**：10个模型分成两组。每个误差指标的分数=100×该指标最小误差/本模型误差，外圈100表示该指标的最佳值；不是分类准确率。raw指标见metrics_raw.csv。
2. **02 实际参考雷达图**：8月12家航司为轴；一致程度=100×max(0,1−|预测率/实际率−1|)。外圈为完全一致的实际参考；高估和低估均表现为向内偏离。
3. **03 多模型折线**：上图为实际及预测异常总量，下图为逐月加权MAE。每个月是单独训练、预测下一个月。
4. **04 误差热图**：10个模型×8个月，显示原始加权MAE；品红边框标出采用模型。
5. **05 小提琴与箱线图**：每模型101条case的绝对误差。点为case，箱体为四分位范围、中线为中位数，须线按1.5IQR；不是置信区间。此图不按行李量加权。
6. **06 多指标折线**：雷达图相同分数的另一种展现，可直接比较不同指标的权衡。
7. **07 保险组合图**：左图98个可行组合与7点前沿；右图M0775的六情景收益率。

除图02为8月外，预测模型对比使用2026年1–8月101条共同测试记录。性能曲线和数值均取自现有结果，不重新训练。保险图沿用现有六情景计算。
'''
(OUT/'图表说明.md').write_text(notes,encoding='utf-8')
(OUT/'provenance.json').write_text(json.dumps({'style_reference':'user provided colour and multi-panel examples','numerical_inputs_changed':False,'manuscript_edited':False,'new_figures':7,'normalization':'minimum metric / model metric *100; case agreement 100*max(0,1-absolute relative error)','palette':colors},indent=2),encoding='utf-8')
print('Exported seven diverse figures in PNG/SVG with source CSV and normalization definitions.')
