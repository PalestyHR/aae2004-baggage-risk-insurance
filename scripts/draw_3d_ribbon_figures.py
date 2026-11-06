"""Reference-inspired 3D filled-line figures of existing numerical records."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.patches import Patch
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'visualizations/ribbons_3d'
for f in ['figures','source_data']:(OUT/f).mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],'font.size':9,
 'svg.fonttype':'none','legend.frameon':False,'axes.labelcolor':'#30343C','text.color':'#272B33',
 'figure.facecolor':'white','savefig.facecolor':'white'})
PALETTE=['#7B3864','#5E4FA2','#5D9FDB','#77BEC9','#66C2A5','#B4DE8D']
def ribbon(name,title,x,series,xlabels,xlabel,zlabel,shortlabels=None,azim=-53):
 fig=plt.figure(figsize=(11,8.2));ax=fig.add_axes([.02,.03,.83,.86],projection='3d')
 labels=list(series);ys=np.arange(len(labels));verts=[]
 for label in labels:
  vals=np.asarray(series[label],dtype=float);assert len(vals)==len(x) and np.isfinite(vals).all() and (vals>=0).all()
  verts.append([(x[0],0),*zip(x,vals),(x[-1],0)])
 fills=PolyCollection(verts,facecolors=PALETTE[:len(labels)],edgecolors='none',alpha=.25)
 ax.add_collection3d(fills,zs=ys,zdir='y')
 for j,label in enumerate(labels):
  vals=np.asarray(series[label]);col=PALETTE[j]
  ax.plot(x,np.full(len(x),j),vals,color=col,lw=1.45,zorder=10-j)
  ax.plot([x[0],x[0]],[j,j],[0,vals[0]],color=col,lw=.7,alpha=.8)
  ax.plot([x[-1],x[-1]],[j,j],[0,vals[-1]],color=col,lw=.7,alpha=.8)
 ax.set_xlim(float(x[0]),float(x[-1]));ax.set_ylim(-.25,len(labels)-.75)
 ax.set_zlim(0,max(float(np.max(v)) for v in series.values())*1.12)
 ax.set_xticks(x if len(x)<=12 else np.linspace(x[0],x[-1],5))
 if xlabels is not None:ax.set_xticklabels(xlabels,rotation=25,ha='right',fontsize=8)
 ax.set_yticks(ys);ax.set_yticklabels(shortlabels or labels,fontsize=8)
 ax.set_xlabel(xlabel,labelpad=16);ax.set_zlabel(zlabel,labelpad=10)
 ax.tick_params(axis='y',pad=1);ax.tick_params(axis='z',labelsize=8)
 ax.view_init(elev=25,azim=azim);ax.set_box_aspect((1.8,1.2,1.05))
 for axis in [ax.xaxis,ax.yaxis,ax.zaxis]:
  axis.pane.set_facecolor((.96,.965,.975,1));axis.pane.set_edgecolor('#E2E5EB')
  axis._axinfo['grid']['color']=(.80,.82,.86,.62)
 fig.suptitle(title,x=.46,y=.965,fontfamily='Times New Roman',fontstyle='italic',fontweight='bold',fontsize=18)
 handles=[Patch(facecolor=PALETTE[i],edgecolor=PALETTE[i],alpha=.40,label=label) for i,label in enumerate(labels)]
 fig.legend(handles=handles,loc='upper left',bbox_to_anchor=(.79,.92),fontsize=8.5)
 fig.savefig(OUT/'figures'/(name+'.png'),dpi=450,bbox_inches='tight')
 fig.savefig(OUT/'figures'/(name+'.svg'),bbox_inches='tight');plt.close(fig)

monthly=pd.read_csv(ROOT/'visualizations/style_v2/source_data/monthly_all_models.csv')
methods={'seasonal_naive':'Seasonal naive','blend50':'50/50 blend','HistGBR':'Histogram boosting','XGBoost':'XGBoost','LightGBM':'Symmetric LightGBM','adopted':'Adopted model'}
series={label:monthly[monthly.model_id==code].sort_values('month').weighted_MAE.to_numpy() for code,label in methods.items()}
ribbon('01_monthly_model_errors','Monthly prediction error',np.arange(8),series,['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug'],'2026 target month','Weighted MAE (bags / 1,000)',shortlabels=['Seasonal','Blend','HGB','XGB','Sym. LGB','Adopted'])
monthly[monthly.model_id.isin(methods)].to_csv(OUT/'source_data/monthly_error_records.csv',index=False)

pred=pd.read_csv(ROOT/'results/baseline_comparison_case_predictions.csv');aug=pred[pred.target_month=='2026-08-01'].copy()
order=aug[aug.model_id=='adopted'].sort_values('target_rate_per_1000').carrier.tolist()
series={}
for code,label in methods.items():
 g=aug[aug.model_id==code].set_index('carrier').loc[order]
 series[label]=abs(g.predicted_rate-g.target_rate_per_1000).to_numpy()
ribbon('02_august_carrier_errors','August error across operating carriers',np.arange(12),series,order,'BTS operating carrier','Absolute error (bags / 1,000)',shortlabels=['Seasonal','Blend','HGB','XGB','Sym. LGB','Adopted'])
aug[aug.model_id.isin(methods)].to_csv(OUT/'source_data/august_error_records.csv',index=False)

curves=pd.read_csv(ROOT/'visualizations/source_data/model1_tree_prefix_metrics.csv')
series={'Training weighted MAE':curves[curves['set']=='train'].weighted_MAE_deployed.to_numpy(),
        'August test weighted MAE':curves[curves['set']=='test_august'].weighted_MAE_deployed.to_numpy(),
        'August test weighted RMSE':curves[curves['set']=='test_august'].weighted_RMSE_deployed.to_numpy()}
ribbon('03_boosting_metric_layers','Frozen model performance by iteration',np.arange(201),series,None,'Boosting iterations','Error (bags / 1,000)',shortlabels=['Train MAE','Test MAE','Test RMSE'])
curves.to_csv(OUT/'source_data/frozen_tree_prefix_metrics.csv',index=False)

profits=pd.read_csv(ROOT/'results/reference/insurance_design_history/optimized_menu_v2/selected_operator_results.csv')
carrier_order=['WN','DL','AA','UA','OO','AS','F9','B6','G4','YX','MQ','OH']
ss={'beta1.0_base':'beta 1 · baseline risk','beta2.0_base':'beta 2 · baseline risk','beta3.0_base':'beta 3 · baseline risk',
    'beta1.0_higher':'beta 1 · higher risk','beta2.0_higher':'beta 2 · higher risk','beta3.0_higher':'beta 3 · higher risk'}
series={label:profits[profits.scenario==code].set_index('carrier').loc[carrier_order].expected_profit.to_numpy()/1e6 for code,label in ss.items()}
ribbon('04_insurance_scenario_profits','M0775 net contribution after system costs',np.arange(12),series,carrier_order,'BTS scale case','Expected profit (million USD)',shortlabels=['B1 base','B2 base','B3 base','B1 high','B2 high','B3 high'])
profits.to_csv(OUT/'source_data/m0775_operator_scenario_profits.csv',index=False)
(OUT/'图表说明.md').write_text('''# 三维分层折线素材

仿照参考图，使用紫、蓝、青绿渐变色、半透明填充和三维错层排列。每张均有450dpi PNG、可编辑SVG及对应CSV。

| 图 | 内容 | 数值方向 |
|---|---|---|
| 01 | 六种方法的2026年1–8月逐月误差 | 越低越好 |
| 02 | 六种方法在8月12家航司上的绝对误差 | 越低越好 |
| 03 | 冻结模型逐轮Train MAE、Test MAE、Test RMSE | 越低越好 |
| 04 | 采用套餐M0775在六种情景、12个规模案例的预计利润 | 越高越好 |

横轴为月份、迭代次数或航司类别；纵深仅用于分层放置不同系列。航司之间的连线用于展示，不是时间顺序。所有折线只连接已有点，未补随机波动或平滑数据。
图03由保存的树前缀恢复，August test是测试诊断。图04沿用当前保险情景计算。源数据保留完整数值，图04绘图时将美元换算成百万美元。
''',encoding='utf-8')
(OUT/'checks.json').write_text(json.dumps({'figures':4,'data_sources':'existing frozen model and adopted-menu records','new_training':False,'manuscript_edits':False,'fill_baseline':0,'point_interpolation':'straight lines only','output_formats':['PNG450dpi','SVG','CSV']},indent=2),encoding='utf-8')
print('Exported four 3D ribbon figures with source records.')
