"""New figure/case-study materials only; never edits manuscript text."""
from pathlib import Path
import shutil,json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'writing_materials'
for folder in ['figures','source_data']:(OUT/folder).mkdir(parents=True,exist_ok=True)
for source in (ROOT/'visualizations/figures').iterdir():
 if source.suffix in ['.png','.svg']:shutil.copy2(source,OUT/'figures'/source.name)
for source in (ROOT/'visualizations/source_data').glob('*.csv'):shutil.copy2(source,OUT/'source_data'/source.name)
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],'font.size':10,
                     'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','legend.frameon':False})
def save(fig,name):
 fig.savefig(OUT/'figures'/(name+'.png'),dpi=450,bbox_inches='tight');fig.savefig(OUT/'figures'/(name+'.svg'),bbox_inches='tight');plt.close(fig)
aug=pd.read_csv(ROOT/'results/reference/model1/latest_month_2026_08/carrier_predictions_and_trends.csv')
data=aug[['carrier','july_actual_mishandled','target_mishandled_bags','anomaly_count_ml','target_rate_per_1000','rate_ml','actual_count_change_percent','predicted_count_change_percent']].copy()
data=data.rename(columns={'target_mishandled_bags':'aug_actual_count','anomaly_count_ml':'aug_predicted_count','target_rate_per_1000':'aug_actual_rate','rate_ml':'aug_predicted_rate'})
data['count_relative_error_percent']=100*(data.aug_predicted_count/data.aug_actual_count-1)
data.to_csv(OUT/'source_data/model1_august_cases_all.csv',index=False)
selected=data[data.carrier.isin(['UA','AA','G4'])].copy();selected.to_csv(OUT/'source_data/model1_selected_cases.csv',index=False)
g=data.sort_values('aug_actual_rate');x=np.arange(len(g));fig,ax=plt.subplots(figsize=(8.5,4.3),layout='constrained')
ax.bar(x-.18,g.aug_actual_rate,.36,color='#A9B7C0',label='Observed')
ax.bar(x+.18,g.aug_predicted_rate,.36,color='#537A99',label='Forecast')
ax.set_xticks(x,g.carrier);ax.set(ylabel='Mishandled bags per 1,000 enplaned bags',xlabel='BTS operating carrier',title='August 2026: observed and forecast baggage-mishandling rates')
ax.legend();ax.grid(axis='y',alpha=.16);ax.set_axisbelow(True);save(fig,'06_august_carrier_comparison')

source=ROOT/'results/reference/insurance_design_history/optimized_menu_v2'
profits=pd.read_csv(source/'selected_operator_results.csv')
base=profits[(profits.carrier=='AA')&(profits.scenario=='beta1.0_base')].iloc[0]
costs=base.revenue-base.claims-base.expected_profit-50000
waterfall=pd.DataFrame([{'component':'Premium revenue','amount_usd':base.revenue},
 {'component':'Expected payouts','amount_usd':-base.claims},
 {'component':'Variable expenses','amount_usd':-costs},
 {'component':'System cost','amount_usd':-50000},
 {'component':'Expected profit','amount_usd':base.expected_profit}])
assert np.isclose(waterfall.amount_usd.iloc[:4].sum(),waterfall.amount_usd.iloc[4])
waterfall.to_csv(OUT/'source_data/model2_AA_base_cost_breakdown.csv',index=False)
fig,ax=plt.subplots(figsize=(8.5,4.3),layout='constrained')
running=0
for i,row in waterfall.iterrows():
 value=row.amount_usd/1000
 if i in [0,4]:bottom=0;height=value
 else:bottom=running+value;height=-value
 ax.bar(i,height,bottom=bottom,width=.64,color='#537A99' if i in [0,4] else '#C5A68C')
 ax.text(i,bottom+height+18,f'{abs(value):,.1f}',ha='center',va='bottom',fontsize=9)
 if i<4:running+=value
ax.set_xticks(range(5),['Premium\nrevenue','Expected\npayouts','Variable\nexpenses','System\ncost','Expected\nprofit'])
ax.set(ylabel='Thousand USD',title='M0775 product scenario: AA-scale cost and profit breakdown')
ax.set_ylim(0,base.revenue/1000*1.15);ax.grid(axis='y',alpha=.16);ax.set_axisbelow(True);save(fig,'07_product_cost_profit_breakdown')

extra=pd.read_csv(source/'selected_extra_sensitivity.csv')
financial=extra[(extra.carrier=='AA')&extra.reference_total_purchase_probability.isin([.02,.1])&extra.system_cost.isin([50000,200000])].copy()
financial.to_csv(OUT/'source_data/model2_AA_cost_purchase_cases.csv',index=False)
shutil.copy2(ROOT/'results/baseline_comparison_2026_01_08.csv',OUT/'source_data/baseline_comparison_2026_01_08.csv')
shutil.copy2(source/'selected_menu.csv',OUT/'source_data/adopted_menu.csv')
catalog=[
 {'id':'P1','use':'Introduction','case':'Thai Airways Auto Reflight','facts':'SITA reports a three-minute task reduced to one second per bag across nine airports.','source':'https://www.sita.aero/about-us/pressroom/news-releases/tech-drove-down-mishandled-bag-rates-by-23-in-2025-but-mishandling-still-costs-the-industry-%246.3-billion-a-year','source_location':'29 June 2026 release, real-world results paragraph'},
 {'id':'P2','use':'Background or Discussion','case':'Air France tracking process example','facts':'LYS-CDG-ATL example explains baggage/ULD scanning, container inheritance and exchange of tracking messages.','source':'https://www.iata.org/contentassets/d22bcfe86aa54f22a2919b0e4224f68f/reso753-implementation-guide---2023_issue-4.0.pdf','source_location':'Section14.1, printed pages49-51, PDF pages54-56'},
 {'id':'P3','use':'Discussion','case':'Etihad offline baggage scanning','facts':'The guide describes centralizing loaded-bag inventories when some airports cannot send BRS messages; offline records have limited timestamps.','source':'https://www.iata.org/contentassets/d22bcfe86aa54f22a2919b0e4224f68f/reso753-implementation-guide---2023_issue-4.0.pdf','source_location':'Section14.2, printed pages52-53, PDF pages57-58'}]
pd.DataFrame(catalog).to_csv(OUT/'source_data/public_cases_and_sources.csv',index=False)
notes=['# 写作素材索引','', '本目录只提供素材，没有改动已有正文。所有图均有PNG和SVG；对应数据在source_data。',
 '', '## 可视化素材', '',
 '| 图号 | 内容 | 建议用途 |','|---|---|---|',
 '| 01 | 逐轮loss | 预测模型训练表现 |','| 02 | 逐轮MAE / RMSE | 预测模型表现 |',
 '| 03 | 逐月回测及基准 | 跨月比较 |','| 04 | 套餐帕累托前沿 | 方案选择 |',
 '| 05 | 六情景利润与利润率 | 财务结果 |','| 06 | 8月12家航司实际与预测异常率 | 预测案例 |',
 '| 07 | AA规模的收入、赔付、费用和利润分解 | 保险案例 |',
 '', '图01、02由保存模型的树前缀恢复；8月是测试诊断。图04、05、07是对应参数下的计算结果。原始图表说明随包提供。',
 '', '## 本项目可以写的case', '',
 '| 案例 | 关键数字 | 可以讨论什么 |','|---|---|---|']
for code in ['UA','AA','G4']:
 r=data[data.carrier==code].iloc[0]
 topic={'UA':'高异常率航司的数量预测较接近实际','AA':'预测到下降趋势，但下降幅度估计不足','G4':'低数量案例中，异常数量的升降方向预测有偏差'}[code]
 notes.append(f'| {code} 2026年8月 | 预测异常{r.aug_predicted_count:,.0f}件；实际{r.aug_actual_count:,.0f}件；相对误差{r.count_relative_error_percent:+.2f}% | {topic} |')
notes+=['', '这些案例来自同一批BTS回测，直接配图06和model1_selected_cases.csv即可。',
 '', '**保险case：AA规模、采用菜单M0775。**',
 f'基准情景的保费收入为US${base.revenue:,.2f}，预计赔付US${base.claims:,.2f}，变动费用US${costs:,.2f}，系统成本US$50,000，预计利润US${base.expected_profit:,.2f}，利润率{100*base.margin:.2f}%。配图07可展示收益如何覆盖成本。',
 '再比较model2_AA_cost_purchase_cases.csv中的购买率和系统成本变化，适合讨论产品在什么条件下能够回本。',
 '', '## 可引用的公开case','']
for r in catalog:notes.append(f"- **{r['case']}**：{r['facts']} 来源：[原文]({r['source']})；位置：{r['source_location']}。")
notes+=['','公开案例用于背景或讨论；它们与本项目自己的预测/收益案例分开引用。可使用来源链接，不需要把原报告整篇插入作业。']
(OUT/'素材索引.md').write_text('\n'.join(notes)+'\n',encoding='utf-8')
shutil.copy2(ROOT/'visualizations/README_图表说明.md',OUT/'原有曲线说明.md')
(OUT/'provenance.json').write_text(json.dumps({'new_figures':2,'existing_figures_copied':5,'manuscript_edited':False,'currency':'USD','empirical_cases':'BTS August2026 backtest','financial_cases':'M0775 scenario calculations','public_source_dates_checked':'2026-11-07'},indent=2),encoding='utf-8')
print(selected.to_string(index=False));print('Prepared seven figure pairs, case tables and source index.')
