"""Recompute matched Jan-Aug2026 baseline metrics from saved predictions."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
REF=ROOT/'results/reference'
pipe=pd.read_csv(REF/'model1/predictions.csv');pipe=pipe[pipe.year==2026].copy()
families=pd.read_csv(REF/'integrated_model_comparison/predictions.csv');families=families[families.year==2026]
assert len(pipe)==101 and pipe.row_id.nunique()==101
actual=pipe.set_index('row_id')
labels={'seasonal_naive':'Seasonal naive (same month last year)',
        'blend50':'50/50 recent-seasonal blend','ElasticNet':'ElasticNet',
        'Poisson_GLM':'Poisson GLM (exposure-weighted fit)',
        'RandomForest':'Random Forest','ExtraTrees':'Extra Trees',
        'HistGBR':'Histogram Gradient Boosting','LightGBM':'LightGBM (symmetric squared loss)',
        'XGBoost':'XGBoost (squared loss)','adopted':'Adopted LightGBM (under:over = 1:2)'}
rows=[];records=[]
for name in labels:
 if name=='adopted': pred=actual.rate_ml.to_numpy()
 else:
  b=families[families.model==name].set_index('row_id')
  assert len(b)==101 and set(b.index)==set(actual.index)
  b=b.loc[actual.index]
  assert np.allclose(b.target_rate_per_1000,actual.target_rate_per_1000)
  assert np.allclose(b.target_enplaned_bags,actual.target_enplaned_bags)
  pred=b.prediction.to_numpy()
 y=actual.target_rate_per_1000.to_numpy();w=actual.target_enplaned_bags.to_numpy();e=pred-y
 count=pred*actual.forecast_baggage_volume.to_numpy()/1000
 cy=actual.target_mishandled_bags.to_numpy()
 row={'model_id':name,'method':labels[name],'n':len(y),'rate_weighted_MAE':float(np.average(abs(e),weights=w)),
      'rate_weighted_RMSE':float(np.sqrt(np.average(e**2,weights=w))),
      'rate_MAE':float(abs(e).mean()),'rate_RMSE':float(np.sqrt((e**2).mean())),
      'rate_weighted_bias':float(np.average(e,weights=w)),
      'count_WAPE_percent_same_forecast_exposure':float(100*abs(count-cy).sum()/cy.sum())}
 rows.append(row)
 out=actual[['target_month','carrier','target_rate_per_1000','target_mishandled_bags','target_enplaned_bags','forecast_baggage_volume']].copy()
 out['model_id']=name;out['predicted_rate']=pred;out['predicted_count_same_exposure']=count;records.append(out.reset_index())
r=pd.DataFrame(rows);r.to_csv(ROOT/'results/baseline_comparison_2026_01_08.csv',index=False)
pd.concat(records,ignore_index=True).to_csv(ROOT/'results/baseline_comparison_case_predictions.csv',index=False)
known=pd.read_csv(REF/'integrated_model_comparison/metrics.csv')
for row in rows:
 if row['model_id']=='adopted':continue
 ref=known[(known.year==2026)&(known.model==row['model_id'])].iloc[0]
 assert abs(row['rate_weighted_MAE']-ref.weighted_MAE)<1e-10
assert np.isclose(rows[-1]['rate_weighted_MAE'],.6375740849258186,atol=1e-10)
lines=['# Model 1 Baseline Comparison Table','',
 'Test period: January–August 2026. **101 matched carrier-month cases**. Monthly expanding-window refits use only earlier target months, assuming previous-month data are available. All metrics below are recomputed from saved predictions; no retraining or model selection was performed.',
 '', '## Main comparison','',
 '| Method | n | Rate weighted MAE ↓ | Rate weighted RMSE ↓ | Count WAPE (%) ↓ |',
 '|---|---:|---:|---:|---:|']
for x in rows:
 lines.append(f"| {x['method']} | {x['n']} | {x['rate_weighted_MAE']:.6f} | {x['rate_weighted_RMSE']:.6f} | {x['count_WAPE_percent_same_forecast_exposure']:.3f} |")
lines+=['','Rate metrics are in **mishandled bags per 1,000 enplaned bags**. Count WAPE uses the **same forecast baggage exposure for every method**, namely the 50/50 latest-month and same-month-last-year blend selected on 2025 validation. No actual target-month exposure is used to produce count predictions. Actual exposure weights rate evaluation only.',
 '', '## Additional rate diagnostics','',
 '| Method | Unweighted MAE ↓ | Unweighted RMSE ↓ | Weighted bias |',
 '|---|---:|---:|---:|']
for x in rows:lines.append(f"| {x['method']} | {x['rate_MAE']:.6f} | {x['rate_RMSE']:.6f} | {x['rate_weighted_bias']:+.6f} |")
lines+=['','Weighted bias = weighted mean(predicted − observed): positive is overprediction. Simple methods use their named historical rates; learned methods use the same 30 core features and training-only preprocessing. Poisson uses training exposure weights; other fits use equal row weights. The adopted loss ratio was chosen on 2025 validation.',
 '', '**Interpretation:** Histogram Gradient Boosting has the lowest full-period weighted MAE in this comparison; the adopted model is not uniformly best. Its full-period weighted MAE improves slightly on symmetric LightGBM, while weighted RMSE worsens. August-only MAE 0.347635 is a different evaluation subset and must not replace the full-period 0.637574.',
 '', 'The 2026 period has already been inspected in earlier exploration; this is not an untouched final holdout. Traffic-volume evaluation ends in July because August passenger labels are unavailable; traffic is not pooled into the rate/count table.',
 '', 'Sources: `results/reference/integrated_model_comparison/predictions.csv` and `results/reference/model1/predictions.csv`. Full-precision values and 1,010 aligned case predictions are exported under `results/baseline_comparison*`.']
(ROOT/'docs').mkdir(exist_ok=True)
(ROOT/'docs/Model1_Baseline_Comparison_2026_Jan_Aug.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(r[['method','rate_weighted_MAE','rate_weighted_RMSE','count_WAPE_percent_same_forecast_exposure']].to_string(index=False))
