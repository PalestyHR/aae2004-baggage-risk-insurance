# Model 1 Baseline Comparison Table

Test period: January–August 2026. **101 matched carrier-month cases**. Monthly expanding-window refits use only earlier target months, assuming previous-month data are available. All metrics below are recomputed from saved predictions; no retraining or model selection was performed.

## Main comparison

| Method | n | Rate weighted MAE ↓ | Rate weighted RMSE ↓ | Count WAPE (%) ↓ |
|---|---:|---:|---:|---:|
| Seasonal naive (same month last year) | 101 | 0.860345 | 0.999467 | 20.354 |
| 50/50 recent-seasonal blend | 101 | 0.697223 | 0.887871 | 17.977 |
| ElasticNet | 101 | 0.716507 | 0.991806 | 17.742 |
| Poisson GLM (exposure-weighted fit) | 101 | 0.706555 | 1.004255 | 18.040 |
| Random Forest | 101 | 0.659568 | 0.925244 | 16.665 |
| Extra Trees | 101 | 0.678233 | 0.903590 | 17.155 |
| Histogram Gradient Boosting | 101 | 0.620367 | 0.845798 | 16.155 |
| LightGBM (symmetric squared loss) | 101 | 0.647520 | 0.875529 | 15.969 |
| XGBoost (squared loss) | 101 | 0.635317 | 0.882772 | 16.012 |
| Adopted LightGBM (under:over = 1:2) | 101 | 0.637574 | 0.886649 | 15.836 |

Rate metrics are in **mishandled bags per 1,000 enplaned bags**. Count WAPE uses the **same forecast baggage exposure for every method**, namely the 50/50 latest-month and same-month-last-year blend selected on 2025 validation. No actual target-month exposure is used to produce count predictions. Actual exposure weights rate evaluation only.

## Additional rate diagnostics

| Method | Unweighted MAE ↓ | Unweighted RMSE ↓ | Weighted bias |
|---|---:|---:|---:|
| Seasonal naive (same month last year) | 0.834807 | 1.011862 | +0.391625 |
| 50/50 recent-seasonal blend | 0.689533 | 0.886786 | +0.242602 |
| ElasticNet | 0.750315 | 0.999720 | +0.023013 |
| Poisson GLM (exposure-weighted fit) | 0.809423 | 1.066113 | +0.025325 |
| Random Forest | 0.684080 | 0.932370 | +0.142820 |
| Extra Trees | 0.672847 | 0.886116 | +0.236411 |
| Histogram Gradient Boosting | 0.654173 | 0.854687 | +0.087858 |
| LightGBM (symmetric squared loss) | 0.672796 | 0.878507 | +0.119235 |
| XGBoost (squared loss) | 0.673213 | 0.894598 | +0.079647 |
| Adopted LightGBM (under:over = 1:2) | 0.688772 | 0.914432 | -0.075093 |

Weighted bias = weighted mean(predicted − observed): positive is overprediction. Simple methods use their named historical rates; learned methods use the same 30 core features and training-only preprocessing. Poisson uses training exposure weights; other fits use equal row weights. The adopted loss ratio was chosen on 2025 validation.

**Interpretation:** Histogram Gradient Boosting has the lowest full-period weighted MAE in this comparison; the adopted model is not uniformly best. Its full-period weighted MAE improves slightly on symmetric LightGBM, while weighted RMSE worsens. August-only MAE 0.347635 is a different evaluation subset and must not replace the full-period 0.637574.

The 2026 period has already been inspected in earlier exploration; this is not an untouched final holdout. Traffic-volume evaluation ends in July because August passenger labels are unavailable; traffic is not pooled into the rate/count table.

Sources: `results/reference/integrated_model_comparison/predictions.csv` and `results/reference/model1/predictions.csv`. Full-precision values and 1,010 aligned case predictions are exported under `results/baseline_comparison*`.
