# A Baggage Risk Management Framework for Cathay Pacific

Monthly baggage-risk forecasting and scenario-based pricing of optional baggage protection. The experiments use U.S. Bureau of Transportation Statistics (BTS) carrier-month data, with Cathay Pacific as the proposed application context.

## Overview

1. Forecast the next month's baggage-mishandling rate with LightGBM.
2. Forecast baggage and passenger volumes using historical exposure models.
3. Convert the forecast risk into scenario claim probabilities.
4. Optimise a three-tier protection menu and evaluate profitability.

[Model equations and data construction](MODEL_SUMMARY.md) · [Data sources](DATA_SOURCES.md) · [Full baseline comparison](docs/Model1_Baseline_Comparison_2026_Jan_Aug.md)

## Main results

All methods are compared on the same **101 carrier-month records from January–August 2026**. Rate errors are weighted by loaded baggage volume.

| Model | WMAE ↓ | WRMSE ↓ | Count WAPE (%) ↓ |
|---|---:|---:|---:|
| Seasonal naive | 0.8603 | 0.9995 | 20.354 |
| 50/50 historical blend | 0.6972 | 0.8879 | 17.977 |
| Histogram Gradient Boosting | **0.6204** | **0.8458** | 16.155 |
| LightGBM, symmetric loss | 0.6475 | 0.8755 | 15.969 |
| XGBoost | 0.6353 | 0.8828 | 16.012 |
| **Our Model** | **0.6376** | **0.8866** | **15.836** |

WMAE and WRMSE are measured in bags per 1,000 loaded bags. Our Model uses an underestimation-to-overestimation penalty ratio of **1:2**, selected using 2025 validation WMAE. The [penalty-ratio experiment](results/reference/asymmetric_loss_comparison/metrics.csv) records the alternatives.

The August 2026 forecast supplies the insurance experiment with 12 airline-scale cases. The selected menu is **M0775**.

| Plan | Premium (USD) | Delay allowance (USD) | Cumulative final-loss benefit (USD) |
|---|---:|---:|---:|
| Basic | 3.75 | 50 | 300 |
| Standard | 6.00 | 100 | 600 |
| Premium | 11.00 | 200 | 1,200 |

The search evaluates **1,836 menus**, retains **98 feasible menus**, and finds **7 Pareto-optimal menus**. Across six scenarios, the selected menu's combined monthly expected profit ranges from **USD 2.74 million to USD 4.02 million**. Scenario assumptions and full outputs are in [configs](configs/) and [insurance results](results/reference/insurance/).

## Installation

Use Python 3.12.

```bash
python -m venv .venv
```

Activate the environment with `.venv\Scripts\Activate.ps1` on PowerShell or `source .venv/bin/activate` on Linux/macOS, then install dependencies.

```bash
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .
```

## Reproduction

Run these commands from the repository root, in order.

```bash
python scripts/reproduce_risk.py
python scripts/reproduce_insurance.py
python scripts/check_reproduction.py
```

The first command refits the forecast model with an expanding monthly window. The second runs menu optimisation and the additional demand/cost scenarios. The last compares the regenerated predictions, selected menu and financial outputs with the included reference results.

| Command | Output |
|---|---|
| `reproduce_risk.py` | `results/reproduced_risk/` and the saved predictor in `models/` |
| `reproduce_insurance.py` | `results/reproduced_insurance/` |
| `check_reproduction.py` | `results/reproduction_check.json` |
| `simulate_selected_menu.py` | `results/selected_menu_monte_carlo/` |
| `export_baseline_comparison.py` | `results/baseline_comparison*.csv` |

Run the selected-menu Monte Carlo experiment and existing tests with:

```bash
python scripts/simulate_selected_menu.py
python -m unittest discover -s tests -v
```

Monte Carlo uses 30,000 draws for each of six scenarios and 12 airline cases. The baseline export recomputes metrics from the saved model predictions.

To retrain all baseline methods or repeat the seven penalty-ratio comparisons:

```bash
python scripts/train_baselines.py
python scripts/compare_loss_ratios.py
```

These commands write to `results/reproduced_baselines/` and `results/reproduced_loss_ratios/`. Each uses the same chronological folds and configurations as its reference experiment.

## Data

The supervised dataset has **1,226 real carrier-month records** and **30 core features**.

| Split | Target years | Records |
|---|---|---:|
| Initial training | 2020–2024 | 957 |
| Validation | 2025 | 168 |
| Rolling test | January–August 2026 | 101 |

2019 observations initialise historical features. Monthly refits use earlier target months. Feature preprocessing is fitted on each training window. Synthetic feature-masking copies are stored separately and are excluded from the adopted model. Passenger-volume labels are available through July 2026.

Detailed source tables, transformations and column definitions are linked from [DATA_SOURCES.md](DATA_SOURCES.md). Demand, claim conversion and cost inputs are specified scenario parameters in `configs/`.

## Code and results

```text
configs/              Feature definitions, assumptions and selected menu
src/baggage_project/  Frozen predictor and insurance mathematics
scripts/              Training, optimisation, simulation and plotting entry points
tests/                Existing insurance calculation tests
data/                 Processed datasets, baggage source CSV and source records
models/               Saved August LightGBM predictor and preprocessing
results/reference/    Forecasts, baseline experiments and selected-menu results
visualizations/       Figures and numerical plotting data
docs/                 Full model comparison
archive/scripts/      Original data collection and preparation records
```

For figure regeneration, install `requirements-viz.txt`, then run:

```bash
python scripts/export_baseline_comparison.py
python scripts/export_visualization_curves.py
python scripts/draw_visual_style_v2.py
python scripts/draw_3d_ribbon_figures.py
```

Iteration curves are reconstructed from the saved model's tree prefixes. The 2026 experiments form an exploratory rolling evaluation. Financial simulations describe the operating conditions specified in the configuration files.
