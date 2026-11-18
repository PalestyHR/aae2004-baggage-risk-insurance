<h1 align="center">A Baggage Risk Management Framework<br>for Cathay Pacific</h1>

<p align="center">
  <a href="#overview">Overview</a> ·
  <a href="#models">Models</a> ·
  <a href="#quick-start">Quick Start</a> ·
  <a href="#main-results">Results</a> ·
  <a href="#data">Data</a>
</p>

## Overview

This project combines monthly baggage-risk forecasting with the pricing of an optional baggage protection product. Experiments use U.S. BTS carrier-month data, with Cathay Pacific as the proposed application context.

Historical operations → Risk and volume forecasts → Claim probabilities → Menu optimisation → Profitability simulation

[Method and equations](MODEL_SUMMARY.md) · [Data sources](DATA_SOURCES.md) · [Full baseline comparison](docs/Model1_Baseline_Comparison_2026_Jan_Aug.md)

## Models

| Component | Implementation | Experiment |
|---|---|---|
| **Forecasting Model** | [forecasting_model.py](models/forecasting_model.py) | [reproduce_risk.py](scripts/reproduce_risk.py) |
| **Insurance Model** | [insurance.py](models/insurance.py) | [reproduce_insurance.py](scripts/reproduce_insurance.py) |

The forecasting model contains the LightGBM parameters, asymmetric loss, preprocessing, fitting and prediction. The insurance model contains claim conversion, purchasing probabilities, expected profit and Monte Carlo simulation. Saved weights and preprocessing values are in [`checkpoints/`](checkpoints/).

<details>
<summary>Use the forecasting model in Python</summary>

```python
import json
from pathlib import Path

import pandas as pd
from models.forecasting_model import ForecastingModel

features = json.loads(Path("configs/feature_spec.json").read_text())["core_features"]
data = pd.read_csv("data/processed/lag1/all_real.csv")
train = data[data.target_month < "2026-08-01"]
test = data[data.target_month == "2026-08-01"]

model = ForecastingModel(features)
model.fit(train)
predicted_rates = model.predict(test)
```

</details>

## Quick start

Use **Python 3.12**. The processed datasets are included, and the experiments run on CPU.

```bash
git clone https://github.com/PalestyHR/aae2004-baggage-risk-insurance.git
cd aae2004-baggage-risk-insurance
python -m venv .venv
```

Activate with `.venv\Scripts\Activate.ps1` on PowerShell or `source .venv/bin/activate` on Linux/macOS.

```bash
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .

python scripts/reproduce.py
```

The command trains the forecasting model and nine baselines, repeats seven loss ratios under both weighting schemes, optimises the protection menu, runs 72 Monte Carlo cases, and rebuilds the comparison table from the new predictions. It checks the results against the included references and stops if a step fails.

A successful run ends with **`Reproduction complete: PASS`**. The verification summary is saved in `results/reproduction_check.json`. Expected selections are **1:2** for the under/over loss ratio and **M0775** for the protection menu.

| Result | Location |
|---|---|
| Forecast predictions and metrics | `results/reproduced_risk/` |
| Baseline predictions and metrics | `results/reproduced_baselines/` |
| Loss-ratio selection and comparison | `results/reproduced_loss_ratios/` |
| Menu, six scenarios and sensitivity cases | `results/reproduced_insurance/` |
| Monte Carlo results | `results/selected_menu_monte_carlo/` |
| Full baseline comparison | `results/baseline_comparison_2026_01_08.csv` |

<details>
<summary>Additional experiments</summary>

| Experiment | Command | Output |
|---|---|---|
| Baseline training | `python scripts/train_baselines.py` | `results/reproduced_baselines/` |
| Loss-ratio comparison | `python scripts/compare_loss_ratios.py` | `results/reproduced_loss_ratios/` |
| Monte Carlo simulation | `python scripts/simulate_selected_menu.py` | `results/selected_menu_monte_carlo/` |
| Existing tests | `python -m unittest discover -s tests -v` | Terminal results |

Monte Carlo uses 30,000 draws for each of six scenarios and 12 airline cases. Baseline training and loss-ratio experiments use the recorded chronological folds and configurations.

</details>

## Main results

### Baggage risk forecasting

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

### Baggage protection pricing

The August 2026 forecast supplies the insurance experiment with 12 airline-scale cases. The selected menu is **M0775**.

| Plan | Premium (USD) | Delay allowance (USD) | Cumulative final-loss benefit (USD) |
|---|---:|---:|---:|
| Basic | 3.75 | 50 | 300 |
| Standard | 6.00 | 100 | 600 |
| Premium | 11.00 | 200 | 1,200 |

The search evaluates **1,836 menus**, retains **98 feasible menus**, and finds **7 Pareto-optimal menus**. Across six scenarios, the selected menu's combined monthly expected profit ranges from **USD 2.74 million to USD 4.02 million**. Scenario assumptions and full outputs are in [configs](configs/) and [insurance results](results/reference/insurance/).

## Data

The supervised dataset has **1,226 real carrier-month records** and **30 core features**.

| Split | Target years | Records |
|---|---|---:|
| Initial training | 2020–2024 | 957 |
| Validation | 2025 | 168 |
| Rolling test | January–August 2026 | 101 |

2019 observations initialise historical features. Monthly refits use earlier target months. Feature preprocessing is fitted on each training window. Synthetic feature-masking copies are stored separately and are excluded from the adopted model. Passenger-volume labels are available through July 2026.

Detailed source tables, transformations and column definitions are linked from [DATA_SOURCES.md](DATA_SOURCES.md). Demand, claim conversion and cost inputs are specified scenario parameters in `configs/`.

## Repository structure

```text
models/          Forecasting and insurance implementations
scripts/         Training, evaluation and simulation entry points
configs/         Features, financial assumptions and selected menu
checkpoints/     LightGBM weights and fitted preprocessing values
data/            Datasets and source records
results/         Reference results and reproduced outputs
visualizations/  Figures and numerical plotting data
tests/           Calculation tests
```

Detailed comparisons are in `docs/`. Original data preparation records are retained in `archive/scripts/`. The original `baggage_project` import paths are supported through `src/baggage_project/`.

<details>
<summary>Regenerate figures</summary>

```bash
python -m pip install -r requirements-viz.txt
python scripts/export_baseline_comparison.py
python scripts/export_visualization_curves.py
python scripts/draw_visual_style_v2.py
python scripts/draw_3d_ribbon_figures.py
```

Iteration curves are reconstructed from saved model tree prefixes. Baseline tables are recomputed from saved predictions.

</details>

The 2026 results form an exploratory rolling evaluation. Financial simulations use the demand, claim and cost assumptions specified in `configs/`.
