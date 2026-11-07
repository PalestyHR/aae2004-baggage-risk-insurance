# AAE2004 Baggage Risk and Protection Models

## Project resources

- [Model definitions and experimental results](MODEL_SUMMARY.md)
- [January–August 2026 baseline comparison](docs/Model1_Baseline_Comparison_2026_Jan_Aug.md)
- [Processed datasets](data/processed/) and [data sources](DATA_SOURCES.md)
- [Figures and plot source data](visualizations/)

This project uses public U.S. BTS records to evaluate monthly baggage-mishandling forecasts. Cathay Pacific is the proposed application context, not the tested population. Financial outputs are scenario estimates, not observed sales or underwriting results.

## Adopted design

- Forecast pipeline: 30 features; monthly expanding LightGBM; asymmetric under/over loss **1:2**; validated blend forecasts for traffic and baggage exposure.
- Full January–August 2026 test: rate WMAE **0.637574**, WRMSE **0.886649**; count WAPE **15.835726%**.
- August 2026 case: rate WMAE **0.347635 bags per 1,000**; count WAPE **8.935854%**. Earlier rolling tests remain in the evidence files.
- Product: optional fixed extra support payments, not reimbursement of declared baggage value.
- Menu **M0775**, all amounts USD: prices **3.75 / 6 / 11**, delay benefits **50 / 100 / 200**, final-loss cumulative benefits **300 / 600 / 1,200**.
- The financial design is conditional on uncalibrated demand, claim conversion and cost assumptions. It does not guarantee profitability.

## Run

Use Python 3.12. Install the pinned numerical environment:

```sh
python -m venv .venv
# Activate .venv for your shell, then:
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .
python scripts/reproduce_risk.py
python scripts/reproduce_insurance.py
python scripts/check_reproduction.py
python -m unittest discover -s tests -v
```

Optional scenario simulation: `python scripts/simulate_selected_menu.py`.

Curves and editable source data are in `visualizations/`. Install `requirements-viz.txt` and run `python scripts/export_visualization_curves.py` to regenerate them. Per-round risk diagnostics are reconstructed from frozen tree prefixes; they are not original saved training logs.

Reproduced outputs go under `results/reproduced_*` (ignored by Git), preserving `results/reference`. The risk script also exports a text LightGBM model and preprocessing JSON in `models/`; no pickle is required. The insurance script uses reproduced forecasts when available, otherwise the frozen reference interface.

## Files

| Path | Contents |
|---|---|
| `visualizations/` | Plot exports and numerical source records |
| `data/processed/` | Real train/validation/test sets, optional synthetic training copies, dictionary and source manifests |
| `data/raw/` | Original user-supplied BTS baggage CSV |
| `data/reference/` | Small factual market-reference tables; no fitted price elasticity |
| `models/` | Frozen August risk booster and preprocessing |
| `src/baggage_project/` | Risk loading and insurance mathematics |
| `configs/` | Feature spec, scenario assumptions and current selected menu |
| `results/reference/` | Full numerical history, including explicitly superseded experiments |
| `results/selected_menu_monte_carlo/` | Conditional simulations of the adopted menu |
| `archive/scripts/` | Historical source snapshots, not maintained portable entry points |

## Data and interpretation

The real supervised dataset contains **1,226 carrier-month cases**: 957 train, 168 validation, 101 test. Synthetic copies are separate and unused in the adopted forecasting model. Never use current target volume or synthetic flags as predictive features.

The 2026 test period has been inspected in earlier experiments. This is exploratory evaluation. August traffic truth is unavailable; passenger-volume testing ends in July. Mishandling includes multiple incident categories and does not identify permanent loss or policy claims.

Raw large external downloads, third-party full reports, classroom PDFs and local environments are omitted. Source URLs, hashes, transformations and historical collection scripts are retained. See [DATA_SOURCES.md](DATA_SOURCES.md). No blanket redistribution licence is asserted for third-party material.
