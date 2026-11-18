# Data

## Sources

| Source | Use |
|---|---|
| [BTS mishandled baggage](https://transtats.bts.gov/Fields.asp?gnoyr_VQ=GGD) | Loaded and mishandled baggage counts by operating carrier and month |
| [BTS delay causes](https://www.transtats.bts.gov/OT_Delay/OT_DelayCause1.asp) | Monthly carrier and airport operational features |
| [BTS T-100 domestic segments](https://www.transtats.bts.gov/DL_SelectFields.aspx?gnoyr_VQ=GEE) | Passenger segments, seats, flights and airport activity |

Download records and source identifiers are retained in [`source_manifest.json`](data/processed/source_manifest.json). Source terms apply to the corresponding data.

## Construction

1. Retain baggage records from 2019 onward using the operating-carrier reporting format.
2. Aggregate operational records by DOT carrier identifier and calendar month.
3. Combine delay statistics with T-100 domestic scheduled passenger-service records (CLASS F).
4. Construct prior-month and seasonal features using earlier observations. Use 2019 records to initialise historical features for 2020 targets.
5. Form the monthly target as `1000 * mishandled_bags / loaded_bags` and retain rows with the required historical information.

The resulting dataset contains **1,226 real carrier-month cases** and **30 selected features**. Passenger-volume observations extend through July 2026. The source manifest also records exploratory weather downloads, which are excluded from the selected features.

## Files used in reproduction

| File | Contents |
|---|---|
| `data/processed/lag1/all_real.csv` | Complete supervised carrier-month dataset |
| `data/processed/lag1/train_real.csv` | 957 initial training cases, 2020–2024 |
| `data/processed/lag1/validation_real.csv` | 168 validation cases, 2025 |
| `data/processed/lag1/test_real.csv` | 101 rolling test cases, January–August 2026 |
| `data/processed/carrier_month_observed_NOT_model_input.csv` | Historical volumes used to construct exposure forecasts and evaluation targets |
| `data/processed/data_dictionary.csv` | Column definitions, units, timing and sources |
| `configs/feature_spec.json` | Feature selection used by the training scripts |
| `data/raw/bts_mishandled_baggage.csv` | Original baggage source table |

The reproduction command starts from these included processed files. Each monthly fit estimates preprocessing on its own training window. Synthetic rows are excluded from the reported training and evaluation.
