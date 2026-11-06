# Data sources and boundaries

| Source | Included | Use and limits |
|---|---|---|
| User-supplied BTS Mishandled Baggage CSV | Original CSV and processed aggregates | Retain 2019 onward FORM-1; report count before 2019 is a different unit |
| BTS delay-cause download | Derived monthly features and manifests | Airport/carrier/month sums; weather attribution is not observed meteorology |
| BTS T100 domestic segment | Derived passenger/seat/airport statistics and manifests | CLASS F; passenger-segments, not unique people; through July 2026 |
| NOAA GSOD | Derived airport-month weather and coverage | 20 airports; retrieved observations end August 2025; excluded from adopted core features |
| USTIA / ABI market publications | Small factual CSV extracts and links | Aggregate market context only; no empirically identified demand elasticity |

Source entry points:

- https://transtats.bts.gov/Fields.asp?gnoyr_VQ=GGD
- https://www.bts.gov/content/mishandled-baggage-reports-filed-passengers-largest-us-air-carriersa
- https://www.transtats.bts.gov/OT_Delay/OT_DelayCause1.asp
- https://www.transtats.bts.gov/DL_SelectFields.aspx?gnoyr_VQ=GEE
- https://www.ncei.noaa.gov/data/global-summary-of-the-day/access/
- https://www.ncei.noaa.gov/pub/data/noaa/isd-history.csv

`data/processed/source_manifest.json` preserves precise download URLs and checksums. Each third-party source retains its terms. Full third-party PDFs, course materials and browser/session HTML are not redistributed here. Government-origin aggregate observations and our derived tables are included for this private research project; verify terms before making any public release.

Training reproduction starts from the included processed snapshots. Rebuilding all web downloads is a separate operation; historical download and build scripts in `archive/scripts/data_*` retain the original workspace layout and may require adapting paths. They are records of collection, not a promise that external URLs will remain available.
