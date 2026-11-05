# Data sources

## Eurostat: Sewage sludge production and disposal (env_ww_spd)

- Landing page: https://ec.europa.eu/eurostat/databrowser/view/env_ww_spd/default/table
- Licence: **CC BY 4.0 (Eurostat reuse policy)**
- Attribution: Eurostat: Sewage sludge production and disposal (env_ww_spd), https://ec.europa.eu/eurostat/databrowser/view/env_ww_spd/default/table (licence: CC BY 4.0 (Eurostat reuse policy))
- Download method: `{'kind': 'http_csv', 'url': 'https://ec.europa.eu/eurostat/api/dissemination/sdmx/2.1/data/env_ww_spd/?format=SDMX-CSV&compressed=false', 'max_mb': 300}`
- Inspected by the factory on 2026-11-05 (sha256 of the inspected sample `a4598e082bb38a8f…`)

| Column | Validated type |
|---|---|
| `ww_tpar` | string |
| `ww_tp` | string |
| `unit` | string |
| `geo` | string |
| `TIME_PERIOD` | integer |
| `OBS_VALUE` | float |
| `OBS_FLAG` | string |

### Limitations

- Automated inspection: Snapshot inspected 2026-11-05; the source may change after this date.
- Long format: codes need labels (SLD_LAND=landfill, SLD_INCN=incineration, SLD_AGR=agricultural use, SLP=production).
- Country-level annual totals only; no plant-level detail. Coverage varies by country and year (gaps, flags).

## Eurostat: Population connected to wastewater treatment (env_ww_con)

- Landing page: https://ec.europa.eu/eurostat/databrowser/view/env_ww_con/default/table
- Licence: **CC BY 4.0 (Eurostat reuse policy)**
- Attribution: Eurostat: Population connected to wastewater treatment (env_ww_con), https://ec.europa.eu/eurostat/databrowser/view/env_ww_con/default/table (licence: CC BY 4.0 (Eurostat reuse policy))
- Download method: `{'kind': 'http_csv', 'url': 'https://ec.europa.eu/eurostat/api/dissemination/sdmx/2.1/data/env_ww_con/?format=SDMX-CSV&compressed=false', 'max_mb': 300}`
- Inspected by the factory on 2026-11-05 (sha256 of the inspected sample `958b5267e8bc471b…`)

| Column | Validated type |
|---|---|
| `ww_tp` | string |
| `unit` | string |
| `geo` | string |
| `TIME_PERIOD` | integer |
| `OBS_VALUE` | float |
| `OBS_FLAG` | string |

### Limitations

- Automated inspection: Snapshot inspected 2026-11-05; the source may change after this date.
- Mixes units (PC percentage vs THS thousand persons); tools must filter one unit.
- Country-level annual; reporting gaps for several countries and years.

## Alternatives considered and rejected

- **Eurostat: Renewables and wastes supply - biogases (nrg_cb_rw)**: Query used wrong energy-balance code (PRD); Eurostat returned HTTP 400. Superseded by eurostat-biogas-production with code IPRD.
- **introvoyz041/Biogas-Production-Machine-Learning-Analysis**: Contains ML model evaluation metrics (MAE/RMSE/R2, params), not biogas or plant data; not useful for a Cambi use case.
