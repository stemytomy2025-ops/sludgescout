# SludgeScout


Ask questions in plain English about how European countries produce and dispose of sewage
sludge. An AI agent chooses the right analysis, runs it on official Eurostat data, and shows
exactly how each answer was computed.

A Claude agent answers by calling validated Python tools over two real Eurostat datasets,
**Sewage sludge production and disposal (env_ww_spd)** and **Population connected to
wastewater treatment (env_ww_con)** (both CC BY 4.0). The agent is instructed that every number
in an answer must come from a tool result, and the app shows which tools ran, with their inputs
and outputs. See [docs/about.md](docs/about.md) for how the agent works.

## Example questions

- Which five countries landfilled the most sewage sludge (thousand tonnes) in the latest year with data?
- How has Poland's sludge going to incineration changed since 2010?
- What is the disposal-route mix in Poland in 2022, and which rows is that based on?
- Which countries still have under 70% of the population on secondary treatment?

## How it works

| Tool | Operation | What it does |
|---|---|---|
| `rank_countries_by_sludge` | aggregate | Rank countries by sludge for ONE disposal route (ww_tpar: SLD_LAND landfill, SLD_INCN incineration, SLD_AGR agricultural use, SLD_COMP compost, SLD_OTH other, SLD total disposed, SLP total produced) and ONE year (TIME_PERIOD). Always filter unit (THS_T thousand tonnes or KG_HAB kg per capita) and ww_tp (URB_TP urban plants). |
| `sludge_route_by_year` | aggregate | Yearly sludge values for ONE country (geo) and ONE route (ww_tpar codes as in rank_countries_by_sludge). Always filter unit (THS_T thousand tonnes or KG_HAB kg per capita) and ww_tp (URB_TP urban plants). |
| `sludge_route_mix` | aggregate | Compare ww_tpar values within ONE country and ONE year. SLP (produced) and SLD (total disposed) are totals, not routes: report routes against them, never add them to the routes. Always filter unit (THS_T thousand tonnes or KG_HAB kg per capita) and ww_tp (URB_TP urban plants). |
| `cite_sludge_rows` | filter_rows | Return the exact Eurostat rows (with OBS_FLAG estimate/break flags) behind a figure, for citation. Filter at least geo and TIME_PERIOD. |
| `treatment_coverage_by_country` | aggregate | Rank countries by ONE treatment level (ww_tp: URB_OTH_CS_T2 secondary, URB_OTH_CS_T3 tertiary, URB_OTH_CS_NC not connected, TOTAL connected) in ONE year. Always filter unit (PC percent of population or THS thousand persons). |

- The model (`claude-opus-5-5`) only chooses tools and arguments. Arguments are validated against
  the real data (allowed metrics, filter values, date ranges, feature bounds); it never writes
  or runs code.
- Each question is limited to 6 model turns and 8 tool calls.
- See [docs/architecture.md](docs/architecture.md) for details.

## Quickstart

Requires Python 3.11+.

```bash
python -m venv .venv
```

Activate it (`.venv\Scripts\Activate.ps1` on Windows, `source .venv/bin/activate` elsewhere),
then:

```bash
pip install -e ".[dev]"
```

```bash
python scripts/fetch_data.py
```

```bash
streamlit run app.py
```

For the chat tab, set `ANTHROPIC_API_KEY` in your environment or in
`.streamlit/secrets.toml` (see `.streamlit/secrets.toml.example`). The tool runner, evaluation
and data tabs work without a key.

## Tests and evaluation

```bash
pytest
```

Tests run offline on SYNTHETIC fixtures with a fake Anthropic client. After downloading the
data, the real-data evals run too.

```bash
python scripts/pin_expectations.py
```

```bash
python scripts/run_eval.py
```

`pin_expectations.py` recomputes the expected answers from the raw data with pyarrow and plain
Python, independently of the tool code. `run_eval.py --agent` also runs agent-level evals
(calls the API, costs tokens).

## Data and attribution

Eurostat: Sewage sludge production and disposal (env_ww_spd), https://ec.europa.eu/eurostat/databrowser/view/env_ww_spd/default/table (licence: CC BY 4.0 (Eurostat reuse policy)). Full details, inspection evidence and limitations:
[docs/data_sources.md](docs/data_sources.md). The data is downloaded at setup time and is not
committed to this repository.

## Deployment (not performed)

The app is ready for Streamlit Community Cloud: `requirements.txt` installs the package, and
the API key goes into the app's Secrets settings. On first start the app downloads and
validates the Eurostat data automatically; **Download / refresh data** in the sidebar refreshes it.
