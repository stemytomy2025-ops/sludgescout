## What SludgeScout does

SludgeScout answers market questions about European sewage sludge in plain English. For example:

- *Which countries landfilled the most sewage sludge in 2024?*
- *How has Poland's sludge going to incineration changed since 2010?*
- *Which countries still have a low share of the population on secondary treatment?*

It works from official Eurostat statistics, not from the AI model's memory. Every answer can
be traced to the calculation and data behind it.

## How the AI agent works

SludgeScout is an **agent**, not a chatbot: it decides how to answer each question.

1. **Understands the question.** Claude (Anthropic's `claude-opus-5-5` model) reads the
   question and works out what data it needs: which country, year, disposal route and unit.
2. **Chooses the analysis.** It picks one of five purpose-built tools and fills in their
   inputs (see below). It can chain several tools in one answer, for example ranking countries
   first and then pulling the exact rows behind the top result.
3. **Runs it on the real data.** Each tool is a fixed, tested calculation over the downloaded
   Eurostat tables. The model never writes or runs code.
4. **Checks and corrects itself.** Every input is validated against values that exist in the
   data. If one is wrong, for example a year with no data, the tool says why, and the agent fixes
   the input and tries again.
5. **Answers with the evidence.** The agent is instructed that every number it states must
   come from a tool result. If no tool can answer, it says so instead of guessing.

Under each answer, **"How this answer was computed"** shows every tool the agent called, the
exact inputs it used, the results, and how many model turns and tokens the answer took.

### Guardrails

- At most 6 model turns and 8 tool calls per question, so a question cannot run away.
- Only the five analysis tools are available. The agent cannot browse the web, change data or
  run code.
- Text in the data is treated as data, never as instructions to the agent.
- Only public, aggregated country statistics are used. There is no personal data.

## The analysis tools

| Tool | What it answers |
|---|---|
| Rank countries by sludge | Which countries produce or dispose of the most sludge by a given route (landfill, incineration, agriculture, compost) in a given year |
| Sludge by year | How one country's sludge figure has changed over time |
| Disposal-route mix | How one country splits its sludge between routes in a given year |
| Source rows | The exact Eurostat rows behind a figure, including estimate flags |
| Treatment coverage | Which countries have the highest or lowest share of the population on primary, secondary or tertiary wastewater treatment |

## The data

- **Eurostat: Sewage sludge production and disposal** (`env_ww_spd`): 12,929 rows, 1970–2024.
- **Eurostat: Population connected to wastewater treatment** (`env_ww_con`): 17,181 rows,
  1970–2024.

Both are published by Eurostat under CC BY 4.0. The app downloads them from Eurostat's API
and checks their structure before use. The **Data & sources** tab shows the full attribution,
the validated columns and the download record.

## How accuracy is checked

Each tool is tested against a separate, independent calculation on the same downloaded data,
and all five checks pass. These checks cover the calculations. The agent's free-text answers
have not yet been formally evaluated, so please review important figures using "How this
answer was computed".

## Limitations

- **Country-level, annual figures only.** There is no data on individual treatment plants.
- **Reporting varies.** Countries report in different years, so the latest year with data
  differs by country. Some values are marked by Eurostat as estimates or as breaks in a series.
- **No forecasts.** The tools describe what has been reported. They do not predict future years.
- **Raw figures, not ratios.** Tools return reported tonnages and percentages. Shares such as
  "percentage of sludge landfilled" are not calculated inside the tools.
- **One year at a time.** The agent cannot yet compare each country's own most recent year in a
  single step.
