"""Agent tools: each binds one operation to fixed, inspected columns.

Input models are generated from the real data (allowed metrics, filter values, date ranges,
feature bounds), so the model's arguments are validated before any operation runs. Errors
come back as ToolError messages the model can read and correct.
"""

from __future__ import annotations

from datetime import date
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, ValidationError, create_model

from sludgescout import extraction
from sludgescout import operations as ops
from sludgescout.config import AppConfig, ToolConfig

FilterValue = str | int | float | bool
STRICT = ConfigDict(extra="forbid")


class ToolError(Exception):
    pass


def _literal(values: list[Any]) -> Any:
    return Literal[tuple(values)]  # type: ignore[valid-type]


def _values(df: pd.DataFrame, column: str, limit: int = 30) -> list[Any]:
    return [ops.clean(v) for v in sorted(df[column].dropna().unique().tolist())][:limit]


def _date_range(df: pd.DataFrame, time: str) -> tuple[date, date]:
    ts = pd.to_datetime(df[time])
    return ts.min().date(), ts.max().date()


def _range_fields(df: pd.DataFrame, time: str | None) -> dict[str, Any]:
    if not time:
        return {}
    lo, hi = _date_range(df, time)
    note = f"Inclusive. Data covers {lo} to {hi}."
    return {
        "start": (date | None, Field(None, description=f"Start date. {note}")),
        "end": (date | None, Field(None, description=f"End date. {note}")),
    }


MAX_LISTED_VALUES = 50


def _allowed_values(df: pd.DataFrame, column: str) -> str:
    """Describe a filter column's values without hiding any: a truncated list reads as complete."""
    values = _values(df, column, limit=len(df))
    if not ops.is_text(df[column]) and len(values) > 12:
        return (
            f"{column}: {values[0]} to {values[-1]} "
            f"({len(values)} observed values, not every value in between)"
        )
    if len(values) <= MAX_LISTED_VALUES:
        return f"{column}: {values}"
    shown = values[:MAX_LISTED_VALUES]
    return f"{column}: {shown} and {len(values) - len(shown)} more (any observed value is accepted)"


def _filter_field(df: pd.DataFrame, filterable: list[str]) -> dict[str, Any]:
    if not filterable:
        return {}
    allowed = "; ".join(_allowed_values(df, c) for c in filterable)
    return {
        "filters": (
            dict[str, FilterValue],
            Field(default_factory=dict, description=f"Equality filters. Allowed: {allowed}"),
        )
    }


def _scenario_fields(df: pd.DataFrame, features: list[str]) -> dict[str, Any]:
    """One optional field per feature: observed values for codes, observed bounds otherwise."""
    fields: dict[str, Any] = {}
    for feature in features:
        series = df[feature]
        if ops.is_text(series) or series.nunique() <= 12:
            kind: Any = _literal(_values(df, feature)) | None
            desc = f"Optional scenario value. Observed: {_values(df, feature)}."
            fields[feature] = (kind, Field(None, description=desc))
        else:
            lo_f, hi_f = float(series.min()), float(series.max())
            desc = f"Optional scenario value between {lo_f:g} and {hi_f:g}."
            fields[feature] = (float | None, Field(None, ge=lo_f, le=hi_f, description=desc))
    return fields


def build_input_model(cfg: ToolConfig, df: pd.DataFrame) -> type[BaseModel]:
    p = cfg.params
    fields: dict[str, Any] = {}
    if cfg.operation == "aggregate" and p.get("count_only"):
        fields["group"] = (
            _literal(p["groups"]),
            Field(p["group"], description="What to count rows by."),
        )
        fields |= _filter_field(df, p.get("filterable", []))
    elif cfg.operation == "aggregate":
        metrics = p["metrics"]
        fields["metric"] = (
            _literal(metrics),
            Field(metrics[0], description="Column to summarise."),
        )
        fields["agg"] = (ops.Agg, Field("mean", description="Aggregation per group."))
        fields |= _filter_field(df, p.get("filterable", []))
    elif cfg.operation == "trend":
        fields["freq"] = (ops.Freq, Field("day", description="Period length."))
        fields["agg"] = (ops.Agg, Field("sum", description="Aggregation per period."))
        fields |= _range_fields(df, p["time"]) | _filter_field(df, p.get("filterable", []))
    elif cfg.operation == "outliers":
        fields["threshold"] = (float, Field(3.5, ge=2, le=10, description="Robust |z| cut-off."))
        fields["direction"] = (Literal["high", "low", "both"], Field("both"))
        fields["limit"] = (int, Field(10, ge=1, le=ops.MAX_ROWS))
        fields |= _range_fields(df, p["time"])
    elif cfg.operation == "filter_rows":
        fields["sort_by"] = (_literal(p["columns"]) | None, Field(None))
        fields["descending"] = (bool, Field(True))
        fields["limit"] = (int, Field(10, ge=1, le=ops.MAX_ROWS))
        fields |= _range_fields(df, p.get("time")) | _filter_field(df, p.get("filterable", []))
    elif cfg.operation == "forecast":
        lo, hi = _date_range(df, p["time"])
        fields["date"] = (date, Field(description=f"Day to forecast. Data covers {lo} to {hi}."))
        if p.get("hour"):
            fields["hour"] = (int, Field(ge=0, le=23, description="Hour of day, 0-23."))
        fields |= _scenario_fields(df, p.get("features", []))
    elif cfg.operation == "retrieve":
        fields["query"] = (str, Field(min_length=1, max_length=500, description="Search terms."))
        fields["limit"] = (int, Field(10, ge=1, le=ops.MAX_ROWS))
        fields |= _filter_field(df, p.get("filterable", []))
    elif cfg.operation == "llm_extract":
        fields["sentence"] = (
            str | None,
            Field(None, min_length=1, max_length=2000, description="Sentence to extract from."),
        )
        fields["example_id"] = (
            str | None,
            Field(None, description=f"Or the {p['id']} of a sentence in the corpus."),
        )
    elif cfg.operation == "evaluate":
        desc = (
            "'saved' scores predictions from scripts/run_extraction.py on the holdout. "
            "'gold_self_check' scores the human annotations against themselves to test the "
            "scorer: it says nothing about the model."
        )
        fields["source"] = (Literal["saved", "gold_self_check"], Field("saved", description=desc))
        fields["limit"] = (int, Field(10, ge=1, le=ops.MAX_ROWS, description="Worst examples."))
    elif cfg.operation == "woe_iv":
        desc = "Characteristic to bin. Omit to rank all characteristics by Information Value."
        fields["feature"] = (_literal(p["features"]) | None, Field(None, description=desc))
    elif cfg.operation == "scorecard_performance":
        fields["deciles"] = (int, Field(10, ge=4, le=20, description="Calibration bands."))
    elif cfg.operation == "score_breakdown":
        if p.get("id"):
            ids = df[p["id"]]
            desc = (
                f"Optional {p['id']} of an existing account "
                f"({ids.min()} to {ids.max()}). Omit to score a typical or what-if applicant."
            )
            fields["account_id"] = (int | None, Field(None, description=desc))
        fields |= _scenario_fields(df, p["features"])
    else:  # pragma: no cover - config validation prevents this
        raise ValueError(f"unsupported operation {cfg.operation}")
    return create_model(f"{cfg.name}_input", __config__=STRICT, **fields)


class Tool:
    def __init__(
        self,
        cfg: ToolConfig,
        df: pd.DataFrame,
        frames: dict[str, pd.DataFrame] | None = None,
    ):
        self.cfg = cfg
        for item in cfg.params.get("derive", []):
            df = df.assign(**{item["name"]: ops.derive_label(df[item["from"]], item["patterns"])})
        self.df = df
        self.frames = frames or {}
        self.llm: extraction.LLM | None = None
        self.input_model = build_input_model(cfg, df)
        self._forecaster: ops.ForecastModel | None = None

    @property
    def name(self) -> str:
        return self.cfg.name

    def definition(self) -> dict[str, Any]:
        """Anthropic tool definition (name, description, JSON input schema)."""
        return {
            "name": self.cfg.name,
            "description": self.cfg.description,
            "input_schema": self.input_model.model_json_schema(),
        }

    def _filters(self, filters: dict[str, Any]) -> dict[str, Any]:
        """Map filter values onto real column values; reject unknown columns or values."""
        allowed_columns = self.cfg.params.get("filterable", [])
        converted = {}
        for column, value in filters.items():
            if column not in allowed_columns:
                raise ToolError(f"cannot filter on {column!r}; allowed: {allowed_columns}")
            observed = {_key(v): v for v in self.df[column].dropna().unique().tolist()}
            if _key(value) not in observed:
                raise ToolError(
                    f"{value!r} is not a value of {column!r}; observed: "
                    f"{list(observed.values())[:20]}"
                )
            converted[column] = observed[_key(value)]
        return converted

    def run(self, raw: dict[str, Any] | None) -> dict[str, Any]:
        try:
            args = self.input_model.model_validate(raw or {})
        except ValidationError as exc:
            details = "; ".join(
                f"{'.'.join(map(str, e['loc'])) or 'input'}: {e['msg']}" for e in exc.errors()
            )
            raise ToolError(f"invalid input for {self.name}: {details}") from exc
        try:
            return self._dispatch(args)
        except ops.OperationError as exc:
            raise ToolError(str(exc)) from exc

    def _dispatch(self, a: BaseModel) -> dict[str, Any]:
        p, df, op = self.cfg.params, self.df, self.cfg.operation
        filters = self._filters(getattr(a, "filters", {}) or {})
        if op == "aggregate" and p.get("count_only"):
            return ops.aggregate(df, a.group, a.group, "count", filters)
        if op == "aggregate":
            return ops.aggregate(df, p["group"], a.metric, a.agg, filters)
        if op == "retrieve":
            return ops.search_text(df, p["text_columns"], p["columns"], a.query, a.limit, filters)
        if op == "llm_extract":
            return self._extract(a)
        if op == "evaluate":
            return self._evaluate(a)
        if op == "trend":
            return ops.trend(
                df, p["time"], p["metric"], a.freq, a.agg, p.get("hour"), a.start, a.end, filters
            )
        if op == "outliers":
            return ops.outliers(
                df,
                p["time"],
                p["metric"],
                p.get("hour"),
                p.get("group_by", []),
                a.threshold,
                a.direction,
                a.start,
                a.end,
                a.limit,
            )
        if op == "filter_rows":
            return ops.filter_rows(
                df,
                p["columns"],
                filters,
                p.get("time"),
                p.get("hour"),
                getattr(a, "start", None),
                getattr(a, "end", None),
                a.sort_by,
                a.descending,
                a.limit,
            )
        if op in SCORECARD_OPERATIONS:
            card = scorecard(df, p)
            if op == "woe_iv":
                return ops.woe_iv(card, a.feature)
            if op == "scorecard_performance":
                return ops.scorecard_performance(card, a.deciles)
            overrides = {f: getattr(a, f) for f in p["features"]}
            return ops.score_breakdown(card, getattr(a, "account_id", None), overrides)
        # forecast
        if self._forecaster is None:
            self._forecaster = ops.ForecastModel(
                df,
                p["time"],
                p["metric"],
                p.get("hour"),
                p.get("features", []),
                p.get("holdout_days", 28),
            )
        overrides = {f: getattr(a, f) for f in p.get("features", [])}
        return self._forecaster.predict(a.date, getattr(a, "hour", None), overrides)

    def _gold(self) -> dict[str, ops.Rule]:
        p = self.cfg.params
        return ops.gold_rules(
            self.df,
            self.frames[p["relations_dataset"]],
            p["entity_columns"],
            p["relation_columns"],
            p.get("none_label", "none"),
        )

    def _evaluate(self, a: BaseModel) -> dict[str, Any]:
        share = self.cfg.params.get("holdout_share", 0.2)
        holdout = {k: v for k, v in self._gold().items() if ops.is_holdout(k, share)}
        if a.source == "gold_self_check":
            result = ops.score_extraction(holdout, holdout, a.limit)
            return {"operation": "evaluate", "source": a.source, "is_model_result": False} | result
        saved = extraction.load_predictions()
        predictions = {k: extraction.to_tuples(v) for k, v in saved["predictions"].items()}
        result = ops.score_extraction(holdout, predictions, a.limit)
        run = {k: saved.get(k) for k in ("model", "effort", "prompt_version", "run_at", "usage")}
        return {
            "operation": "evaluate",
            "source": a.source,
            "is_model_result": True,
            "run": run,
            "failed_examples": len(saved.get("failures", {})),
        } | result

    def few_shot_examples(self, count: int = 3) -> list[dict[str, Any]]:
        """Gold examples from outside the holdout, chosen by id so the prompt is stable."""
        p = self.cfg.params
        if p.get("relations_dataset") not in self.frames:
            return []
        share = p.get("holdout_share", 0.2)
        text = dict(zip(self.df[p["id"]].astype(str), self.df[p["text"]], strict=True))
        examples = []
        for key, rule in sorted(self._gold().items()):
            if ops.is_holdout(key, share) or not (rule["entities"] and rule["relations"]):
                continue
            examples.append(
                {
                    "sentence": str(text[key]),
                    "rule": {
                        "entities": [{"text": t, "type": k} for k, t in sorted(rule["entities"])],
                        "relations": [
                            {"subject": s, "object": o, "type": k}
                            for k, s, o in sorted(rule["relations"])
                        ],
                    },
                }
            )
            if len(examples) == count:
                break
        return examples

    def _extract(self, a: BaseModel) -> dict[str, Any]:
        p = self.cfg.params
        if self.llm is None:
            raise ToolError("this tool calls Claude: set ANTHROPIC_API_KEY first")
        if (a.sentence is None) == (a.example_id is None):
            raise ToolError("give exactly one of sentence or example_id")
        sentence = a.sentence
        if a.example_id is not None:
            match = self.df[self.df[p["id"]].astype(str) == a.example_id]
            if match.empty:
                raise ToolError(f"no sentence with {p['id']} = {a.example_id!r}")
            sentence = str(match[p["text"]].iloc[0])
        return extraction.extract(
            self.llm, sentence, p["entity_types"], p["relation_types"], self.few_shot_examples()
        )


SCORECARD_OPERATIONS = {"woe_iv", "scorecard_performance", "score_breakdown"}
# Values hold the frame too, so its id() cannot be reused by another frame while cached.
_SCORECARDS: dict[tuple[int, str], tuple[pd.DataFrame, ops.Scorecard]] = {}


def scorecard(df: pd.DataFrame, p: dict[str, Any]) -> ops.Scorecard:
    """One fitted scorecard per frame and configuration, shared by all scorecard tools."""
    keys = ("target", "features", "id", "test_share", "max_bins", "seed")
    settings = {k: p[k] for k in keys if k in p}
    cache_key = (id(df), repr(sorted(settings.items())))
    if cache_key not in _SCORECARDS:
        _SCORECARDS[cache_key] = (
            df,
            ops.Scorecard(
                df,
                p["target"],
                p["features"],
                p.get("id"),
                test_share=p.get("test_share", 0.3),
                max_bins=p.get("max_bins", 5),
                seed=p.get("seed", 0),
            ),
        )
    return _SCORECARDS[cache_key][1]


def _key(value: Any) -> str:
    """Compare filter values across types: 1, 1.0 and '1' are the same value."""
    if isinstance(value, bool):
        return str(value).lower()
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value).strip().lower()
    return str(int(number)) if number.is_integer() else str(number)


class ToolRegistry:
    def __init__(self, tools: list[Tool]):
        self.tools = {t.name: t for t in tools}

    @classmethod
    def from_config(cls, config: AppConfig, frames: dict[str, pd.DataFrame]) -> ToolRegistry:
        return cls([Tool(t, frames[t.dataset_id], frames) for t in config.tools])

    def configure_llm(self, llm: extraction.LLM) -> None:
        """Give tools that call Claude a client; every other tool works without one."""
        for tool in self.tools.values():
            tool.llm = llm

    def definitions(self) -> list[dict[str, Any]]:
        return [t.definition() for t in self.tools.values()]

    def execute(self, name: str, raw: dict[str, Any] | None) -> dict[str, Any]:
        if name not in self.tools:
            raise ToolError(f"unknown tool {name!r}; available: {sorted(self.tools)}")
        return self.tools[name].run(raw)
