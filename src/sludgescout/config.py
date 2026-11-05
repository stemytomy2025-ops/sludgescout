"""Typed access to app_config.json, the single description of datasets, tools and model."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "app_config.json"
DATA_DIR = ROOT / "data" / "raw"
FALLBACK_BETA = "server-side-fallback-2026-07-01"

ColumnType = Literal["integer", "float", "boolean", "date", "datetime", "string"]
Operation = Literal[
    "filter_rows",
    "aggregate",
    "trend",
    "forecast",
    "outliers",
    "woe_iv",
    "scorecard_performance",
    "score_breakdown",
    "retrieve",
    "llm_extract",
    "evaluate",
]


class SourceConfig(BaseModel):
    kind: Literal["huggingface_parquet", "http_csv"]
    dataset: str | None = None  # Hugging Face dataset id
    config: str = "default"
    split: str = "train"
    url: str | None = None  # direct CSV URL
    max_mb: int = Field(200, ge=1)


class DatasetConfig(BaseModel):
    id: str
    name: str
    landing_page: str
    license: str
    attribution: str
    source: SourceConfig
    expected_schema: dict[str, ColumnType]
    expected_min_rows: int = Field(1, ge=1)
    limitations: list[str] = []
    synthetic: bool = False


class ToolConfig(BaseModel):
    name: str
    description: str
    operation: Operation
    dataset_id: str
    params: dict[str, Any] = {}


class AppConfig(BaseModel):
    title: str
    tagline: str
    model: str
    effort: Literal["low", "medium", "high", "xhigh", "max"] = "medium"
    max_turns: int = Field(6, ge=1, le=20)
    max_tool_calls: int = Field(8, ge=1, le=30)
    use_refusal_fallback: bool = True
    is_test_fixture: bool = False
    datasets: list[DatasetConfig]
    tools: list[ToolConfig]
    example_questions: list[str] = []
    # Audience-facing options: hide the "Run a tool" and "Evaluation" tabs, and show a
    # Markdown file (relative to the app root) in an "About this project" tab.
    show_developer_tabs: bool = True
    about_file: str | None = None

    def dataset(self, dataset_id: str) -> DatasetConfig:
        for ds in self.datasets:
            if ds.id == dataset_id:
                return ds
        raise KeyError(f"unknown dataset {dataset_id!r}")

    def about_text(self, root: Path = ROOT) -> str | None:
        if not self.about_file:
            return None
        path = root / self.about_file
        if not path.is_file():
            raise FileNotFoundError(f"about_file {self.about_file!r} not found under {root}")
        return path.read_text(encoding="utf-8")


def load_config(path: Path = CONFIG_PATH) -> AppConfig:
    return AppConfig.model_validate(json.loads(path.read_text(encoding="utf-8")))
