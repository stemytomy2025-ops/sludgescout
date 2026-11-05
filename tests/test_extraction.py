"""Retrieval, gold parsing, scoring and extraction on small SYNTHETIC frames; no network."""

from __future__ import annotations

import json

import pandas as pd
import pytest
from conftest import FakeClient, response, text

from sludgescout import extraction
from sludgescout import operations as ops
from sludgescout.config import ToolConfig
from sludgescout.tools import Tool, ToolError

# SYNTHETIC sentences and annotations, written for these tests.
ENTITIES = pd.DataFrame(
    {
        "example_id": ["s1", "s2"],
        "content": ["A door must be 800 mm wide.", "The lift shall serve each level."],
        "processed_content": ["A door must be 800 mm wide .", "The lift shall serve each level ."],
        "label": [
            "O B-object O O B-value I-value B-property O",
            "O B-object O B-quality I-quality I-quality O",
        ],
    }
)
RELATIONS = pd.DataFrame(
    {
        "example_id": ["s1", "s1", "s2"],
        "tagged_sentence": [
            "A door must be <e1>800 mm</e1> <e2>wide</e2>.",
            "A <e2>door</e2> must be <e1>800 mm</e1> wide.",
            "The <e1>lift</e1> shall <e2>serve each level</e2>.",
        ],
        "relation_type": ["greater-equal", "none", "necessity"],
    }
)
ENTITY_COLUMNS = {"id": "example_id", "tokens": "processed_content", "labels": "label"}
RELATION_COLUMNS = {"id": "example_id", "tagged": "tagged_sentence", "type": "relation_type"}
PARAMS = {
    "id": "example_id",
    "text": "content",
    "entity_types": ["object", "property", "quality", "value"],
    "relation_types": ["greater-equal", "necessity"],
    "relations_dataset": "rel",
    "entity_columns": ENTITY_COLUMNS,
    "relation_columns": RELATION_COLUMNS,
    "holdout_share": 1.0,
}


def gold():
    return ops.gold_rules(ENTITIES, RELATIONS, ENTITY_COLUMNS, RELATION_COLUMNS)


def make_tool(operation: str) -> Tool:
    cfg = ToolConfig(
        name="tool", description="d", operation=operation, dataset_id="ent", params=PARAMS
    )
    return Tool(cfg, ENTITIES, {"rel": RELATIONS})


def llm(*responses) -> extraction.LLM:
    return extraction.LLM(FakeClient(*responses), "test-model", "low", False)


def test_search_ranks_matching_rows_and_omits_non_matches():
    out = ops.search_text(ENTITIES, ["content"], ["example_id"], "lift level")
    assert [r["example_id"] for r in out["rows"]] == ["s2"]
    assert out["matched"] == 1


def test_derive_label_uses_first_matching_pattern():
    series = pd.Series(["1_UK_DocA", "2_Finnish_x", "z"])
    out = ops.derive_label(series, {"UK": "_UK_", "FI": "Fin"})
    assert out.tolist() == ["UK", "FI", "other"]


def test_bio_spans_and_stray_inside_tags():
    spans = ops.bio_entities(["a", "b", "c", "d"], ["B-x", "I-x", "O", "I-y"])
    assert spans == {("x", "a b"), ("y", "d")}


def test_gold_excludes_none_relations_and_reads_both_tag_orders():
    assert gold()["s1"]["relations"] == {("greater-equal", "800 mm", "wide")}
    assert ops.tagged_relation("<e2>b</e2> then <e1>a</e1>", "t") == ("t", "a", "b")


def test_score_hand_computed():
    predicted = {"s1": {"entities": {("object", "door"), ("value", "9 mm")}, "relations": set()}}
    out = ops.score_extraction({"s1": gold()["s1"]}, predicted)
    exact = out["entities"]["exact"]
    # 1 of 2 predicted is right; gold has door, 800 mm, wide.
    assert (exact["true_positives"], exact["predicted"], exact["actual"]) == (1, 2, 3)
    assert exact["precision"] == 0.5
    assert exact["recall"] == pytest.approx(0.3333)
    assert exact["f1"] == 0.4
    assert out["relations"]["exact"]["recall"] == 0.0
    assert out["examples_scored"] == 1


def test_partial_match_needs_same_type_and_half_the_tokens():
    assert ops._partial_tp({("x", "a b")}, {("x", "a b c d")}) == 1
    assert ops._partial_tp({("x", "a")}, {("x", "a b c")}) == 0
    assert ops._partial_tp({("y", "a b")}, {("x", "a b")}) == 0


def test_self_check_is_perfect_and_labelled_not_a_model_result():
    out = make_tool("evaluate").run({"source": "gold_self_check"})
    assert out["is_model_result"] is False
    assert out["entities"]["exact"]["f1"] == 1.0
    assert out["relations"]["exact"]["f1"] == 1.0


def test_saved_scoring_uses_predictions_and_is_labelled(tmp_path, monkeypatch):
    monkeypatch.setattr(extraction, "PREDICTIONS_PATH", tmp_path / "p.json")
    with pytest.raises(ToolError, match="run_extraction"):
        make_tool("evaluate").run({})
    rule = {"entities": [{"text": "lift", "type": "object"}], "relations": []}
    extraction.save_predictions({"s2": rule}, {}, llm(), {"input_tokens": 1})
    out = make_tool("evaluate").run({})
    assert out["is_model_result"] is True
    assert out["run"]["model"] == "test-model"
    assert out["examples_scored"] == 1
    assert out["examples_available"] == 2
    assert out["entities"]["exact"]["precision"] == 1.0


def test_extract_validates_and_flags_invented_spans():
    tool = make_tool("llm_extract")
    rule = {
        "entities": [{"text": "door", "type": "object"}, {"text": "window", "type": "object"}],
        "relations": [],
    }
    tool.llm = llm(response(text(json.dumps(rule))))
    out = tool.run({"example_id": "s1"})
    assert out["rule"] == rule
    assert out["spans_not_in_sentence"] == ["window"]
    request = tool.llm.client.messages.calls[0]
    assert request["output_config"]["format"]["type"] == "json_schema"
    assert "tool_choice" not in request
    assert request["messages"][0]["content"] == "A door must be 800 mm wide."


def test_few_shot_examples_come_from_outside_the_holdout():
    tool = make_tool("llm_extract")
    assert tool.few_shot_examples() == []  # holdout_share 1.0: everything is held out
    tool.cfg.params = PARAMS | {"holdout_share": 0.0}
    sentences = [e["sentence"] for e in tool.few_shot_examples()]
    assert "A door must be 800 mm wide." in sentences


def test_extract_rejects_types_outside_the_schema():
    tool = make_tool("llm_extract")
    bad = {"entities": [{"text": "door", "type": "invented"}], "relations": []}
    tool.llm = llm(response(text(json.dumps(bad))))
    with pytest.raises(ToolError, match="invalid rule"):
        tool.run({"example_id": "s1"})


@pytest.mark.parametrize(
    ("stop_reason", "message"), [("refusal", "declined"), ("max_tokens", "max_tokens")]
)
def test_extract_reports_refusal_and_truncation(stop_reason, message):
    tool = make_tool("llm_extract")
    tool.llm = llm(response(text(""), stop_reason=stop_reason))
    with pytest.raises(ToolError, match=message):
        tool.run({"sentence": "x"})


def test_extract_needs_a_client_and_exactly_one_input():
    tool = make_tool("llm_extract")
    with pytest.raises(ToolError, match="ANTHROPIC_API_KEY"):
        tool.run({"sentence": "x"})
    tool.llm = llm()
    with pytest.raises(ToolError, match="exactly one"):
        tool.run({"sentence": "x", "example_id": "s1"})
    with pytest.raises(ToolError, match="no sentence"):
        tool.run({"example_id": "missing"})
