"""LLM extraction of entities and relations into validated structures.

The response is constrained by a JSON schema (structured outputs), then validated again with
Pydantic. Every call reports whether each span really occurs in the sentence, because a
schema-valid answer can still be invented. Output is a model's reading of the text, not a
compliance judgement.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ValidationError, create_model

from sludgescout import operations as ops
from sludgescout.config import FALLBACK_BETA, ROOT, AppConfig

PROMPT_VERSION = "v1"
PREDICTIONS_PATH = ROOT / "evals" / "predictions.json"


@dataclass
class LLM:
    client: Any
    model: str
    effort: str
    use_refusal_fallback: bool = True

    @classmethod
    def from_config(cls, client: Any, config: AppConfig) -> LLM:
        return cls(client, config.model, config.effort, config.use_refusal_fallback)


def rule_model(entity_types: Sequence[str], relation_types: Sequence[str]) -> type[BaseModel]:
    entity_kind: Any = Literal[tuple(entity_types)]  # type: ignore[valid-type]
    relation_kind: Any = Literal[tuple(relation_types)]  # type: ignore[valid-type]
    entity = create_model("Entity", text=(str, ...), type=(entity_kind, ...))
    relation = create_model(
        "Relation", subject=(str, ...), object=(str, ...), type=(relation_kind, ...)
    )
    return create_model(
        "ExtractedRule", entities=(list[entity], ...), relations=(list[relation], ...)
    )


def _closed(schema: Any) -> Any:
    """Structured outputs require every object to forbid extra properties."""
    if isinstance(schema, dict):
        if schema.get("type") == "object":
            schema["additionalProperties"] = False
        for value in schema.values():
            _closed(value)
    elif isinstance(schema, list):
        for value in schema:
            _closed(value)
    return schema


def to_tuples(rule: dict[str, Any]) -> dict[str, set[tuple[str, ...]]]:
    """A rule in the same comparable form as the human annotations."""
    return {
        "entities": {(e["type"], ops.normalise(e["text"])) for e in rule["entities"]},
        "relations": {
            (r["type"], ops.normalise(r["subject"]), ops.normalise(r["object"]))
            for r in rule["relations"]
        },
    }


def system_prompt(
    entity_types: Sequence[str],
    relation_types: Sequence[str],
    examples: Sequence[dict[str, Any]] = (),
) -> str:
    lines = [
        "Extract structured knowledge from one sentence of a regulation.",
        f"Entity types: {', '.join(entity_types)}.",
        f"Relation types: {', '.join(relation_types)}.",
        "Copy each entity's text exactly from the sentence. A relation's subject and object "
        "must be entity texts. Omit anything the sentence does not state.",
        "Return empty lists when there is nothing to extract.",
    ]
    for example in examples:
        lines += ["", "Sentence: " + example["sentence"], "Output: " + json.dumps(example["rule"])]
    return "\n".join(lines)


def extract(
    llm: LLM,
    sentence: str,
    entity_types: Sequence[str],
    relation_types: Sequence[str],
    examples: Sequence[dict[str, Any]] = (),
) -> dict[str, Any]:
    model = rule_model(entity_types, relation_types)
    params: dict[str, Any] = {
        "model": llm.model,
        "max_tokens": 16000,
        "system": system_prompt(entity_types, relation_types, examples),
        "messages": [{"role": "user", "content": sentence}],
        "output_config": {
            "effort": llm.effort,
            "format": {"type": "json_schema", "schema": _closed(model.model_json_schema())},
        },
    }
    if llm.use_refusal_fallback:
        response = llm.client.beta.messages.create(
            **params, betas=[FALLBACK_BETA], fallbacks="default"
        )
    else:
        response = llm.client.messages.create(**params)
    if response.stop_reason == "refusal":
        raise ops.OperationError("the model declined to extract from this sentence")
    if response.stop_reason == "max_tokens":
        raise ops.OperationError("the extraction hit the max_tokens limit")
    text = next((b.text for b in response.content if b.type == "text"), "")
    try:
        rule = model.model_validate_json(text).model_dump()
    except ValidationError as exc:
        raise ops.OperationError(f"the model returned an invalid rule: {exc}") from exc
    source = ops.normalise(sentence)
    usage = getattr(response, "usage", None)
    return {
        "operation": "llm_extract",
        "sentence": sentence,
        "rule": rule,
        "spans_not_in_sentence": [
            e["text"] for e in rule["entities"] if ops.normalise(e["text"]) not in source
        ],
        "model": llm.model,
        "prompt_version": PROMPT_VERSION,
        "input_tokens": getattr(usage, "input_tokens", 0) or 0,
        "output_tokens": getattr(usage, "output_tokens", 0) or 0,
        "note": "Model output, not a compliance judgement.",
    }


def save_predictions(
    predictions: dict[str, dict[str, Any]],
    failures: dict[str, str],
    llm: LLM,
    usage: dict[str, int],
    path: Path | None = None,
) -> None:
    payload = {
        "run_at": datetime.now(UTC).isoformat(),
        "model": llm.model,
        "effort": llm.effort,
        "prompt_version": PROMPT_VERSION,
        "usage": usage,
        "failures": failures,
        "predictions": predictions,
    }
    target = path or PREDICTIONS_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_predictions(path: Path | None = None) -> dict[str, Any]:
    target = path or PREDICTIONS_PATH
    if not target.is_file():
        raise ops.OperationError(
            "no saved predictions yet. Run `python scripts/run_extraction.py` (uses API tokens), "
            "or use source='gold_self_check' to test the scorer."
        )
    return json.loads(target.read_text(encoding="utf-8"))
