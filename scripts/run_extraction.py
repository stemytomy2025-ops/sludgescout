"""Run the LLM extractor over a capped holdout sample and save predictions.

This calls the Claude API once per sentence, so it costs tokens. The score is then read with
the evaluate tool in the app.

Usage: python scripts/run_extraction.py --limit 50
"""

from __future__ import annotations

import argparse
import sys

import anthropic

from sludgescout import extraction
from sludgescout import operations as ops
from sludgescout.config import load_config
from sludgescout.data import DataError, load_dataset
from sludgescout.tools import ToolRegistry


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=50, help="Holdout sentences to run.")
    args = parser.parse_args()

    config = load_config()
    try:
        frames = {ds.id: load_dataset(ds) for ds in config.datasets}
    except DataError as exc:
        print(f"ERROR: {exc}")
        return 1
    registry = ToolRegistry.from_config(config, frames)
    tool = next((t for t in registry.tools.values() if t.cfg.operation == "llm_extract"), None)
    if tool is None:
        print("ERROR: this app has no llm_extract tool")
        return 1
    llm = extraction.LLM.from_config(anthropic.Anthropic(), config)
    registry.configure_llm(llm)

    p = tool.cfg.params
    ids = sorted(str(v) for v in tool.df[p["id"]])
    share = p.get("holdout_share", 0.2)
    holdout = [i for i in ids if ops.is_holdout(i, share)][: args.limit]
    predictions: dict[str, dict] = {}
    failures: dict[str, str] = {}
    usage = {"input_tokens": 0, "output_tokens": 0}
    for n, example_id in enumerate(holdout, 1):
        try:
            result = tool.run({"example_id": example_id})
        except Exception as exc:  # one failed sentence must not lose the whole run
            failures[example_id] = f"{type(exc).__name__}: {exc}"
            print(f"{n}/{len(holdout)} {example_id}: FAILED {failures[example_id]}")
            continue
        predictions[example_id] = result["rule"]
        usage["input_tokens"] += result["input_tokens"]
        usage["output_tokens"] += result["output_tokens"]
        print(f"{n}/{len(holdout)} {example_id}: ok")
    extraction.save_predictions(predictions, failures, llm, usage)
    print(
        f"{len(predictions)} saved, {len(failures)} failed. Tokens: "
        f"{usage['input_tokens']:,} in / {usage['output_tokens']:,} out"
    )
    return 1 if failures and not predictions else 0


if __name__ == "__main__":
    sys.exit(main())
