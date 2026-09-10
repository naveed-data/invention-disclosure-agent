#!/usr/bin/env python3
"""Regression/correctness harness on Langfuse Datasets + Experiments.

This is the Langfuse-native counterpart to run_eval.py. Instead of overwriting a
single local eval_report.md on every run, each invocation is a named "experiment
run" against a persistent Langfuse dataset ("prior-art-eval-cases" — the same 5
hand-written invention disclosures and expected confidence labels as
expected_results.md). Successive runs (e.g. after a prompt or model change) show
up side by side in the Langfuse UI for comparison — this is the "regression test"
the JD asks for made concrete and versioned, rather than a markdown file that gets
replaced each time.

Every case also runs the real pipeline, so it picks up the same per-claim
groundedness and citation-hallucination scores that run.py attaches during a
normal run (see research_agent.py / run_pipeline in run.py) — a "correctness"
regression run and a "groundedness" regression run are the same run here, not two
separate systems.

Usage (from the project root, with the venv active and OPENAI_API_KEY + LANGFUSE_*
set):
    python eval/langfuse_eval.py --run-name "baseline"
    python eval/langfuse_eval.py --run-name "gpt-4o-mini research model"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv  # noqa: E402
from langfuse import get_client  # noqa: E402
from langfuse.openai import OpenAI  # noqa: E402

from run import run_pipeline  # noqa: E402
from run_logger import RunLogger  # noqa: E402

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

DATASET_NAME = "prior-art-eval-cases"
RANK = {"high": 3, "medium": 2, "low": 1, "none": 0}

EXPECTED = {
    1: "high",
    2: "high",
    3: "medium",
    4: "none",
    5: "medium",
}


def _case_confidence(report) -> str:
    if not report.claims:
        return "none"
    best = max(report.claims, key=lambda c: RANK[c.confidence])
    return best.confidence


def ensure_dataset(langfuse) -> None:
    """Creates the dataset + its 5 items on first run; a no-op on every run after that."""
    try:
        langfuse.get_dataset(DATASET_NAME)
        return
    except Exception:  # noqa: BLE001 - "dataset not found" isn't a typed exception in the SDK
        pass

    eval_dir = Path(__file__).resolve().parent
    langfuse.create_dataset(
        name=DATASET_NAME,
        description="5 hand-written invention disclosures with expected confidence labels (see expected_results.md)",
    )
    for case_num, expected in EXPECTED.items():
        sample_path = eval_dir / f"sample{case_num}.txt"
        langfuse.create_dataset_item(
            dataset_name=DATASET_NAME,
            input=sample_path.read_text(encoding="utf-8"),
            expected_output=expected,
            metadata={"case_num": case_num, "sample_file": sample_path.name},
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the prior-art eval suite as a Langfuse dataset experiment")
    parser.add_argument(
        "--run-name", required=True, help="Label for this experiment run, e.g. a prompt or model version under test"
    )
    args = parser.parse_args()

    langfuse = get_client()
    ensure_dataset(langfuse)
    dataset = langfuse.get_dataset(DATASET_NAME)

    client = OpenAI()
    log_dir = Path(__file__).resolve().parent.parent / "logs"

    def task(*, item, **kwargs):
        logger = RunLogger(log_dir=str(log_dir))
        try:
            _extraction, report = run_pipeline(item.input, logger, client)
        except RuntimeError as e:
            logger.log_error(str(e))
            logger.finalize()
            return "ERROR"
        logger.finalize()
        return _case_confidence(report)

    def correctness_evaluator(*, input, output, expected_output=None, **kwargs):  # noqa: A002
        return {
            "name": "correctness",
            "value": 1.0 if output == expected_output else 0.0,
            "comment": f"expected={expected_output} actual={output}",
        }

    result = dataset.run_experiment(
        name=args.run_name,
        description="Regression suite: 5 fixed invention disclosures vs. hand-written expected confidence",
        task=task,
        evaluators=[correctness_evaluator],
    )
    print(result.format())
    langfuse.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
