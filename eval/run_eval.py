#!/usr/bin/env python3
"""Runs the full pipeline on every eval/sampleN.txt case and writes eval_report.md.

Usage (from the project root, with the venv active and ANTHROPIC_API_KEY set):
    python eval/run_eval.py

Case-level confidence = the highest per-claim confidence returned for that case
(see expected_results.md for why). Confidence rank for "highest": high > medium > low > none.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv  # noqa: E402
from langfuse import get_client  # noqa: E402
from langfuse.openai import OpenAI  # noqa: E402 - drop-in client: auto-traces every call, incl. cost

from run import run_pipeline  # noqa: E402
from run_logger import RunLogger  # noqa: E402

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

RANK = {"high": 3, "medium": 2, "low": 1, "none": 0}

EXPECTED = {
    1: "high",
    2: "high",
    3: "medium",
    4: "none",
    5: "medium",
}


def case_confidence(report) -> str:
    if not report.claims:
        return "none"
    best = max(report.claims, key=lambda c: RANK[c.confidence])
    return best.confidence


def main() -> int:
    eval_dir = Path(__file__).resolve().parent
    client = OpenAI()
    rows = []

    for case_num in sorted(EXPECTED):
        sample_path = eval_dir / f"sample{case_num}.txt"
        disclosure_text = sample_path.read_text(encoding="utf-8")
        logger = RunLogger(log_dir=str(eval_dir.parent / "logs"))
        print(f"--- Running case {case_num} ({sample_path.name}) --- log: {logger.path}")
        try:
            extraction, report = run_pipeline(disclosure_text, logger, client)
            summary = logger.finalize()
            actual = case_confidence(report)
            json_path = logger.path.with_suffix(".json")
            json_path.write_text(
                json.dumps(
                    {"extraction": extraction.model_dump(), "report": report.model_dump(), "summary": summary},
                    indent=2,
                ),
                encoding="utf-8",
            )
        except RuntimeError as e:
            logger.log_error(str(e))
            logger.finalize()
            actual = "ERROR"
            print(f"  case {case_num} failed: {e}")

        expected = EXPECTED[case_num]
        match = "✅" if actual == expected else "❌"
        rows.append((case_num, sample_path.name, expected, actual, match))
        print(f"  expected={expected} actual={actual} {match}")

    report_lines = [
        "# Eval Report — Generated After Running",
        "",
        "Compares each case's actual result against the hand-written expectation in "
        "`expected_results.md` (written before this pipeline was ever run).",
        "",
        "| Case | Sample | Expected Confidence | Actual Confidence | Match? |",
        "|---|---|---|---|---|",
    ]
    for case_num, name, expected, actual, match in rows:
        report_lines.append(f"| {case_num} | {name} | {expected} | {actual} | {match} |")

    passed = sum(1 for r in rows if r[4] == "✅")
    report_lines += ["", f"**{passed}/{len(rows)} cases matched expectation.**"]

    out_path = eval_dir / "eval_report.md"
    out_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(f"\nWrote {out_path}")

    get_client().flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
