#!/usr/bin/env python3
"""CLI entry point: full pipeline end to end.

    python run.py --disclosure eval/sample1.txt

Requires OPENAI_API_KEY in the environment (or a local .env file — loaded
automatically via python-dotenv). USPTO_ODP_API_KEY is optional (patent
search degrades gracefully to an empty/error result without it — see
tools.py and README for why).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from langfuse import get_client, observe
from langfuse.openai import OpenAI  # drop-in OpenAI client: auto-traces every call.create(), incl. token cost/latency

from extraction_agent import run_extraction
from research_agent import run_research
from run_logger import RunLogger
from schemas import ExtractionOutput, ResearchReport

load_dotenv()


@observe(name="prior_art_pipeline")
def run_pipeline(disclosure_text: str, logger: RunLogger, client: OpenAI) -> tuple[ExtractionOutput, ResearchReport]:
    extraction = run_extraction(disclosure_text, logger, client)
    report = run_research(extraction, logger, client)

    # Aggregate hallucination-rate score for the whole run: what fraction of citations the
    # model produced across all claims didn't correspond to a real tool result and were
    # stripped by the code-level guardrail in research_agent.py.
    langfuse = get_client()
    total_citation_checks = logger.citations_verified + logger.citations_stripped
    if total_citation_checks:
        langfuse.create_score(
            name="citation_hallucination_rate",
            value=logger.citations_stripped / total_citation_checks,
            trace_id=langfuse.get_current_trace_id(),
            data_type="NUMERIC",
            comment=f"{logger.citations_stripped}/{total_citation_checks} model-cited sources were not found in real tool results",
        )
    langfuse.update_current_span(metadata={"domain": extraction.domain, "num_claims": len(extraction.key_claims)})

    return extraction, report


def render_human_readable(extraction: ExtractionOutput, report: ResearchReport) -> str:
    lines = []
    lines.append(f"PRIOR-ART REPORT: {extraction.title}")
    lines.append(f"Domain: {extraction.domain}")
    lines.append(f"Keywords: {', '.join(extraction.keywords)}")
    lines.append("=" * 70)
    for i, claim_result in enumerate(report.claims, start=1):
        lines.append(f"\nClaim {i}: {claim_result.claim}")
        lines.append(f"Confidence: {claim_result.confidence.upper()}")
        if not claim_result.matches:
            lines.append("  No verified matches found.")
        for m in claim_result.matches:
            lines.append(f"  - [{m.source_type}] {m.title} ({m.relevance} relevance)")
            lines.append(f"      {m.id_or_url}")
            lines.append(f"      Reasoning: {m.reasoning}")
        lines.append(f"  Conclusion: {claim_result.conclusion}")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Invention disclosure prior-art research pipeline")
    parser.add_argument("--disclosure", required=True, help="Path to a raw invention disclosure text file")
    parser.add_argument("--log-dir", default="logs", help="Directory for the run log (default: logs)")
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY is not set in the environment (or in a local .env file).", file=sys.stderr)
        return 1

    disclosure_path = Path(args.disclosure)
    if not disclosure_path.exists():
        print(f"ERROR: disclosure file not found: {disclosure_path}", file=sys.stderr)
        return 1
    disclosure_text = disclosure_path.read_text(encoding="utf-8")

    logger = RunLogger(args.log_dir)
    client = OpenAI()

    try:
        extraction, report = run_pipeline(disclosure_text, logger, client)
    except RuntimeError as e:
        logger.log_error(str(e))
        logger.finalize()
        print(f"PIPELINE FAILED: {e}", file=sys.stderr)
        return 1
    finally:
        # Flush buffered traces/scores now — this is a short-lived CLI process,
        # not a long-running server, so nothing else will flush them on exit.
        get_client().flush()

    summary = logger.finalize()

    print(render_human_readable(extraction, report))
    print("\n" + "=" * 70)
    print(
        f"Run summary: {summary['llm_calls']} LLM calls, {summary['tool_calls']} tool calls, "
        f"~{summary['approx_tokens']} tokens, {summary['runtime_seconds']}s runtime, "
        f"{summary['citations_verified']} citations verified, {summary['citations_stripped']} stripped"
    )
    print(f"Run log: {logger.path}")

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = Path(args.log_dir) / f"run_{ts}.json"
    json_path.write_text(
        json.dumps(
            {
                "extraction": extraction.model_dump(),
                "report": report.model_dump(),
                "summary": summary,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Structured output: {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
