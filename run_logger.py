"""Structured, human-readable run log shared by run.py and both agents.

One instance per pipeline run. Writes to logs/run_<timestamp>.log as events
happen (not buffered to the end) so the file is inspectable mid-run, and
also keeps counters for the final observability summary.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path


class RunLogger:
    def __init__(self, log_dir: str = "logs"):
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.path = Path(log_dir) / f"run_{ts}.log"
        self._fh = open(self.path, "a", encoding="utf-8")
        self.start_time = time.monotonic()

        self.llm_calls = 0
        self.tool_calls = 0
        self.approx_tokens = 0
        self.citations_verified = 0
        self.citations_stripped = 0

        self._write("RUN START", ts)

    def _write(self, tag: str, message: str) -> None:
        line = f"[{datetime.now(timezone.utc).isoformat(timespec='seconds')}] {tag}: {message}"
        self._fh.write(line + "\n")
        self._fh.flush()

    def log_llm_call(self, agent: str, input_summary: str, validation: str, approx_tokens: int = 0) -> None:
        self.llm_calls += 1
        self.approx_tokens += approx_tokens
        self._write(
            "LLM_CALL",
            f"agent={agent} input='{input_summary}' validation={validation} approx_tokens={approx_tokens}",
        )

    def log_tool_call(self, claim_idx: int, tool_name: str, query: str, source: str, raw_count: int, error: str | None) -> None:
        self.tool_calls += 1
        status = f"error={error}" if error else "ok"
        self._write(
            "TOOL_CALL",
            f"claim={claim_idx} tool={tool_name} query='{query}' source={source} results={raw_count} {status}",
        )

    def log_decision(self, claim_idx: int, message: str) -> None:
        self._write("DECISION", f"claim={claim_idx} {message}")

    def log_citation_check(self, claim_idx: int, id_or_url: str, verified: bool) -> None:
        if verified:
            self.citations_verified += 1
        else:
            self.citations_stripped += 1
        self._write(
            "CITATION_CHECK",
            f"claim={claim_idx} id_or_url='{id_or_url}' verified={verified}",
        )

    def log_error(self, message: str) -> None:
        self._write("ERROR", message)

    def finalize(self) -> dict:
        runtime = time.monotonic() - self.start_time
        summary = {
            "llm_calls": self.llm_calls,
            "tool_calls": self.tool_calls,
            "approx_tokens": self.approx_tokens,
            "runtime_seconds": round(runtime, 2),
            "citations_verified": self.citations_verified,
            "citations_stripped": self.citations_stripped,
        }
        self._write(
            "RUN_SUMMARY",
            f"llm_calls={summary['llm_calls']} tool_calls={summary['tool_calls']} "
            f"approx_tokens={summary['approx_tokens']} runtime_seconds={summary['runtime_seconds']} "
            f"citations_verified={summary['citations_verified']} citations_stripped={summary['citations_stripped']}",
        )
        self._fh.close()
        return summary
