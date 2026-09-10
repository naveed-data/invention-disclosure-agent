"""Agent 1 — Extraction Agent.

Single LLM call. No tools, no retrieval — everything needed is already in
the disclosure text. Low temperature for consistent, near-deterministic
structuring. Output is pydantic-validated with one retry on failure before
hard-failing with a clear error (bad data is never silently passed on).
"""

from __future__ import annotations

import json
import os

from langfuse import observe
from openai import OpenAI
from pydantic import ValidationError

from run_logger import RunLogger
from schemas import DOMAIN_TAXONOMY, ExtractionOutput

MODEL = os.environ.get("OPENAI_EXTRACTION_MODEL", "gpt-4o-mini")  # cheap/fast: structuring is near-deterministic

SYSTEM_PROMPT = f"""You are the Extraction Agent in a prior-art research pipeline for invention disclosures.

Read the raw invention disclosure text below and extract a structured summary. Do not invent facts \
that are not stated or clearly implied by the text. Output ONLY a single JSON object, no prose, no \
markdown code fences, matching exactly this shape:

{{
  "title": "short descriptive title of the invention",
  "domain": "best-fit technical/therapeutic domain",
  "key_claims": ["discrete, atomic claim 1", "discrete, atomic claim 2", "..."],
  "keywords": ["search-ready keyword or phrase", "..."]
}}

Guidance:
- "key_claims" should split the invention into 1-4 independently-searchable claims (e.g. the core \
mechanism, a specific component, a novel combination) — not restate the whole disclosure as one claim.
- "keywords" should be terms suitable for literature/patent search engines (technical terms, not \
marketing language).
- Prefer, when it reasonably fits, one of these domain categories: {", ".join(DOMAIN_TAXONOMY)}. If \
none fit well, use the most accurate short domain label instead of forcing a bad fit.
- Output valid JSON only.
"""


def _extract_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    return json.loads(text)


@observe(name="extraction")
def run_extraction(disclosure_text: str, logger: RunLogger, client: OpenAI) -> ExtractionOutput:
    """Runs Agent 1 end to end: LLM call -> parse -> validate -> (retry once) -> hard fail."""

    input_summary = disclosure_text.strip().replace("\n", " ")[:100]

    def _call(extra_user_note: str | None = None) -> tuple[str, int]:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": disclosure_text}]
        if extra_user_note:
            messages.append({"role": "user", "content": extra_user_note})
        resp = client.chat.completions.create(
            model=MODEL,
            max_tokens=1024,
            temperature=0.0,
            messages=messages,
        )
        approx_tokens = resp.usage.prompt_tokens + resp.usage.completion_tokens
        return resp.choices[0].message.content or "", approx_tokens

    raw_text, tokens = _call()
    try:
        data = _extract_json(raw_text)
        result = ExtractionOutput.model_validate(data)
        logger.log_llm_call("extraction", input_summary, "pass", tokens)
        return result
    except (json.JSONDecodeError, ValidationError) as first_error:
        logger.log_llm_call("extraction", input_summary, f"fail: {first_error}", tokens)
        logger.log_decision(0, "Agent 1 output failed validation, retrying once with error appended")

        retry_note = (
            "Your previous output failed schema validation with this error:\n"
            f"{first_error}\n\nRe-emit ONLY the corrected JSON object, matching the required shape exactly."
        )
        raw_text_2, tokens_2 = _call(retry_note)
        try:
            data_2 = _extract_json(raw_text_2)
            result = ExtractionOutput.model_validate(data_2)
            logger.log_llm_call("extraction", input_summary, "pass (retry)", tokens_2)
            return result
        except (json.JSONDecodeError, ValidationError) as second_error:
            logger.log_llm_call("extraction", input_summary, f"fail (retry): {second_error}", tokens_2)
            logger.log_error(f"Agent 1 hard-failed after retry: {second_error}")
            raise RuntimeError(
                "Extraction Agent failed schema validation twice. Refusing to pass unvalidated data "
                f"downstream. Last error: {second_error}"
            ) from second_error
