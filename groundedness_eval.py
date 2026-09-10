"""LLM-as-judge groundedness/faithfulness scorer.

Answers the "outputs remain traceable to verifiable source material" requirement:
for each surviving match on a ClaimResult, checks whether `Match.reasoning` is
actually supported by the real retrieved title/snippet, and whether `conclusion`
overstates what the matches support.

This is a different check than the code-level citation guardrail already in
research_agent.py. That guardrail verifies the *id_or_url itself* was really
returned by a tool call (catches invented sources). This judge verifies what the
model *said about* a real source is actually true of that source's content
(catches a real citation with a misrepresented/overstated reasoning attached to it).

Runs as an independent judge call (temperature 0, JSON-only) so it isn't the same
generation re-approving itself. Never raises: a judge failure returns None and the
pipeline proceeds without a groundedness score for that claim, rather than failing
the run over an evaluator problem.
"""

from __future__ import annotations

import json
import os

from openai import OpenAI

from schemas import ClaimResult, GroundednessJudgment

JUDGE_MODEL = os.environ.get("OPENAI_JUDGE_MODEL", "gpt-4o-mini")

JUDGE_SYSTEM_PROMPT = """You are a strict fact-checking judge. You will be given a claim, the prior-art \
matches an upstream research agent reported for it (each with the reasoning it gave), and the ACTUAL \
retrieved title+snippet for each match. Your only job is to check whether the reasoning is actually \
supported by the real snippet text, and whether the conclusion overstates what the matches support.

For each match, decide "grounded": true only if the reasoning's claims are actually backed by the real \
snippet content. If the snippet is empty/thin and the reasoning asserts specifics the snippet doesn't \
contain, grounded=false. Be skeptical, not charitable — a plausible-sounding reasoning is not the same \
as a grounded one.

Output ONLY a single JSON object, no prose, no markdown fences, matching exactly this shape:
{
  "verdicts": [{"id_or_url": "...", "grounded": true, "justification": "one sentence"}],
  "conclusion_overstates": false
}
"""


def _extract_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    return json.loads(text)


def judge_groundedness(
    claim_result: ClaimResult,
    retrieved_by_id: dict[str, dict],
    client: OpenAI,
) -> GroundednessJudgment | None:
    """Returns None if there's nothing to judge (no surviving matches) or the judge call fails."""
    if not claim_result.matches:
        return None

    matches_payload = [
        {
            "id_or_url": m.id_or_url,
            "agent_reasoning": m.reasoning,
            "actual_retrieved_title": retrieved_by_id.get(m.id_or_url, {}).get("title", "(not found in retrieval log)"),
            "actual_retrieved_snippet": retrieved_by_id.get(m.id_or_url, {}).get(
                "snippet", "(not found in retrieval log)"
            ),
        }
        for m in claim_result.matches
    ]

    user_payload = {
        "claim": claim_result.claim,
        "conclusion": claim_result.conclusion,
        "confidence": claim_result.confidence,
        "matches": matches_payload,
    }

    try:
        resp = client.chat.completions.create(
            model=JUDGE_MODEL,
            max_tokens=1024,
            temperature=0.0,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(user_payload)},
            ],
        )
        raw_text = resp.choices[0].message.content or ""
        data = _extract_json(raw_text)
        return GroundednessJudgment.model_validate(data)
    except Exception:  # noqa: BLE001 - judge must never crash the pipeline
        return None


def groundedness_score(judgment: GroundednessJudgment) -> float:
    """Fraction of matches whose reasoning was actually grounded in the real snippet, 0.0-1.0."""
    if not judgment.verdicts:
        return 1.0
    grounded_count = sum(1 for v in judgment.verdicts if v.grounded)
    return grounded_count / len(judgment.verdicts)
