"""Agent 2 — Prior-Art Research Agent.

This is the agentic loop + RAG component. For each key_claim extracted by
Agent 1, the model decides which tool(s) to call, inspects real results,
judges relevance, and either refines its query, tries another tool, or
concludes — capped at 3 tool calls per claim (cost/latency guardrail).

After the model produces its structured per-claim answer, a code-level
guardrail cross-checks every `id_or_url` against the tool results actually
logged for that claim during this run. Anything that doesn't match a real
logged result is stripped before it reaches the final report — hallucinated
citations are especially dangerous in a legal/IP context, so this is
enforced in code, not just prompted for.
"""

from __future__ import annotations

import json
import os

from langfuse import get_client, observe
from openai import OpenAI
from pydantic import ValidationError

from groundedness_eval import groundedness_score, judge_groundedness
from run_logger import RunLogger
from schemas import ClaimResult, ExtractionOutput, ResearchReport
from tools import TOOL_REGISTRY

MODEL = os.environ.get("OPENAI_RESEARCH_MODEL", "gpt-4o")  # open-ended tool selection + judgment needs a stronger model
MAX_TOOL_CALLS_PER_CLAIM = 3
MAX_ROUNDS_PER_CLAIM = 4  # backstop; the tool-call budget above is what actually governs stopping

TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "search_papers",
            "description": "Search academic literature (Semantic Scholar) for papers relevant to a query.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query, technical terms only"},
                    "limit": {"type": "integer", "description": "Max results, default 5"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_pubmed",
            "description": "Search biomedical literature (NCBI PubMed) for papers relevant to a query.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query, technical terms only"},
                    "limit": {"type": "integer", "description": "Max results, default 5"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_patents",
            "description": (
                "Search USPTO patent applications for a query. May return an error/empty result if no "
                "API key is configured for this environment — treat that as 'tool unavailable', never as "
                "evidence that no prior patent art exists."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query, technical terms only"},
                    "limit": {"type": "integer", "description": "Max results, default 5"},
                },
                "required": ["query"],
            },
        },
    },
]

SYSTEM_PROMPT = """You are the Prior-Art Research Agent inside a DIMS-style invention-disclosure pipeline.

You are researching ONE specific claim extracted from an invention disclosure. Your job is to \
determine whether prior art (a patent or published paper) already discloses this specific claim, \
using the search tools available to you.

Process (this is an agentic loop, not a fixed script):
1. Call exactly ONE tool per turn -- never multiple tools in the same turn. Craft a targeted query \
(don't just repeat the claim verbatim -- extract the technical core of it), and pick the single most \
promising tool for this step.
2. Inspect the ACTUAL returned titles/abstracts before doing anything else. Judge relevance based on \
that real content, not on keyword overlap alone.
3. Based on what you just saw, decide your next move: refine the query and search again, try a \
different tool, or conclude now. State that reasoning briefly before your next tool call (e.g. "the \
first search returned 0 relevant results, trying PubMed with a narrower query").
4. You have a hard budget of 3 tool calls total for this claim, spent one at a time across up to 3 \
turns. Don't spend a call just to rephrase trivially -- only call again if the previous result was \
insufficient to judge relevance.

When you stop calling tools (by choice or because your budget is exhausted), you will be asked to \
produce a final structured judgment. At that point:
- Only cite sources that actually appeared in your tool results in this conversation. Never invent a \
title, id, or URL.
- If nothing relevant was found, that is a valid and important answer: confidence "none" with an \
empty matches list. Do not fabricate a marginal match just to avoid an empty result.
- Your reasoning for each match must reference what the retrieved title/abstract actually says, and \
your conclusion must not overstate what the matches actually support.
"""

FINAL_ANSWER_INSTRUCTION = """Your tool budget for this claim is now used up (or you chose to stop searching).

Based ONLY on the actual tool results above, output ONLY a single JSON object (no prose, no markdown \
fences) in exactly this shape:

{{
  "claim": "{claim}",
  "matches": [
    {{"source_type": "patent|paper", "title": "...", "id_or_url": "...", "relevance": "high|medium|low", "reasoning": "..."}}
  ],
  "confidence": "high|medium|low|none",
  "conclusion": "one sentence, must not overstate what the matches actually support"
}}

Every id_or_url must be copied exactly from a tool result you actually received above. If you found \
nothing relevant, use confidence "none" and matches: [].
"""


def _extract_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    return json.loads(text)


def _usage_tokens(resp) -> int:
    return resp.usage.prompt_tokens + resp.usage.completion_tokens


def _assistant_message_dict(msg) -> dict:
    d = {"role": "assistant", "content": msg.content}
    if msg.tool_calls:
        d["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.function.name, "arguments": tc.function.arguments},
            }
            for tc in msg.tool_calls
        ]
    return d


@observe(name="research_claim")
def _research_single_claim(
    claim: str,
    extraction: ExtractionOutput,
    claim_idx: int,
    logger: RunLogger,
    client: OpenAI,
) -> ClaimResult:
    context = (
        f"Invention title: {extraction.title}\n"
        f"Domain: {extraction.domain}\n"
        f"Keywords: {', '.join(extraction.keywords)}\n\n"
        f"Claim to research: {claim}"
    )
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": context}]
    valid_ids: set[str] = set()
    retrieved_by_id: dict[str, dict] = {}  # id_or_url -> {"title", "snippet"} for groundedness judging
    tool_calls_used = 0

    for round_idx in range(1, MAX_ROUNDS_PER_CLAIM + 1):
        resp = client.chat.completions.create(
            model=MODEL,
            max_tokens=1024,
            temperature=0.2,
            tools=TOOLS_SCHEMA,
            messages=messages,
        )
        logger.log_llm_call(
            "research",
            f"claim {claim_idx} round {round_idx}: {claim[:60]}",
            "n/a (tool round)",
            _usage_tokens(resp),
        )
        msg = resp.choices[0].message
        messages.append(_assistant_message_dict(msg))

        if msg.content:
            logger.log_decision(claim_idx, f"model reasoning: {msg.content.strip()[:300]}")

        tool_calls = msg.tool_calls or []
        if not tool_calls:
            logger.log_decision(
                claim_idx, f"model concluded research after {tool_calls_used} tool call(s), no further tool use"
            )
            break

        for tc in tool_calls:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}

            if tool_calls_used >= MAX_TOOL_CALLS_PER_CLAIM:
                logger.log_decision(
                    claim_idx,
                    f"skipped tool call '{tc.function.name}' - per-claim budget ({MAX_TOOL_CALLS_PER_CLAIM}) exhausted",
                )
                output = {"error": "tool-call budget exhausted for this claim", "results": [], "raw_count": 0}
            else:
                query = args.get("query", "")
                limit = args.get("limit", 5)
                fn = TOOL_REGISTRY.get(tc.function.name)
                if fn is None:
                    output = {
                        "error": f"unknown tool {tc.function.name}",
                        "results": [],
                        "raw_count": 0,
                        "source": tc.function.name,
                    }
                else:
                    output = fn(query, limit)
                    tool_calls_used += 1
                logger.log_tool_call(
                    claim_idx,
                    tc.function.name,
                    query,
                    output.get("source", tc.function.name),
                    output.get("raw_count", 0),
                    output.get("error"),
                )
                for r in output.get("results", []):
                    valid_ids.add(r["id_or_url"])
                    retrieved_by_id[r["id_or_url"]] = {"title": r.get("title", ""), "snippet": r.get("snippet", "")}
                if output.get("raw_count", 0) == 0:
                    logger.log_decision(claim_idx, f"0 results from {tc.function.name} for query '{query}'")

            messages.append({"role": "tool", "tool_call_id": tc.id, "content": json.dumps(output)})

        if tool_calls_used >= MAX_TOOL_CALLS_PER_CLAIM:
            logger.log_decision(claim_idx, f"tool-call budget ({MAX_TOOL_CALLS_PER_CLAIM}) reached, forcing conclusion")
            break

    # Final structured-answer step (no tools) with one retry on schema failure.
    messages.append({"role": "user", "content": FINAL_ANSWER_INSTRUCTION.format(claim=claim.replace('"', "'"))})

    def _final_call(extra_note: str | None = None) -> tuple[str, int]:
        msgs = list(messages)
        if extra_note:
            msgs.append({"role": "user", "content": extra_note})
        resp = client.chat.completions.create(model=MODEL, max_tokens=1024, temperature=0.0, messages=msgs)
        text = resp.choices[0].message.content or ""
        return text, _usage_tokens(resp)

    raw_text, tokens = _final_call()
    try:
        data = _extract_json(raw_text)
        claim_result = ClaimResult.model_validate(data)
        logger.log_llm_call("research", f"claim {claim_idx} final answer", "pass", tokens)
    except (json.JSONDecodeError, ValidationError) as first_error:
        logger.log_llm_call("research", f"claim {claim_idx} final answer", f"fail: {first_error}", tokens)
        logger.log_decision(claim_idx, "final answer failed validation, retrying once")
        retry_note = f"Previous output failed schema validation: {first_error}\nRe-emit ONLY the corrected JSON object."
        raw_text_2, tokens_2 = _final_call(retry_note)
        try:
            data_2 = _extract_json(raw_text_2)
            claim_result = ClaimResult.model_validate(data_2)
            logger.log_llm_call("research", f"claim {claim_idx} final answer", "pass (retry)", tokens_2)
        except (json.JSONDecodeError, ValidationError) as second_error:
            logger.log_llm_call("research", f"claim {claim_idx} final answer", f"fail (retry): {second_error}", tokens_2)
            logger.log_error(f"Agent 2 hard-failed on claim {claim_idx} after retry: {second_error}")
            raise RuntimeError(
                f"Research Agent failed schema validation twice on claim {claim_idx!r}. "
                f"Refusing to pass unvalidated data downstream. Last error: {second_error}"
            ) from second_error

    # Force the claim text to exactly match what we asked about, regardless of model paraphrasing.
    claim_result.claim = claim

    # --- Guardrail: verify every citation against real logged tool results for this claim ---
    verified_matches = []
    for m in claim_result.matches:
        is_verified = m.id_or_url in valid_ids
        logger.log_citation_check(claim_idx, m.id_or_url, is_verified)
        if is_verified:
            verified_matches.append(m)
        else:
            logger.log_decision(
                claim_idx, f"stripped unverified citation '{m.id_or_url}' - not found in logged tool results"
            )
    claim_result.matches = verified_matches

    if not verified_matches and claim_result.confidence != "none":
        logger.log_decision(
            claim_idx,
            f"downgrading confidence {claim_result.confidence!r} -> 'none' after citation-verification stripped all matches",
        )
        claim_result.confidence = "none"
        claim_result.conclusion = "No verifiable prior art was found for this claim in the sources searched."

    # Structural consistency guardrail: "none" must mean empty matches, never a leftover
    # weak/tangential hit the model decided not to actually stand behind.
    if claim_result.confidence == "none" and claim_result.matches:
        logger.log_decision(
            claim_idx, "confidence was 'none' but matches were non-empty - clearing matches for consistency"
        )
        claim_result.matches = []

    # --- Langfuse scoring: groundedness (does reasoning match the real retrieved content?) ---
    # and citation hallucination rate (what fraction of the model's own citations, before the
    # guardrail above stripped them, didn't correspond to a real tool result for this claim).
    langfuse = get_client()
    trace_id = langfuse.get_current_trace_id()
    observation_id = langfuse.get_current_observation_id()

    judgment = judge_groundedness(claim_result, retrieved_by_id, client)
    if judgment is not None:
        score = groundedness_score(judgment)
        ungrounded = [v for v in judgment.verdicts if not v.grounded]
        langfuse.create_score(
            name="groundedness",
            value=score,
            trace_id=trace_id,
            observation_id=observation_id,
            data_type="NUMERIC",
            comment=(
                f"{len(ungrounded)}/{len(judgment.verdicts)} match(es) not supported by the real retrieved snippet"
                + ("; conclusion overstates matches" if judgment.conclusion_overstates else "")
            ),
        )
        logger.log_decision(
            claim_idx, f"groundedness judge score={score:.2f} conclusion_overstates={judgment.conclusion_overstates}"
        )

    return claim_result


def run_research(extraction: ExtractionOutput, logger: RunLogger, client: OpenAI) -> ResearchReport:
    claim_results = []
    for idx, claim in enumerate(extraction.key_claims, start=1):
        logger.log_decision(idx, f"starting research for claim: {claim[:80]}")
        claim_results.append(_research_single_claim(claim, extraction, idx, logger, client))
    return ResearchReport(title=extraction.title, domain=extraction.domain, claims=claim_results)
