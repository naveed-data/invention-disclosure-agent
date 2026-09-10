# Invention Disclosure Prior-Art Research Agent

A DIMS-style multi-agent pipeline that takes a raw invention disclosure and produces a
structured, cited, confidence-scored prior-art report — grounded in real, verifiable public
sources, with explicit code-level guardrails against hallucinated citations, full observability
of agent behavior, and a small evaluation harness proving it actually works.

## Problem statement

UT Southwestern's Innovation Hub runs DIMS (Discovery Information Management System), which
manages faculty research "Discovery Tracks" — the pipeline from invention disclosure to
commercialization (patents, licensing). When a researcher submits an invention disclosure, the
Innovation Hub must determine whether prior art already exists (a patent or published paper that
already discloses the same or a very similar invention). Today that's a slow, manual process
done by tech transfer staff and patent attorneys, prone to missed matches from keyword mismatch
and inconsistent search coverage. This project is a working prototype of an agentic system that
does that first-pass research automatically, while being explicit about what it actually found
versus what it's guessing at.

## Architecture

```
Raw Disclosure Text
        |
        v
[Agent 1: Extraction Agent]  -- no tools, no retrieval --
        |  -> Structured JSON (title, domain, key_claims[], keywords[])
        v
[Agent 2: Prior-Art Research Agent]  -- tool-calling loop, RAG lives here --
        |  -> Cited Prior-Art Report (JSON + human-readable)
        v
Final Output + Eval Report + Run Log
```

**Agent 1 — Extraction Agent** (`extraction_agent.py`): one LLM call, no tools, low
temperature. Everything it needs is already in the disclosure text, so it doesn't retrieve
anything — it just structures the text into a title, domain, a set of discrete/atomic
`key_claims`, and search-ready `keywords`. Output is pydantic-validated; on failure it retries
once with the validation error appended to the prompt, then hard-fails rather than passing bad
data downstream.

**Agent 2 — Prior-Art Research Agent** (`research_agent.py`): this is the agentic loop and the
RAG component. For each `key_claim`, the model decides which tool to call and with what query,
inspects the actual returned titles/abstracts, judges relevance from that real content, and
decides to refine the query, try a different tool, or conclude — one tool call per turn, up to a
hard budget of 3 tool calls per claim. Once it stops (by choice or budget), a final structured
judgment is pydantic-validated (one retry on failure), and every citation is then cross-checked
in code against the tool results actually logged for that claim — anything that doesn't match a
real logged result is stripped before it reaches the report.

**Why two agents, not one, not three:** narrow, near-deterministic extraction and open-ended
iterative research are different problems with different failure modes, so they get different
models, temperatures, and guardrails. A third "orchestrator" agent was considered and rejected —
the handoff between Agent 1 and Agent 2 is a straight sequential pipe with no branching decision
to make, so a reasoning agent there would add cost and latency with no real capability gain. (See
"What's next" below for where a legitimate third agent does make sense.)

## Tool layer (`tools.py`) — and why these specific sources

- **`search_papers`** — Semantic Scholar Graph API. Public, no key required, but its
  per-IP rate limit is easy to hit from a shared/proxied network (this happened repeatedly
  during development — see `eval/eval_report.md`). One retry-with-backoff is built in; a free
  `SEMANTIC_SCHOLAR_API_KEY` raises the limit substantially.
- **`search_pubmed`** — NCBI PubMed E-utilities. Public, no key, no meaningful rate limit —
  this ended up carrying most of the biomedical literature search during testing and is a
  reliable second source, not just a stretch goal.
- **`search_patents`** — targets the USPTO Open Data Portal Patent Search API. **Note:** the
  spec's originally-named PatentsView endpoint (`api.patentsview.org`) is retired — it now
  redirects to a marketing page, confirmed while building this. PatentsView has been folded into
  USPTO's ODP, whose real API (`api.uspto.gov/api/v1/patent/applications/search`) is live but
  requires a free self-service key from `data.uspto.gov`. Set `USPTO_ODP_API_KEY` to enable it;
  without it, the tool degrades gracefully to an empty result + logged error, exactly per the
  "never crash the loop" guardrail — you'll see that happen in every run log until a key is
  added.

Every tool function returns `{query, source, raw_count, results, error}` and never raises —
timeouts, HTTP errors, and rate limits are all caught and logged as an error flag instead.

**Why public sources only, for now:** this is the legitimate scope for a standalone demo outside
the institution. The tool interface (a plain Python function returning normalized results) is
exactly what would extend to internal DIMS records and Azure AI Search in production, with no
change to the agent loop that calls it.

## Guardrails (real code, not just prompting)

1. **Citation verification** — every `id_or_url` in the final report is checked against the
   tool results actually logged for that claim during that run; anything unverified is stripped
   (`research_agent.py`, citation-check block). This is the guardrail I'd defend hardest in an
   interview: a hallucinated citation in a legal/IP prior-art report isn't a cosmetic bug, it's
   the kind of error that could get a real patent filing challenged or a licensing decision made
   on a source that doesn't exist.
2. **Structural "none" consistency** — if confidence comes back `none`, matches are forced to
   `[]` even if the model left a stray weak hit attached, so "no prior art found" is never
   silently paired with a fabricated-looking match.
3. **Explicit no-match path** — `eval/sample4.txt` is deliberately an unusual, hyper-specific
   invention combination designed to have no real prior art, and is used to confirm the pipeline
   produces `confidence: "none"` with an empty `matches` list rather than forcing a weak match to
   avoid an empty result.
4. **Schema validation on every LLM call**, both agents, pydantic-enforced, one retry on
   failure, then a hard, clear failure — never silently passing invalid structured data forward.
5. **Cost/latency caps** — max 3 tool calls per claim, logged and enforced in code
   (`MAX_TOOL_CALLS_PER_CLAIM` in `research_agent.py`), plus a full LLM-call/tool-call/token/
   runtime summary at the end of every run.

## Observability (`run_logger.py`)

Every run writes `logs/run_<timestamp>.log` as events happen (not buffered to the end), capturing
in order: every LLM call (agent, input summary, validation pass/fail, approx tokens), every tool
call (query, source, result count, timestamp, error if any), every decision the Research Agent
makes (including its own stated reasoning between tool calls — "the first search returned 0
relevant results, trying PubMed with a narrower query" is a real line you'll see in these logs,
not a made-up example), every citation-verification check, and a final run summary. A matching
`run_<timestamp>.json` alongside it holds the full structured output. This log is the artifact
that proves the reasoning loop is real, not a scripted demo — read one end to end before showing
it live.

## Evaluation harness (`eval/`)

`expected_results.md` was written by hand, before the pipeline was ever run, with 5 cases: two
expected to surface strong real prior art (CRISPR-Cas9 correction of the sickle-cell HBB mutation;
an ionizable-LNP nucleoside-modified-mRNA vaccine — both essentially describe real, approved or
late-stage therapeutic mechanisms), one expected to surface only partial/related art (an
RL-driven closed-loop insulin dosing algorithm — the hardware category is extremely well
established, the specific RL mechanism less so), one deliberately designed to surface nothing (an
oddly-specific staggered-probiotic photonic-hydrogel microneedle patch for a named
ultra-rare disease), and one more partial case (a chatbot+wearable-HRV triage system).

`eval/run_eval.py` runs the full pipeline on all 5 and writes `eval/eval_report.md` — an
expected-vs-actual table plus honest analysis of the two cases that didn't match, tracing each
mismatch to a specific, log-visible cause (case-level confidence being a coarser metric than the
system's real per-claim resolution, and a live Semantic Scholar rate-limit during that run) rather
than glossing over it. Regenerate it any time with:

```bash
python eval/run_eval.py
```

## Langfuse integration (tracing, groundedness, regression)

Every run is traced to [Langfuse](https://langfuse.com) via a drop-in `OpenAI` client
(`langfuse.openai.OpenAI` in `run.py`), so every LLM call is captured automatically with
token usage and cost — no manual instrumentation of the extraction or research agents
needed. `run_pipeline()` is wrapped in `@observe(name="prior_art_pipeline")`, and
`_research_single_claim()` in `@observe(name="research_claim")`, so a single trace groups
the whole run into per-claim spans.

Three scores get attached to each trace, mapped directly onto claims a prior-art tool
has to defend:

- **`groundedness`** (`groundedness_eval.py`) — an independent LLM judge compares each
  match's `reasoning` against the *actual* retrieved title/snippet (not the model's own
  citation) and scores what fraction is really supported. This is a different check than
  the code-level citation guardrail already in `research_agent.py`: that guardrail catches
  an invented `id_or_url`; this catches a real citation with a reasoning that
  misrepresents what the source actually says.
- **`citation_hallucination_rate`** (`run.py`) — the fraction of the model's own citations,
  across the whole run, that didn't correspond to a real tool result and were stripped by
  that guardrail before reaching the report.
- **`correctness`** (`eval/langfuse_eval.py`) — expected-vs-actual confidence on the same 5
  hand-written cases as `eval/run_eval.py`, but run as a named Langfuse **dataset
  experiment** instead of a local markdown file, so successive prompt/model changes are
  compared as runs in the Langfuse UI rather than overwriting one report:

  ```bash
  python eval/langfuse_eval.py --run-name "baseline"
  ```

Requires `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, and `LANGFUSE_HOST` in `.env` (see
below). If they're unset the Langfuse client no-ops rather than failing the run — tracing
is additive observability, not a hard dependency of the pipeline.

## How to run it

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt   # anthropic-free: openai, pydantic, requests, python-dotenv, langfuse

# create a .env file (gitignored) with:
#   OPENAI_API_KEY=sk-...
#   USPTO_ODP_API_KEY=...        # optional, enables real patent search
#   SEMANTIC_SCHOLAR_API_KEY=... # optional, avoids shared-IP rate limiting
#   LANGFUSE_PUBLIC_KEY=pk-lf-... # optional, enables tracing/eval scores (see below)
#   LANGFUSE_SECRET_KEY=sk-lf-...
#   LANGFUSE_HOST=https://us.cloud.langfuse.com

python run.py --disclosure eval/sample1.txt
```

This prints a human-readable report to stdout, writes `logs/run_<timestamp>.log` and
`logs/run_<timestamp>.json`, and prints the LLM-call/tool-call/token/runtime summary. Model
choice is configurable without code changes: `OPENAI_EXTRACTION_MODEL` (default `gpt-4o-mini`,
Agent 1) and `OPENAI_RESEARCH_MODEL` (default `gpt-4o`, Agent 2 — needs stronger tool-use
judgment than Agent 1's near-deterministic structuring task).

## Production Path

In production this would swap the direct OpenAI client for the Azure OpenAI Service SDK (or
Azure AI Foundry) — same chat-completions-with-tools architecture, same agent logic in
`extraction_agent.py`/`research_agent.py`, just a different client construction and
deployment-name-based model reference instead of a model string, which also puts the whole
pipeline inside UTSW's existing Azure tenancy/compliance boundary rather than an external API.
The tool layer would extend past the two public sources here to internal DIMS records and
institutional repositories via Azure AI Search (vector + hybrid search over UTSW's own prior
disclosures, licensing history, and faculty publication records), registered in the same
`TOOL_REGISTRY` shape so Agent 2's loop doesn't need to change to use them. The inline prompt
strings in `extraction_agent.py`/`research_agent.py` would move to a versioned prompt store —
a prompt ID plus a changelog per agent, so a prompt regression is a diffable, revertible change
instead of a silent behavior shift in a code file. And the per-run
LLM-calls/tool-calls/tokens/runtime summary that's currently a line in a log file would feed a
small dashboard (cost and latency per run, per agent, over time), which is the natural next step
once this is handling real disclosure volume instead of 5 hand-written eval cases.

## What's next

With more time, the natural next addition is a **Report Synthesis Agent** — a third agent that
takes Agent 2's raw per-claim JSON and turns it into a plain-language executive summary for a
tech transfer officer who isn't going to read structured JSON. This is deliberately framed as
future work, not something to build into tonight's scope: the two-agent split here already
covers extraction and research cleanly, and a synthesis step is additive (a straightforward
transform of already-validated, already-cited data) rather than something that needs its own
tool-calling loop or guardrails.
# prior-art-research-agent
# prior-art-research-agent
