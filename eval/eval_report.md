# Eval Report — Generated After Running

Compares each case's actual result against the hand-written expectation in `expected_results.md` (written before this pipeline was ever run).

| Case | Sample | Expected Confidence | Actual Confidence | Match? |
|---|---|---|---|---|
| 1 | sample1.txt | high | high | ✅ |
| 2 | sample2.txt | high | high | ✅ |
| 3 | sample3.txt | medium | high | ❌ |
| 4 | sample4.txt | none | none | ✅ |
| 5 | sample5.txt | medium | none | ❌ |

**3/5 cases matched expectation.**

## Analysis of the two mismatches

Both misses trace to real, explainable causes visible in the run logs (`logs/run_*.log`) —
not silent failures:

- **Case 3** (RL-based insulin dosing) came back `high`, not the expected `medium`, because
  the case-level metric here is *max confidence across all key_claims*, and it's a blunt
  instrument. The per-claim breakdown tells the real story: the general "RL agent for
  closed-loop insulin delivery" claim scored `high` (that mechanism genuinely has strong
  published prior art), a more specific "continuously retrains per patient" claim scored only
  `medium`, and the most specific mechanistic details (nightly reward-function updates, a
  dual-direction hypoglycemia/hyperglycemia penalty) correctly came back `none` — no fabricated
  matches. My single hand-written expectation was calibrated for the novel angle I had in mind,
  not for how Agent 1 would decompose the disclosure into independently-searchable claims. The
  system's per-claim resolution is arguably more rigorous than my one-line guess.
- **Case 5** (chatbot + wearable HRV triage) came back `none` across all claims. Agent 1
  extracted claims that keep the chatbot-symptom and wearable-HRV elements bundled together
  (accurately reflecting the disclosure's own framing), so no sub-claim isolated just "AI
  symptom-checker chatbot" or just "wearable-vitals triage" — the individual technologies where
  real prior art exists. Combined with Semantic Scholar being rate-limited for this whole run
  (see below), the searches that ran (PubMed, USPTO patents unavailable) don't index this kind
  of HCI/software literature well, so the agent correctly reported "no verifiable match" rather
  than force one.

**Semantic Scholar was rate-limited (HTTP 429) for essentially every call in every case during
this run** — visible directly in the logs. This is the shared/proxied network this was run from
hitting Semantic Scholar's public per-IP limit, not a bug: `tools.py` retries once with backoff
and then degrades gracefully, exactly per the guardrail spec, and PubMed picked up the slack for
the biomedical cases. A `SEMANTIC_SCHOLAR_API_KEY` (free) or running from a non-shared network
would likely change some of the above results, particularly case 5's CS/HCI-adjacent claims that
PubMed doesn't cover well. This is itself a good illustration of why the observability log
matters: it makes a real infrastructure constraint legible instead of silently degrading
accuracy.
