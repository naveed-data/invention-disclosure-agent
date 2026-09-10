# Expected Results — Written Before Running

These expectations were written by hand, before the pipeline was ever executed, based on domain
knowledge of what's actually publicly known/patented in each area. This is the falsifiable baseline
the eval report (`eval_report.md`, generated after running) is checked against.

**Case-level confidence** in the eval report = the highest per-claim confidence returned across all
of that case's `key_claims` (i.e. "did the system find the invention's most-exposed angle?"). A case
can have multiple claims with different confidences; the table below is about the case as a whole.

| Case | Disclosure summary | Expected confidence | Rationale |
|---|---|---|---|
| 1 | Ex vivo CRISPR-Cas9 RNP editing of HBB in autologous HSCs for sickle cell disease | **high** | This is essentially the mechanism behind an FDA-approved therapy (Casgevy/exa-cel) and a large public patent/literature estate (Doudna/Charpentier foundational CRISPR patents, multiple sickle-cell base/gene-editing papers). Should surface clear, real, verifiable matches. |
| 2 | Ionizable-lipid LNP delivering nucleoside-modified mRNA encoding a viral antigen, for prophylactic vaccination | **high** | This is the Moderna/BioNTech-Pfizer COVID-19 vaccine platform mechanism almost exactly (Karikó & Weissman modified-nucleoside mRNA work; well-documented ionizable LNP patents). Should surface clear, real, verifiable matches. |
| 3 | Closed-loop insulin delivery where dosing is driven by a continuously self-retraining reinforcement-learning agent (not PID/MPC) | **medium** | CGM+pump closed-loop ("artificial pancreas") systems are extremely well established (Medtronic 670G/780G, Tandem Control-IQ), so *related* art is abundant -- but most disclosed control algorithms are PID/MPC-based, not an RL agent that retrains nightly per-patient. Expect matches on the closed-loop hardware/algorithm space generally, but not an exact hit on the RL-specific mechanism -- i.e. partial/related, not a direct disclosure. |
| 4 | Biodegradable microneedle patch delivering a staggered 3-strain engineered probiotic consortium, release timed by a color-changing photonic pH-responsive hydrogel, for a named ultra-rare (<1:1,000,000) pediatric skin disorder | **none** | Deliberately constructed as an unusual, highly specific combination (staggered engineered-probiotic dosing + visible photonic pH indicator + microneedle delivery + one named ultra-rare disease) that is very unlikely to have a direct real-world match. This case exercises the "no match found" guardrail -- expect `confidence: none` and an empty `matches` list, not a fabricated marginal match. |
| 5 | AI chatbot fusing free-text symptom intake with wearable HRV data for same-day primary-care triage scoring | **medium** | Symptom-checker chatbots (Ada Health, Babylon, Buoy) and wearable-vitals-based triage are each independently well documented, but the specific fusion of conversational free-text symptoms *with* continuous HRV time-series in one triage model is a narrower, less-established combination. Expect related-but-not-identical matches on the two component technologies rather than a single exact hit. |
