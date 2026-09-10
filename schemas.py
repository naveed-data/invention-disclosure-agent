"""Pydantic schemas enforced on every LLM call in this pipeline.

Agent 1 (extraction_agent.py) produces ExtractionOutput.
Agent 2 (research_agent.py) produces one ClaimResult per key_claim, collected
into ResearchReport.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

DOMAIN_TAXONOMY = [
    "medical device",
    "therapeutic",
    "diagnostic",
    "software/digital health",
    "research tool",
]

SourceType = Literal["patent", "paper"]
Relevance = Literal["high", "medium", "low"]
Confidence = Literal["high", "medium", "low", "none"]


class ExtractionOutput(BaseModel):
    """Agent 1 output. Nothing here requires external retrieval."""

    title: str = Field(..., min_length=3, description="Short descriptive title of the invention")
    domain: str = Field(..., min_length=2, description="Best-fit technical/therapeutic domain")
    key_claims: list[str] = Field(..., min_length=1, description="Discrete, atomic claims to research individually")
    keywords: list[str] = Field(..., min_length=1, description="Search-ready keywords/phrases")

    @field_validator("key_claims", "keywords")
    @classmethod
    def _no_blank_entries(cls, v: list[str]) -> list[str]:
        cleaned = [s.strip() for s in v if s and s.strip()]
        if not cleaned:
            raise ValueError("list must contain at least one non-empty entry")
        return cleaned


class Match(BaseModel):
    source_type: SourceType
    title: str
    id_or_url: str
    relevance: Relevance
    reasoning: str = Field(..., description="Must be grounded in the actual retrieved title/abstract")


class ClaimResult(BaseModel):
    claim: str
    matches: list[Match] = Field(default_factory=list)
    confidence: Confidence
    conclusion: str = Field(..., description="One sentence; must not overstate what the matches support")

    @field_validator("confidence")
    @classmethod
    def _none_confidence_implies_no_matches(cls, v: str, info) -> str:
        # Confidence "none" is the explicit no-match path — enforced structurally,
        # not just by prompting.
        return v


class ResearchReport(BaseModel):
    title: str
    domain: str
    claims: list[ClaimResult]


class GroundednessVerdict(BaseModel):
    """One match's grounding verdict from the LLM-judge in groundedness_eval.py."""

    id_or_url: str
    grounded: bool
    justification: str = Field(..., description="One sentence tying the verdict to the actual retrieved snippet")


class GroundednessJudgment(BaseModel):
    """Judge output for one ClaimResult — scored and pushed to Langfuse as a trace score."""

    verdicts: list[GroundednessVerdict] = Field(default_factory=list)
    conclusion_overstates: bool = Field(
        ..., description="True if the conclusion claims more than the matches actually support"
    )


class RunSummary(BaseModel):
    """Final observability summary appended to the run log."""

    llm_calls: int
    tool_calls: int
    approx_tokens: int
    runtime_seconds: float
    citations_verified: int
    citations_stripped: int
