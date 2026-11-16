"""Pydantic schemas for all agent inputs and outputs.

Every agent's I/O is a Pydantic model so we get:
  - JSON schema for structured LLM output
  - Automatic validation
  - Clean serialisation to JSONL / Parquet
"""
from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ── Enums ──────────────────────────────────────────────────────────────────

class AuthorityTier(str, Enum):
    """Source authority ranking, from most to least trustworthy."""
    TIER_1 = "tier_1"  # Audited annual report, regulatory filing, BRSR
    TIER_2 = "tier_2"  # Official but unaudited (investor pres., press release)
    TIER_3 = "tier_3"  # Third-party credible (news, analyst report)
    TIER_4 = "tier_4"  # Low authority (blog, social media, forum)


class Severity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# ── Evidence Agent output ──────────────────────────────────────────────────

class EvidenceRecord(BaseModel):
    """Output of the Evidence Agent (rule-based)."""
    record_id: str
    company: str
    fiscal_year: str
    source_type: str
    authority_tier: AuthorityTier
    source_date: Optional[str] = None   # ISO date or fiscal year string
    is_current: bool = True             # False → stale
    passage: str                        # The extracted text passage
    metadata: dict = Field(default_factory=dict)


# ── Risk Extractor output ─────────────────────────────────────────────────

class RiskItem(BaseModel):
    """A single extracted risk."""
    risk_type: str = Field(
        description="Category: Supply-chain, Environmental, Liquidity, "
                    "Financial, Cybersecurity, or Operational"
    )
    severity: Severity = Severity.MEDIUM
    claim: str = Field(description="The specific risk claim, quoted from passage")
    supporting_evidence_ids: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)
    reasoning: str = ""


class RiskExtraction(BaseModel):
    """Full output of the Risk Extractor for one record."""
    record_id: str
    model_used: str
    risks: list[RiskItem] = Field(default_factory=list)
    raw_confidence: float = Field(ge=0.0, le=1.0, default=0.5)
    logprobs: Optional[list[float]] = None


# ── Critic output ─────────────────────────────────────────────────────────

class CriticFlag(BaseModel):
    """Output of the Defect-Aware Critic for one record."""
    record_id: str
    model_used: str

    staleness_flag: bool = False
    staleness_reasoning: str = ""

    conflict_flag: bool = False
    conflict_reasoning: str = ""

    missingness_flag: bool = False
    missingness_reasoning: str = ""

    low_authority_flag: bool = False
    low_authority_reasoning: str = ""

    overall_reliability: float = Field(ge=0.0, le=1.0, default=1.0)


# ── Aggregator output ─────────────────────────────────────────────────────

class AggregatedScore(BaseModel):
    """Output of the Aggregator (deterministic code)."""
    record_id: str
    company: str
    risk_score: float = Field(ge=0.0, le=1.0, default=0.0)
    reliability_score_additive: float = Field(ge=0.0, le=1.0, default=1.0)
    reliability_score_interaction: float = Field(ge=0.0, le=1.0, default=1.0)
    defect_count: int = 0
    defect_flags: dict = Field(default_factory=dict)
    route: str = ""


# ── Experiment result (one row of the results log) ────────────────────────

class ExperimentResult(BaseModel):
    """One row of the experiment JSONL log."""
    record_id: str
    condition_id: int                   # 0..15
    condition_label: str                # e.g. "S+C", "clean"
    extractor_model: str
    critic_model: str
    extraction: RiskExtraction
    critic_flags: CriticFlag
    aggregated: AggregatedScore
    ground_truth_label: Optional[str] = None
