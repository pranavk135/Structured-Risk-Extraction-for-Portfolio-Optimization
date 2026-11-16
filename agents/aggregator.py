"""Aggregator Agent — deterministic, no LLM.

Two scoring modes:
  1. Additive:          sum of fixed penalties per defect.
  2. Interaction-aware: includes pairwise and higher-order interaction terms.

Keeps the risk score separate from the reliability score.
"""
from __future__ import annotations

from schemas import AggregatedScore, CriticFlag, RiskExtraction

# ── Fixed penalty weights (report these in the paper) ─────────────────────

PENALTIES: dict[str, float] = {
    "staleness":     0.15,
    "conflict":      0.25,
    "missingness":   0.20,
    "low_authority": 0.25,
}

# Interaction boosts when defects co-occur
INTERACTIONS: dict[tuple[str, ...], float] = {
    ("staleness", "conflict"):                          0.10,
    ("staleness", "missingness"):                       0.08,
    ("conflict", "missingness"):                        0.12,
    ("staleness", "low_authority"):                     0.05,
    ("conflict", "low_authority"):                      0.10,
    ("missingness", "low_authority"):                   0.08,
    ("staleness", "conflict", "missingness"):           0.15,
    ("staleness", "conflict", "low_authority"):         0.10,
    ("staleness", "missingness", "low_authority"):      0.08,
    ("conflict", "missingness", "low_authority"):       0.12,
    ("staleness", "conflict", "missingness", "low_authority"): 0.20,
}

SEVERITY_MAP: dict[str, float] = {
    "low": 0.25, "medium": 0.50, "high": 0.75, "critical": 1.0,
}


# ── Helpers ────────────────────────────────────────────────────────────────

def _active_defects(critic: CriticFlag) -> dict[str, bool]:
    return {
        "staleness":     critic.staleness_flag,
        "conflict":      critic.conflict_flag,
        "missingness":   critic.missingness_flag,
        "low_authority": critic.low_authority_flag,
    }


# ── Scoring functions ─────────────────────────────────────────────────────

def score_additive(critic: CriticFlag) -> float:
    """Reliability = 1.0 − Σ penalties for active defects (clamped to [0, 1])."""
    defects = _active_defects(critic)
    penalty = sum(PENALTIES[d] for d, on in defects.items() if on)
    return max(0.0, min(1.0, 1.0 - penalty))


def score_interaction(critic: CriticFlag) -> float:
    """Reliability with additive + pairwise / higher-order interaction terms."""
    defects = _active_defects(critic)
    active = [d for d, on in defects.items() if on]

    penalty = sum(PENALTIES[d] for d in active)

    for combo, boost in INTERACTIONS.items():
        if all(d in active for d in combo):
            penalty += boost

    return max(0.0, min(1.0, 1.0 - penalty))


def compute_risk_score(extraction: RiskExtraction) -> float:
    """Aggregate risk severity: max severity weighted by confidence."""
    if not extraction.risks:
        return 0.0
    weighted = [
        SEVERITY_MAP.get(r.severity.value, 0.5) * r.confidence
        for r in extraction.risks
    ]
    return max(weighted) if weighted else 0.0


def route_decision(reliability: float, materiality: str) -> str:
    """2-D triage routing."""
    high_mat = materiality.strip().lower() == "high"
    high_rel = reliability >= 0.70

    if high_mat and high_rel:
        return "High-priority direct action"
    if high_mat and not high_rel:
        return "Urgent human audit queue"
    if not high_mat and high_rel:
        return "Routine monitoring"
    return "Low-priority evidence gap"


# ── Public API ─────────────────────────────────────────────────────────────

def aggregate(
    record_id: str,
    company: str,
    extraction: RiskExtraction,
    critic: CriticFlag,
    materiality: str = "Medium",
) -> AggregatedScore:
    """Full aggregation: risk score + reliability (both modes) + routing."""
    defects = _active_defects(critic)
    active_count = sum(1 for v in defects.values() if v)

    rel_add = score_additive(critic)
    rel_int = score_interaction(critic)
    risk    = compute_risk_score(extraction)
    route   = route_decision(rel_int, materiality)

    return AggregatedScore(
        record_id=record_id,
        company=company,
        risk_score=risk,
        reliability_score_additive=rel_add,
        reliability_score_interaction=rel_int,
        defect_count=active_count,
        defect_flags=defects,
        route=route,
    )
