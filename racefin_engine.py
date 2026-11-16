"""RACE-Fin engine — bridge between the old heuristic demo and the new
multi-agent pipeline.

Keeps the original three helper functions so existing demo code doesn't
break, and adds a thin wrapper around the real agents for live mode.
"""
from __future__ import annotations

from schemas import CriticFlag, RiskExtraction, AggregatedScore
from agents import aggregator


# ── Legacy helpers (kept for backward compatibility) ──────────────────────

def score_evidence(claim: dict, defects: dict) -> dict:
    """Transparent illustrative penalties for UI demonstration only."""
    score = 1.0
    penalties = {
        "Staleness (S)":     0.15,
        "Conflict (C)":      0.25,
        "Missingness (M)":   0.20,
        "Low Authority (A)": 0.25,
    }
    for defect, active in defects.items():
        if active:
            score -= penalties.get(defect, 0)
    score = max(0.0, min(1.0, score))
    route = "Human review" if score < 0.70 else "Eligible for routine verification"
    return {"reliability": score, "route": route}


def triage_label(materiality: str, reliability: float) -> str:
    high_m = materiality == "High"
    high_r = reliability >= 0.70
    if high_m and high_r:
        return "High-priority direct action"
    if high_m and not high_r:
        return "Urgent human audit queue"
    if not high_m and high_r:
        return "Routine monitoring"
    return "Low-priority evidence gap"


def compare_xbrl(reported: float, structured: float, tolerance_pct: float = 0.5) -> dict:
    denom = max(abs(structured), 1e-9)
    diff = abs(reported - structured) / denom * 100
    return {"match": diff <= tolerance_pct, "difference_pct": diff}


# ── New: agent-backed scoring ─────────────────────────────────────────────

def score_with_agents(
    record_id: str,
    company: str,
    extraction: RiskExtraction,
    critic_flags: CriticFlag,
    materiality: str = "Medium",
) -> AggregatedScore:
    """Delegate to the real Aggregator agent."""
    return aggregator.aggregate(
        record_id=record_id,
        company=company,
        extraction=extraction,
        critic=critic_flags,
        materiality=materiality,
    )
