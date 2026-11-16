"""Orchestrator: chains the 4-agent pipeline for one record or a batch.

Evidence → Risk Extractor → Critic → Aggregator
"""
from __future__ import annotations

import pandas as pd

from agents import evidence_agent, risk_extractor, critic, aggregator
from schemas import CriticFlag, EvidenceRecord, RiskExtraction, AggregatedScore


def run_single(
    row: dict,
    extractor_model: str,
    critic_model: str,
    use_critic: bool = True,
    logprobs: bool = False,
) -> dict:
    """Run the full 4-agent pipeline on one disclosure record.

    Args:
        row:             A dict representing one CSV row.
        extractor_model: LiteLLM model string for the risk extractor.
        critic_model:    LiteLLM model string for the critic.
        use_critic:      False → skip critic (ablation study).
        logprobs:        Request log-probabilities from open models.

    Returns:
        Dict with keys: evidence, extraction, critic_flags, aggregated.
    """
    # ── Step 1: Evidence Agent (rule-based) ────────────────────────────────
    evidence: EvidenceRecord = evidence_agent.process(row)

    # ── Step 2: Risk Extractor (LLM) ──────────────────────────────────────
    extraction: RiskExtraction = risk_extractor.extract(
        record_id=row["record_id"],
        passage=row["claim_text"],
        evidence_context=row.get("evidence_context", ""),
        model=extractor_model,
        source_type=row.get("source_type", ""),
        logprobs=logprobs,
    )

    # ── Step 3: Critic (LLM, different family) — skippable for ablation ───
    if use_critic:
        critic_flags: CriticFlag = critic.critique(
            record_id=row["record_id"],
            passage=row["claim_text"],
            evidence_context=row.get("evidence_context", ""),
            source_type=row.get("source_type", ""),
            extracted_risks=[r.model_dump() for r in extraction.risks],
            model=critic_model,
        )
    else:
        # Ablation: fall back to ground-truth defect flags from the CSV
        critic_flags = CriticFlag(
            record_id=row["record_id"],
            model_used="none (ablation)",
            staleness_flag=bool(int(row.get("S_staleness", 0))),
            conflict_flag=bool(int(row.get("C_conflict", 0))),
            missingness_flag=bool(int(row.get("M_missingness", 0))),
            low_authority_flag=bool(int(row.get("A_low_authority", 0))),
        )

    # ── Step 4: Aggregator (deterministic code) ───────────────────────────
    aggregated: AggregatedScore = aggregator.aggregate(
        record_id=row["record_id"],
        company=row["company"],
        extraction=extraction,
        critic=critic_flags,
        materiality=row.get("materiality", "Medium"),
    )

    return {
        "evidence":     evidence,
        "extraction":   extraction,
        "critic_flags": critic_flags,
        "aggregated":   aggregated,
    }


def run_batch(
    df: pd.DataFrame,
    extractor_model: str,
    critic_model: str,
    use_critic: bool = True,
    logprobs: bool = False,
    progress_callback=None,
) -> list[dict]:
    """Run the pipeline on every row in *df*.

    Args:
        progress_callback: Optional callable(current: int, total: int).

    Returns:
        List of per-record result dicts.
    """
    results: list[dict] = []
    total = len(df)

    for i, (_, row) in enumerate(df.iterrows()):
        result = run_single(
            row.to_dict(),
            extractor_model=extractor_model,
            critic_model=critic_model,
            use_critic=use_critic,
            logprobs=logprobs,
        )
        results.append(result)
        if progress_callback:
            progress_callback(i + 1, total)

    return results
