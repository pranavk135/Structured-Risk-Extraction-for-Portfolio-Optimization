"""Ranking stability: Kendall τ, Spearman ρ, and top-k overlap.

Compares company-level risk rankings between clean and defective conditions.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import kendalltau, spearmanr


def kendall_tau(ranking_a, ranking_b) -> dict:
    """Kendall τ rank correlation."""
    tau, p = kendalltau(ranking_a, ranking_b)
    return {"tau": float(tau), "p_value": float(p)}


def spearman_rho(ranking_a, ranking_b) -> dict:
    """Spearman ρ rank correlation."""
    rho, p = spearmanr(ranking_a, ranking_b)
    return {"rho": float(rho), "p_value": float(p)}


def top_k_overlap(ranking_a, ranking_b, k: int = 5) -> float:
    """Fraction of items in both top-k sets (by descending score)."""
    a = np.asarray(ranking_a)
    b = np.asarray(ranking_b)
    top_a = set(np.argsort(-a)[:k])
    top_b = set(np.argsort(-b)[:k])
    return len(top_a & top_b) / k


def ranking_stability_report(
    clean_scores: list | np.ndarray,
    defective_scores: list | np.ndarray,
    company_names: list[str],
    k: int = 5,
) -> dict:
    """Full ranking-stability analysis.

    Args:
        clean_scores:     Company risk scores under the clean condition.
        defective_scores: Company risk scores under a defective condition.
        company_names:    Company names for readability.
        k:                Top-k for overlap measure.

    Returns:
        Dict with Kendall τ, Spearman ρ, top-k overlap, biggest movers.
    """
    cs = np.asarray(clean_scores, dtype=float)
    ds = np.asarray(defective_scores, dtype=float)

    tau = kendall_tau(cs, ds)
    rho = spearman_rho(cs, ds)
    overlap = top_k_overlap(cs, ds, k=k)

    # Rank changes (rank 1 = highest score)
    clean_ranks = np.argsort(np.argsort(-cs)) + 1
    defective_ranks = np.argsort(np.argsort(-ds)) + 1
    changes = (defective_ranks - clean_ranks).tolist()

    biggest_movers = sorted(
        zip(company_names, changes),
        key=lambda x: abs(x[1]),
        reverse=True,
    )[:k]

    return {
        "kendall_tau": tau,
        "spearman_rho": rho,
        f"top_{k}_overlap": overlap,
        "biggest_rank_changes": biggest_movers,
    }
