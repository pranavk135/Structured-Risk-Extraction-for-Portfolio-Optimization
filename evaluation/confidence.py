"""Confidence evaluation: ECE, Brier score, calibration curves, and
comparison of raw vs evidence-conditioned confidence.
"""
from __future__ import annotations

import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


# ── Core metrics ──────────────────────────────────────────────────────────

def expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Expected Calibration Error (ECE)."""
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_prob > lo) & (y_prob <= hi)
        if mask.sum() == 0:
            continue
        ece += mask.sum() / len(y_true) * abs(y_true[mask].mean() - y_prob[mask].mean())
    return float(ece)


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Brier score (lower is better)."""
    return float(np.mean((np.asarray(y_prob) - np.asarray(y_true)) ** 2))


# ── Calibration methods ──────────────────────────────────────────────────

def platt_scaling(y_prob: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """Platt (logistic) calibration — equivalent to temperature scaling."""
    lr = LogisticRegression(C=1.0, max_iter=200, solver="lbfgs")
    lr.fit(np.asarray(y_prob).reshape(-1, 1), np.asarray(y_true))
    return lr.predict_proba(np.asarray(y_prob).reshape(-1, 1))[:, 1]


def isotonic_calibration(y_prob: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """Isotonic regression calibration."""
    ir = IsotonicRegression(out_of_bounds="clip")
    ir.fit(np.asarray(y_prob), np.asarray(y_true))
    return ir.transform(np.asarray(y_prob))


def compute_calibration_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """Calibration curve data for plotting."""
    frac_pos, mean_pred = calibration_curve(
        y_true, y_prob, n_bins=n_bins, strategy="uniform"
    )
    return mean_pred, frac_pos


# ── Comparison ────────────────────────────────────────────────────────────

def compare_calibration_methods(
    y_true: np.ndarray | list,
    raw_confidence: np.ndarray | list,
    evidence_conditioned: np.ndarray | list,
) -> dict:
    """Compare raw vs evidence-conditioned confidence on ECE and Brier.

    Returns:
        Dict with metrics for each method (raw, evidence_conditioned,
        platt_scaled, isotonic).
    """
    yt = np.asarray(y_true, dtype=float)
    raw = np.asarray(raw_confidence, dtype=float)
    cond = np.asarray(evidence_conditioned, dtype=float)

    # Fit Platt on the raw scores as a baseline
    raw_platt = platt_scaling(raw, yt)

    return {
        "raw": {
            "ece":   expected_calibration_error(yt, raw),
            "brier": brier_score(yt, raw),
        },
        "evidence_conditioned": {
            "ece":   expected_calibration_error(yt, cond),
            "brier": brier_score(yt, cond),
        },
        "platt_scaled": {
            "ece":   expected_calibration_error(yt, raw_platt),
            "brier": brier_score(yt, raw_platt),
        },
    }
