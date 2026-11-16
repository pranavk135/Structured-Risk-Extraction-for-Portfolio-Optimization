"""Tests for RACE-Fin Multi-Agent Pipeline & Compounding Logic."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import pytest

from agents import evidence_agent, aggregator
from schemas import CriticFlag, RiskExtraction, RiskItem, Severity
from pipeline.defect_injector import generate_conditions, create_experiment_matrix
from evaluation.confidence import expected_calibration_error, brier_score
from evaluation.ranking_stability import kendall_tau


def test_evidence_agent_staleness():
    stale_text = "The cited operational report is dated FY2021-22 and no current update is supplied."
    is_stale, stale_date = evidence_agent.check_staleness(stale_text, current_fy="2025-26")
    assert is_stale is True
    assert stale_date == "2021-22"

    current_text = "Annual report FY2025-26; current period official filing."
    is_stale_curr, _ = evidence_agent.check_staleness(current_text, current_fy="2025-26")
    assert is_stale_curr is False


def test_evidence_agent_authority_classification():
    assert evidence_agent.classify_authority("Audited annual report").value == "tier_1"
    assert evidence_agent.classify_authority("Investor presentation").value == "tier_2"
    assert evidence_agent.classify_authority("Analyst report").value == "tier_3"
    assert evidence_agent.classify_authority("Third-party blog post").value == "tier_4"


def test_non_linear_compounding_amplification():
    """Verify that interaction-aware scoring amplifies penalty when multiple defects co-occur."""
    # 1. Single defect: Staleness only
    single_critic = CriticFlag(
        record_id="T01", model_used="test",
        staleness_flag=True, conflict_flag=False,
        missingness_flag=False, low_authority_flag=False
    )
    add_single = aggregator.score_additive(single_critic)
    int_single = aggregator.score_interaction(single_critic)
    assert abs(add_single - int_single) < 1e-6, "Single defect should have additive == interaction"

    # 2. Compound defect: Staleness + Conflict + Missingness
    compound_critic = CriticFlag(
        record_id="T02", model_used="test",
        staleness_flag=True, conflict_flag=True,
        missingness_flag=True, low_authority_flag=False
    )
    add_comp = aggregator.score_additive(compound_critic)
    int_comp = aggregator.score_interaction(compound_critic)
    assert int_comp < add_comp, "Compound defects must amplify penalty: interaction < additive"


def test_factorial_conditions_generation():
    full_16 = generate_conditions(pilot=False)
    assert len(full_16) == 16, "Full factorial must generate 2^4 = 16 conditions"

    pilot_8 = generate_conditions(pilot=True)
    assert len(pilot_8) == 8, "Pilot must generate exactly 8 conditions"


def test_calibration_metrics():
    y_true = np.array([1, 0, 1, 0, 1])
    y_prob = np.array([0.9, 0.1, 0.8, 0.2, 0.7])
    ece = expected_calibration_error(y_true, y_prob)
    brier = brier_score(y_true, y_prob)
    assert 0.0 <= ece <= 1.0
    assert 0.0 <= brier <= 1.0
