"""Defect Injector: programmatic injection of 16 defect conditions.

Generates modified disclosure records with controlled defect combinations
for experimental evaluation (2^4 full factorial design).
"""
from __future__ import annotations

import itertools
import pandas as pd

from config import DEFECT_NAMES

# ── Injection templates ───────────────────────────────────────────────────
# Each template describes how to modify the evidence_context and source_type
# when a defect is turned on for a clean record.

INJECTION_TEMPLATES: dict[str, dict[str, str]] = {
    "S_staleness": {
        "context_prefix": (
            "The cited evidence is from FY2021-22 and has not been updated. "
        ),
        "source_suffix": " (archived)",
    },
    "C_conflict": {
        "context_prefix": (
            "A second source reports a materially different figure for the "
            "same metric and period. "
        ),
        "source_suffix": " + conflicting filing",
    },
    "M_missingness": {
        "context_prefix": (
            "Key supporting data (tables, metrics, or breakdowns) are "
            "omitted from the excerpt. "
        ),
        "source_suffix": " (incomplete)",
    },
    "A_low_authority": {
        "context_prefix": (
            "The claim originates from an unverified third-party source. "
        ),
        "source_override": "Unverified blog post",
    },
}


# ── Condition generators ─────────────────────────────────────────────────

def generate_conditions(pilot: bool = False) -> list[dict]:
    """Return the list of defect conditions.

    Full:  16 conditions (2^4 factorial).
    Pilot:  8 conditions (clean + 4 singles + 3 strongest pairs).
    """
    conditions: list[dict] = []
    for i, combo in enumerate(itertools.product([0, 1], repeat=4)):
        defect_dict = dict(zip(DEFECT_NAMES, combo))
        active = [DEFECT_NAMES[j] for j, v in enumerate(combo) if v]
        label = (
            "+".join(d.split("_")[0].upper() for d in active) if active else "clean"
        )
        conditions.append(
            {"condition_id": i, "condition_label": label, **defect_dict}
        )

    if pilot:
        # clean + 4 singles + 3 strongest pairs
        keep_ids = {
            0,   # clean   (0000)
            1,   # A only  (0001)
            2,   # M only  (0010)
            4,   # C only  (0100)
            8,   # S only  (1000)
            5,   # C+A     (0101)
            6,   # C+M     (0110)
            10,  # S+C     (1010)
        }
        conditions = [c for c in conditions if c["condition_id"] in keep_ids]

    return conditions


# ── Defect injection ──────────────────────────────────────────────────────

def inject_defects(row: dict, condition: dict) -> dict:
    """Apply a defect condition to a clean disclosure record.

    Modifies evidence_context and source_type according to the templates.
    """
    modified = row.copy()
    modified["condition_id"] = condition["condition_id"]
    modified["condition_label"] = condition["condition_label"]

    for defect_name in DEFECT_NAMES:
        flag = condition.get(defect_name, 0)
        modified[defect_name] = flag

        if flag and defect_name in INJECTION_TEMPLATES:
            tmpl = INJECTION_TEMPLATES[defect_name]

            if "context_prefix" in tmpl:
                modified["evidence_context"] = (
                    tmpl["context_prefix"] + modified.get("evidence_context", "")
                )
            if "source_override" in tmpl:
                modified["source_type"] = tmpl["source_override"]
            elif "source_suffix" in tmpl:
                modified["source_type"] = (
                    modified.get("source_type", "") + tmpl["source_suffix"]
                )

    return modified


# ── Experiment matrix ─────────────────────────────────────────────────────

def create_experiment_matrix(
    df: pd.DataFrame,
    conditions: list[dict] | None = None,
    pilot: bool = False,
) -> pd.DataFrame:
    """Cross every record × every condition.

    Args:
        df:         Original dataset (clean records).
        conditions: Custom list, or None to use full / pilot defaults.
        pilot:      If True and conditions is None, use 8 pilot conditions.

    Returns:
        DataFrame with one row per (record, condition) pair.
    """
    if conditions is None:
        conditions = generate_conditions(pilot=pilot)

    rows: list[dict] = []
    for _, record in df.iterrows():
        for cond in conditions:
            rows.append(inject_defects(record.to_dict(), cond))

    return pd.DataFrame(rows)
