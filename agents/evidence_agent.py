"""Evidence Agent — rule-based, no LLM.

Parses source type → authority tier, checks date → staleness,
and attaches metadata. Deterministic and explainable.
"""
from __future__ import annotations

import re
from schemas import EvidenceRecord, AuthorityTier

# ── Source-type → authority tier mapping ───────────────────────────────────
_SOURCE_TIER: list[tuple[str, AuthorityTier]] = [
    # Tier 1: audited / regulatory
    ("audited annual report", AuthorityTier.TIER_1),
    ("annual report",         AuthorityTier.TIER_1),
    ("regulatory filing",     AuthorityTier.TIER_1),
    ("brsr",                  AuthorityTier.TIER_1),
    ("10-k",                  AuthorityTier.TIER_1),
    ("10k",                   AuthorityTier.TIER_1),
    # Tier 2: official but unaudited
    ("investor presentation", AuthorityTier.TIER_2),
    ("press release",         AuthorityTier.TIER_2),
    ("filing",                AuthorityTier.TIER_2),
    # Tier 3: third-party credible
    ("news",                  AuthorityTier.TIER_3),
    ("analyst report",        AuthorityTier.TIER_3),
    # Tier 4: low authority
    ("blog",                  AuthorityTier.TIER_4),
    ("social media",          AuthorityTier.TIER_4),
    ("forum",                 AuthorityTier.TIER_4),
    ("website",               AuthorityTier.TIER_4),
    ("unofficial",            AuthorityTier.TIER_4),
]

# Matches "FY2021-22", "FY 2023-24", "FY2022-2023"
_FY_RE = re.compile(r"FY\s*(\d{4})[-–](\d{2,4})")
_CURRENT_FY = "2025-26"


# ── Helpers ────────────────────────────────────────────────────────────────

def classify_authority(source_type: str) -> AuthorityTier:
    """Map a free-text source_type to an AuthorityTier."""
    src = source_type.lower()
    for keyword, tier in _SOURCE_TIER:
        if keyword in src:
            return tier
    return AuthorityTier.TIER_3          # conservative default


def check_staleness(
    evidence_context: str,
    current_fy: str = _CURRENT_FY,
) -> tuple[bool, str | None]:
    """Return (is_stale, stale_fy_string) by checking for old FY references."""
    matches = _FY_RE.findall(evidence_context)
    if not matches:
        return False, None

    # Normalise current FY for comparison
    cy_start, cy_end = current_fy.split("-")
    cy_norm = f"{cy_start}-{cy_end}"           # "2025-26"

    for start_yr, end_part in matches:
        # Normalise the match
        end_short = end_part[-2:]               # take last 2 digits
        fy_norm = f"{start_yr}-{end_short}"     # e.g. "2021-22"
        if fy_norm != cy_norm:
            return True, fy_norm

    return False, None


# ── Public API ─────────────────────────────────────────────────────────────

def process(row: dict) -> EvidenceRecord:
    """Turn one CSV row (dict) into a validated EvidenceRecord."""
    authority = classify_authority(row.get("source_type", ""))
    is_stale, stale_fy = check_staleness(row.get("evidence_context", ""))

    return EvidenceRecord(
        record_id=row["record_id"],
        company=row["company"],
        fiscal_year=row["fiscal_year"],
        source_type=row["source_type"],
        authority_tier=authority,
        source_date=stale_fy,
        is_current=not is_stale,
        passage=row["claim_text"],
        metadata={
            "evidence_context": row.get("evidence_context", ""),
            "industry":         row.get("industry", ""),
            "risk_category":    row.get("risk_category", ""),
            "materiality":      row.get("materiality", ""),
        },
    )
