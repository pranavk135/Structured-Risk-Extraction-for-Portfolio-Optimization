"""Shared core logic for the RACE-Fin platform (data, risk scan, SCAM math, optimiser registry, account)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for sub in ("src/disclosure_audit", "src/risk_allocation"):
    p = str(ROOT / sub)
    if p not in sys.path:
        sys.path.insert(0, p)
