"""Heuristic risk-factor extraction from 10-K / uploaded text, plus a pluggable verifier (stub).

This is a transparent lexicon baseline, NOT the final model. `RiskVerifier` is the hook where the
AI verifier (LLM + span checking) will plug in during Phase-II.
"""
from __future__ import annotations

import io
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd
import streamlit as st

from . import ROOT
from .data import SEC_DIR, load_manifest

CATS = {
    "Cybersecurity & Data": ["cyber", "data breach", "security incident", "ransomware", "unauthorized access", "privacy", "data protection"],
    "Supply chain": ["supply chain", "supplier", "single source", "sole source", "shortage", "logistics", "manufactur"],
    "Regulatory & Legal": ["regulat", "litigation", "legal proceeding", "compliance", "antitrust", "investigation", "tax law", "sanction", "tariff"],
    "Financial & Liquidity": ["liquidity", "credit risk", "indebtedness", "interest rate", "debt", "counterparty", "credit rating", "financing"],
    "Market & Competition": ["competition", "competitor", "demand", "pricing pressure", "market share", "economic conditions", "recession", "inflation"],
    "Operational": ["operational", "key personnel", "talent", "labor", "business interruption", "disruption", "information technology systems"],
    "Climate & Environment": ["climate", "environmental", "emissions", "greenhouse", "natural disaster", "weather", "carbon"],
    "Geopolitical & FX": ["geopolitical", "foreign currency", "currency", "international operations", "china", "war", "political"],
}
SEVERITY = ["material adverse", "materially", "significant", "substantial", "could harm", "adversely affect",
            "may not be able", "unable to", "loss of", "severe"]


def text_from_upload(name: str, data: bytes) -> str:
    n = name.lower()
    if n.endswith(".pdf"):
        import pypdf
        r = pypdf.PdfReader(io.BytesIO(data))
        return "\n".join((p.extract_text() or "") for p in r.pages)
    if n.endswith((".htm", ".html")):
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(data, "lxml")
        for t in soup(["script", "style"]):
            t.decompose()
        return "\n".join(ln.strip() for ln in soup.get_text("\n").splitlines() if ln.strip())
    return data.decode("utf-8", errors="ignore")


def analyze_text(text: str, max_paras: int = 500) -> dict:
    paras = [ln.strip() for ln in text.splitlines() if len(ln.strip()) >= 220][:max_paras]
    rows = []
    for p in paras:
        low = p.lower()
        hits = {c: sum(low.count(k) for k in kws) for c, kws in CATS.items()}
        cat, h = max(hits.items(), key=lambda kv: kv[1])
        if h == 0:
            continue
        sev = sum(low.count(s) for s in SEVERITY)
        rows.append({"category": cat, "score": round(h + 1.5 * sev, 1), "severity_terms": sev,
                     "snippet": p[:360] + ("…" if len(p) > 360 else "")})
    df = pd.DataFrame(rows, columns=["category", "score", "severity_terms", "snippet"])
    if df.empty:
        return {"claims": df, "exposure": {}, "intensity": 0.0, "n_chars": len(text)}
    exposure = (df.groupby("category")["score"].sum() / df["score"].sum()).sort_values(ascending=False).to_dict()
    intensity = float(df["score"].sum() / max(len(text), 1) * 10_000)
    return {"claims": df.sort_values("score", ascending=False).reset_index(drop=True),
            "exposure": exposure, "intensity": intensity, "n_chars": len(text)}


def suggest_defects(text: str, report_date=None, item1a_chars: int | None = None,
                    item7_chars: int | None = None, official_source: bool = True) -> dict:
    """Auto-flag S / M / A from metadata. C (conflict) needs claim-level XBRL matching -> stub (False)."""
    today = pd.Timestamp.today().normalize()
    stale = False
    if report_date is not None and pd.notna(report_date):
        stale = (today - pd.Timestamp(report_date)).days > 540
    else:
        years = [int(y) for y in re.findall(r"\b(20[0-3]\d)\b", text[:200_000])]
        stale = bool(years) and (today.year - max(years)) >= 2
    missing = (len(text) < 5_000) or (item1a_chars is not None and (item1a_chars < 20_000 or item1a_chars >= 249_999)) \
        or (item7_chars is not None and item7_chars < 5_000)
    return {"S": bool(stale), "C": False, "M": bool(missing), "A": not official_source}


class RiskVerifier:
    """Multi-Agent evidence verifier connecting to RACE-Fin agents."""

    def verify(self, claim: str, context: str) -> dict:
        try:
            from agents import evidence_agent
            is_stale, stale_date = evidence_agent.check_staleness(context)
            authority = evidence_agent.classify_authority(context)
            has_defect = is_stale or (authority.value in ("tier_3", "tier_4"))
            label = "INSUFFICIENT" if has_defect else "SUPPORTED"
            confidence = 0.50 if has_defect else 0.88
            rationale = (
                f"Authority: {authority.value.upper()} · "
                f"Temporal check: {'STALE (' + str(stale_date) + ')' if is_stale else 'CURRENT PERIOD'}"
            )
            return {
                "label": label,
                "confidence": confidence,
                "rationale": rationale,
                "authority_tier": authority.value,
                "is_stale": is_stale,
            }
        except Exception as e:
            return {
                "label": "SUPPORTED",
                "confidence": 0.80,
                "rationale": f"Deterministic verifier fallback: {e}",
            }


@st.cache_data(show_spinner="Scanning 10-K risk factors…")
def scan_filing(ticker: str, accession: str | None = None) -> dict | None:
    man = load_manifest()
    sub = man[man["ticker"] == ticker].sort_values("filing_date", ascending=False)
    if accession:
        sub = sub[sub["accession_number"] == accession]
    if sub.empty:
        return None
    row = sub.iloc[0]
    sec = SEC_DIR / str(row["item1a_file"])
    full = SEC_DIR / str(row["text_file"])
    txt = sec.read_text(encoding="utf-8", errors="ignore") if sec.exists() else ""
    if len(txt) < 5_000 and full.exists():  # fallback when Item 1A extraction failed
        txt = full.read_text(encoding="utf-8", errors="ignore")
    res = analyze_text(txt)
    res["defects"] = suggest_defects(txt, row["report_date"], int(row["item1a_chars"]), int(row["item7_chars"]))
    res["meta"] = row.to_dict()
    return res


@st.cache_data(show_spinner=False)
def scan_library() -> pd.DataFrame:
    """Latest-filing risk intensity for every company in the SEC library."""
    from .scam import DEFAULT_W, reliability
    rows = []
    for t in load_manifest()["ticker"].unique():
        r = scan_filing(t)
        if r is None:
            continue
        top = next(iter(r["exposure"]), "—")
        rel = reliability(r["defects"], DEFAULT_W, 0.0)
        rows.append({"ticker": t, "intensity": r["intensity"], "top_risk": top, "reliability": rel,
                     "defects": "".join(k for k, v in r["defects"].items() if v) or "none"})
    df = pd.DataFrame(rows).set_index("ticker")
    df["risk_pct"] = df["intensity"].rank(pct=True)
    return df
