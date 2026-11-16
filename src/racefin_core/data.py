"""Data loaders: S&P 500 price snapshot, per-company metrics, SEC 10-K manifest."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import streamlit as st

from . import ROOT
from portfolio_engine import read_price_universe  # noqa: E402

MARKET_CSV = ROOT / "data" / "market" / "SP500Data.csv"
SEC_DIR = ROOT / "data" / "sec_10k"
TD = 252


@st.cache_data(show_spinner=False)
def load_prices():
    raw, eligible, info = read_price_universe(str(MARKET_CSV))
    px = raw[eligible].ffill()
    return px, info


@st.cache_data(show_spinner=False)
def load_names() -> dict:
    p = SEC_DIR / "company_tickers.json"
    if not p.exists():
        return {}
    data = json.loads(p.read_text(encoding="utf-8"))
    return {str(v["ticker"]).upper(): str(v["title"]).title() for v in data.values()}


@st.cache_data(show_spinner=False)
def load_manifest() -> pd.DataFrame:
    p = SEC_DIR / "filings_manifest.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_xbrl() -> pd.DataFrame:
    p = SEC_DIR / "xbrl_key_facts.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


@st.cache_data(show_spinner=False)
def company_table() -> pd.DataFrame:
    """One row per ticker: price, 1Y return, vol, Sharpe, drawdown, momentum, risk bucket, sparkline."""
    px, _ = load_prices()
    px = px.dropna(axis=1, how="all")
    n = min(TD, len(px) - 2)
    ret = px.pct_change().iloc[-n:]
    win = px.iloc[-n - 1:]
    vol = ret.std() * np.sqrt(TD)
    out = pd.DataFrame({
        "price": px.iloc[-1],
        "ret_1y": px.iloc[-1] / px.iloc[-n - 1] - 1,
        "ret_3m": px.iloc[-1] / px.iloc[-64] - 1,
        "vol": vol,
        "sharpe": ret.mean() / ret.std().replace(0, np.nan) * np.sqrt(TD),
        "max_dd": (win / win.cummax() - 1).min(),
    }).dropna()
    q1, q2 = out["vol"].quantile([1 / 3, 2 / 3])
    out["risk_bucket"] = np.where(out["vol"] <= q1, "Low", np.where(out["vol"] <= q2, "Medium", "High"))
    names, man = load_names(), load_manifest()
    sectors = man.drop_duplicates("ticker").set_index("ticker")["sector"].to_dict() if not man.empty else {}
    out["name"] = [names.get(t, t) for t in out.index]
    out["sector"] = [sectors.get(t, "—") for t in out.index]
    out["spark"] = [px[t].iloc[-120::3].round(2).tolist() for t in out.index]
    out.index.name = "ticker"
    return out
