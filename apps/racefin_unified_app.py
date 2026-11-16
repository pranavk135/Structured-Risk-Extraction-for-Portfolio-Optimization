from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "src" / "disclosure_audit"))
sys.path.insert(0, str(BASE_DIR / "src" / "risk_allocation"))

from racefin_engine import compare_xbrl, score_evidence, triage_label  # noqa: E402
from portfolio_engine import (  # noqa: E402
    DEFAULT_TICKERS,
    TRADING_DAYS,
    prepare_prices,
    read_price_universe,
    run_comparison,
)

DISCLOSURE_CSV = BASE_DIR / "data" / "synthetic_disclosures" / "racefin_demo_disclosures.csv"
MARKET_CSV = BASE_DIR / "data" / "market" / "SP500Data.csv"
SEC_MANIFEST_CSV = BASE_DIR / "data" / "sec_10k" / "filings_manifest.csv"
SEC_XBRL_CSV = BASE_DIR / "data" / "sec_10k" / "xbrl_key_facts.csv"
SEC_DIR = BASE_DIR / "data" / "sec_10k"

SAGE = "#4D7C5B"
SLATE = "#5C738A"
SAND = "#C5B388"
RUST = "#B36A5E"
CHARCOAL = "#111827"
MUTED = "#6B7280"
PALETTE = [SAGE, SLATE, SAND, RUST, "#7A8B99", "#9CA3AF", "#6B8E78", "#A37B73"]

st.set_page_config(
    page_title="RACE-Fin | Portfolio & Risk Intelligence",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp {
        background-color: #F7F7F5;
        color: #111827;
        font-family: -apple-system, BlinkMacSystemFont, "Inter", "Segoe UI", sans-serif;
    }
    header[data-testid="stHeader"] {
        background-color: #F7F7F5 !important;
        color: #111827 !important;
    }
    section[data-testid="stSidebar"] {
        background-color: #FFFFFF !important;
        border-right: 1px solid #E5E7EB !important;
    }
    .rf-card {
        background: #FFFFFF;
        border: 1px solid #EAEAEA;
        border-radius: 16px;
        padding: 18px 20px;
        margin-bottom: 14px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.03);
    }
    .rf-eyebrow {
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #6B7280;
        margin-bottom: 6px;
    }
    .rf-hero-val {
        font-size: 36px;
        font-weight: 700;
        color: #111827;
        line-height: 1.1;
        margin-bottom: 8px;
    }
    .rf-pill-ok {
        display: inline-block;
        background: #E8F2EB;
        color: #2E6F40;
        font-size: 12px;
        font-weight: 600;
        padding: 4px 12px;
        border-radius: 999px;
    }
    .rf-pill-warn {
        display: inline-block;
        background: #FBECE9;
        color: #A64438;
        font-size: 12px;
        font-weight: 600;
        padding: 4px 12px;
        border-radius: 999px;
    }
    .rf-pill-neutral {
        display: inline-block;
        background: #F1F3F5;
        color: #374151;
        font-size: 12px;
        font-weight: 600;
        padding: 4px 12px;
        border-radius: 999px;
    }
    .rf-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 9px 0;
        border-bottom: 1px solid #F1F1EF;
        font-size: 13.5px;
    }
    .rf-row:last-child { border-bottom: none; }
    .rf-row-label { color: #4B5563; font-weight: 500; }
    .rf-row-val { color: #111827; font-weight: 700; }
    .rf-row-bad { color: #B91C1C; font-weight: 700; }
    .rf-topbar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 2px 0 12px 0;
        border-bottom: 1px solid #E5E7EB;
        margin-bottom: 16px;
    }
    .rf-brand { font-size: 22px; font-weight: 700; color: #111827; letter-spacing: -0.02em; }
    .rf-sub { font-size: 12.5px; color: #6B7280; }
    .rf-sec-head {
        font-size: 15px;
        font-weight: 700;
        color: #111827;
        margin: 8px 0 10px 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_market():
    return read_price_universe(str(MARKET_CSV))


@st.cache_data(show_spinner=False)
def load_disclosures():
    return pd.read_csv(DISCLOSURE_CSV)


@st.cache_data(show_spinner=False)
def load_sec_data():
    manifest = pd.read_csv(SEC_MANIFEST_CSV) if SEC_MANIFEST_CSV.exists() else pd.DataFrame()
    xbrl = pd.read_csv(SEC_XBRL_CSV) if SEC_XBRL_CSV.exists() else pd.DataFrame()
    return manifest, xbrl


def risk_score_from_vol(ann_vol: float) -> int:
    return int(np.clip(round(15 + ann_vol * 260), 15, 98))


def risk_tier(score: int) -> tuple[str, str]:
    if score < 45:
        return "Low risk", "Defensive diversification limits portfolio swings."
    if score < 70:
        return "Moderate risk", "Balanced exposure across clustered risk factors."
    return "Elevated risk", "Concentrated equity volatility drives most swings."


def build_area_chart(wealth_series: pd.Series) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=wealth_series.index,
            y=wealth_series.values,
            mode="lines",
            line=dict(color=SAGE, width=2.8, shape="spline"),
            fill="tozeroy",
            fillcolor="rgba(77, 124, 91, 0.14)",
            hovertemplate="%{x|%b %d, %Y}<br><b>$%{y:,.2f}</b><extra></extra>",
        )
    )
    fig.update_layout(
        height=230,
        margin=dict(l=6, r=6, t=10, b=6),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=False, zeroline=False, tickfont=dict(size=11, color=MUTED)),
        yaxis=dict(showgrid=True, gridcolor="#EFEFED", zeroline=False, tickprefix="$", tickfont=dict(size=11, color=MUTED)),
        showlegend=False,
    )
    return fig


def build_donut_chart(weights: pd.Series) -> go.Figure:
    w_sorted = weights.sort_values(ascending=False)
    fig = go.Figure(
        data=[
            go.Pie(
                labels=w_sorted.index.tolist(),
                values=w_sorted.values,
                hole=0.62,
                marker=dict(colors=PALETTE[: len(w_sorted)], line=dict(color="#FFFFFF", width=2)),
                textinfo="none",
                hovertemplate="<b>%{label}</b>: %{percent}<extra></extra>",
            )
        ]
    )
    fig.update_layout(
        height=210,
        margin=dict(l=0, r=0, t=4, b=4),
        paper_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
    )
    return fig


def build_risk_gauge(score: int) -> go.Figure:
    fig = go.Figure(
        data=[
            go.Pie(
                values=[score, 100 - score],
                hole=0.74,
                sort=False,
                direction="clockwise",
                marker=dict(colors=[SLATE if score < 70 else RUST, "#EAECEF"]),
                textinfo="none",
                hoverinfo="skip",
            )
        ]
    )
    fig.update_layout(
        height=140,
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        annotations=[
            dict(text=f"<b>{score}</b>", x=0.5, y=0.5, font=dict(size=25, color=CHARCOAL), showarrow=False)
        ],
    )
    return fig


def render_stacked_bar(label: str, weights: pd.Series) -> str:
    segs = []
    for i, (tk, val) in enumerate(weights.sort_values(ascending=False).items()):
        pct = max(val * 100.0, 0.5)
        col = PALETTE[i % len(PALETTE)]
        segs.append(f'<div title="{tk}: {val:.1%}" style="width:{pct:.2f}%;background:{col};height:11px;"></div>')
    return (
        f'<div style="margin-bottom:9px;">'
        f'<div style="font-size:12px;color:#4B5563;font-weight:500;margin-bottom:4px;">{label}</div>'
        f'<div style="display:flex;width:100%;border-radius:999px;overflow:hidden;background:#EAECEF;">'
        + "".join(segs)
        + "</div></div>"
    )


raw_prices, eligible_tickers, market_info = load_market()
df_disc = load_disclosures()
df_sec_manifest, df_sec_xbrl = load_sec_data()
default_tickers = [t for t in DEFAULT_TICKERS if t in eligible_tickers] or eligible_tickers[:8]

# Sidebar Controls
with st.sidebar:
    st.markdown("### Portfolio Optimiser")
    selected_tickers = st.multiselect("Asset universe (S&P 500)", options=eligible_tickers, default=default_tickers)
    capital = st.number_input("Portfolio capital ($)", min_value=1_000.0, max_value=50_000_000.0, value=250_000.0, step=10_000.0)
    strategy_choice = st.radio("Primary optimizer", ["HRP", "PCA"], horizontal=True)
    train_frac = st.slider("In-sample train split", 0.65, 0.85, 0.80, 0.05)

    st.divider()
    st.markdown("### Disclosure Risk Evaluator")
    comp_filter = st.selectbox("Company", ["All companies"] + sorted(df_disc["company"].unique().tolist()))
    subset_df = df_disc if comp_filter == "All companies" else df_disc[df_disc["company"] == comp_filter]
    rec_ids = subset_df["record_id"].tolist()
    chosen_id = st.selectbox(
        "Scenario",
        rec_ids,
        format_func=lambda rid: f"{rid} · {df_disc.loc[df_disc['record_id'] == rid, 'risk_category'].iloc[0]}",
    )

port_res = None
port_err = None
if len(selected_tickers) >= 3:
    try:
        cleaned = prepare_prices(raw_prices, selected_tickers)
        port_res = run_comparison(cleaned, capital=capital, train_fraction=train_frac)
    except Exception as exc:
        port_err = str(exc)
else:
    port_err = "Select at least 3 tickers in the sidebar to run HRP / PCA optimization."

rec = df_disc[df_disc["record_id"] == chosen_id].iloc[0]

# Top Bar
badge_html = '<span class="rf-pill-ok">Risk validated</span>' if port_res is not None else '<span class="rf-pill-warn">Select ≥ 3 assets</span>'
st.markdown(
    f'<div class="rf-topbar">'
    f'<div><div class="rf-brand">RACE-Fin</div>'
    f'<div class="rf-sub">Unified Portfolio Optimiser (HRP / PCA) &amp; Evidence-Reliability Risk Evaluator</div></div>'
    f'<div>{badge_html}</div>'
    f'</div>',
    unsafe_allow_html=True,
)

# ==============================================================================
# ROW 1: PORTFOLIO OPTIMISER (LEFT)  +  RISK CHECK & REBALANCE (RIGHT)
# ==============================================================================
if port_err:
    st.warning(port_err)
else:
    strat_wealth = port_res.wealth[strategy_choice]
    end_val = float(strat_wealth.iloc[-1])
    tot_ret = float(port_res.metrics.loc[strategy_choice, "Total return"])
    ann_vol = float(port_res.metrics.loc[strategy_choice, "Annualised volatility"])
    sharpe = float(port_res.metrics.loc[strategy_choice, "Sharpe ratio"])
    opt_dd = float(port_res.metrics.loc[strategy_choice, "Maximum drawdown"])
    weights_sel = port_res.weights[strategy_choice].sort_values(ascending=False)

    eq_w = pd.Series(1.0 / len(selected_tickers), index=port_res.weights["HRP"].index)
    eq_ret = port_res.test_returns.loc[:, eq_w.index].dot(eq_w)
    eq_vol = float(eq_ret.std(ddof=1) * np.sqrt(TRADING_DAYS))
    eq_score = risk_score_from_vol(eq_vol)
    opt_score = risk_score_from_vol(ann_vol)
    tier_title, tier_desc = risk_tier(opt_score)

    col_left, col_right = st.columns([1.2, 1.0], gap="medium")

    with col_left:
        st.markdown('<div class="rf-sec-head">1 · Portfolio &amp; Optimiser</div>', unsafe_allow_html=True)
        delta_cls = "rf-pill-ok" if tot_ret >= 0 else "rf-pill-warn"
        arrow = "▲" if tot_ret >= 0 else "▼"
        st.markdown(
            f'<div class="rf-card">'
            f'<div class="rf-eyebrow">PORTFOLIO VALUE ({strategy_choice} HOLDOUT)</div>'
            f'<div class="rf-hero-val">${end_val:,.2f}</div>'
            f'<span class="{delta_cls}">{arrow} {tot_ret:+.2%} out-of-sample</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

        tf = st.radio("Holdout window", ["1M", "3M", "6M", "ALL"], index=3, horizontal=True, label_visibility="collapsed")
        days_map = {"1M": 21, "3M": 63, "6M": 126, "ALL": len(strat_wealth)}
        st.plotly_chart(build_area_chart(strat_wealth.iloc[-days_map[tf] :]), use_container_width=True)

        # Donut + Weights inside clean card
        st.markdown('<div class="rf-card"><div class="rf-eyebrow">OPTIMIZED ASSET ALLOCATION</div>', unsafe_allow_html=True)
        d1, d2 = st.columns([1.0, 1.0])
        with d1:
            st.plotly_chart(build_donut_chart(weights_sel), use_container_width=True)
        with d2:
            rows_html = ""
            for i, (tk, wv) in enumerate(weights_sel.items()):
                dot = PALETTE[i % len(PALETTE)]
                rows_html += (
                    f'<div class="rf-row" style="padding:4px 0;font-size:12.5px;">'
                    f'<span class="rf-row-label"><span style="color:{dot};">●</span> {tk}</span>'
                    f'<span class="rf-row-val">{wv:.1%}</span>'
                    f'</div>'
                )
            st.markdown(rows_html, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_right:
        st.markdown('<div class="rf-sec-head">2 · Risk Check &amp; Rebalance</div>', unsafe_allow_html=True)
        st.markdown('<div class="rf-card"><div class="rf-eyebrow">PORTFOLIO RISK SCORE</div>', unsafe_allow_html=True)
        g1, g2 = st.columns([0.85, 1.45])
        with g1:
            st.plotly_chart(build_risk_gauge(opt_score), use_container_width=True)
        with g2:
            st.markdown(
                f'<div style="padding-top:20px;">'
                f'<div style="font-size:18px;font-weight:700;color:#111827;">{tier_title}</div>'
                f'<div style="font-size:13px;color:#4B5563;margin-top:4px;">{tier_desc}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown(
            f'<div class="rf-card">'
            f'<div class="rf-row"><span class="rf-row-label">Volatility ({strategy_choice})</span><span class="rf-row-val">{ann_vol:.1%}</span></div>'
            f'<div class="rf-row"><span class="rf-row-label">Max drawdown</span><span class="rf-row-bad">{opt_dd:.1%}</span></div>'
            f'<div class="rf-row"><span class="rf-row-label">Sharpe ratio</span><span class="rf-row-val">{sharpe:.2f}</span></div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        delta_badge = f"Risk {eq_score} → {opt_score}"
        pill = "rf-pill-ok" if opt_score <= eq_score else "rf-pill-neutral"
        alt_strat = "PCA" if strategy_choice == "HRP" else "HRP"
        bars_html = (
            render_stacked_bar("Current (Equal-Weight Baseline)", eq_w)
            + render_stacked_bar(f"Optimized ({strategy_choice})", weights_sel)
            + render_stacked_bar(f"Alternative ({alt_strat})", port_res.weights[alt_strat])
        )
        st.markdown(
            f'<div class="rf-card">'
            f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;">'
            f'<span class="rf-eyebrow" style="margin:0;">PROPOSED REBALANCE</span>'
            f'<span class="{pill}">{delta_badge}</span>'
            f'</div>'
            f'{bars_html}'
            f'</div>',
            unsafe_allow_html=True,
        )

        if st.button("Review & apply optimized weights", type="primary", use_container_width=True):
            rebal_df = pd.DataFrame(
                {
                    "Ticker": weights_sel.index,
                    "Equal-Weight": [eq_w[t] for t in weights_sel.index],
                    f"{strategy_choice} Weight": weights_sel.values,
                    "Target ($)": weights_sel.values * capital,
                }
            )
            st.dataframe(
                rebal_df.style.format({"Equal-Weight": "{:.1%}", f"{strategy_choice} Weight": "{:.1%}", "Target ($)": "${:,.0f}"}),
                hide_index=True,
                use_container_width=True,
            )

# ==============================================================================
# ROW 2: EVIDENCE-RELIABILITY RISK EVALUATOR (ALWAYS VISIBLE BELOW!)
# ==============================================================================
st.divider()
st.markdown('<div class="rf-sec-head">3 · Evidence-Reliability Risk Evaluator (S × C × M × A)</div>', unsafe_allow_html=True)

ev_left, ev_mid, ev_right = st.columns([1.25, 1.0, 1.0], gap="medium")

with ev_left:
    st.markdown(
        f'<div class="rf-card">'
        f'<div class="rf-eyebrow">CLAIM · {rec["record_id"]} · {rec["company"]}</div>'
        f'<div style="font-size:15px;font-weight:700;color:#111827;margin-bottom:6px;">{rec["claim_text"]}</div>'
        f'<div style="font-size:12.5px;color:#4B5563;">{rec["industry"]} · {rec["fiscal_year"]} · <b>{rec["risk_category"]}</b> · {rec["source_type"]}</div>'
        f'<div style="font-size:12.5px;color:#374151;margin-top:8px;padding-top:8px;border-top:1px solid #F1F1EF;">'
        f'<b>Evidence:</b> {rec["evidence_context"]}'
        f'</div></div>',
        unsafe_allow_html=True,
    )

with ev_mid:
    st.markdown('<div class="rf-card"><div class="rf-eyebrow">INTERACTIVE DEFECT TOGGLES</div>', unsafe_allow_html=True)
    f_s = st.checkbox("Staleness (S)", value=bool(rec["S_staleness"]), key=f"s_{chosen_id}")
    f_c = st.checkbox("Conflict (C)", value=bool(rec["C_conflict"]), key=f"c_{chosen_id}")
    f_m = st.checkbox("Missingness (M)", value=bool(rec["M_missingness"]), key=f"m_{chosen_id}")
    f_a = st.checkbox("Low Authority (A)", value=bool(rec["A_low_authority"]), key=f"a_{chosen_id}")
    rep_val = st.number_input("Narrative reported ($M)", value=124.5, step=0.5, key="rep_xbrl")
    str_val = st.number_input("XBRL structured ($M)", value=124.0, step=0.5, key="str_xbrl")
    xres = compare_xbrl(rep_val, str_val)
    xpill = "rf-pill-ok" if xres["match"] else "rf-pill-warn"
    st.markdown(f'<span class="{xpill}">XBRL diff: {xres["difference_pct"]:.2f}% ({"Match" if xres["match"] else "Flagged"})</span></div>', unsafe_allow_html=True)

with ev_right:
    defects_dict = {"Staleness (S)": f_s, "Conflict (C)": f_c, "Missingness (M)": f_m, "Low Authority (A)": f_a}
    ev = score_evidence(rec["claim_text"], defects_dict)
    rel = ev["reliability"]
    triage = triage_label(rec["materiality"], rel)
    tpill = "rf-pill-ok" if rel >= 0.70 else "rf-pill-warn"
    st.markdown(
        f'<div class="rf-card">'
        f'<div class="rf-eyebrow">EVIDENCE RELIABILITY &amp; TRIAGE</div>'
        f'<div class="rf-hero-val">{rel:.0%}</div>'
        f'<span class="{tpill}">{triage}</span>'
        f'<div style="margin-top:12px;">'
        f'<div class="rf-row"><span class="rf-row-label">Risk Materiality</span><span class="rf-row-val">{rec["materiality"]}</span></div>'
        f'<div class="rf-row"><span class="rf-row-label">Route</span><span class="rf-row-val">{ev["route"]}</span></div>'
        f'<div class="rf-row"><span class="rf-row-label">Active Defects</span><span class="rf-row-val">{sum(defects_dict.values())} / 4</span></div>'
        f'</div></div>',
        unsafe_allow_html=True,
    )

# ==============================================================================
# ROW 3: REAL SEC 10-K & XBRL DATASET EXPLORER
# ==============================================================================
if not df_sec_manifest.empty:
    with st.expander(f"4 · SEC 10-K & XBRL Real Filings Explorer ({len(df_sec_manifest)} filings · {len(df_sec_xbrl):,} XBRL facts)", expanded=False):
        sec_ticker = st.selectbox("Select SEC 10-K Company", sorted(df_sec_manifest["ticker"].unique().tolist()))
        sub_m = df_sec_manifest[df_sec_manifest["ticker"] == sec_ticker]
        st.dataframe(
            sub_m[["ticker", "company_name", "sector", "filing_date", "report_date", "item1a_chars", "item7_chars", "edgar_url"]],
            hide_index=True,
            use_container_width=True,
        )
        if not sub_m.empty:
            latest_row = sub_m.iloc[0]
            i1a_path = SEC_DIR / str(latest_row["item1a_file"])
            if i1a_path.exists():
                txt = i1a_path.read_text(encoding="utf-8", errors="ignore")
                st.text_area(f"Item 1A Risk Factors — {sec_ticker} ({latest_row['filing_date']})", txt[:3500], height=180)
        if not df_sec_xbrl.empty:
            sub_x = df_sec_xbrl[df_sec_xbrl["ticker"] == sec_ticker].sort_values(["fy", "metric"], ascending=[False, True])
            st.dataframe(sub_x[["fy", "metric", "xbrl_tag", "val", "unit", "end", "filed"]], hide_index=True, use_container_width=True)
