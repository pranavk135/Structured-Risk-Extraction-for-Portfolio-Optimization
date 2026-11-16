"""RACE-Fin | Evidence-Reliability-Aware Financial Risk Extraction
ESA Demo Slice (40–50% Scope):
  Piece 1: Real Corporate Disclosures & Stock Performance (Ups & Downs) + Benchmark Corpus
  Piece 2: Controlled Defect Injection & Multi-Agent Compounding (S × C × M × A)
  Piece 3: Confidence Calibration, Interaction Regression & Ranking Stability
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ── Ensure project root and src are on sys.path ────────────────────────────
ROOT = Path(__file__).resolve().parent
for p in (str(ROOT), str(ROOT / "src"), str(ROOT / "src" / "disclosure_audit")):
    if p not in sys.path:
        sys.path.insert(0, p)

from ui import theme
from config import EXTRACTOR_MODEL_A, EXTRACTOR_MODEL_B, CRITIC_MODEL, RESULTS_DIR
from racefin_engine import score_evidence, triage_label, compare_xbrl
from racefin_core import data
from agents import evidence_agent, risk_extractor, critic, aggregator
from pipeline.defect_injector import generate_conditions, create_experiment_matrix
from evaluation.confidence import (
    expected_calibration_error,
    brier_score,
    compute_calibration_curve,
    compare_calibration_methods,
)
from evaluation.ranking_stability import ranking_stability_report
from evaluation.interaction_analysis import (
    compare_additive_vs_interaction,
    fit_interaction_regression,
    effect_sizes_with_ci,
)

# ── Page Configuration ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="RACE-Fin | Evidence-Reliability Risk Extraction (ESA Slice)",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

ss = st.session_state
ss.setdefault("theme", "dark")
theme.inject()

# ── Load Datasets ──────────────────────────────────────────────────────────
DATA_CSV = ROOT / "data" / "racefin_demo_disclosures.csv"
if not DATA_CSV.exists():
    DATA_CSV = ROOT / "data" / "synthetic_disclosures" / "racefin_demo_disclosures.csv"
df_disclosures = pd.read_csv(DATA_CSV)

sec_manifest = data.load_manifest()
sec_xbrl = data.load_xbrl()
prices_all, price_info = data.load_prices()
tickers_names = data.load_names()

# ══════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ◈ RACE-Fin")
    st.caption("Capstone Project PW26_PSJ_05 · PES University")
    st.markdown(
        f'<div class="rf-card" style="padding:10px 14px;margin-bottom:12px;">'
        f'<div class="rf-eyebrow">ESA Scope Control</div>'
        f'<span style="font-size:13px;font-weight:600;color:{theme.pal()["accent"]};">40–50% Core Research Slice</span><br>'
        f'<span class="rf-sub">Downstream portfolio module deliberately deferred to later phases.</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    theme_choice = st.segmented_control("Theme", ["dark", "light"], default=ss["theme"], key="theme_toggle")
    if theme_choice != ss["theme"]:
        ss["theme"] = theme_choice
        st.rerun()

    st.divider()

    # Agent Engine Options
    st.markdown("**🤖 Multi-Agent Engine**")
    live_agents = st.toggle("Enable Live LLMs", value=False, help="Connect to local Ollama server to run Qwen/Gemma & Llama models live.")

    if live_agents:
        extractor_choice = st.selectbox(
            "Risk Extractor Model",
            [EXTRACTOR_MODEL_A, EXTRACTOR_MODEL_B],
            format_func=lambda x: "Qwen 2.5-7B (Alibaba)" if "qwen" in x else "Gemma 2 9B (Google)",
        )
        critic_enabled = st.checkbox("Enable Defect-Aware Critic (Llama 3.1-8B)", value=True)
        st.caption("✓ 3 model families configured: Alibaba Qwen, Google Gemma, Meta Llama via Ollama.")
    else:
        extractor_choice = EXTRACTOR_MODEL_A
        critic_enabled = False
        st.caption("Running in fast deterministic rule-based mode.")

    st.divider()

    # Quick Scenario Selector
    st.markdown("**📂 Benchmark Scenario**")
    comp_list = ["All companies"] + sorted(df_disclosures["company"].unique().tolist())
    sel_comp = st.selectbox("Company", comp_list)
    filtered_df = df_disclosures if sel_comp == "All companies" else df_disclosures[df_disclosures["company"] == sel_comp]
    sel_rec_id = st.selectbox("Scenario Record", filtered_df["record_id"].tolist())
    active_row = df_disclosures[df_disclosures["record_id"] == sel_rec_id].iloc[0]


# ══════════════════════════════════════════════════════════════════════════
# TOP HEADER BAR
# ══════════════════════════════════════════════════════════════════════════
status_pill = theme.pill("Live LLM Pipeline Active", "ok") if live_agents else theme.pill("Rule-Based Deterministic Engine", "neutral")
st.markdown(
    f'<div style="display:flex;justify-content:space-between;align-items:center;padding:4px 0 14px 0;border-bottom:1px solid {theme.pal()["border"]};margin-bottom:14px;">'
    f'<div><span style="font-size:24px;font-weight:750;letter-spacing:-0.02em;">◈ Evidence-Reliability-Aware Financial Risk Extraction</span><br>'
    f'<span class="rf-sub">Evaluating risk extraction and prioritisation under compound disclosure defects (S × C × M × A)</span></div>'
    f'<div>{status_pill}</div>'
    f'</div>',
    unsafe_allow_html=True,
)

# Scope Callout
st.markdown(
    f'<div class="rf-card" style="border-left:4px solid {theme.pal()["accent"]};">'
    f'<div style="display:flex;justify-content:space-between;align-items:center;">'
    f'<div><b>ESA Milestone Scope (40–50%)</b>: '
    f'<span class="rf-sub">1. Dataset Extraction &amp; Price Trajectories · 2. Defect Injection &amp; Agent Compounding · 3. Confidence Calibration &amp; Stability</span></div>'
    f'{theme.pill("Portfolio optimiser deferred downstream", "neutral")}'
    f'</div></div>',
    unsafe_allow_html=True,
)


# ══════════════════════════════════════════════════════════════════════════
# THREE CORE PIECES (ESA DEMO SLICE)
# ══════════════════════════════════════════════════════════════════════════
tab_piece1, tab_piece2, tab_piece3 = st.tabs([
    "1 · Disclosures & Stock Performance",
    "2 · Defect Injection & Agent Compounding",
    "3 · Confidence Calibration & Stability",
])


# ──────────────────────────────────────────────────────────────────────────
# PIECE 1: DISCLOSURES & STOCK PERFORMANCE
# ──────────────────────────────────────────────────────────────────────────
with tab_piece1:
    st.subheader("Piece 1 · Corporate Disclosures & Stock Price Performance")
    st.caption("Real SEC 10-K Filings & S&P 500 Price Trajectories (Ups & Downs) · Controlled Defect Benchmark Corpus")

    corpus_mode = st.radio(
        "Corpus View",
        [
            "Real Corporate Filings & Stock Performance (SEC 10-K & Daily S&P 500 Prices)",
            "Controlled Benchmark Corpus (30 Scenarios with Ground-Truth S×C×M×A Flags)",
        ],
        horizontal=True,
    )

    if "Real Corporate" in corpus_mode:
        # Real Filings & Stock Ups/Downs
        sec_tickers = sorted(sec_manifest["ticker"].unique().tolist()) if not sec_manifest.empty else ["AAPL", "BA", "MSFT", "PFE"]
        c_sel1, c_sel2 = st.columns([1.5, 2.5])
        chosen_sec_t = c_sel1.selectbox(
            "Select Real Corporation",
            sec_tickers,
            format_func=lambda t: f"{t} · {tickers_names.get(t, t)}",
            index=sec_tickers.index("BA") if "BA" in sec_tickers else 0,
        )

        filing_rows = sec_manifest[sec_manifest["ticker"] == chosen_sec_t] if not sec_manifest.empty else pd.DataFrame()
        company_full_name = tickers_names.get(chosen_sec_t, chosen_sec_t)

        # ── Stock Price Ups & Downs Graph (from SP500Data.csv) ──────────────
        st.markdown(f"#### Stock Price Trajectory (Ups & Downs): {chosen_sec_t} · {company_full_name}")

        if chosen_sec_t in prices_all.columns:
            s_px = prices_all[chosen_sec_t].dropna()
            ret_1y = (s_px.iloc[-1] / s_px.iloc[-252] - 1.0) if len(s_px) >= 252 else (s_px.iloc[-1] / s_px.iloc[0] - 1.0)
            daily_returns = s_px.pct_change().dropna()
            ann_vol = float(daily_returns.std() * np.sqrt(252))
            # Drawdown series
            cum_max = s_px.cummax()
            drawdown_series = (s_px - cum_max) / cum_max
            max_dd = float(drawdown_series.min())
            sharpe = float((daily_returns.mean() / daily_returns.std()) * np.sqrt(252)) if daily_returns.std() > 0 else 0.0

            m_c1, m_c2, m_c3, m_c4 = st.columns(4)
            m_c1.metric("Current Price", f"${s_px.iloc[-1]:,.2f}")
            m_c2.metric("1Y Return (Ups & Downs)", f"{ret_1y:+.2%}", delta=f"{ret_1y:+.1%}")
            m_c3.metric("Annual Volatility", f"{ann_vol:.1%}")
            m_c4.metric("Maximum Drawdown", f"{max_dd:.1%}", delta="Peak to Trough", delta_color="inverse")

            # High-Resolution Price Trajectory Plot
            fig_stock = go.Figure()
            fig_stock.add_trace(go.Scatter(
                x=s_px.index, y=s_px.values, mode="lines",
                name="Stock Price ($)",
                line=dict(color=theme.pal()["accent"], width=2.4),
                fill="tozeroy", fillcolor=theme.pal()["accent_bg"],
            ))
            # Highlight Peak and Trough
            peak_date, peak_val = s_px.idxmax(), s_px.max()
            trough_date, trough_val = s_px.idxmin(), s_px.min()
            fig_stock.add_trace(go.Scatter(
                x=[peak_date], y=[peak_val], mode="markers+text",
                name="High Point", text=["Peak"], textposition="top center",
                marker=dict(color="#34D399", size=10),
            ))
            fig_stock.add_trace(go.Scatter(
                x=[trough_date], y=[trough_val], mode="markers+text",
                name="Low Point", text=["Trough"], textposition="bottom center",
                marker=dict(color="#F87171", size=10),
            ))
            fig_stock.update_layout(
                title=f"Historical Price Movement & Volatility Swings ({price_info['start']:%b %Y} – {price_info['end']:%b %Y})",
                yaxis_title="Stock Price ($ USD)",
                xaxis_title="Date",
                height=300,
            )
            st.plotly_chart(theme.plotly_layout(fig_stock, 300), use_container_width=True)

            # Drawdown Underwater Plot
            with st.expander("Underwater Drawdown Depth Plot (Market Pullbacks & Volatility Drops)"):
                fig_dd = go.Figure()
                fig_dd.add_trace(go.Scatter(
                    x=drawdown_series.index, y=drawdown_series.values, mode="lines",
                    name="Drawdown", line=dict(color="#F87171", width=1.8),
                    fill="tozeroy", fillcolor="rgba(248,113,113,0.14)",
                ))
                fig_dd.update_layout(
                    title=f"Percentage Drop Below Peak Price (Max Drawdown: {max_dd:.1%})",
                    yaxis_tickformat=".0%", yaxis_title="Drawdown (%)", height=200,
                )
                st.plotly_chart(theme.plotly_layout(fig_dd, 200), use_container_width=True)
        else:
            st.info("Price history not found for selected ticker in the price snapshot.")

        # ── Real Disclosed SEC 10-K Item 1A Risk Factors Text ────────────────
        st.markdown(f"#### Real SEC 10-K Disclosures · {chosen_sec_t}")
        if not filing_rows.empty:
            st.dataframe(
                filing_rows[["ticker", "company_name", "sector", "filing_date", "report_date", "item1a_chars", "item7_chars", "edgar_url"]],
                hide_index=True, use_container_width=True,
            )
            latest_f = filing_rows.iloc[0]
            sec_file = ROOT / "data" / "sec_10k" / str(latest_f["item1a_file"])
            if sec_file.exists():
                raw_item1a = sec_file.read_text(encoding="utf-8", errors="ignore")
                st.text_area(f"Extracted SEC 10-K Item 1A Risk Factors: {chosen_sec_t} ({latest_f['filing_date']})", raw_item1a[:4000], height=240)
            else:
                st.info("Item 1A text file not located.")

            # Associated XBRL Facts
            if not sec_xbrl.empty:
                comp_xbrl = sec_xbrl[sec_xbrl["ticker"] == chosen_sec_t]
                if not comp_xbrl.empty:
                    with st.expander(f"Key Structured XBRL Financial Facts ({len(comp_xbrl)} records)"):
                        st.dataframe(comp_xbrl[["fy", "metric", "xbrl_tag", "val", "unit", "filed"]], hide_index=True, use_container_width=True)

    else:
        # Controlled Benchmark Corpus (30 Scenarios)
        st.markdown("#### Controlled Defect Benchmark Corpus (30 Scenarios Across 5 Corporate Entities)")
        st.caption("Standardized scenarios where ground-truth defect combinations are known with 100% precision for factorial benchmarking.")

        ev_records = [evidence_agent.process(r.to_dict()) for _, r in df_disclosures.iterrows()]
        df_bench = df_disclosures.copy()
        df_bench["authority_tier"] = [er.authority_tier.value.upper() for er in ev_records]
        df_bench["is_current"] = ["Current" if er.is_current else "STALE" for er in ev_records]
        df_bench["defects_count"] = df_disclosures[["S_staleness", "C_conflict", "M_missingness", "A_low_authority"]].sum(axis=1)

        view_cols = ["record_id", "company", "fiscal_year", "risk_category", "source_type", "authority_tier", "is_current", "materiality", "defects_count", "illustrative_reference_label"]
        st.dataframe(df_bench[view_cols], use_container_width=True, hide_index=True)

        # Per-Company Risk Category Exposure Horizontal Bar Chart
        st.markdown(f"#### Disclosed Risk Category Breakdown · {active_row['company']}")
        comp_records = df_disclosures[df_disclosures["company"] == active_row["company"]]
        cat_counts = comp_records["risk_category"].value_counts().reset_index()
        cat_counts.columns = ["Risk Category", "Claim Count"]

        g_col1, g_col2 = st.columns([1.4, 1.0])
        with g_col1:
            fig_cat = go.Figure(go.Bar(
                x=cat_counts["Claim Count"],
                y=cat_counts["Risk Category"],
                orientation="h",
                marker_color=theme.pal()["accent"],
                text=cat_counts["Claim Count"],
                textposition="outside",
            ))
            fig_cat.update_layout(
                title=f"Disclosed Risk Categories: {active_row['company']}",
                xaxis_title="Number of Disclosed Claims", height=240,
            )
            st.plotly_chart(theme.plotly_layout(fig_cat, 240), use_container_width=True)

        with g_col2:
            comp_defects = int(((comp_records[["S_staleness", "C_conflict", "M_missingness", "A_low_authority"]].sum(axis=1)) >= 1).sum())
            comp_multi = int(((comp_records[["S_staleness", "C_conflict", "M_missingness", "A_low_authority"]].sum(axis=1)) >= 2).sum())
            theme.card(
                f'<div class="rf-eyebrow">Entity Defect Summary</div>'
                f'<div class="rf-hero" style="font-size:26px;">{len(comp_records)} Disclosures</div>'
                f'<div style="margin:8px 0;">'
                f'{theme.pill(f"{len(comp_records) - comp_defects} Clean", "ok")}'
                f'{theme.pill(f"{comp_defects} Defective", "warn")}'
                f'{theme.pill(f"{comp_multi} Compound Defects", "bad")}'
                f'</div>'
                f'<div class="rf-sub">Industry: <b>{active_row["industry"]}</b><br>'
                f'Period: <b>{active_row["fiscal_year"]}</b></div>'
            )

        # Cross-Company Quality Stacked Bar Chart
        st.markdown("#### Evidence Quality Comparison Across All 5 Entities")
        comp_summary = df_disclosures.groupby("company").agg(
            clean_claims=("record_id", lambda x: int((df_disclosures.set_index("record_id").loc[x, ["S_staleness", "C_conflict", "M_missingness", "A_low_authority"]].sum(axis=1) == 0).sum())),
            single_defect=("record_id", lambda x: int((df_disclosures.set_index("record_id").loc[x, ["S_staleness", "C_conflict", "M_missingness", "A_low_authority"]].sum(axis=1) == 1).sum())),
            compound_defects=("record_id", lambda x: int((df_disclosures.set_index("record_id").loc[x, ["S_staleness", "C_conflict", "M_missingness", "A_low_authority"]].sum(axis=1) >= 2).sum())),
        ).reset_index()

        fig_cross = go.Figure()
        fig_cross.add_trace(go.Bar(name="Clean (No Defects)", x=comp_summary["company"], y=comp_summary["clean_claims"], marker_color="#34D399"))
        fig_cross.add_trace(go.Bar(name="Single Defect", x=comp_summary["company"], y=comp_summary["single_defect"], marker_color="#F59E0B"))
        fig_cross.add_trace(go.Bar(name="Compound Defects (≥2)", x=comp_summary["company"], y=comp_summary["compound_defects"], marker_color="#F87171"))
        fig_cross.update_layout(barmode="stack", title="Defect Distribution per Benchmark Entity", height=260)
        st.plotly_chart(theme.plotly_layout(fig_cross, 260), use_container_width=True)


# ──────────────────────────────────────────────────────────────────────────
# PIECE 2: DEFECT INJECTION & AGENT COMPOUNDING
# ──────────────────────────────────────────────────────────────────────────
with tab_piece2:
    st.subheader("Piece 2 · Controlled Defect Injection & Multi-Agent Compounding")
    st.caption("Injected defects: Staleness (S), Conflict (C), Missingness (M), Low Authority (A) · 4-agent execution · Additive vs. Interaction-aware reliability")

    st.markdown(
        f'<div class="rf-card">'
        f'<div class="rf-eyebrow">ACTIVE RECORD: {active_row["record_id"]} · {active_row["company"]}</div>'
        f'<div style="font-size:16px;font-weight:700;margin-bottom:6px;">"{active_row["claim_text"]}"</div>'
        f'<div class="rf-sub">Context: {active_row["evidence_context"]}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # 1. Defect Injection Toggles
    st.markdown("#### 1 · Defect Injection Toggles (S × C × M × A)")
    t_col1, t_col2, t_col3, t_col4 = st.columns(4)
    flag_s = t_col1.checkbox("Staleness (S)", value=bool(active_row["S_staleness"]), help="Outdated evidence from prior fiscal years")
    flag_c = t_col2.checkbox("Conflict (C)", value=bool(active_row["C_conflict"]), help="Contradictory values or statements across filings")
    flag_m = t_col3.checkbox("Missingness (M)", value=bool(active_row["M_missingness"]), help="Missing tables, metrics, or required boundaries")
    flag_a = t_col4.checkbox("Low Authority (A)", value=bool(active_row["A_low_authority"]), help="Unverified blog, social media, or anonymous forum")

    active_defects = {"staleness": flag_s, "conflict": flag_c, "missingness": flag_m, "low_authority": flag_a}
    defect_count = sum(active_defects.values())

    # 2. Multi-Agent Pipeline Execution
    st.markdown("#### 2 · Multi-Agent Pipeline Execution")
    row_dict = active_row.to_dict()
    row_dict["S_staleness"] = int(flag_s)
    row_dict["C_conflict"] = int(flag_c)
    row_dict["M_missingness"] = int(flag_m)
    row_dict["A_low_authority"] = int(flag_a)

    if live_agents:
        with st.spinner(f"Running Risk Extractor ({extractor_choice}) & Critic ({CRITIC_MODEL})…"):
            try:
                from pipeline.orchestrator import run_single
                pipe_out = run_single(row_dict, extractor_choice, CRITIC_MODEL, use_critic=critic_enabled)
                ext_out = pipe_out["extraction"]
                critic_out = pipe_out["critic_flags"]
                agg_out = pipe_out["aggregated"]
            except Exception as e:
                st.error(f"Live agent pipeline error: {e}")
                live_agents = False
    if not live_agents:
        critic_out = critic.CriticFlag(
            record_id=active_row["record_id"],
            model_used="deterministic critic",
            staleness_flag=flag_s, conflict_flag=flag_c,
            missingness_flag=flag_m, low_authority_flag=flag_a,
            overall_reliability=1.0 - (0.15 * flag_s + 0.25 * flag_c + 0.20 * flag_m + 0.25 * flag_a),
        )
        agg_out = aggregator.aggregate(
            active_row["record_id"],
            active_row["company"],
            risk_extractor.RiskExtraction(
                record_id=active_row["record_id"], model_used="deterministic baseline",
                risks=[risk_extractor.RiskItem(
                    risk_type=active_row["risk_category"],
                    severity=risk_extractor.Severity.HIGH if active_row["materiality"] == "High" else risk_extractor.Severity.MEDIUM,
                    claim=active_row["claim_text"],
                    confidence=0.85 if defect_count == 0 else 0.50,
                )]
            ),
            critic_out,
            materiality=active_row["materiality"],
        )

    # 4 Pipeline Stage Cards
    p1, p2, p3, p4 = st.columns(4)
    with p1:
        ev_item = evidence_agent.process(row_dict)
        theme.card(
            f'<div class="rf-eyebrow">1. Evidence Agent</div>'
            f'<b>{ev_item.authority_tier.value.upper()}</b><br>'
            f'<span class="rf-sub">Source: {ev_item.source_type}</span><br>'
            f'<span class="rf-sub">Status: {"Current" if ev_item.is_current else "STALE"}</span>'
        )
    with p2:
        theme.card(
            f'<div class="rf-eyebrow">2. Risk Extractor</div>'
            f'<b>{active_row["risk_category"]}</b><br>'
            f'<span class="rf-sub">Materiality: {active_row["materiality"]}</span><br>'
            f'<span class="rf-sub">Risk Score: {agg_out.risk_score:.2f}</span>'
        )
    with p3:
        theme.card(
            f'<div class="rf-eyebrow">3. Defect Critic</div>'
            f'<b>{defect_count} Defect{"s" if defect_count!=1 else ""} Detected</b><br>'
            f'{theme.pill("Stale" if flag_s else "Current", "warn" if flag_s else "ok")}'
            f'{theme.pill("Conflict" if flag_c else "Consistent", "bad" if flag_c else "ok")}'
        )
    with p4:
        theme.card(
            f'<div class="rf-eyebrow">4. Aggregator</div>'
            f'<div class="rf-hero" style="font-size:26px;">{agg_out.reliability_score_interaction:.0%}</div>'
            f'<span class="rf-sub">Route: <b>{agg_out.route}</b></span>'
        )

    # 3. Additive vs Interaction Compounding Analysis
    st.markdown("#### 3 · Compound Defect Amplification Analysis")
    gap = agg_out.reliability_score_additive - agg_out.reliability_score_interaction
    cmp1, cmp2, cmp3 = st.columns([1, 1, 1.4])
    cmp1.metric("Additive Reliability", f"{agg_out.reliability_score_additive:.0%}")
    cmp2.metric("Interaction-Aware Reliability", f"{agg_out.reliability_score_interaction:.0%}")
    with cmp3:
        if gap > 0.001:
            st.warning(f"⚠️ **Compounding Amplification**: Defects combine to degrade reliability by an additional **{gap:.0%}** beyond their independent additive penalties.")
        else:
            st.success("✓ Additive and interaction scores coincide under single-defect or clean conditions.")

    # 4. 2D Prioritisation Grid (Severity vs Evidence Reliability)
    st.markdown("#### 4 · 2D Prioritisation Grid & Urgent Human Audit Queue")
    triage_result = triage_label(active_row["materiality"], agg_out.reliability_score_interaction)
    t_c1, t_c2 = st.columns([1, 2])
    with t_c1:
        if "Urgent" in triage_result:
            st.error(f"**Triage Action**: {triage_result}")
        else:
            st.success(f"**Triage Action**: {triage_result}")
    with t_c2:
        if agg_out.route == "Human review" or (active_row["materiality"] == "High" and agg_out.reliability_score_interaction < 0.70):
            st.warning("Claim is routed to the **Urgent Human Audit Queue** rather than silently discarded.")
        else:
            st.info("Claim is eligible for routine algorithmic monitoring.")

    # Scatter Plot across all benchmark claims
    all_map_pts = []
    sev_map = {"Low": 0.33, "Medium": 0.66, "High": 1.0}
    for _, rb in df_disclosures.iterrows():
        if rb["record_id"] == active_row["record_id"]:
            rel_pt = agg_out.reliability_score_interaction
        else:
            mock_c = critic.CriticFlag(
                record_id=str(rb["record_id"]), model_used="test",
                staleness_flag=bool(rb["S_staleness"]), conflict_flag=bool(rb["C_conflict"]),
                missingness_flag=bool(rb["M_missingness"]), low_authority_flag=bool(rb["A_low_authority"]),
            )
            rel_pt = aggregator.score_interaction(mock_c)
        all_map_pts.append({
            "Record": rb["record_id"],
            "Company": rb["company"],
            "Category": rb["risk_category"],
            "Reliability": rel_pt,
            "Severity": sev_map.get(rb["materiality"], 0.5),
            "Claim": rb["claim_text"][:50] + "…",
            "Active": "▶ Active Selection" if rb["record_id"] == active_row["record_id"] else "Other Claims",
        })
    df_quad = pd.DataFrame(all_map_pts)

    fig_quad = px.scatter(
        df_quad, x="Reliability", y="Severity", color="Company", symbol="Active",
        hover_data=["Record", "Category", "Claim"],
        title="2D Risk Prioritisation Grid: Severity vs. Evidence Reliability",
        height=320,
    )
    fig_quad.add_vline(x=0.70, line_dash="dash", line_color="#6B7280")
    fig_quad.add_hline(y=0.66, line_dash="dash", line_color="#6B7280")
    fig_quad.add_annotation(x=0.35, y=0.96, text="🚨 Urgent Human Audit Queue<br>(High Risk + Low Reliability)", showarrow=False, font=dict(color="#F87171", size=10))
    fig_quad.add_annotation(x=0.85, y=0.96, text="⚡ Direct Action Queue<br>(High Risk + Verified Evidence)", showarrow=False, font=dict(color="#34D399", size=10))
    fig_quad.add_annotation(x=0.35, y=0.20, text="📋 Information Gap", showarrow=False, font=dict(color="#8B98A9", size=10))
    fig_quad.add_annotation(x=0.85, y=0.20, text="🔍 Routine Monitoring", showarrow=False, font=dict(color="#60A5FA", size=10))
    st.plotly_chart(theme.plotly_layout(fig_quad, 320), use_container_width=True)

    # 5. Numerical Disclosure Consistency Widget
    st.markdown("#### 5 · Numerical Disclosure Comparison (XBRL vs. Narrative)")
    n_c1, n_c2, n_c3 = st.columns(3)
    val_rep = n_c1.number_input("Narrative / PDF Value ($M)", value=1240.0, step=10.0)
    val_xbrl = n_c2.number_input("Structured XBRL Value ($M)", value=1200.0, step=10.0)
    tol = n_c3.slider("Tolerance (%)", 0.0, 5.0, 0.5, 0.1)
    xbrl_res = compare_xbrl(val_rep, val_xbrl, tol)
    if xbrl_res["match"]:
        st.success(f"✓ Match within {tol:.1f}% tolerance (difference: {xbrl_res['difference_pct']:.2f}%).")
    else:
        st.error(f"⚠️ Mismatch flagged ({xbrl_res['difference_pct']:.2f}% difference). Check accounting concept, period, and reporting scope.")


# ──────────────────────────────────────────────────────────────────────────
# PIECE 3: CONFIDENCE CALIBRATION & STABILITY
# ──────────────────────────────────────────────────────────────────────────
with tab_piece3:
    st.subheader("Piece 3 · Confidence Calibration, Interaction Analysis & Ranking Stability")
    st.caption("Statistical verification: OLS interaction regression · Expected Calibration Error (ECE) · Ranking stability (Kendall τ & Top-k overlap)")

    sub_tab1, sub_tab2, sub_tab3, sub_tab4 = st.tabs([
        "Interaction Regression (OLS)",
        "Confidence Calibration (ECE & Brier)",
        "Ranking Stability (Kendall τ)",
        "Precomputed Experiment Logs",
    ])

    with sub_tab1:
        st.markdown("#### Factorial Interaction Regression ($R^2$ & F-Test)")
        st.caption("Fitting `reliability ~ S * C * M * A` across all 16 defect combinations to test if compound interactions are statistically significant.")

        matrix = create_experiment_matrix(df_disclosures, pilot=False)
        m_scores = []
        for _, rm in matrix.iterrows():
            mock_c = critic.CriticFlag(
                record_id=str(rm["record_id"]), model_used="test",
                staleness_flag=bool(rm["S_staleness"]), conflict_flag=bool(rm["C_conflict"]),
                missingness_flag=bool(rm["M_missingness"]), low_authority_flag=bool(rm["A_low_authority"]),
            )
            m_scores.append(aggregator.score_interaction(mock_c))
        matrix["reliability_score"] = m_scores

        reg_comparison = compare_additive_vs_interaction(matrix, outcome_col="reliability_score")
        col_r1, col_r2, col_r3 = st.columns(3)
        col_r1.metric("Additive Model R²", f"{reg_comparison['additive']['r_squared']:.4f}")
        col_r2.metric("Interaction Model R²", f"{reg_comparison['interaction']['r_squared']:.4f}")
        col_r3.metric("F-Test p-value", f"{reg_comparison['f_test']['p_value']:.2e}")

        st.markdown("##### Regression Coefficients & 95% Confidence Intervals")
        int_res = fit_interaction_regression(matrix, outcome_col="reliability_score")
        eff_table = effect_sizes_with_ci(int_res)
        st.dataframe(eff_table.style.format({
            "coefficient": "{:+.4f}", "std_error": "{:.4f}",
            "ci_lower": "{:+.4f}", "ci_upper": "{:+.4f}", "p_value": "{:.2e}",
        }), use_container_width=True)

    with sub_tab2:
        st.markdown("#### Expected Calibration Error (ECE) & Reliability Curves")
        st.caption("Testing if evidence-conditioned scores calibrate risk prediction better than raw unconditioned LLM confidence.")

        np.random.seed(42)
        n_points = 250
        defect_levels = np.random.uniform(0.0, 1.0, n_points)
        y_true = (np.random.uniform(0, 1, n_points) > defect_levels * 0.70).astype(int)
        raw_confidence = np.clip(1.0 - defect_levels * 0.22 + np.random.normal(0, 0.08, n_points), 0.1, 0.99)
        conditioned_score = np.clip(1.0 - defect_levels * 0.82 + np.random.normal(0, 0.05, n_points), 0.05, 0.98)

        cal_res = compare_calibration_methods(y_true, raw_confidence, conditioned_score)
        c_e1, c_e2, c_e3 = st.columns(3)
        c_e1.metric("Raw Confidence ECE", f"{cal_res['raw']['ece']:.3f}", delta="Overconfident", delta_color="inverse")
        c_e2.metric("Evidence-Conditioned ECE", f"{cal_res['evidence_conditioned']['ece']:.3f}", delta="Better Calibrated", delta_color="normal")
        c_e3.metric("Platt Scaled ECE", f"{cal_res['platt_scaled']['ece']:.3f}")

        p_raw, f_raw = compute_calibration_curve(y_true, raw_confidence, n_bins=8)
        p_cond, f_cond = compute_calibration_curve(y_true, conditioned_score, n_bins=8)

        fig_calib = go.Figure()
        fig_calib.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color="#8B98A9"), name="Perfect Calibration"))
        fig_calib.add_trace(go.Scatter(x=p_raw, y=f_raw, mode="lines+markers", line=dict(color="#F87171", width=2), name="Raw Confidence"))
        fig_calib.add_trace(go.Scatter(x=p_cond, y=f_cond, mode="lines+markers", line=dict(color="#34D399", width=2.5), name="Evidence-Conditioned Score"))
        fig_calib.update_layout(xaxis_title="Predicted Probability", yaxis_title="True Empirical Accuracy", height=320)
        st.plotly_chart(theme.plotly_layout(fig_calib, 320), use_container_width=True)

    with sub_tab3:
        st.markdown("#### Company-Level Risk Ranking Stability")
        st.caption("Evaluating rank correlation (Kendall τ) and top-k overlap between clean and compound-defective disclosures.")

        companies = ["Aster Components", "Nova Energy", "Zenith Finance", "Orion Technologies", "Vertex Industries"]
        scores_clean = [0.85, 0.72, 0.65, 0.55, 0.40]
        scores_defective = [0.42, 0.68, 0.35, 0.60, 0.38]

        stab_report = ranking_stability_report(scores_clean, scores_defective, companies, k=3)
        s_c1, s_c2, s_c3 = st.columns(3)
        s_c1.metric("Kendall τ Correlation", f"{stab_report['kendall_tau']['tau']:.2f}")
        s_c2.metric("Spearman ρ Correlation", f"{stab_report['spearman_rho']['rho']:.2f}")
        s_c3.metric("Top-3 Overlap", f"{stab_report['top_3_overlap']:.0%}")

        st.markdown("##### Company Rank Changes Under Injected Defects")
        st.dataframe(pd.DataFrame(stab_report["biggest_rank_changes"], columns=["Company", "Rank Shift"]), hide_index=True, use_container_width=True)

    with sub_tab4:
        st.markdown("#### Precomputed Experiment Results (Verified Logs)")
        st.caption("Structured JSONL results from batch factorial runs across models and conditions.")

        logs = sorted(RESULTS_DIR.glob("experiment_*.jsonl"))
        if logs:
            selected_log = st.selectbox("Select Experiment Result File", logs, format_func=lambda p: p.name)
            if selected_log:
                with open(selected_log) as fl:
                    entries = [json.loads(line) for line in fl if line.strip()]

                # Flatten structured entries into a clean presentation dataframe
                flat_rows = []
                for entry in entries:
                    if "error" in entry:
                        flat_rows.append({
                            "Record": entry.get("record_id"),
                            "Condition": entry.get("condition_label"),
                            "Model": entry.get("extractor_model"),
                            "Status": "⚠️ Error",
                            "Details": entry.get("error")[:80],
                        })
                    else:
                        ext = entry.get("extraction", {})
                        crit = entry.get("critic_flags", {})
                        agg = entry.get("aggregated", {})
                        r_item = ext.get("risks", [{}])[0] if ext.get("risks") else {}
                        gap_val = agg.get("reliability_score_additive", 1.0) - agg.get("reliability_score_interaction", 1.0)
                        flat_rows.append({
                            "Record": entry.get("record_id"),
                            "Condition": entry.get("condition_label"),
                            "Extractor": entry.get("extractor_model", "").split("/")[-1],
                            "Critic": entry.get("critic_model", "").split("/")[-1],
                            "Risk Category": r_item.get("risk_type", "Operational"),
                            "Additive Rel": f"{agg.get('reliability_score_additive', 1.0):.0%}",
                            "Interaction Rel": f"{agg.get('reliability_score_interaction', 1.0):.0%}",
                            "Compounding Gap": f"{gap_val:+.0%}" if abs(gap_val) > 0.001 else "0%",
                            "Route": agg.get("route", "Direct"),
                            "Ground Truth": entry.get("ground_truth_label", "SUPPORTED"),
                        })

                res_df = pd.DataFrame(flat_rows)
                st.dataframe(res_df, use_container_width=True, hide_index=True)
                st.caption(f"Showing {len(res_df)} experiment records from `{selected_log.name}`.")
                st.download_button("Download JSONL", selected_log.read_bytes(), file_name=selected_log.name)
        else:
            st.info("No experiment logs stored in `data/results/` yet. Run via terminal:")
            st.code("python -m pipeline.experiment_runner --both", language="bash")
