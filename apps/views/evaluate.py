import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ui.shell import setup, advanced

setup()
from racefin_core import data, riskscan, scam  # noqa: E402
from racefin_engine import compare_xbrl, triage_label  # noqa: E402
from ui import theme  # noqa: E402

ss = st.session_state
theme.steps(1, len(ss["shortlist"]))
st.title("Evaluate risk")
st.caption("Find the risks a company discloses, check how trustworthy the evidence is, then shortlist stocks.")

man = data.load_manifest()
tbl = data.company_table()
names = data.load_names()

# ---------------- 1. source ----------------
src = st.segmented_control("Source", ["SEC 10-K library", "Upload a document"], default="SEC 10-K library", key="ev_src")
text, meta, defects, label = "", {}, {}, ""
if src == "Upload a document":
    up = st.file_uploader("Annual report / 10-K / risk section (PDF, HTML or TXT)", type=["pdf", "htm", "html", "txt"])
    if not up:
        st.info("Upload a document to extract its risk factors. The multi-agent evidence verifier inspects source authority and temporal staleness.")
        st.stop()
    text = riskscan.text_from_upload(up.name, up.read())
    res = riskscan.analyze_text(text)
    res["defects"] = riskscan.suggest_defects(text, official_source=False)
    label = up.name
else:
    tick = sorted(man["ticker"].unique())
    c1, c2 = st.columns([2, 1])
    t = c1.selectbox("Company", tick, format_func=lambda x: f"{x} · {names.get(x, x)}", index=tick.index("AAPL") if "AAPL" in tick else 0)
    fil = man[man["ticker"] == t].sort_values("filing_date", ascending=False)
    acc_no = None
    if advanced():
        acc_no = c2.selectbox("Filing", fil["accession_number"], format_func=lambda a: f"FY{str(fil.set_index('accession_number').loc[a, 'report_date'])[:4]} · filed {fil.set_index('accession_number').loc[a, 'filing_date']}")
    res = riskscan.scan_filing(t, acc_no)
    label = f"{t} · {names.get(t, t)}"
    if res is None or res["claims"].empty:
        st.warning("No risk text could be extracted for this filing.")
        st.stop()

claims, exposure, defects = res["claims"], res["exposure"], res["defects"]
if claims.empty:
    st.warning("No risk statements detected in this document.")
    st.stop()

# ---------------- 2. results ----------------
rel = scam.reliability(defects, scam.DEFAULT_W, 0.0)
left, right = st.columns([1.5, 1])
with left:
    st.markdown(f'<div class="rf-eyebrow">Risk factors found · {label}</div>', unsafe_allow_html=True)
    cats = list(exposure.items())[:8][::-1]
    fig = go.Figure(go.Bar(x=[v for _, v in cats], y=[k for k, _ in cats], orientation="h",
                           marker_color=theme.pal()["accent"], text=[f"{v:.0%}" for _, v in cats], textposition="outside"))
    fig.update_xaxes(tickformat=".0%", range=[0, max(v for _, v in cats) * 1.25])
    st.plotly_chart(theme.plotly_layout(fig, 300), use_container_width=True)
with right:
    kind = "ok" if rel >= 0.7 else "warn"
    flags = "".join(theme.pill(scam.NAMES[k], "bad") for k, v in defects.items() if v) or theme.pill("No defects detected", "ok")
    theme.card(f'<div class="rf-eyebrow">Evidence reliability</div><div class="rf-hero">{rel:.0%}</div>'
               f'<div style="margin:8px 0">{flags}</div>'
               f'<div class="rf-sub">Auto-detected from filing date, section completeness and source. Conflict check needs claim-level XBRL matching (stub).</div>')
    theme.card(theme.rows([("Risk statements", len(claims)), ("Risk intensity / 10k chars", f"{res['intensity']:.1f}"),
                                   ("Top theme", next(iter(exposure)))]))

st.markdown("#### Top risk statements")
for _, r in claims.head(5 if not advanced() else 8).iterrows():
    sev = "bad" if r["severity_terms"] >= 3 else ("warn" if r["severity_terms"] >= 1 else "neutral")
    theme.card(f'{theme.pill(r["category"], "neutral")}{theme.pill(f"severity {r["severity_terms"]}", sev)}'
               f'<div class="rf-sub" style="margin-top:6px;color:inherit">{r["snippet"]}</div>')
with st.expander("AI verifier (preview)"):
    st.write("Each statement will be checked against its source span and XBRL facts, then labelled SUPPORTED / REFUTED / INSUFFICIENT.")
    v = riskscan.RiskVerifier().verify(claims.iloc[0]["snippet"], text or "")
    st.json(v)

# ---------------- 3. stocks to consider ----------------
st.divider()
st.markdown("### Stocks to consider")
st.caption("Illustrative screening score from risk intensity, evidence reliability and price history — not investment advice.")
lib = data.company_table()
scan = riskscan.scan_library()
cand = scan.join(lib[["name", "price", "ret_1y", "vol"]], how="inner")
if advanced():
    w1, w2, w3, w4 = st.columns(4)
    wr = w1.slider("Low disclosed risk", 0.0, 1.0, 0.4); we = w2.slider("Evidence reliability", 0.0, 1.0, 0.25)
    wm = w3.slider("1Y return", 0.0, 1.0, 0.2); wv = w4.slider("Low volatility", 0.0, 1.0, 0.15)
else:
    wr, we, wm, wv = 0.4, 0.25, 0.2, 0.15
cand["score"] = (wr * (1 - cand["risk_pct"]) + we * cand["reliability"] + wm * cand["ret_1y"].rank(pct=True)
                 + wv * (1 - cand["vol"].rank(pct=True))) / (wr + we + wm + wv)
cand = cand.sort_values("score", ascending=False)

topn = st.slider("How many to shortlist", 3, min(12, len(cand)), min(6, len(cand)))
show = cand.head(topn)
for i in range(0, len(show), 3):
    cols = st.columns(3)
    for col, (t, r) in zip(cols, list(show.iloc[i:i + 3].iterrows())):
        with col:
            rk = "ok" if r["reliability"] >= 0.7 else "warn"
            theme.card(f'<div class="rf-h" style="margin:0">{t} <span class="rf-sub">{r["name"][:22]}</span></div>'
                       f'<div class="rf-hero" style="font-size:26px">{r["score"]*100:.0f}<span class="rf-sub"> / 100</span></div>'
                       f'{theme.pill(r["top_risk"], "neutral")}{theme.pill(f"evidence {r["reliability"]:.0%}", rk)}'
                       f'{theme.rows([("1Y return", f"{r["ret_1y"]:+.1%}"), ("Volatility", f"{r["vol"]:.1%}")])}', "hov")
if st.button("Send shortlist to Optimiser →", type="primary"):
    ss["shortlist"] = show.index.tolist()
    st.success(f"Shortlisted {len(show)} stocks. Open **Optimise** in the sidebar.")

# ---------------- 4. advanced: SCAM lab ----------------
if advanced():
    st.divider()
    st.markdown("### SCAM lab — Staleness · Conflict · Missingness · low-Authority")
    t1, t2, t3 = st.tabs(["Interactive scorer", "Factorial & half-fraction", "Claim table"])
    with t1:
        a, b = st.columns([1, 1.2])
        with a:
            st.caption("Penalty per defect")
            w = {f: st.slider(scam.NAMES[f], 0.0, 0.5, scam.DEFAULT_W[f], 0.01, key=f"w_{f}") for f in scam.F}
            gamma = st.slider("Compound penalty γ (per co-occurring pair)", 0.0, 0.2, 0.05, 0.01,
                              help="γ > 0 makes defects interact non-additively.")
            mat = st.radio("Risk materiality", ["High", "Low"], horizontal=True)
            thr = st.slider("Reliability threshold", 0.4, 0.9, 0.7, 0.05)
        with b:
            st.caption("Toggle defects (pre-filled from the document)")
            x = {f: st.toggle(scam.NAMES[f], value=bool(defects.get(f)), key=f"x_{f}_{label}") for f in scam.F}
            r_ = scam.reliability(x, w, gamma)
            parts = scam.penalty_breakdown(x, w, gamma)
            fig = go.Figure(go.Waterfall(x=["Start"] + [p for p, _ in parts] + ["Reliability"],
                                         y=[1.0] + [v for _, v in parts] + [0],
                                         measure=["absolute"] + ["relative"] * len(parts) + ["total"],
                                         increasing_marker_color="#34D399", decreasing_marker_color="#F87171",
                                         totals_marker_color="#60A5FA"))
            fig.update_yaxes(tickformat=".0%", range=[0, 1.05])
            st.plotly_chart(theme.plotly_layout(fig, 250), use_container_width=True)
            tri = triage_label(mat, r_ if r_ >= thr else min(r_, 0.69))
            st.markdown(theme.pill(f"Reliability {r_:.0%}", "ok" if r_ >= thr else "bad") + theme.pill(tri, "warn"), unsafe_allow_html=True)
            xa, xb = st.columns(2)
            rep = xa.number_input("Narrative figure ($M)", value=124.5); stv = xb.number_input("XBRL figure ($M)", value=124.0)
            xr = compare_xbrl(rep, stv)
            st.markdown(theme.pill(f"XBRL diff {xr['difference_pct']:.2f}% — {'match' if xr['match'] else 'flagged'}", "ok" if xr["match"] else "bad"), unsafe_allow_html=True)
    with t2:
        ff = scam.full_factorial(w, gamma)  # uses the sliders from the Interactive scorer tab
        grid = ff.pivot_table(index=["S", "C"], columns=["M", "A"], values="R")
        fig = go.Figure(go.Heatmap(z=grid.values, x=[f"M={m} A={a}" for m, a in grid.columns],
                                   y=[f"S={s} C={c}" for s, c in grid.index], colorscale="RdYlGn", zmin=0, zmax=1,
                                   text=[[f"{v:.0%}" for v in row] for row in grid.values], texttemplate="%{text}"))
        st.plotly_chart(theme.plotly_layout(fig, 300), use_container_width=True)
        half = ff[ff["half_fraction"]]
        e_full, e_half = scam.effects(ff), scam.effects(half)
        eff = pd.DataFrame({"Full 2⁴ (16 runs)": e_full, "Half 2⁴⁻¹ (8 runs)": e_half}).loc[scam.F + ["S×C", "S×M", "S×A"]]
        eff["Aliased with (half)"] = [scam.ALIASES[i] for i in eff.index]
        st.caption("Effect = mean reliability with factor at '1' minus at '0'. Half-fraction (A = S·C·M) halves the cost; "
                   "main effects match when the model has no 3-way interaction.")
        st.dataframe(eff.style.format({"Full 2⁴ (16 runs)": "{:+.3f}", "Half 2⁴⁻¹ (8 runs)": "{:+.3f}"}), use_container_width=True)
    with t3:
        st.dataframe(claims, use_container_width=True, hide_index=True)
