import plotly.graph_objects as go
import streamlit as st

from ui.shell import setup, advanced

setup()
from racefin_core import account as acc, algos, data  # noqa: E402
from ui import theme  # noqa: E402

ss = st.session_state
prof = ss["profile"]
tbl = data.company_table()
prices, _ = data.load_prices()

theme.steps(2, len(ss["shortlist"]))
st.title("Optimise portfolio")

# ---- universe: shortlist from Evaluate > watchlist > sensible default
universe = [t for t in (ss["shortlist"] or ss["watchlist"]) if t in prices.columns]
src_note = "your Evaluate shortlist" if ss["shortlist"] else ("your watchlist" if ss["watchlist"] else "a default large-cap set")
if len(universe) < 3:
    universe = [t for t in ["AAPL", "MSFT", "JPM", "JNJ", "XOM", "KO", "WMT", "BA"] if t in prices.columns]
    src_note = "a default large-cap set"


def show_result(bt, aid, capital):
    w = bt["weights"][aid].sort_values(ascending=False)
    m = bt["metrics"].loc[aid]
    wl = bt["wealth"][aid]
    a, b = st.columns([1.35, 1])
    with a:
        theme.card(f'<div class="rf-eyebrow">Out-of-sample value · {algos.BY_ID[aid].name}</div>'
                   f'<div class="rf-hero">${wl.iloc[-1]:,.0f}</div>'
                   f'{theme.pill(f"{m["Total return"]:+.1%} holdout", "ok" if m["Total return"] >= 0 else "bad")}')
        fig = go.Figure(go.Scatter(x=wl.index, y=wl.values, mode="lines", line=dict(color=theme.pal()["accent"], width=2.4),
                                   fill="tozeroy", fillcolor=theme.pal()["accent_bg"]))
        fig.update_yaxes(range=[min(wl.min(), capital) * 0.97, wl.max() * 1.02])
        st.plotly_chart(theme.plotly_layout(fig, 240), use_container_width=True)
    with b:
        sc = algos.risk_score(m["Annualised volatility"])
        tier = "Low" if sc < 45 else ("Moderate" if sc < 70 else "Elevated")
        fig = go.Figure(go.Pie(labels=w.index, values=w.values, hole=0.66, sort=False,
                               marker=dict(colors=theme.SERIES, line=dict(color=theme.pal()["surface"], width=2)), textinfo="none"))
        fig.update_layout(annotations=[dict(text=f"<b>{sc}</b><br><span style='font-size:11px'>{tier} risk</span>", showarrow=False,
                                            font=dict(size=22, color=theme.pal()["text"]))], showlegend=False)
        st.plotly_chart(theme.plotly_layout(fig, 200), use_container_width=True)
        st.markdown(theme.rows([("Volatility", f"{m['Annualised volatility']:.1%}"), ("Max drawdown", f"{m['Maximum drawdown']:.1%}"),
                                ("Sharpe", f"{m['Sharpe ratio']:.2f}")]), unsafe_allow_html=True)
    st.dataframe(w.rename("Weight").to_frame().assign(**{"Amount ($)": w * capital}).style.format({"Weight": "{:.1%}", "Amount ($)": "${:,.0f}"}),
                 use_container_width=True)
    if st.button("Use this plan →", type="primary", key=f"use_{aid}"):
        ss["plan"] = {"algo": aid, "weights": w.to_dict(), "capital": capital}
        st.success("Plan saved. Go to **Profile & Account** to invest it in your paper account.")


# =============================== SIMPLE ===============================
if not advanced():
    st.caption(f"Choose how adventurous you want to be. Universe: {src_note} ({len(universe)} stocks).")
    default_bucket = prof.bucket()
    bucket = st.segmented_control("Style", list(algos.BUCKETS), default=default_bucket, key="bucket")
    bucket = bucket or default_bucket
    st.markdown(f'<div class="rf-sub">{algos.BUCKETS[bucket]}  · matches your profile: <b>{default_bucket}</b></div>', unsafe_allow_html=True)
    capital = st.number_input("Amount to invest ($)", 1000.0, 50_000_000.0, float(prof.capital), 5000.0)
    algo = algos.default_algo(bucket)
    if st.button("Build my portfolio", type="primary"):
        bt = algos.backtest(prices[universe].dropna(), [algo.id], capital)
        ss["last_bt"] = (bt, algo.id, capital)
    if "last_bt" in ss:
        show_result(*ss["last_bt"])
    st.caption("Backtest = weights fitted on the first 80% of history, evaluated on the last 20%. Past performance does not predict the future.")
    st.stop()

# =============================== ADVANCED ===============================
c1, c2 = st.columns([2, 1])
universe = c1.multiselect("Universe", list(prices.columns), default=universe)
capital = c2.number_input("Capital ($)", 1000.0, 50_000_000.0, float(prof.capital), 5000.0)
train = st.slider("Training split", 0.6, 0.9, 0.8, 0.05)

st.markdown("#### Algorithm library")
bucket_f = st.segmented_control("Filter by bucket", ["All"] + list(algos.BUCKETS), default="All")
lib = [a for a in algos.REGISTRY if bucket_f in (None, "All") or bucket_f in a.tags]
cols = st.columns(3)
chosen = []
for i, a in enumerate(lib):
    with cols[i % 3]:
        tags = "".join(theme.pill(t, "ok" if t == "Low risk" else ("warn" if t == "Balanced" else "bad")) for t in a.tags)
        status = theme.pill("ready", "ok") if a.status == "ready" else theme.pill("coming soon", "neutral")
        theme.card(f'<div class="rf-h">{a.name}</div>{tags}{status}<div class="rf-sub" style="margin-top:6px">{a.blurb}</div>',
                   "rf-stub" if a.status == "stub" else "")
        if a.status == "ready":
            if st.checkbox("Compare", value=a.id in ("hrp", "pca"), key=f"cmp_{a.id}"):
                chosen.append(a.id)
        else:
            st.caption("Placeholder — to be implemented from the paper review")

if len(universe) < 3 or not chosen:
    st.warning("Pick at least 3 stocks and one ready algorithm.")
    st.stop()
if st.button("Run comparison", type="primary"):
    ss["last_bt_adv"] = (algos.backtest(prices[universe].dropna(), chosen, capital, train), capital)
if "last_bt_adv" in ss:
    bt, cap = ss["last_bt_adv"]
    fig = go.Figure()
    for i, aid in enumerate(bt["wealth"].columns):
        fig.add_trace(go.Scatter(x=bt["wealth"].index, y=bt["wealth"][aid], name=algos.BY_ID[aid].name,
                                 line=dict(color=theme.SERIES[i % 8], width=2.2)))
    st.plotly_chart(theme.plotly_layout(fig, 300), use_container_width=True)
    st.dataframe(bt["metrics"].rename(index=lambda i: algos.BY_ID[i].name).style.format("{:.2%}", subset=["Total return", "Annualised return", "Annualised volatility", "Maximum drawdown"]).format("{:.2f}", subset=["Sharpe ratio"]),
                 use_container_width=True)
    pick = st.selectbox("Inspect / use algorithm", list(bt["weights"]), format_func=lambda i: algos.BY_ID[i].name)
    show_result(bt, pick, cap)
