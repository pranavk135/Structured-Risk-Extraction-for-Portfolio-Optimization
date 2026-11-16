import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from ui.shell import setup, advanced

setup()
from racefin_core import data, riskscan  # noqa: E402
from ui import theme  # noqa: E402

PER_PAGE = 12
tbl = data.company_table()
prices, info = data.load_prices()

theme.steps(0, len(st.session_state["shortlist"]))
st.title("Explore companies")
st.caption(f"{len(tbl)} S&P 500 stocks · price snapshot {info['start']:%b %Y} – {info['end']:%b %Y} (live feed is a Phase-II stub)")


@st.dialog("Company details", width="large")
def details(t: str):
    r = tbl.loc[t]
    st.markdown(f"### {t} · {r['name']}")
    s = prices[t].dropna().iloc[-504:]
    fig = go.Figure(go.Scatter(x=s.index, y=s.values, mode="lines", line=dict(color=theme.pal()["accent"], width=2.4),
                               fill="tozeroy", fillcolor=theme.pal()["accent_bg"]))
    st.plotly_chart(theme.plotly_layout(fig, 260), use_container_width=True)
    a, b, c, d = st.columns(4)
    a.metric("Price", f"${r['price']:,.2f}"); b.metric("1Y return", f"{r['ret_1y']:+.1%}")
    c.metric("Volatility", f"{r['vol']:.1%}"); d.metric("Max drawdown", f"{r['max_dd']:.1%}")
    res = riskscan.scan_filing(t) if t in set(data.load_manifest()["ticker"]) else None
    if res:
        st.markdown("**Top disclosed risk themes (latest 10-K)**")
        st.markdown("".join(theme.pill(f"{k} {v:.0%}", "warn") for k, v in list(res["exposure"].items())[:4]), unsafe_allow_html=True)
    else:
        st.caption("No 10-K in the library for this company yet.")
    wl = st.session_state["watchlist"]
    if st.button("★ Remove from watchlist" if t in wl else "☆ Add to watchlist", use_container_width=True):
        wl.remove(t) if t in wl else wl.append(t)
        st.rerun()


# ---------------- filters ----------------
f1, f2, f3, f4 = st.columns([2.2, 1.2, 1.4, 1.2])
q = f1.text_input("Search", placeholder="Ticker or company…", label_visibility="collapsed")
risk = f2.selectbox("Risk", ["Any risk", "Low", "Medium", "High"], label_visibility="collapsed")
sort = f3.selectbox("Sort", ["Top 1Y return", "Lowest volatility", "Best Sharpe", "A–Z"], label_visibility="collapsed")
only_wl = f4.toggle("★ Watchlist", value=False)

df = tbl
if q:
    m = q.lower()
    df = df[df.index.str.lower().str.contains(m) | df["name"].str.lower().str.contains(m)]
if risk != "Any risk":
    df = df[df["risk_bucket"] == risk]
if only_wl:
    df = df[df.index.isin(st.session_state["watchlist"])]
df = {"Top 1Y return": df.sort_values("ret_1y", ascending=False), "Lowest volatility": df.sort_values("vol"),
      "Best Sharpe": df.sort_values("sharpe", ascending=False), "A–Z": df.sort_index()}[sort]

if advanced():
    with st.expander("Risk–return map of all stocks", expanded=False):
        fig = px.scatter(tbl.reset_index(), x="vol", y="ret_1y", color="risk_bucket", hover_name="ticker",
                         color_discrete_map={"Low": "#34D399", "Medium": "#F59E0B", "High": "#F87171"},
                         labels={"vol": "Annualised volatility", "ret_1y": "1Y return"})
        st.plotly_chart(theme.plotly_layout(fig, 340), use_container_width=True)

if df.empty:
    st.info("No companies match. Try clearing the filters.")
    st.stop()

pages = max(1, -(-len(df) // PER_PAGE))
pg = st.session_state.get("ex_page", 1)
pg = min(pg, pages)
chunk = df.iloc[(pg - 1) * PER_PAGE: pg * PER_PAGE]

kind = {"Low": "ok", "Medium": "warn", "High": "bad"}
for i in range(0, len(chunk), 3):
    cols = st.columns(3)
    for col, (t, r) in zip(cols, list(chunk.iloc[i:i + 3].iterrows())):
        with col:
            star = "★ " if t in st.session_state["watchlist"] else ""
            extra = f'<div class="rf-row"><span>Sharpe · Max DD</span><span><b>{r["sharpe"]:.2f} · {r["max_dd"]:.0%}</b></span></div>' if advanced() else ""
            st.markdown(f"""<div class="rf-card hov">
<div style="display:flex;justify-content:space-between;align-items:flex-start">
 <div><div class="rf-h" style="margin:0">{star}{t}</div><div class="rf-sub">{r['name'][:26]}</div></div>
 {theme.spark(r['spark'])}
</div>
<div style="margin:8px 0 6px"><span class="rf-hero" style="font-size:24px">${r['price']:,.2f}</span>
 {theme.pill(f"{r['ret_1y']:+.1%} 1Y", "ok" if r['ret_1y'] >= 0 else "bad")}</div>
{theme.pill(r['risk_bucket'] + ' risk', kind[r['risk_bucket']])}{theme.pill(r['sector'], 'neutral') if r['sector'] != '—' else ''}
<div class="rf-row"><span>Volatility</span><span><b>{r['vol']:.1%}</b></span></div>{extra}</div>""", unsafe_allow_html=True)
            if st.button("Details", key=f"d_{t}", use_container_width=True):
                details(t)

st.segmented_control("Page", list(range(1, pages + 1)), key="ex_page", default=1, label_visibility="collapsed") if pages > 1 else None
st.caption(f"Showing {len(chunk)} of {len(df)} · page {pg}/{pages}")
