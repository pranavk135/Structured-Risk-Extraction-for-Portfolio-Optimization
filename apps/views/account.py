import pandas as pd
import streamlit as st

from ui.shell import setup, advanced

setup()
from racefin_core import account as acc, data  # noqa: E402
from ui import theme  # noqa: E402

ss = st.session_state
tbl = data.company_table()
theme.steps(3, len(ss["shortlist"]))
st.title("Profile & Account")
st.caption("Simulated trading account — no real money or brokerage connection. Prices come from the dataset snapshot.")

prof, acct = ss["profile"], ss["account"]
left, right = st.columns([1, 1.25], gap="large")

with left:
    st.markdown("#### About you")
    with st.form("profile_form"):
        name = st.text_input("Name", prof.name)
        c1, c2 = st.columns(2)
        age = c1.number_input("Age", 16, 100, prof.age)
        inc = c2.number_input("Monthly income ($)", 0.0, 10_000_000.0, float(prof.monthly_income), 1000.0)
        cap = c1.number_input("Investable capital ($)", 0.0, 100_000_000.0, float(prof.capital), 5000.0)
        hor = c2.number_input("Horizon (years)", 1, 40, prof.horizon_years)
        risk = st.select_slider("Risk comfort", [1, 2, 3, 4, 5], prof.risk_tolerance,
                                format_func=lambda v: {1: "1 · Very low", 2: "2 · Low", 3: "3 · Medium", 4: "4 · High", 5: "5 · Very high"}[v])
        goal = st.selectbox("Goal", acc.GOALS, acc.GOALS.index(prof.goal))
        exp = st.selectbox("Experience", acc.LEVELS, acc.LEVELS.index(prof.experience))
        if st.form_submit_button("Save profile", type="primary"):
            ss["profile"] = prof = acc.Profile(name, int(age), inc, cap, int(hor), int(risk), goal, exp)
            acc.save_profile(prof)
            st.success("Saved.")
    theme.card(f'<div class="rf-eyebrow">Suggested style</div><div class="rf-hero" style="font-size:26px">{prof.bucket()}</div>'
               f'<div class="rf-sub">Derived from risk comfort, horizon, age and goal. Used as the default in Optimise.</div>')

with right:
    px = tbl["price"]
    held = sum(p["shares"] * float(px.get(t, 0)) for t, p in acct["positions"].items())
    cost = sum(p["shares"] * p["avg"] for p in acct["positions"].values())
    pnl = held - cost
    theme.card(f'<div class="rf-eyebrow">Paper account</div><div class="rf-hero">${acct["cash"] + held:,.2f}</div>'
               f'{theme.pill(f"P&L {pnl:+,.0f}", "ok" if pnl >= 0 else "bad")}{theme.pill(f"Cash ${acct["cash"]:,.0f}", "neutral")}'
               f'{theme.pill(f"Invested ${held:,.0f}", "neutral")}')
    f1, f2, f3 = st.columns([1.4, 1, 1])
    amt = f1.number_input("Add virtual cash ($)", 0.0, 100_000_000.0, float(prof.capital), 10_000.0, label_visibility="collapsed")
    if f2.button("Fund account", use_container_width=True):
        acct["cash"] += amt; acc.save_account(acct); st.rerun()
    if f3.button("Reset", use_container_width=True):
        ss["account"] = acct = acc.new_account(); acc.save_account(acct); st.rerun()

    plan = ss.get("plan")
    st.markdown("#### Pending plan")
    if not plan:
        st.info("No plan yet. Build one in **Optimise** and click *Use this plan*.")
    else:
        st.markdown(theme.pill(plan["algo"].upper(), "ok") + theme.pill(f"${plan['capital']:,.0f}", "neutral")
                    + "".join(theme.pill(f"{t} {w:.0%}", "neutral") for t, w in plan["weights"].items()), unsafe_allow_html=True)
        if acct["cash"] < 1:
            st.warning("Fund your account first.")
        elif st.button("Execute plan (paper trade)", type="primary"):
            prices = {t: tbl.loc[t, "price"] for t in plan["weights"] if t in tbl.index}
            acct = acc.execute_plan(acct, {t: plan["weights"][t] for t in prices}, prices, plan["capital"], plan["algo"])
            ss["account"] = acct; acc.save_account(acct); ss["plan"] = None
            st.success("Executed."); st.rerun()

    st.markdown("#### Holdings")
    if acct["positions"]:
        df = pd.DataFrame([{"Ticker": t, "Shares": p["shares"], "Avg cost": p["avg"], "Price": float(px.get(t, 0)),
                            "Value": p["shares"] * float(px.get(t, 0))} for t, p in acct["positions"].items()])
        st.dataframe(df.style.format({"Shares": "{:.3f}", "Avg cost": "${:,.2f}", "Price": "${:,.2f}", "Value": "${:,.0f}"}),
                     hide_index=True, use_container_width=True)
    else:
        st.caption("No holdings yet.")
    if advanced() and acct["trades"]:
        with st.expander("Trade history"):
            st.dataframe(pd.DataFrame(acct["trades"]), hide_index=True, use_container_width=True)
