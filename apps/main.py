import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st  # noqa: E402

st.set_page_config(page_title="RACE-Fin", page_icon="◈", layout="wide", initial_sidebar_state="expanded")

from ui.shell import setup  # noqa: E402

setup()

with st.sidebar:
    st.markdown("## ◈ RACE-Fin")
    st.caption("Risk-aware investing, evidence first.")
    st.segmented_control("Mode", ["Simple", "Advanced"], key="mode", default="Simple", help="Simple = guided. Advanced = full controls + SCAM lab.")
    st.segmented_control("Theme", ["dark", "light"], key="theme", default="dark")
    p = st.session_state["profile"]
    st.markdown(
        f'<div class="rf-card"><div class="rf-eyebrow">Your profile</div><b>{p.name or "Guest"}</b><br>'
        f'<span class="rf-sub">Suggested bucket: {p.bucket()}</span></div>', unsafe_allow_html=True)

nav = st.navigation([
    st.Page("views/explore.py", title="Explore", icon=":material/travel_explore:", default=True),
    st.Page("views/evaluate.py", title="Evaluate risk", icon=":material/fact_check:"),
    st.Page("views/optimise.py", title="Optimise", icon=":material/tune:"),
    st.Page("views/account.py", title="Profile & Account", icon=":material/account_circle:"),
])
nav.run()
