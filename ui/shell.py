"""Per-page bootstrap: sys.path, session defaults, theme. Safe to call on every page."""
import sys
from pathlib import Path

APPS = Path(__file__).resolve().parents[1]
ROOT = APPS.parent
for p in (str(APPS), str(ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

import streamlit as st  # noqa: E402

from racefin_core import account as acc  # noqa: E402
from ui import theme  # noqa: E402


def setup():
    ss = st.session_state
    ss.setdefault("theme", "dark")
    ss.setdefault("mode", "Simple")
    ss.setdefault("watchlist", [])
    ss.setdefault("shortlist", [])
    ss.setdefault("plan", None)
    if "profile" not in ss:
        ss["profile"] = acc.load_profile()
        ss["account"] = acc.load_account()
    theme.inject()


def advanced() -> bool:
    return st.session_state.get("mode") == "Advanced"
