"""Design system: dark (default) + light palettes, one CSS injector, tiny HTML helpers."""
import streamlit as st

PAL = {
    "dark": dict(bg="#0B0F14", surface="#121820", surface2="#18212C", border="#243040", text="#E6EDF3",
                 muted="#8B98A9", accent="#34D399", accent_bg="rgba(52,211,153,.14)", warn="#F59E0B",
                 warn_bg="rgba(245,158,11,.14)", bad="#F87171", bad_bg="rgba(248,113,113,.14)", blue="#60A5FA",
                 input="#0F151D"),
    "light": dict(bg="#F6F7F4", surface="#FFFFFF", surface2="#F1F3F0", border="#E2E5E0", text="#111827",
                  muted="#6B7280", accent="#0F9D6E", accent_bg="rgba(15,157,110,.12)", warn="#B45309",
                  warn_bg="rgba(180,83,9,.12)", bad="#B91C1C", bad_bg="rgba(185,28,28,.10)", blue="#2563EB",
                  input="#FFFFFF"),
}
SERIES = ["#34D399", "#60A5FA", "#F59E0B", "#F87171", "#A78BFA", "#2DD4BF", "#FB923C", "#94A3B8"]


def pal() -> dict:
    return PAL[st.session_state.get("theme", "dark")]


def inject():
    p = pal()
    st.markdown(f"""
<style>
.stApp, header[data-testid="stHeader"] {{ background:{p['bg']} !important; color:{p['text']}; }}
section[data-testid="stSidebar"] {{ background:{p['surface']} !important; border-right:1px solid {p['border']}; }}
.stApp p, .stApp label, .stApp span, .stApp li, .stApp h1, .stApp h2, .stApp h3, .stApp h4,
.stApp [data-testid="stMarkdownContainer"], .stApp [data-testid="stWidgetLabel"] {{ color:{p['text']}; }}
.stApp [data-testid="stCaptionContainer"], .stApp small {{ color:{p['muted']} !important; }}
div[data-baseweb="input"], div[data-baseweb="select"] > div, div[data-baseweb="textarea"], textarea,
.stNumberInput input, .stTextInput input {{ background:{p['input']} !important; color:{p['text']} !important;
  border-color:{p['border']} !important; }}
[data-testid="stSelectbox"] div[data-baseweb="select"] div, [data-testid="stMultiSelect"] div[data-baseweb="select"] div,
[data-testid="stSegmentedControl"] button, button[data-testid^="stBaseButton-segmented"],
[data-testid="stFileUploaderDropzone"] {{ background:{p['input']} !important; color:{p['text']} !important; border-color:{p['border']} !important; }}
button[data-testid="stBaseButton-segmentedControlActive"] {{ background:{p['accent_bg']} !important; color:{p['accent']} !important; border-color:{p['accent']} !important; }}
div[data-baseweb="popover"] *, ul[role="listbox"] {{ background:{p['surface']} !important; color:{p['text']} !important; }}
.stButton > button, .stDownloadButton > button {{ background:{p['surface2']}; color:{p['text']}; border:1px solid {p['border']};
  border-radius:10px; font-weight:600; }}
.stButton > button:hover {{ border-color:{p['accent']}; color:{p['accent']}; }}
.stButton > button[kind="primary"] {{ background:{p['accent']}; color:#06130D; border:none; }}
.stButton > button[kind="primary"]:hover {{ filter:brightness(1.08); color:#06130D; }}
div[data-testid="stDataFrame"] {{ border:1px solid {p['border']}; border-radius:12px; }}
div[data-baseweb="tab-list"] {{ gap:10px; border-bottom:1px solid {p['border']}; margin-bottom:18px; }}
button[data-baseweb="tab"] {{ background:transparent !important; color:{p['muted']} !important; font-weight:600; font-size:14px; padding:10px 18px; }}
button[data-baseweb="tab"][aria-selected="true"] {{ color:{p['accent']} !important; border-bottom: 2px solid {p['accent']} !important; }}
.block-container {{ padding-top:1.8rem; max-width:1340px; }}
/* components */
.rf-card {{ background:{p['surface']}; border:1px solid {p['border']}; border-radius:16px; padding:16px 18px; margin-bottom:12px; }}
.rf-card:hover.hov {{ border-color:{p['accent']}; }}
.rf-eyebrow {{ font-size:11px; font-weight:700; letter-spacing:.09em; text-transform:uppercase; color:{p['muted']}; margin-bottom:6px; }}
.rf-hero {{ font-size:34px; font-weight:750; line-height:1.1; }}
.rf-h {{ font-size:16px; font-weight:700; margin:2px 0 8px; }}
.rf-sub {{ color:{p['muted']}; font-size:13px; }}
.rf-pill {{ display:inline-block; font-size:12px; font-weight:650; padding:3px 10px; border-radius:999px; margin:0 6px 4px 0; }}
.rf-ok {{ background:{p['accent_bg']}; color:{p['accent']}; }} .rf-warn {{ background:{p['warn_bg']}; color:{p['warn']}; }}
.rf-bad {{ background:{p['bad_bg']}; color:{p['bad']}; }} .rf-neutral {{ background:{p['surface2']}; color:{p['muted']}; }}
.rf-row {{ display:flex; justify-content:space-between; padding:7px 0; border-bottom:1px solid {p['border']}; font-size:13.5px; }}
.rf-row:last-child {{ border-bottom:none; }} .rf-row span:first-child {{ color:{p['muted']}; }}
.rf-steps {{ display:flex; gap:8px; margin:4px 0 18px; flex-wrap:wrap; }}
.rf-step {{ padding:6px 14px; border-radius:999px; border:1px solid {p['border']}; font-size:12.5px; color:{p['muted']}; background:{p['surface']}; }}
.rf-step.on {{ border-color:{p['accent']}; color:{p['accent']}; background:{p['accent_bg']}; }}
.rf-stub {{ border-style:dashed; opacity:.75; }}
</style>""", unsafe_allow_html=True)


def pill(text, kind="neutral"):
    return f'<span class="rf-pill rf-{kind}">{text}</span>'


def card(inner: str, cls: str = "") -> None:
    st.markdown(f'<div class="rf-card {cls}">{inner}</div>', unsafe_allow_html=True)


def rows(pairs) -> str:
    return "".join(f'<div class="rf-row"><span>{k}</span><span><b>{v}</b></span></div>' for k, v in pairs)


def spark(vals, color=None, w=120, h=34) -> str:
    if len(vals) < 2:
        return ""
    lo, hi = min(vals), max(vals)
    rng = (hi - lo) or 1
    pts = " ".join(f"{i * w / (len(vals) - 1):.1f},{h - (v - lo) / rng * (h - 4) - 2:.1f}" for i, v in enumerate(vals))
    color = color or (pal()["accent"] if vals[-1] >= vals[0] else pal()["bad"])
    return f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}"><polyline fill="none" stroke="{color}" stroke-width="2" points="{pts}"/></svg>'


def steps(active: int, shortlist: int = 0):
    labels = ["1 · Explore", f"2 · Evaluate risk{f' ({shortlist})' if shortlist else ''}", "3 · Optimise", "4 · Invest"]
    st.markdown('<div class="rf-steps">' + "".join(
        f'<span class="rf-step {"on" if i == active else ""}">{l}</span>' for i, l in enumerate(labels)) + "</div>",
        unsafe_allow_html=True)


def plotly_layout(fig, h=260):
    p = pal()
    fig.update_layout(height=h, margin=dict(l=8, r=8, t=8, b=8), paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", font=dict(color=p["text"], size=12),
                      xaxis=dict(gridcolor=p["border"], zeroline=False, tickfont=dict(color=p["text"])),
                      yaxis=dict(gridcolor=p["border"], zeroline=False, tickfont=dict(color=p["text"])),
                      legend=dict(orientation="h", y=1.12, font=dict(color=p["text"])))
    return fig
