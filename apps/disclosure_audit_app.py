import pandas as pd
import streamlit as st
from pathlib import Path
from racefin_engine import score_evidence, triage_label, compare_xbrl

st.set_page_config(page_title="RACE-Fin | Disclosure Reliability Demo", page_icon="🧾", layout="wide")
DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "synthetic_disclosures" / "racefin_demo_disclosures.csv"
df = pd.read_csv(DATA_PATH)

st.title("RACE-Fin | Disclosure-to-Risk Prioritisation")
st.caption("Expanded Review-2 prototype · 30 synthetic disclosure scenarios · rule-based demonstration, not a calibrated research result")

with st.sidebar:
    st.header("Dataset explorer")
    company_options = ["All companies"] + sorted(df["company"].unique().tolist())
    selected_company = st.selectbox("Company", company_options)
    filtered = df if selected_company == "All companies" else df[df["company"] == selected_company]
    record_id = st.selectbox("Disclosure scenario", filtered["record_id"].tolist())
    row = df[df["record_id"] == record_id].iloc[0]
    st.divider()
    st.markdown("**Defect flags for selected scenario**")
    st.caption("Loaded from the synthetic scenario dataset. You can override them to explore alternative combinations.")
    s = st.checkbox("Staleness (S)", value=bool(row["S_staleness"]))
    c = st.checkbox("Conflict (C)", value=bool(row["C_conflict"]))
    m = st.checkbox("Missingness (M)", value=bool(row["M_missingness"]))
    a = st.checkbox("Low Authority (A)", value=bool(row["A_low_authority"]))

st.info("All company names and disclosure scenarios in this expanded dataset are fictional and synthetic. They are not actual corporate filings.")

# Dataset overview
st.subheader("Dataset overview")
k1, k2, k3, k4 = st.columns(4)
k1.metric("Companies", df["company"].nunique())
k2.metric("Disclosure scenarios", len(df))
k3.metric("Risk categories", df["risk_category"].nunique())
k4.metric("Compound-defect scenarios", int(((df[["S_staleness","C_conflict","M_missingness","A_low_authority"]].sum(axis=1)) >= 2).sum()))

with st.expander("Browse all synthetic disclosure records"):
    view = df[["record_id","company","risk_category","claim_text","source_type","materiality",
               "S_staleness","C_conflict","M_missingness","A_low_authority"]].copy()
    st.dataframe(view, use_container_width=True, hide_index=True)
    st.download_button("Download full synthetic dataset (CSV)", df.to_csv(index=False), "racefin_demo_disclosures.csv", "text/csv")

st.divider()
st.subheader("1 · Selected disclosure and extracted risk claim")
c1, c2, c3 = st.columns(3)
c1.write("**Company**")
c1.write(row["company"])
c2.write("**Fiscal period**")
c2.write(row["fiscal_year"])
c3.write("**Risk category**")
c3.write(row["risk_category"])

st.markdown("**Illustrative extracted claim**")
st.info(row["claim_text"])
st.markdown("**Evidence context / provenance scenario**")
st.write(row["evidence_context"])
st.caption(f"Record: {row['record_id']} · Source type: {row['source_type']} · Industry: {row['industry']}")

st.subheader("2 · Evidence defect profile")
defects = {"Staleness (S)": s, "Conflict (C)": c, "Missingness (M)": m, "Low Authority (A)": a}
cols = st.columns(4)
for col, (label, active) in zip(cols, defects.items()):
    col.metric(label, "Flagged" if active else "Not flagged")

st.subheader("3 · Reliability assessment and routing")
result = score_evidence({}, defects)
a1, a2, a3 = st.columns(3)
a1.metric("Demo evidence reliability", f"{result['reliability']:.0%}")
a2.metric("Materiality", row["materiality"])
a3.metric("Recommended route", result["route"])
st.progress(result["reliability"])
st.caption("The score uses fixed illustrative penalties. It is not learned, calibrated, or validated and must not be reported as an experimental result.")

materiality = st.select_slider("Analyst-assessed materiality", options=["Low","Medium","High"],
                               value=row["materiality"])
triage = triage_label(materiality, result["reliability"])
if "Urgent" in triage:
    st.warning(f"2D triage outcome: {triage}")
else:
    st.success(f"2D triage outcome: {triage}")
if result["route"] == "Human review" or (materiality == "High" and result["reliability"] < 0.70):
    st.warning("The claim is escalated for human verification rather than silently discarded.")

st.subheader("4 · Company-level risk overview")
company_df = df if selected_company == "All companies" else df[df["company"] == selected_company]
summary = company_df.groupby(["company","risk_category"], as_index=False).agg(
    disclosure_records=("record_id","count"),
    compound_defect_records=("record_id", lambda x: int((df.set_index("record_id").loc[x, ["S_staleness","C_conflict","M_missingness","A_low_authority"]].sum(axis=1) >= 2).sum()))
)
st.dataframe(summary, use_container_width=True, hide_index=True)
st.caption("This is a summary of the constructed demonstration scenarios, not a measured profile of real companies.")

st.subheader("5 · Numerical disclosure comparison (XBRL-style)")
st.caption("Enter values to demonstrate a numerical consistency check. This widget does not ingest actual XBRL filings.")
x1, x2, x3 = st.columns(3)
with x1:
    concept = st.selectbox("Concept", ["Revenue","Total debt","Operating profit","Emissions"])
with x2:
    reported = st.number_input("Narrative / PDF value", min_value=0.0, value=1250.0, step=10.0)
with x3:
    structured = st.number_input("Structured XBRL-style value", min_value=0.0, value=1200.0, step=10.0)
tolerance = st.number_input("Tolerance (%)", min_value=0.0, max_value=20.0, value=0.5, step=0.1)
check = compare_xbrl(reported, structured, tolerance)
if check["match"]:
    st.success(f"{concept}: values match within {tolerance:.1f}% tolerance.")
else:
    st.error(f"{concept}: mismatch candidate ({check['difference_pct']:.2f}% difference). Check concept, unit, period and reporting scope before treating this as a true conflict.")

with st.expander("Implementation status and limitations"):
    st.markdown("""
    **Implemented in this prototype**
    - Loads 30 synthetic disclosure scenarios across five fictional companies.
    - Displays selected claims, evidence context, source labels and fiscal period.
    - Loads S/C/M/A scenario flags and permits manual overrides.
    - Shows a transparent heuristic score, review route and materiality/reliability triage.
    - Provides a company/category overview and CSV download.
    - Demonstrates numerical value comparison.

    **Not implemented / planned**
    - Real annual-report/BRSR PDF ingestion and live XBRL parsing.
    - Automated LLM claim extraction and automated defect detection.
    - Expert-annotated ground truth, calibration and compound-defect statistical experiments.
    - Persistent provenance knowledge graph and multi-agent audit.
    - Integration of verified corporate-risk signals into HRP/PCA portfolio allocation.

    **Data warning:** All records are synthetic demonstration scenarios, not real company disclosures or research ground truth.
    """)
st.caption("HRP/PCA portfolio notebooks remain a separate downstream mathematical prototype.")
