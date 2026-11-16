# Dataset notes

All 30 records are **synthetic and fictional**. Company names, claims, evidence contexts, fiscal periods, and reference labels are created solely for demonstrating the RACE-Fin interface. They are not real issuer disclosures, NSE filings, BRSR reports, SEC filings, or verified facts.

The four binary defect columns use 1 = defect present and 0 = not flagged in the constructed scenario:
- S_staleness
- C_conflict
- M_missingness
- A_low_authority

`illustrative_reference_label` is an intended scenario label for UI demonstration only. It is not expert-annotated ground truth and must not be used for research-performance claims.
