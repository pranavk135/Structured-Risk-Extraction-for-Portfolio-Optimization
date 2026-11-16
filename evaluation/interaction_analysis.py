"""Interaction analysis: OLS regression with interaction terms.

Fits additive-only and full-interaction models, compares them via F-test,
and extracts effect sizes with confidence intervals.
"""
from __future__ import annotations

import pandas as pd
import statsmodels.formula.api as smf


# ── Public API ─────────────────────────────────────────────────────────────

def fit_interaction_regression(
    df: pd.DataFrame,
    outcome_col: str = "reliability_score",
) -> object:
    """Fit OLS with all main effects and interactions up to 4-way.

    Expects columns S_staleness, C_conflict, M_missingness, A_low_authority
    (binary 0/1) and the outcome column.

    Returns:
        A fitted statsmodels OLSResults.
    """
    formula = (
        f"{outcome_col} ~ S_staleness * C_conflict"
        f" * M_missingness * A_low_authority"
    )
    return smf.ols(formula, data=df).fit()


def fit_additive_regression(
    df: pd.DataFrame,
    outcome_col: str = "reliability_score",
) -> object:
    """Fit OLS with main effects only (no interactions)."""
    formula = (
        f"{outcome_col} ~ S_staleness + C_conflict"
        f" + M_missingness + A_low_authority"
    )
    return smf.ols(formula, data=df).fit()


def compare_additive_vs_interaction(
    df: pd.DataFrame,
    outcome_col: str = "reliability_score",
) -> dict:
    """Compare the additive-only and interaction-aware models.

    Returns:
        Dict with R², AIC, BIC for each model plus an F-test.
    """
    add_model = fit_additive_regression(df, outcome_col)
    int_model = fit_interaction_regression(df, outcome_col)

    f_stat, f_pval, df_diff = int_model.compare_f_test(add_model)

    return {
        "additive": {
            "r_squared":     float(add_model.rsquared),
            "adj_r_squared": float(add_model.rsquared_adj),
            "aic":           float(add_model.aic),
            "bic":           float(add_model.bic),
        },
        "interaction": {
            "r_squared":     float(int_model.rsquared),
            "adj_r_squared": float(int_model.rsquared_adj),
            "aic":           float(int_model.aic),
            "bic":           float(int_model.bic),
        },
        "f_test": {
            "f_statistic": float(f_stat),
            "p_value":     float(f_pval),
            "df_diff":     float(df_diff),
        },
        "interaction_summary": int_model.summary().as_text(),
    }


def effect_sizes_with_ci(model, alpha: float = 0.05) -> pd.DataFrame:
    """Extract coefficients, CIs, and significance from a fitted model.

    Returns:
        DataFrame sorted by p-value.
    """
    ci = model.conf_int(alpha=alpha)
    results = pd.DataFrame(
        {
            "coefficient": model.params,
            "std_error":   model.bse,
            "ci_lower":    ci[0],
            "ci_upper":    ci[1],
            "p_value":     model.pvalues,
            "significant": model.pvalues < alpha,
        }
    )
    return results.sort_values("p_value")
