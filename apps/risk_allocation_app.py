from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from portfolio_engine import DEFAULT_TICKERS, prepare_prices, read_price_universe, run_comparison

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "market" / "SP500Data.csv"

st.set_page_config(page_title="RACE-Fin | Portfolio Optimisation", page_icon="📈", layout="wide")
st.title("RACE-Fin | Portfolio Optimisation Demo")
st.caption("Review-stage prototype · HRP vs PCA Eigen-Risk Allocation")
st.info(
    "This demonstrates the downstream portfolio-allocation component of RACE-Fin. "
    "It adapts the HRP and PCA comparison from the mentor-provided notebook. "
    "The corporate-risk extraction and evidence-verification modules are not connected to allocation in this early demo."
)

@st.cache_data(show_spinner=False)
def load_universe(path: str):
    return read_price_universe(path)

try:
    raw_prices, eligible_tickers, dataset_info = load_universe(str(DATA_PATH))
except Exception as exc:
    st.error(f"Could not load the supplied S&P 500 dataset: {exc}")
    st.stop()

available_defaults = [ticker for ticker in DEFAULT_TICKERS if ticker in eligible_tickers]
if len(available_defaults) < 3:
    available_defaults = eligible_tickers[:8]

with st.sidebar:
    st.header("Portfolio setup")
    selected_tickers = st.multiselect(
        "Select stocks / tickers",
        options=eligible_tickers,
        default=available_defaults,
        help="Choose at least three stocks from the mentor-provided historical S&P 500 sample.",
    )
    capital = st.number_input(
        "Initial portfolio value",
        min_value=1_000.0,
        max_value=100_000_000.0,
        value=100_000.0,
        step=5_000.0,
    )
    st.caption("Training / testing split: 80% / 20% (as in the mentor's HRP notebook).")
    run_button = st.button("Run portfolio comparison", type="primary", use_container_width=True)
    st.divider()
    st.caption("Methods: Hierarchical Risk Parity (HRP) and PCA Eigen-Risk.")

st.markdown("### Dataset")
info_cols = st.columns(4)
info_cols[0].metric("Price universe", "S&P 500 sample")
info_cols[1].metric("Usable tickers", f"{dataset_info['universe_size']}")
info_cols[2].metric("Daily observations", f"{dataset_info['observations']}")
info_cols[3].metric("Date range", f"{dataset_info['start']:%Y-%m-%d} to {dataset_info['end']:%Y-%m-%d}")
st.caption(f"Source file: `{dataset_info['source']}` · supplied in the mentor's Hierarchical Risk Parity materials.")

# Run only when the user asks; retain the last result during normal Streamlit reruns.
selection_signature = (tuple(selected_tickers), float(capital))
if st.session_state.get("selection_signature") != selection_signature:
    st.session_state.pop("portfolio_result", None)
    st.session_state["selection_signature"] = selection_signature

if run_button:
    try:
        with st.spinner("Preparing historical prices and calculating HRP / PCA allocations..."):
            cleaned_prices = prepare_prices(raw_prices, selected_tickers)
            result = run_comparison(cleaned_prices, capital=capital, train_fraction=0.80)
        st.session_state["portfolio_result"] = result
    except Exception as exc:
        st.session_state.pop("portfolio_result", None)
        st.error(f"Portfolio comparison could not be completed: {exc}")
        st.stop()

result = st.session_state.get("portfolio_result")
if result is None:
    st.markdown("### What this demo does")
    st.markdown(
        "1. Loads historical prices for the selected companies.\n"
        "2. Uses the first 80% of observations to calculate portfolio weights.\n"
        "3. Compares **HRP** and **PCA Eigen-Risk** allocations.\n"
        "4. Evaluates both portfolios on the later 20% holdout period."
    )
    st.warning("Select at least three tickers in the sidebar, then click **Run portfolio comparison**.")
    st.stop()

st.divider()
st.subheader("1. Portfolio allocations")
left, right = st.columns(2)
for column, strategy, title in [
    (left, "HRP", "Hierarchical Risk Parity (HRP)"),
    (right, "PCA", "PCA Eigen-Risk Allocation"),
]:
    with column:
        st.markdown(f"**{title}**")
        allocation = result.weights[strategy].sort_values(ascending=False)
        chart_data = pd.DataFrame({"Ticker": allocation.index.tolist(), "Weight": allocation.values})
        figure = px.bar(chart_data, x="Ticker", y="Weight", text=chart_data["Weight"].map(lambda x: f"{x:.1%}"))
        figure.update_layout(height=350, yaxis_tickformat=".0%", margin=dict(t=20, b=20))
        st.plotly_chart(figure, use_container_width=True)
        st.dataframe(
            pd.DataFrame({"Ticker": allocation.index, "Weight": allocation.values})
              .assign(Weight=lambda frame: frame["Weight"].map(lambda value: f"{value:.2%}")),
            hide_index=True,
            use_container_width=True,
        )

st.subheader("2. Out-of-sample backtest")
st.caption(
    f"Weights are estimated on {result.train_prices.index.min():%Y-%m-%d} to "
    f"{result.train_prices.index.max():%Y-%m-%d}; evaluated on the later holdout period "
    f"{result.test_returns.index.min():%Y-%m-%d} to {result.test_returns.index.max():%Y-%m-%d}."
)
line = go.Figure()
for strategy in ["HRP", "PCA"]:
    line.add_trace(go.Scatter(x=result.wealth.index, y=result.wealth[strategy], mode="lines", name=strategy))
line.update_layout(height=400, xaxis_title="Holdout date", yaxis_title="Portfolio value", legend_title="Method")
st.plotly_chart(line, use_container_width=True)

metric_display = result.metrics.copy()
for column in ["Total return", "Annualised return", "Annualised volatility", "Maximum drawdown"]:
    metric_display[column] = metric_display[column].map(lambda value: f"{value:.2%}")
metric_display["Sharpe ratio"] = metric_display["Sharpe ratio"].map(lambda value: f"{value:.2f}")
st.dataframe(metric_display, use_container_width=True)

weight_comparison = pd.DataFrame({
    "Ticker": result.prices.columns,
    "HRP weight": result.weights["HRP"].reindex(result.prices.columns).values,
    "PCA weight": result.weights["PCA"].reindex(result.prices.columns).values,
})
st.download_button(
    "Download portfolio weights (CSV)",
    data=weight_comparison.to_csv(index=False).encode("utf-8"),
    file_name="racefin_hrp_pca_weights.csv",
    mime="text/csv",
)

with st.expander("Method and project connection"):
    st.markdown(
        "- **HRP:** groups stocks using correlation distance and hierarchical clustering, "
        "then allocates weights through recursive bisection.\n"
        "- **PCA Eigen-Risk:** decomposes the training covariance matrix into principal components, "
        "retains components explaining 95% of variance, and allocates inversely to each asset's "
        "variance contribution across those components. This is a simple PCA-based prototype.\n"
        "- **Backtest:** uses a chronological 80/20 split, matching the mentor notebook's setup. "
        "The holdout portfolio return is calculated from fixed portfolio weights applied to daily asset returns.\n"
        "- **RACE-Fin connection:** this is the downstream portfolio-allocation module. "
        "The upstream disclosure extraction, evidence verification, risk knowledge graph and calibrated reliability "
        "are planned to provide company-screening inputs in a later integration; they are not simulated here."
    )
    st.caption(
        "The supplied S&P 500 prices are historical teaching data (2018–2019), not live prices. "
        "The backtest excludes transaction costs, taxes, slippage and liquidity constraints. "
        "Results are illustrative and are not investment advice."
    )

st.caption("RACE-Fin review-stage prototype · Two methods only: HRP and PCA.")
