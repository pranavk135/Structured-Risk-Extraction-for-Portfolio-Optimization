"""Review-stage portfolio engine: HRP and PCA-based eigen-risk allocation.

The implementation follows the mentor-provided HierarchicalRiskParity notebook:
80/20 chronological split, HRP clustering/recursive bisection, and comparison
with a PCA-based eigen-risk allocation.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform

TRADING_DAYS = 252
DEFAULT_TICKERS = ["AAPL", "MSFT", "JPM", "JNJ", "XOM", "KO", "WMT", "BA"]


@dataclass
class PortfolioResult:
    prices: pd.DataFrame
    train_prices: pd.DataFrame
    test_prices: pd.DataFrame
    train_returns: pd.DataFrame
    test_returns: pd.DataFrame
    weights: dict[str, pd.Series]
    wealth: pd.DataFrame
    metrics: pd.DataFrame


def read_price_universe(path: str | Path) -> tuple[pd.DataFrame, list[str], dict]:
    """Read mentor's S&P 500 CSV and return assets with <=30% missing values."""
    raw = pd.read_csv(path)
    date_col = next((c for c in raw.columns if c.lower() == "date"), None)
    if date_col is None:
        raise ValueError("The price dataset must contain a Date column.")
    raw[date_col] = pd.to_datetime(raw[date_col], errors="coerce")
    raw = (raw.dropna(subset=[date_col])
              .sort_values(date_col)
              .drop_duplicates(date_col, keep="last")
              .set_index(date_col))
    raw = raw.apply(pd.to_numeric, errors="coerce")
    missing_fraction = raw.isna().mean()
    eligible = [c for c in raw.columns if missing_fraction[c] <= 0.30]
    if not eligible:
        raise ValueError("No tickers have enough usable price data.")
    info = {
        "source": Path(path).name,
        "universe_size": len(eligible),
        "observations": len(raw),
        "start": raw.index.min(),
        "end": raw.index.max(),
    }
    return raw, eligible, info


def prepare_prices(raw: pd.DataFrame, selected_assets: list[str]) -> pd.DataFrame:
    if len(selected_assets) < 3:
        raise ValueError("Select at least three tickers.")
    missing = [ticker for ticker in selected_assets if ticker not in raw.columns]
    if missing:
        raise ValueError(f"Tickers not found in the dataset: {', '.join(missing)}")
    selected = raw.loc[:, selected_assets].copy()
    too_missing = selected.columns[selected.isna().mean() > 0.30].tolist()
    if too_missing:
        raise ValueError("More than 30% of prices are missing for: " + ", ".join(too_missing))
    # Match the mentor notebook's forward-fill treatment, then align all assets.
    prices = selected.ffill().dropna(how="any")
    prices = prices.loc[:, (prices > 0).all(axis=0)]
    if prices.shape[1] < 3:
        raise ValueError("Fewer than three assets remain after cleaning.")
    if len(prices) < 100:
        raise ValueError("At least 100 aligned daily observations are required.")
    return prices


def _quasi_diagonal_order(linkage_matrix: np.ndarray, n_assets: int) -> list[int]:
    """Expand the last hierarchical cluster into its ordered original leaves."""
    if n_assets == 1:
        return [0]
    order = [int(linkage_matrix[-1, 0]), int(linkage_matrix[-1, 1])]
    while any(node >= n_assets for node in order):
        expanded: list[int] = []
        for node in order:
            if node < n_assets:
                expanded.append(node)
            else:
                row = linkage_matrix[node - n_assets]
                expanded.extend([int(row[0]), int(row[1])])
        order = expanded
    return order


def _cluster_variance(covariance: pd.DataFrame, tickers: list[str]) -> float:
    block = covariance.loc[tickers, tickers].to_numpy(dtype=float, copy=True)
    variances = np.maximum(np.diag(block), 1e-12)
    inverse_variance = 1.0 / variances
    inverse_variance /= inverse_variance.sum()
    return float(inverse_variance @ block @ inverse_variance)


def hrp_weights(returns: pd.DataFrame) -> pd.Series:
    """Calculate Hierarchical Risk Parity weights using correlation clustering."""
    if returns.shape[1] < 2:
        raise ValueError("HRP requires at least two assets.")
    covariance = returns.cov()
    correlation = returns.corr().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    corr_array = correlation.to_numpy(dtype=float, copy=True)
    np.fill_diagonal(corr_array, 1.0)
    distance = np.sqrt(np.clip((1.0 - corr_array) / 2.0, 0.0, 1.0))
    condensed = squareform(distance, checks=False)
    condensed = np.array(condensed, dtype=float, copy=True)
    tree = linkage(condensed, method="single")
    order = _quasi_diagonal_order(tree, len(correlation.columns))
    ordered_tickers = correlation.columns[order].tolist()

    weights = pd.Series(1.0, index=ordered_tickers, dtype=float)

    def recursive_bisection(items: list[str]) -> None:
        if len(items) <= 1:
            return
        midpoint = len(items) // 2
        left, right = items[:midpoint], items[midpoint:]
        left_var = _cluster_variance(covariance, left)
        right_var = _cluster_variance(covariance, right)
        total_var = left_var + right_var
        left_weight = 0.5 if total_var <= 1e-16 else 1.0 - left_var / total_var
        weights.loc[left] *= left_weight
        weights.loc[right] *= (1.0 - left_weight)
        recursive_bisection(left)
        recursive_bisection(right)

    recursive_bisection(ordered_tickers)
    weights = weights.reindex(returns.columns).fillna(0.0)
    weights = weights.clip(lower=0.0)
    return weights / weights.sum()



def pca_eigenrisk_weights(returns: pd.DataFrame, variance_coverage: float = 0.95) -> pd.Series:
    """PCA eigen-risk allocation: inverse retained-PC variance contribution."""
    if returns.shape[1] < 2:
        raise ValueError("PCA eigen-risk allocation requires at least two assets.")
    covariance = returns.cov().to_numpy(dtype=float, copy=True)
    covariance = np.nan_to_num((covariance + covariance.T) / 2.0,
                               nan=0.0, posinf=0.0, neginf=0.0)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    eigenvectors = eigenvectors[:, order]
    total = float(eigenvalues.sum())
    if total <= 1e-15:
        return pd.Series(1.0 / returns.shape[1], index=returns.columns, name="PCA")
    cumulative = np.cumsum(eigenvalues) / total
    n_components = int(np.searchsorted(cumulative, variance_coverage, side="left") + 1)
    retained_values = eigenvalues[:n_components]
    retained_vectors = eigenvectors[:, :n_components]
    asset_risk = (retained_vectors ** 2) @ retained_values
    asset_risk = np.maximum(asset_risk, 1e-12)
    inverse_risk = 1.0 / asset_risk
    weights = inverse_risk / inverse_risk.sum()
    return pd.Series(weights, index=returns.columns, name="PCA")


def _performance_metrics(daily_returns: pd.Series) -> dict[str, float]:
    returns = daily_returns.dropna().astype(float)
    if returns.empty:
        raise ValueError("No holdout returns are available for evaluation.")
    growth = (1.0 + returns).cumprod()
    total_return = float(growth.iloc[-1] - 1.0)
    annualised_return = float(growth.iloc[-1] ** (TRADING_DAYS / len(returns)) - 1.0) if growth.iloc[-1] > 0 else -1.0
    daily_volatility = float(returns.std(ddof=1)) if len(returns) > 1 else 0.0
    annualised_volatility = daily_volatility * np.sqrt(TRADING_DAYS)
    sharpe = (float(returns.mean() / daily_volatility * np.sqrt(TRADING_DAYS))
              if daily_volatility > 0 else float("nan"))
    drawdown = growth / growth.cummax() - 1.0
    return {
        "Total return": total_return,
        "Annualised return": annualised_return,
        "Annualised volatility": annualised_volatility,
        "Sharpe ratio": sharpe,
        "Maximum drawdown": float(drawdown.min()),
    }


def run_comparison(prices: pd.DataFrame, capital: float = 100_000.0,
                   train_fraction: float = 0.80) -> PortfolioResult:
    """Fit HRP and PCA eigen-risk weights on the first 80%; compare on the later holdout period."""
    if capital <= 0:
        raise ValueError("Initial capital must be greater than zero.")
    if not 0.60 <= train_fraction <= 0.90:
        raise ValueError("Training fraction must be between 60% and 90%.")
    split = int(len(prices) * train_fraction)
    train_prices = prices.iloc[:split].copy()
    test_prices = prices.iloc[split:].copy()
    if len(train_prices) < 80 or len(test_prices) < 30:
        raise ValueError("Not enough observations for the training and holdout periods.")

    train_returns = train_prices.pct_change().dropna()
    test_returns = test_prices.pct_change().dropna()
    if train_returns.empty or test_returns.empty:
        raise ValueError("Could not calculate daily returns from the selected prices.")

    weights = {
        "HRP": hrp_weights(train_returns),
        "PCA": pca_eigenrisk_weights(train_returns),
    }

    wealth = pd.DataFrame(index=test_returns.index)
    metrics_rows: dict[str, dict[str, float]] = {}
    for strategy, allocation in weights.items():
        daily_portfolio_returns = test_returns.loc[:, allocation.index].dot(allocation)
        wealth[strategy] = capital * (1.0 + daily_portfolio_returns).cumprod()
        metrics_rows[strategy] = _performance_metrics(daily_portfolio_returns)

    metrics = pd.DataFrame(metrics_rows).T
    return PortfolioResult(
        prices=prices,
        train_prices=train_prices,
        test_prices=test_prices,
        train_returns=train_returns,
        test_returns=test_returns,
        weights=weights,
        wealth=wealth,
        metrics=metrics,
    )


def run_reliability_downstream_comparison(
    prices: pd.DataFrame,
    risk_scores: dict[str, float] | None = None,
    reliability_scores: dict[str, float] | None = None,
    base_strategy: str = "HRP",
    capital: float = 100_000.0,
    penalty_strength: float = 1.5,
    train_fraction: float = 0.80,
) -> dict:
    """Downstream application: compare three portfolio policies on holdout data.

    Variants:
      1. No risk signal (standard HRP/PCA)
      2. Risk-only ranking (penalises reported risk without reliability check)
      3. Reliability-aware ranking (penalises verified high-risk; discounts unverified defects)
    """
    res = run_comparison(prices, capital=capital, train_fraction=train_fraction)
    base_w = res.weights.get(base_strategy, res.weights["HRP"]).copy()
    tickers = base_w.index.tolist()

    # Defaults if scores not provided
    if risk_scores is None:
        risk_scores = {t: 0.5 for t in tickers}
    if reliability_scores is None:
        reliability_scores = {t: 0.8 for t in tickers}

    r_vec = np.array([float(risk_scores.get(t, 0.5)) for t in tickers])
    rel_vec = np.array([float(reliability_scores.get(t, 0.8)) for t in tickers])

    # Variant 1: No risk signal
    w_no_signal = base_w.copy()

    # Variant 2: Risk-only penalty
    w_risk_only = base_w.to_numpy() * np.exp(-penalty_strength * r_vec)
    w_risk_only = pd.Series(w_risk_only / w_risk_only.sum(), index=tickers)

    # Variant 3: Reliability-aware penalty
    # Penalise high risk only when evidence reliability is high;
    # unverified/defective claims are not blindly acted upon
    effective_risk = r_vec * rel_vec
    w_rel_aware = base_w.to_numpy() * np.exp(-penalty_strength * effective_risk)
    w_rel_aware = pd.Series(w_rel_aware / w_rel_aware.sum(), index=tickers)

    test_ret = res.test_returns.loc[:, tickers]
    wealth_df = pd.DataFrame(index=test_ret.index)
    metrics_dict: dict[str, dict[str, float]] = {}

    policies = {
        "1. No risk signal": w_no_signal,
        "2. Risk-only ranking": w_risk_only,
        "3. Reliability-aware ranking": w_rel_aware,
    }

    for name, w in policies.items():
        daily_ret = test_ret.dot(w)
        wealth_df[name] = capital * (1.0 + daily_ret).cumprod()
        m = _performance_metrics(daily_ret)
        # Add CVaR (Conditional Value at Risk at 95%)
        losses = -daily_ret
        var95 = np.percentile(losses, 95)
        cvar95 = float(losses[losses >= var95].mean())
        m["CVaR (95%)"] = cvar95
        # Turnover relative to baseline
        turnover = float(np.abs(w.to_numpy() - base_w.to_numpy()).sum() / 2.0)
        m["Turnover vs Base"] = turnover
        metrics_dict[name] = m

    return {
        "weights": policies,
        "wealth": wealth_df,
        "metrics": pd.DataFrame(metrics_dict).T,
        "test_prices": res.test_prices,
    }
