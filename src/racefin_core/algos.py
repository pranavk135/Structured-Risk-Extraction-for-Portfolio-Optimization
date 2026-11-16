"""Optimiser registry. `ready` algos run today; `stub` entries are placeholders for the next phase.

Add a new algorithm by writing `fn(train_returns: DataFrame) -> Series` (weights sum to 1) and
registering it below. Tags drive the buckets shown in the UI.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from . import ROOT  # noqa: F401  (sets sys.path for portfolio_engine)
from portfolio_engine import TRADING_DAYS, _performance_metrics, hrp_weights, pca_eigenrisk_weights  # noqa: E402

BUCKETS = {
    "Low risk": "Smoother ride — prioritises stability and diversification.",
    "Balanced": "Middle path — diversified risk with moderate growth.",
    "High return": "Growth-seeking — accepts bigger swings for upside.",
}


@dataclass
class Algo:
    id: str
    name: str
    tags: list[str]
    status: str  # "ready" | "stub"
    blurb: str
    fn: Callable[[pd.DataFrame], pd.Series] | None = None
    risk: str = "Medium"
    default_for: list[str] = field(default_factory=list)


def _norm(s: pd.Series) -> pd.Series:
    s = s.clip(lower=0)
    return s / s.sum()


def equal_weight(r): return pd.Series(1 / r.shape[1], index=r.columns)
def inverse_vol(r): return _norm(1 / r.std().replace(0, np.nan).fillna(r.std().max()))


def momentum_topk(r, k: int = 5):
    k = max(1, min(k, r.shape[1]))
    cum = (1 + r).prod() - 1
    top = cum.nlargest(k).index
    w = pd.Series(0.0, index=r.columns)
    w[top] = 1 / k
    return w


REGISTRY = [
    Algo("hrp", "Hierarchical Risk Parity", ["Low risk", "Balanced"], "ready",
         "Clusters assets by correlation, then splits risk down the tree. Stable, no matrix inversion.",
         hrp_weights, "Low", ["Balanced"]),
    Algo("pca", "PCA Eigen-Risk", ["Balanced"], "ready",
         "Weights assets by exposure to the principal risk factors of the covariance matrix.", pca_eigenrisk_weights, "Medium"),
    Algo("invvol", "Inverse Volatility", ["Low risk"], "ready",
         "Calmer stocks get bigger weights. Simple and robust baseline.", inverse_vol, "Low", ["Low risk"]),
    Algo("equal", "Equal Weight (1/N)", ["Balanced"], "ready",
         "Same amount in every stock. The benchmark every algorithm should beat.", equal_weight, "Medium"),
    Algo("mom", "Momentum Top-K", ["High return"], "ready",
         "Equal weight in the recent winners. Higher return potential, higher drawdown risk.", momentum_topk, "High", ["High return"]),
    # ---- placeholders to be sourced from the papers / next phase -------------------------------
    Algo("minvar", "Minimum Variance (long-only)", ["Low risk"], "stub", "Quadratic program minimising portfolio variance."),
    Algo("erc", "Risk Parity (ERC)", ["Low risk", "Balanced"], "stub", "Equalises each asset's risk contribution."),
    Algo("maxsharpe", "Max-Sharpe (Mean-Variance)", ["High return"], "stub", "Maximises return per unit of risk (needs robust return estimates)."),
    Algo("bl", "Black-Litterman + evidence views", ["Balanced", "High return"], "stub",
         "Blends market prior with views — views can come from the RACE-Fin evidence-reliability score."),
    Algo("cvar", "CVaR Minimisation", ["Low risk"], "stub", "Minimises expected tail loss rather than variance."),
    Algo("rl", "RL Allocation Agent", ["High return"], "stub", "Learned policy; needs careful leakage-safe backtesting."),
]
BY_ID = {a.id: a for a in REGISTRY}


def algos_for(bucket: str, ready_only: bool = True):
    return [a for a in REGISTRY if bucket in a.tags and (a.status == "ready" or not ready_only)]


def default_algo(bucket: str) -> Algo:
    for a in REGISTRY:
        if a.status == "ready" and bucket in a.default_for:
            return a
    return BY_ID["hrp"]


def backtest(prices: pd.DataFrame, algo_ids: list[str], capital: float, train_frac: float = 0.8) -> dict:
    split = int(len(prices) * train_frac)
    train, test = prices.iloc[:split], prices.iloc[split:]
    tr, te = train.pct_change().dropna(), test.pct_change().dropna()
    weights, wealth, metrics = {}, pd.DataFrame(index=te.index), {}
    for aid in algo_ids:
        w = BY_ID[aid].fn(tr)
        w = w[w > 1e-6] / w[w > 1e-6].sum()
        weights[aid] = w
        daily = te.loc[:, w.index].dot(w)
        wealth[aid] = capital * (1 + daily).cumprod()
        metrics[aid] = _performance_metrics(daily)
    return {"weights": weights, "wealth": wealth, "metrics": pd.DataFrame(metrics).T,
            "train_end": train.index[-1], "test_start": test.index[0], "test_end": test.index[-1]}


def risk_score(vol: float) -> int:
    return int(np.clip(round(15 + vol * 260), 15, 98))
