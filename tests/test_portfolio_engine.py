from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "risk_allocation"))

from portfolio_engine import (  # noqa: E402
    DEFAULT_TICKERS,
    hrp_weights,
    pca_eigenrisk_weights,
    prepare_prices,
    read_price_universe,
    run_comparison,
)


def main() -> None:
    csv_path = ROOT / "data" / "market" / "SP500Data.csv"
    if not csv_path.exists():
        csv_path = ROOT / "data" / "SP500Data.csv"
    raw, eligible, info = read_price_universe(csv_path)
    tickers = [ticker for ticker in DEFAULT_TICKERS if ticker in eligible]
    prices = prepare_prices(raw, tickers)
    train_returns = prices.iloc[: int(len(prices) * 0.8)].pct_change().dropna()

    for method in (hrp_weights, pca_eigenrisk_weights):
        weights = method(train_returns)
        assert np.isfinite(weights.to_numpy()).all()
        assert (weights >= -1e-10).all()
        assert abs(float(weights.sum()) - 1.0) < 1e-7

    result = run_comparison(prices)
    assert set(result.weights) == {"HRP", "PCA"}
    assert set(result.metrics.index) == {"HRP", "PCA"}
    assert len(result.wealth) > 20
    assert all(abs(float(w.sum()) - 1.0) < 1e-7 for w in result.weights.values())
    print("PASS: dataset loaded", info)
    print("PASS: HRP and PCA Eigen-Risk weights are valid")
    print("PASS: 80/20 chronological backtest completed")
    print(result.metrics.to_string())


def test_portfolio_engine() -> None:
    main()


if __name__ == "__main__":
    main()
