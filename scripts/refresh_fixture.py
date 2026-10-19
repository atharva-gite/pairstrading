#!/usr/bin/env python3
"""Rebuild the regression fixtures in tests/fixtures/.

    python3 scripts/refresh_fixture.py             # recompute expected metrics from the checked-in prices
    python3 scripts/refresh_fixture.py --download  # re-download prices first (needs network)

Run this only when a model change is intentional, then review the diff of
tests/fixtures/expected_metrics.json in the pull request.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.logic import PairsTradingEngine, download_price_data  # noqa: E402

FIXTURES = ROOT / "tests" / "fixtures"
PRICES_CSV = FIXTURES / "prices.csv"
EXPECTED_JSON = FIXTURES / "expected_metrics.json"

TICKERS = ["KO", "PEP", "AAPL", "MSFT", "SPY"]
START = "2019-01-01"
END = "2026-06-30"
SPLIT = "2024-01-01"
PAIRS = [("KO", "PEP"), ("AAPL", "MSFT")]
PINNED_METRICS = [
    "is_sharpe",
    "oos_sharpe",
    "is_return",
    "oos_return",
    "half_life",
    "hedge_ratio",
    "cointegration_p_value",
    "tether_score",
    "tether_state",
    "oos_tether_state",
    "trade_count",
    "total_pnl",
    "net_beta_abs_mean",
]


def load_prices() -> pd.DataFrame:
    return pd.read_csv(PRICES_CSV, index_col=0, parse_dates=True)


def run_pair(prices: pd.DataFrame, asset_a: str, asset_b: str) -> dict:
    engine = PairsTradingEngine(asset_a, asset_b, "SPY", START, END, split_date=SPLIT)
    engine.data = prices[[asset_a, asset_b, "SPY"]].dropna()
    return engine.run_backtest()["metrics"]


def pinned(metrics: dict) -> dict:
    out = {}
    for key in PINNED_METRICS:
        value = metrics[key]
        out[key] = round(float(value), 6) if isinstance(value, float) else value
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", action="store_true", help="re-download prices from Yahoo Finance")
    args = parser.parse_args()

    FIXTURES.mkdir(parents=True, exist_ok=True)
    if args.download:
        prices = download_price_data(TICKERS, START, END)[TICKERS]
        prices.round(4).to_csv(PRICES_CSV, float_format="%.4f")

    prices = load_prices()
    expected = {f"{a}/{b}": pinned(run_pair(prices, a, b)) for a, b in PAIRS}
    EXPECTED_JSON.write_text(json.dumps(expected, indent=2, sort_keys=True) + "\n")
    print(json.dumps(expected, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
