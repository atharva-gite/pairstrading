import numpy as np
import pytest

from scripts.refresh_fixture import END, SPLIT, START, load_prices
from src.logic import PairsTradingEngine, cost_sensitivity


@pytest.fixture(scope="module")
def ko_pep_engine():
    engine = PairsTradingEngine("KO", "PEP", "SPY", START, END, split_date=SPLIT)
    engine.data = load_prices()[["KO", "PEP", "SPY"]].dropna()
    return engine


def test_cost_grid_pnl_and_sharpe_fall_as_slippage_rises(ko_pep_engine):
    grid = cost_sensitivity(ko_pep_engine, slippage_bps=(0.0, 5.0, 15.0))
    assert grid["slippage_bps"].tolist() == [0.0, 5.0, 15.0]
    assert grid["total_pnl"].is_monotonic_decreasing
    assert grid["total_pnl"].iloc[0] > grid["total_pnl"].iloc[-1]
    assert grid["sharpe_ratio"].is_monotonic_decreasing


def test_cost_grid_matches_base_run_and_leaves_engine_untouched(ko_pep_engine):
    base_pnl = ko_pep_engine.run_backtest()["metrics"]["total_pnl"]
    grid = cost_sensitivity(ko_pep_engine, slippage_bps=(10.0,))
    assert np.isclose(grid["total_pnl"].iloc[0], base_pnl)
    assert ko_pep_engine.slippage == 0.001
