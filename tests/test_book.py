import numpy as np
import pandas as pd
import pytest

from scripts.refresh_fixture import END, SPLIT, START, load_prices
from src.logic import DESK_GROSS_LIMIT, PairsTradingEngine, cost_sensitivity, dollar_notionals


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


def test_leg_cap_scales_both_legs_and_keeps_hedge_ratio():
    beta = pd.Series([1.0, 4.0, -3.0, 0.0], index=pd.date_range("2024-01-01", periods=4))
    capped_a, capped_b = dollar_notionals(beta, capital=100_000.0, max_leg_notional=60_000.0)
    raw_a, raw_b = dollar_notionals(beta, capital=100_000.0)

    assert (np.maximum(capped_a.abs(), capped_b.abs()) <= 60_000.0 + 1e-6).all()
    assert np.allclose(capped_b / capped_a, raw_b / raw_a)
    assert np.isclose(capped_a.iloc[0], raw_a.iloc[0])
    assert np.isclose(capped_b.iloc[1], 60_000.0)
    assert np.isclose(capped_b.iloc[2], -60_000.0)
    assert np.isclose(capped_a.iloc[3], 60_000.0)


def test_leg_cap_applies_to_equal_dollar_book():
    beta = pd.Series([1.0, -1.0], index=pd.date_range("2024-01-01", periods=2))
    a, b = dollar_notionals(beta, capital=100_000.0, sizing_method="Equal Dollar", max_leg_notional=40_000.0)
    assert np.allclose(a, 40_000.0)
    assert np.allclose(b.abs(), 40_000.0)


@pytest.mark.parametrize("capital", [0.0, -5_000.0, DESK_GROSS_LIMIT + 1.0])
def test_engine_rejects_tickets_outside_gross_limit(capital):
    with pytest.raises(ValueError, match="desk limit"):
        PairsTradingEngine("KO", "PEP", "SPY", START, END, capital=capital)


@pytest.mark.parametrize("max_leg_pct", [0.4, 1.2])
def test_engine_rejects_invalid_leg_cap(max_leg_pct):
    with pytest.raises(ValueError, match="max_leg_pct"):
        PairsTradingEngine("KO", "PEP", "SPY", START, END, max_leg_pct=max_leg_pct)


def test_tight_leg_cap_binds_and_shrinks_book(ko_pep_engine):
    loose = ko_pep_engine.run_backtest()
    tight_engine = PairsTradingEngine("KO", "PEP", "SPY", START, END, split_date=SPLIT, max_leg_pct=0.5)
    tight_engine.data = ko_pep_engine.data
    tight = tight_engine.run_backtest()

    assert loose["metrics"]["leg_cap_binding_days"] == 0
    assert tight["metrics"]["leg_cap_binding_days"] > 0
    assert (np.maximum(tight["notional_a"].abs(), tight["notional_b"].abs()) <= 50_000.0 + 1e-6).all()
    assert abs(tight["metrics"]["total_pnl"]) < abs(loose["metrics"]["total_pnl"])
