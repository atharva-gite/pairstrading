import numpy as np
import pandas as pd
import pytest

from src.logic import (
    MAX_TRADABLE_HALF_LIFE,
    PairsTradingEngine,
    apply_multiple_testing,
    build_signals,
    build_trade_blotter,
    calculate_half_life,
    compute_tether_profile,
    pair_pvalue_matrix,
    rolling_cointegration_pvalues,
    rolling_ols_hedge_ratio,
    screen_sector_pairs,
    split_masks,
    tether_from_prices,
)


def _cointegrated_pair(n=300, beta=0.8, seed=11):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2022-01-03", periods=n, freq="B")
    log_b = np.log(50) + np.cumsum(rng.normal(0, 0.01, n))
    noise = np.zeros(n)
    for t in range(1, n):
        noise[t] = 0.85 * noise[t - 1] + rng.normal(0, 0.004)
    log_a = 0.3 + beta * log_b + noise
    return (
        pd.Series(np.exp(log_a), index=dates, name="AAA"),
        pd.Series(np.exp(log_b), index=dates, name="BBB"),
    )


def test_rolling_ols_recovers_constant_hedge_ratio():
    a, b = _cointegrated_pair(beta=0.8)
    beta, alpha = rolling_ols_hedge_ratio(np.log(a), np.log(b), window=60)
    assert len(beta) == len(a)
    assert beta.iloc[:59].isna().all()
    assert np.isclose(beta.dropna().median(), 0.8, atol=0.15)
    assert alpha.dropna().notna().all()


def test_rolling_cointegration_pvalues_short_and_long_samples():
    a, b = _cointegrated_pair(n=300)
    assert rolling_cointegration_pvalues(a.iloc[:100], b.iloc[:100], window=126).empty
    rolling = rolling_cointegration_pvalues(a, b, window=126, step=20)
    assert rolling.name == "rolling_p_value"
    assert len(rolling) == len(range(126, 301, 20))
    assert rolling.between(0.0, 1.0).all()


def test_tether_from_prices_flags_too_few_observations():
    a, b = _cointegrated_pair(n=30)
    out = tether_from_prices(a, b)
    assert out["tether_state"] == "BROKEN"
    assert out["tether_reason"] == "too_few_obs"


@pytest.mark.parametrize(
    ("half_life", "state"),
    [(40.0, "TETHERED"), (60.0, "TETHERED"), (70.0, "WEAK"), (114.0, "WEAK")],
)
def test_slow_half_life_can_never_be_tethered(half_life, state):
    profile = compute_tether_profile(p_value=0.001, adf_p_value=0.001, half_life=half_life)
    assert profile["tether_state"] == state
    if half_life > MAX_TRADABLE_HALF_LIFE:
        assert profile["tether_score"] >= 70
        assert profile["tether_reason"] == "too_slow"


def test_half_life_nan_when_spread_short_or_explosive():
    idx = pd.date_range("2020-01-01", periods=60)
    assert np.isnan(calculate_half_life(pd.Series(np.arange(5.0), index=idx[:5])))
    assert np.isnan(calculate_half_life(pd.Series(np.exp(0.05 * np.arange(60)), index=idx)))


def test_build_signals_exits_through_band_and_stops_out():
    idx = pd.date_range("2020-01-01", periods=8)
    z = pd.Series([np.nan, 2.5, -2.5, -1.0, -0.2, -2.2, -3.5, 0.0], index=idx)
    signals = build_signals(z, entry_z=2.0, exit_z=0.5, stop_loss=3.0)
    assert signals.tolist() == [0, -1, 0, 0, 0, 1, 0, 0]


def test_trade_blotter_marks_stops_and_open_trades():
    idx = pd.date_range("2020-01-01", periods=6)
    positions = pd.Series([0.0, -1.0, -1.0, 0.0, 1.0, 1.0], index=idx)
    zscore = pd.Series([0.0, 2.2, 2.6, 3.2, -2.1, -1.5], index=idx)
    pnl = pd.Series([0.0, 0.0, -30.0, -50.0, 0.0, 5.0], index=idx)
    blotter = build_trade_blotter(positions, zscore, pnl, half_life=float("nan"), stop_loss=3.0)
    assert blotter["reason"].tolist() == ["stop", "open"]
    assert blotter["side"].tolist() == ["Short spread", "Long spread"]
    assert blotter["half_life_mult"].isna().all()


def test_apply_multiple_testing_handles_empty_frame():
    out = apply_multiple_testing(pd.DataFrame({"p_value": []}))
    assert {"p_value_fdr", "fdr_significant", "bonferroni_significant"} <= set(out.columns)
    assert out.empty


def test_split_masks_aligns_timezones():
    idx = pd.date_range("2023-12-28", periods=6, tz="America/New_York")
    mask, ts = split_masks(idx, "2024-01-01")
    assert ts.tzinfo is not None
    assert mask.sum() == 4
    naive_mask, naive_ts = split_masks(idx.tz_localize(None), "2024-01-01T00:00:00+00:00")
    assert naive_ts.tzinfo is None
    assert naive_mask.sum() == 4


def test_pair_pvalue_matrix_is_symmetric_with_zero_diagonal():
    table = pd.DataFrame(
        {"asset_a": ["X", "X"], "asset_b": ["Y", "Z"], "p_value": [0.01, 0.4], "p_value_fdr": [0.02, 0.4]}
    )
    matrix = pair_pvalue_matrix(table, value_col="p_value_fdr")
    assert matrix.loc["X", "Y"] == matrix.loc["Y", "X"] == 0.02
    assert (np.diag(matrix.to_numpy()) == 0.0).all()
    assert pair_pvalue_matrix(pd.DataFrame()).empty


def test_screen_sector_pairs_with_split_reports_live_state():
    a, b = _cointegrated_pair(n=400)
    prices = pd.concat([a, b], axis=1)
    ranked = screen_sector_pairs(prices, split_date=str(a.index[250].date()))
    assert len(ranked) == 1
    assert ranked.iloc[0]["oos_tether_state"] in {"TETHERED", "WEAK", "BROKEN"}


def test_screen_sector_pairs_empty_universe_returns_schema():
    idx = pd.date_range("2020-01-01", periods=10)
    ranked = screen_sector_pairs(pd.DataFrame({"X": np.arange(10.0)}, index=idx))
    assert ranked.empty
    assert "tether_state" in ranked.columns


@pytest.mark.parametrize("method", ["Kalman Filter", "Rolling OLS"])
def test_run_backtest_dynamic_hedge_methods(method):
    a, b = _cointegrated_pair(n=400)
    spy = pd.Series(np.exp(np.log(b) * 1.0), index=a.index)
    engine = PairsTradingEngine(
        "AAA",
        "BBB",
        "SPY",
        str(a.index[0].date()),
        str(a.index[-1].date()),
        lookback=30,
        hedge_method=method,
        split_date=str(a.index[250].date()),
    )
    engine.data = pd.DataFrame({"AAA": a, "BBB": b, "SPY": spy})
    result = engine.run_backtest()
    assert result["hedge_method"] == method
    assert result["metrics"]["walk_forward"] is True
    assert result["notional_a"].ge(0).all()
    assert np.isfinite(result["metrics"]["total_pnl"])
