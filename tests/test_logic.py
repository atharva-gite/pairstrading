import numpy as np
import pandas as pd
from src.logic import apply_kalman_filter, build_signals, calculate_half_life, check_cointegration


def test_check_cointegration_returns_expected_keys():
    x = pd.Series(np.linspace(10, 20, 50), index=pd.date_range("2020-01-01", periods=50))
    y = x * 0.8 + np.random.normal(0, 0.001, 50)
    result = check_cointegration(y, x)

    assert "p_value" in result
    assert "hedge_ratio" in result
    assert isinstance(result["hedge_ratio"], float)


def test_kalman_filter_outputs_same_length():
    x = pd.Series(np.linspace(1, 5, 40))
    y = x * 0.6 + np.random.normal(0, 0.01, 40)
    beta, alpha = apply_kalman_filter(y, x)

    assert len(beta) == len(x)
    assert len(alpha) == len(x)


def test_build_signals_creates_positions():
    z = pd.Series([-3.0, -2.5, -1.0, 0.0, 1.0, 2.5, 3.0], index=pd.date_range("2020-01-01", periods=7))
    signals = build_signals(z, entry_z=2.0, exit_z=0.5, stop_loss=3.0)

    assert signals.iloc[0] == 1
    assert signals.iloc[1] == 1
    assert signals.iloc[-1] == 0


def test_half_life_positive_for_mean_reverting_spread():
    t = pd.date_range("2020-01-01", periods=100)
    spread = pd.Series(1.5 * np.exp(-0.05 * np.arange(100)), index=t)
    half_life = calculate_half_life(spread)

    assert half_life > 0
