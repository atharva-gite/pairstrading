import numpy as np
import pandas as pd
from src.logic import (
    PairsTradingEngine,
    apply_kalman_filter,
    build_signals,
    build_trade_blotter,
    calculate_half_life,
    call_llm,
    check_cointegration,
    compute_tether_profile,
    dollar_notionals,
    generate_desk_briefing,
    parse_desk_briefing,
    apply_multiple_testing,
    screen_sector_pairs,
)


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


def test_dollar_notionals_hedge_ratio_splits_gross_capital():
    beta = pd.Series([1.0, 2.0], index=pd.date_range("2020-01-01", periods=2))
    notional_a, notional_b = dollar_notionals(beta, capital=90_000.0, sizing_method="Hedge Ratio")

    assert np.isclose(notional_a.iloc[0] + abs(notional_b.iloc[0]), 90_000.0)
    assert np.isclose(notional_a.iloc[1], 30_000.0)
    assert np.isclose(notional_b.iloc[1], 60_000.0)


def test_dollar_notionals_equal_dollar_ignores_beta_magnitude():
    beta = pd.Series([2.0, -0.5], index=pd.date_range("2020-01-01", periods=2))
    notional_a, notional_b = dollar_notionals(beta, capital=100_000.0, sizing_method="Equal Dollar")

    assert np.allclose(notional_a, 50_000.0)
    assert np.isclose(notional_b.iloc[0], 50_000.0)
    assert np.isclose(notional_b.iloc[1], -50_000.0)


def test_screen_sector_pairs_ranks_cointegrated_first():
    rng = np.random.default_rng(7)
    dates = pd.date_range("2020-01-01", periods=180)
    x = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 180))), index=dates, name="X")
    y = pd.Series(np.exp(1.1 * np.log(x) + rng.normal(0, 0.005, 180)), index=dates, name="Y")
    z = pd.Series(50 * np.exp(np.cumsum(rng.normal(0, 0.02, 180))), index=dates, name="Z")
    ranked = screen_sector_pairs(pd.concat([x, y, z], axis=1), p_value_threshold=0.05)

    assert "fdr_significant" in ranked.columns
    assert "p_value_fdr" in ranked.columns
    assert len(ranked) == 3
    assert ranked.iloc[0]["pair"] in {"X/Y", "Y/X"}
    assert ranked.iloc[0]["tether_score"] >= ranked.iloc[-1]["tether_score"]


def test_run_backtest_uses_dollar_book_and_buy_hold():
    dates = pd.date_range("2020-01-01", periods=80)
    asset_a = pd.Series(100 * np.exp(np.linspace(0, 0.08, 80) + 0.02 * np.sin(np.arange(80) / 5)), index=dates)
    asset_b = asset_a * 0.92 + 4.0
    spy = asset_a * 1.01
    engine = PairsTradingEngine(
        "AAA",
        "BBB",
        "SPY",
        "2020-01-01",
        "2020-04-20",
        lookback=20,
        capital=100_000.0,
        sizing_method="Hedge Ratio",
    )
    engine.data = pd.DataFrame({"AAA": asset_a, "BBB": asset_b, "SPY": spy})
    result = engine.run_backtest()

    assert "pair_buy_hold_cumulative" in result
    assert "spread_mean" in result
    assert np.isclose(result["metrics"]["capital"], 100_000.0)
    assert "tether_score" in result["metrics"]
    assert "blotter" in result
    assert len(result["positions"]) == 80


def test_tether_profile_flags_broken_relationship():
    broken = compute_tether_profile(p_value=0.40, adf_p_value=0.50, half_life=200.0)
    healthy = compute_tether_profile(p_value=0.01, adf_p_value=0.01, half_life=18.0)
    assert broken["tether_state"] == "BROKEN"
    assert broken["tether_reason"] == "no_cointegration"
    assert healthy["tether_state"] == "TETHERED"
    assert healthy["tether_score"] > broken["tether_score"]
    slow = compute_tether_profile(p_value=0.02, adf_p_value=0.02, half_life=114.0)
    assert slow["tether_reason"] == "too_slow"


def test_trade_blotter_records_round_trip():
    idx = pd.date_range("2020-01-01", periods=6)
    positions = pd.Series([0.0, 1.0, 1.0, 1.0, 0.0, 0.0], index=idx)
    zscore = pd.Series([0.0, -2.2, -1.4, -0.4, -0.2, 0.1], index=idx)
    dollar_pnl = pd.Series([0.0, 0.0, 40.0, 25.0, 10.0, 0.0], index=idx)
    blotter = build_trade_blotter(positions, zscore, dollar_pnl, half_life=10.0, stop_loss=3.0)
    assert len(blotter) == 1
    assert blotter.iloc[0]["side"] == "Long spread"
    assert blotter.iloc[0]["reason"] == "revert"


def test_parse_desk_briefing_reads_fenced_json():
    parsed = parse_desk_briefing(
        '```json\n{"headline": "KO/PEP still tethered", "verdict": "Stay in.", "drivers": "p=0.01", "risks": "break"}\n```'
    )
    assert parsed["headline"].startswith("KO/PEP")
    assert parsed["verdict"] == "Stay in."


def test_call_llm_sends_book_context(monkeypatch):
    captured = {}

    class FakeMessage:
        content = "Grounded answer"

    class FakeChoice:
        message = FakeMessage()

    class FakeResponse:
        choices = [FakeChoice()]

    class FakeClient:
        class chat:
            class completions:
                @staticmethod
                def create(model, messages, temperature=0.2):
                    captured["model"] = model
                    captured["messages"] = messages
                    return FakeResponse()

    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setattr("src.logic.GROQ_CLIENT_FACTORY", lambda api_key: FakeClient())
    result = call_llm(
        [{"role": "user", "content": "Is the tether intact?"}],
        {"pair": "KO/PEP", "metrics": {"tether_state": "TETHERED"}},
        task="research",
    )
    assert result.ok
    assert result.content == "Grounded answer"
    assert "ACTIVE_BOOK_CONTEXT_JSON" in captured["messages"][0]["content"]
    assert "KO/PEP" in captured["messages"][0]["content"]


def test_generate_desk_briefing_parses_json(monkeypatch):
    class FakeMessage:
        content = '{"headline": "H", "verdict": "V", "drivers": "D", "risks": "R"}'

    class FakeChoice:
        message = FakeMessage()

    class FakeResponse:
        choices = [FakeChoice()]

    class FakeClient:
        class chat:
            class completions:
                @staticmethod
                def create(model, messages, temperature=0.2):
                    return FakeResponse()

    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setattr("src.logic.GROQ_CLIENT_FACTORY", lambda api_key: FakeClient())
    briefing = generate_desk_briefing({"pair": "KO/PEP"})
    assert briefing["headline"] == "H"
    assert briefing["verdict"] == "V"
    assert briefing["error"] is None


def test_fdr_is_stricter_than_raw_pvalues():
    ranked = pd.DataFrame({"p_value": [0.01, 0.04, 0.20, 0.49]})
    adjusted = apply_multiple_testing(ranked, alpha=0.05)
    assert adjusted["fdr_significant"].sum() <= (ranked["p_value"] < 0.05).sum()
    assert adjusted["bonferroni_significant"].sum() <= adjusted["fdr_significant"].sum()


def test_walk_forward_reports_is_and_oos_sharpe():
    dates = pd.date_range("2022-01-03", periods=700, freq="B")
    rng = np.random.default_rng(3)
    spy = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(dates)))), index=dates)
    asset_a = spy * 1.01 + rng.normal(0, 0.2, len(dates))
    asset_b = 0.9 * asset_a + rng.normal(0, 0.15, len(dates))
    engine = PairsTradingEngine(
        "AAA",
        "BBB",
        "SPY",
        "2022-01-03",
        "2024-09-01",
        lookback=30,
        capital=100_000.0,
        walk_forward=True,
        split_date="2024-01-01",
        charge_borrow=False,
    )
    engine.data = pd.DataFrame({"AAA": asset_a, "BBB": asset_b, "SPY": spy})
    result = engine.run_backtest()
    assert result["metrics"]["walk_forward"] is True
    assert "is_sharpe" in result["metrics"]
    assert "oos_sharpe" in result["metrics"]
    assert "net_beta" in result


def test_borrow_reduces_pnl_when_short():
    dates = pd.date_range("2020-01-01", periods=80)
    asset_a = pd.Series(100 * np.exp(np.linspace(0, 0.08, 80) + 0.02 * np.sin(np.arange(80) / 5)), index=dates)
    asset_b = asset_a * 0.92 + 4.0
    spy = asset_a * 1.01
    data = pd.DataFrame({"AAA": asset_a, "BBB": asset_b, "SPY": spy})
    common = dict(
        asset_a="AAA",
        asset_b="BBB",
        benchmark="SPY",
        start="2020-01-01",
        end="2020-04-20",
        lookback=20,
        capital=100_000.0,
        walk_forward=False,
    )
    cheap = PairsTradingEngine(**common, charge_borrow=False, borrow_bps=0.0)
    rich = PairsTradingEngine(**common, charge_borrow=True, borrow_bps=500.0)
    cheap.data = data.copy()
    rich.data = data.copy()
    cheap_pnl = cheap.run_backtest()["metrics"]["total_pnl"]
    rich_pnl = rich.run_backtest()["metrics"]["total_pnl"]
    assert rich_pnl <= cheap_pnl + 1e-6

