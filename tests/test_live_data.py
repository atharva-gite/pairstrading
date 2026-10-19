"""Hits Yahoo Finance. Excluded from normal CI; run with `pytest -m live`."""

import datetime as dt

import pytest

from src.logic import PairsTradingEngine

pytestmark = pytest.mark.live


def test_fetch_and_backtest_ko_pep_on_live_prices():
    end = dt.date.today()
    start = end - dt.timedelta(days=3 * 365)
    split = end - dt.timedelta(days=365)
    engine = PairsTradingEngine("KO", "PEP", "SPY", str(start), str(end), split_date=str(split)).fetch()

    assert {"KO", "PEP", "SPY"} <= set(engine.data.columns)
    assert len(engine.data) > 500
    assert engine.data.index.max().date() >= end - dt.timedelta(days=10)

    metrics = engine.run_backtest()["metrics"]
    assert metrics["walk_forward"] is True
    assert metrics["tether_state"] in {"TETHERED", "WEAK", "BROKEN"}
