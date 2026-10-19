"""Frozen walk-forward results on real KO/PEP and AAPL/MSFT closes.

If a change moves these numbers on purpose, rerun scripts/refresh_fixture.py
and review the diff of tests/fixtures/expected_metrics.json.
"""

import json
import math
from pathlib import Path

import pytest

from scripts.refresh_fixture import PAIRS, load_prices, pinned, run_pair

EXPECTED = json.loads((Path(__file__).parent / "fixtures" / "expected_metrics.json").read_text())


@pytest.fixture(scope="module")
def results():
    prices = load_prices()
    return {f"{a}/{b}": pinned(run_pair(prices, a, b)) for a, b in PAIRS}


@pytest.mark.parametrize("pair", sorted(EXPECTED))
def test_walk_forward_metrics_match_frozen_fixture(results, pair):
    actual, expected = results[pair], EXPECTED[pair]
    assert actual.keys() == expected.keys()
    for key, want in expected.items():
        got = actual[key]
        if isinstance(want, float):
            assert math.isclose(got, want, rel_tol=1e-4, abs_tol=1e-6), f"{pair} {key}: {got} != {want}"
        else:
            assert got == want, f"{pair} {key}: {got} != {want}"


def test_aapl_msft_is_the_broken_negative_control(results):
    control = results["AAPL/MSFT"]
    assert control["tether_state"] == "BROKEN"
    assert control["oos_tether_state"] == "BROKEN"
    assert control["cointegration_p_value"] > 0.10


def test_ko_pep_is_tethered_in_discovery(results):
    ko_pep = results["KO/PEP"]
    assert ko_pep["tether_state"] == "TETHERED"
    assert ko_pep["cointegration_p_value"] < 0.05
