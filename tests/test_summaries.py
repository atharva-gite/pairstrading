import json

import pandas as pd
import pytest

from scripts.refresh_fixture import END, SPLIT, START, load_prices
from src import llm
from src.logic import (
    PROVIDERS,
    PairsTradingEngine,
    build_tape_context,
    build_universe_context,
    context_fingerprint,
    generate_view_summary,
    rule_based_summary,
    screen_sector_pairs,
)
from src.summaries import _current_run


@pytest.fixture(autouse=True)
def no_keys(monkeypatch):
    monkeypatch.setattr(llm, "_load_project_dotenv", lambda: None)
    for provider in PROVIDERS.values():
        monkeypatch.delenv(provider.key_env, raising=False)
    monkeypatch.delenv("LLM_PROVIDERS", raising=False)
    monkeypatch.delenv("LLM_PROVIDER", raising=False)


@pytest.fixture(scope="module")
def ko_pep_result():
    engine = PairsTradingEngine("KO", "PEP", "SPY", START, END, split_date=SPLIT)
    engine.data = load_prices()[["KO", "PEP", "SPY"]].dropna()
    return engine.run_backtest()


@pytest.fixture(scope="module")
def staples_screen():
    prices = load_prices()[["KO", "PEP", "AAPL", "MSFT"]].dropna()
    return screen_sector_pairs(prices, split_date=SPLIT)


def fake_provider(monkeypatch, reply):
    class Message:
        content = reply

    class Choice:
        message = Message()

    class Response:
        choices = [Choice()]

    class Client:
        class chat:
            class completions:
                @staticmethod
                def create(model, messages, temperature=0.2):
                    if isinstance(reply, Exception):
                        raise reply
                    return Response()

    monkeypatch.setenv("GROQ_API_KEY", "k")
    monkeypatch.setattr(llm, "LLM_CLIENT_FACTORY", lambda **_: Client())


@pytest.mark.parametrize(
    ("values", "expected"),
    [([0, 1, 1, 1], (1.0, 3)), ([1, 1], (1.0, 2)), ([1, -1, 0], (0.0, 1)), ([], (0.0, 0))],
)
def test_current_run_counts_trailing_state(values, expected):
    assert _current_run(pd.Series(values, dtype=float)) == expected


def test_tape_context_describes_current_book(ko_pep_result):
    context = build_tape_context(ko_pep_result, entry_z=2.0, exit_z=0.5, stop_loss=3.0)
    assert context["pair"] == "KO/PEP"
    assert context["position_now"] in {"long the spread", "short the spread", "flat"}
    assert context["days_in_current_state"] >= 1
    assert len(context["recent_trades"]) == 3
    assert context["bands"] == {"entry": 2.0, "exit": 0.5, "stop": 3.0}
    json.dumps(context)


def test_universe_context_counts_match_screen(staples_screen):
    context = build_universe_context(staples_screen, "Consumer Staples", SPLIT)
    assert context["pairs_tested"] == 6
    assert context["pairs_surviving_fdr"] == int(staples_screen["fdr_significant"].sum())
    assert context["pairs"][0]["pair"] == staples_screen.iloc[0]["pair"]
    json.dumps(context)


def test_rule_based_summaries_quote_the_numbers(ko_pep_result, staples_screen):
    tape = build_tape_context(ko_pep_result, 2.0, 0.5, 3.0)
    tape_text = rule_based_summary("tape", tape)
    assert "KO/PEP" in tape_text and tape["position_now"] in tape_text
    assert "SPY" in tape_text

    universe = build_universe_context(staples_screen, "Consumer Staples", SPLIT)
    universe_text = rule_based_summary("universe", universe)
    assert f"of {universe['pairs_tested']} Consumer Staples pairs" in universe_text
    assert universe["pairs"][0]["pair"] in universe_text


@pytest.mark.parametrize("z", [None, 2.4, 0.2, 1.2])
def test_rule_based_tape_handles_each_stretch(ko_pep_result, z):
    context = build_tape_context(ko_pep_result, 2.0, 0.5, 3.0) | {"zscore_now": z, "recent_trades": []}
    text = rule_based_summary("tape", context)
    assert text.startswith("As of")


def test_summary_falls_back_to_rules_without_keys(ko_pep_result):
    context = build_tape_context(ko_pep_result, 2.0, 0.5, 3.0)
    summary = generate_view_summary("tape", context)
    assert summary["source"] is None
    assert "No AI key found" in summary["error"]
    assert summary["text"] == rule_based_summary("tape", context)
    assert summary["fingerprint"] == context_fingerprint(context)


def test_summary_uses_ai_text_when_available(monkeypatch, staples_screen):
    fake_provider(monkeypatch, "  KO/PEP leads the sector.  ")
    summary = generate_view_summary("universe", build_universe_context(staples_screen, "Consumer Staples", SPLIT))
    assert summary["text"] == "KO/PEP leads the sector."
    assert summary["source"].startswith("Groq")
    assert summary["error"] is None


def test_summary_falls_back_when_ai_errors(monkeypatch, staples_screen):
    fake_provider(monkeypatch, RuntimeError("quota exceeded"))
    context = build_universe_context(staples_screen, "Consumer Staples", SPLIT)
    summary = generate_view_summary("universe", context)
    assert "quota exceeded" in summary["error"]
    assert summary["text"] == rule_based_summary("universe", context)


def test_fingerprint_changes_with_context():
    assert context_fingerprint({"a": 1}) != context_fingerprint({"a": 2})
    assert context_fingerprint({"a": 1, "b": 2}) == context_fingerprint({"b": 2, "a": 1})
