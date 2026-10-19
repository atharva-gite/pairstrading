import json

import numpy as np
import pandas as pd
import pytest

from src import llm
from src.logic import (
    GROQ_MODEL_FALLBACK,
    GROQ_MODEL_PRIMARY,
    build_book_context,
    call_llm,
    generate_desk_briefing,
    json_safe,
    parse_desk_briefing,
    resolve_groq_api_key,
    serialize_book_context,
)


@pytest.fixture(autouse=True)
def isolated_env(monkeypatch):
    monkeypatch.setattr(llm, "_load_project_dotenv", lambda: None)
    for var in ("GROQ_API_KEY", "LLM_PROVIDER", "GROQ_MODEL"):
        monkeypatch.delenv(var, raising=False)


class GroqError(Exception):
    def __init__(self, message, status_code=None):
        super().__init__(message)
        self.status_code = status_code


def fake_client(replies):
    """Client whose create() pops replies in order; an Exception reply is raised."""
    calls = []

    class Message:
        def __init__(self, content):
            self.content = content

    class Choice:
        def __init__(self, content):
            self.message = Message(content)

    class Response:
        def __init__(self, content):
            self.choices = [Choice(content)]

    class Completions:
        @staticmethod
        def create(model, messages, temperature=0.2):
            calls.append(model)
            reply = replies.pop(0)
            if isinstance(reply, Exception):
                raise reply
            return Response(reply)

    class Chat:
        completions = Completions

    class Client:
        chat = Chat

    return Client(), calls


def use_client(monkeypatch, replies):
    client, calls = fake_client(replies)
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setattr(llm, "GROQ_CLIENT_FACTORY", lambda api_key: client)
    return calls


def ask():
    return call_llm([{"role": "user", "content": "why?"}], {"pair": "KO/PEP"})


def test_json_safe_maps_non_finite_and_numpy_values():
    assert json_safe(None) is None
    assert json_safe("x") == "x"
    assert json_safe(float("nan")) is None
    assert json_safe(1.5) == 1.5
    assert json_safe(np.float64(np.inf)) is None
    assert json_safe(np.int64(3)) == 3
    assert json_safe(pd.Timestamp("2024-01-02")) == "2024-01-02 00:00:00"


def test_build_book_context_is_json_serializable():
    idx = pd.date_range("2024-01-01", periods=3)
    result = {
        "asset_a": "KO",
        "asset_b": "PEP",
        "benchmark": "SPY",
        "hedge_method": "Static OLS",
        "sizing_method": "Hedge Ratio",
        "metrics": {"tether_state": "TETHERED", "half_life": np.float64(18.2), "oos_sharpe": float("nan")},
        "blotter": pd.DataFrame({"entry": idx[:2], "exit": idx[1:], "pnl": [10.0, -4.0]}),
        "rolling_p_value": pd.Series([0.01, 0.04, 0.02], index=idx),
    }
    context = build_book_context(result)
    assert context["pair"] == "KO/PEP"
    assert context["metrics"]["oos_sharpe"] is None
    assert context["blotter"][0]["entry"] == "2024-01-01"
    assert context["rolling_p_value_last"] == 0.02
    assert context["rolling_p_value_max"] == 0.04
    assert json.loads(serialize_book_context(context))["pair"] == "KO/PEP"


def test_build_book_context_without_blotter_or_rolling():
    context = build_book_context({"asset_a": "A", "asset_b": "B"})
    assert context["blotter"] == []
    assert context["rolling_p_value_last"] is None


@pytest.mark.parametrize("text", ["", "no json here", "{not: valid}", "[1, 2]"])
def test_parse_desk_briefing_rejects_bad_payloads(text):
    assert parse_desk_briefing(text) is None


def test_resolve_key_prefers_env_and_strips_quotes(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", ' "abc" ')
    assert resolve_groq_api_key() == ("abc", ".env")


def test_resolve_key_falls_back_to_streamlit_secrets():
    assert resolve_groq_api_key(lambda name: "from-secrets") == ("from-secrets", "st.secrets")


def test_resolve_key_survives_secrets_errors():
    def broken(name):
        raise KeyError(name)

    assert resolve_groq_api_key(broken) == (None, "not found")


def test_call_llm_without_key_reports_missing():
    result = ask()
    assert not result.ok
    assert "GROQ_API_KEY is missing" in result.error


def test_call_llm_rejects_unknown_provider(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    result = ask()
    assert not result.ok
    assert "Unsupported LLM_PROVIDER" in result.error


def test_call_llm_empty_response(monkeypatch):
    use_client(monkeypatch, [""])
    assert ask().error == "Groq returned an empty response."


def test_call_llm_falls_back_when_model_not_found(monkeypatch):
    calls = use_client(monkeypatch, [GroqError("model not found", status_code=404), "fallback answer"])
    result = ask()
    assert result.ok and result.content == "fallback answer"
    assert calls == [GROQ_MODEL_PRIMARY, GROQ_MODEL_FALLBACK]


def test_call_llm_fallback_failure_is_reported(monkeypatch):
    use_client(monkeypatch, [GroqError("model does not exist", status_code=404), RuntimeError("down")])
    result = ask()
    assert not result.ok
    assert "down" in result.error


def test_call_llm_fallback_empty_response(monkeypatch):
    use_client(monkeypatch, [GroqError("model_not_found", status_code=404), ""])
    assert ask().error == "Groq returned an empty response."


def test_call_llm_other_errors_do_not_fall_back(monkeypatch):
    calls = use_client(monkeypatch, [GroqError("rate limited", status_code=429)])
    result = ask()
    assert not result.ok
    assert "rate limited" in result.error
    assert calls == [GROQ_MODEL_PRIMARY]


def test_model_not_found_reads_status_from_response(monkeypatch):
    class Response:
        status_code = 404

    exc = GroqError("model not found")
    exc.response = Response()
    calls = use_client(monkeypatch, [exc, "ok"])
    assert ask().ok
    assert calls == [GROQ_MODEL_PRIMARY, GROQ_MODEL_FALLBACK]


def test_desk_briefing_reports_llm_error():
    briefing = generate_desk_briefing({"pair": "KO/PEP"})
    assert briefing["error"] and briefing["verdict"] == ""


def test_desk_briefing_uses_raw_text_when_not_json(monkeypatch):
    use_client(monkeypatch, ["Plain prose verdict."])
    briefing = generate_desk_briefing({"pair": "KO/PEP"})
    assert briefing["verdict"] == "Plain prose verdict."
    assert briefing["error"] is None
