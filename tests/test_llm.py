import json

import numpy as np
import pandas as pd
import pytest

from src import llm
from src.logic import (
    PROVIDERS,
    build_book_context,
    call_llm,
    configured_providers,
    generate_desk_briefing,
    json_safe,
    parse_desk_briefing,
    provider_order,
    resolve_api_key,
    serialize_book_context,
)


@pytest.fixture(autouse=True)
def isolated_env(monkeypatch):
    monkeypatch.setattr(llm, "_load_project_dotenv", lambda: None)
    for provider in PROVIDERS.values():
        monkeypatch.delenv(provider.key_env, raising=False)
        monkeypatch.delenv(provider.model_env, raising=False)
    monkeypatch.delenv("LLM_PROVIDERS", raising=False)
    monkeypatch.delenv("LLM_PROVIDER", raising=False)


class ProviderError(Exception):
    def __init__(self, message, status_code=None):
        super().__init__(message)
        self.status_code = status_code


def use_providers(monkeypatch, replies: dict[str, list]):
    """Give each named provider a key and a fake client that pops its replies; Exceptions are raised."""
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

    def factory(provider, api_key):
        class Completions:
            @staticmethod
            def create(model, messages, temperature=0.2):
                calls.append((provider.name, model))
                reply = replies[provider.name].pop(0)
                if isinstance(reply, Exception):
                    raise reply
                return Response(reply)

        class Chat:
            completions = Completions

        class Client:
            chat = Chat

        return Client()

    for name in replies:
        monkeypatch.setenv(PROVIDERS[name].key_env, f"{name}-key")
    monkeypatch.setattr(llm, "LLM_CLIENT_FACTORY", factory)
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


def test_resolve_key_reads_env_and_strips_quotes(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", ' "abc" ')
    assert resolve_api_key("GEMINI_API_KEY") == "abc"


def test_resolve_key_falls_back_to_streamlit_secrets():
    assert resolve_api_key("GROQ_API_KEY", lambda name: "from-secrets") == "from-secrets"


def test_resolve_key_survives_secrets_errors():
    def broken(name):
        raise KeyError(name)

    assert resolve_api_key("GROQ_API_KEY", broken) is None


def test_every_provider_has_an_openai_compatible_https_endpoint():
    for provider in PROVIDERS.values():
        assert provider.base_url.startswith("https://")
        assert provider.key_env.isupper()
        assert provider.model


def test_provider_order_defaults_and_overrides(monkeypatch):
    names = [p.name for p in provider_order()[0]]
    assert names[0] == "groq" and set(names) == set(PROVIDERS)

    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    assert [p.name for p in provider_order()[0]][:2] == ["gemini", "groq"]

    monkeypatch.setenv("LLM_PROVIDERS", "mistral, cerebras")
    assert [p.name for p in provider_order()[0]] == ["mistral", "cerebras"]


def test_configured_providers_only_lists_those_with_keys(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "g")
    monkeypatch.setenv("OPENROUTER_API_KEY", "o")
    assert [p.name for p, _ in configured_providers()] == ["gemini", "openrouter"]


def test_call_llm_without_any_key_reports_missing():
    result = ask()
    assert not result.ok
    assert "No AI key found" in result.error
    assert "GEMINI_API_KEY" in result.error


def test_call_llm_rejects_unknown_provider(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDERS", "groq,openai")
    result = ask()
    assert not result.ok
    assert "Unsupported LLM provider openai" in result.error


def test_call_llm_reports_provider_and_model(monkeypatch):
    calls = use_providers(monkeypatch, {"gemini": ["grounded"]})
    result = ask()
    assert result.ok and result.content == "grounded"
    assert result.provider == "Google Gemini"
    assert calls == [("gemini", PROVIDERS["gemini"].model)]


def test_model_override_from_env(monkeypatch):
    calls = use_providers(monkeypatch, {"groq": ["ok"]})
    monkeypatch.setenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    assert ask().model == "llama-3.3-70b-versatile"
    assert calls == [("groq", "llama-3.3-70b-versatile")]


def test_rate_limit_falls_through_to_next_provider(monkeypatch):
    calls = use_providers(
        monkeypatch,
        {"groq": [ProviderError("rate limit reached", status_code=429)], "gemini": ["from gemini"]},
    )
    result = ask()
    assert result.ok and result.provider == "Google Gemini"
    assert [name for name, _ in calls] == ["groq", "gemini"]


def test_model_not_found_tries_fallback_model_first(monkeypatch):
    calls = use_providers(monkeypatch, {"groq": [ProviderError("model not found", status_code=404), "fallback"]})
    result = ask()
    assert result.ok and result.content == "fallback"
    assert calls == [("groq", PROVIDERS["groq"].model), ("groq", PROVIDERS["groq"].fallback_model)]


def test_model_not_found_reads_status_from_response(monkeypatch):
    class Response:
        status_code = 404

    exc = ProviderError("model does not exist")
    exc.response = Response()
    calls = use_providers(monkeypatch, {"groq": [exc, "ok"]})
    assert ask().ok
    assert len(calls) == 2


def test_empty_response_moves_on_and_reports_when_last(monkeypatch):
    use_providers(monkeypatch, {"groq": [""]})
    result = ask()
    assert not result.ok
    assert result.error == "Groq: returned an empty response."


def test_all_providers_failing_lists_each_error(monkeypatch):
    use_providers(
        monkeypatch,
        {"groq": [RuntimeError("down")], "mistral": [ProviderError("unauthorized", status_code=401)]},
    )
    result = ask()
    assert not result.ok
    assert result.error.startswith("All AI providers failed.")
    assert "Groq: request failed: down" in result.error
    assert "Mistral: request failed: API key rejected" in result.error


@pytest.mark.parametrize(
    ("exc", "expected"),
    [
        (ProviderError("Rate limit reached for model ... " * 20, status_code=429), "rate limited"),
        (ProviderError("forbidden", status_code=403), "API key rejected"),
        (ProviderError("bad gateway", status_code=502), "provider error (HTTP 502)"),
        (ProviderError("teapot", status_code=418), "HTTP 418"),
        (RuntimeError("x" * 300), "x" * 117 + "..."),
        (RuntimeError(""), "RuntimeError"),
    ],
)
def test_errors_are_shortened_for_the_ui(exc, expected):
    assert llm._describe_error(exc).startswith(expected)


def test_custom_system_prompt_and_context_label(monkeypatch):
    captured = []
    use_providers(monkeypatch, {"groq": ["ok"]})
    real_factory = llm.LLM_CLIENT_FACTORY

    def spy(provider, api_key):
        client = real_factory(provider=provider, api_key=api_key)
        original = client.chat.completions.create

        def create(model, messages, temperature=0.2):
            captured.append(messages)
            return original(model=model, messages=messages, temperature=temperature)

        client.chat.completions.create = create
        return client

    monkeypatch.setattr(llm, "LLM_CLIENT_FACTORY", spy)
    call_llm([{"role": "user", "content": "x"}], {"view": 1}, system="SYS", context_label="VIEW_CONTEXT_JSON")
    assert captured[0][0]["content"].startswith("SYS\n\nVIEW_CONTEXT_JSON:")


def test_desk_briefing_reports_llm_error():
    briefing = generate_desk_briefing({"pair": "KO/PEP"})
    assert briefing["error"] and briefing["verdict"] == ""
    assert briefing["source"] is None


def test_desk_briefing_uses_raw_text_when_not_json(monkeypatch):
    use_providers(monkeypatch, {"cerebras": ["Plain prose verdict."]})
    briefing = generate_desk_briefing({"pair": "KO/PEP"})
    assert briefing["verdict"] == "Plain prose verdict."
    assert briefing["error"] is None
    assert briefing["source"].startswith("Cerebras")
