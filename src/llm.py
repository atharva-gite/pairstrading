"""LLM transport for the desk briefing, Research chat and view summaries.

Every supported provider exposes an OpenAI-compatible chat endpoint, so one
client covers all of them. Providers with a key are tried in order; a rate
limit or outage on one falls through to the next.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from src.stats import MAX_TRADABLE_HALF_LIFE


@dataclass(frozen=True)
class Provider:
    name: str
    label: str
    key_env: str
    base_url: str
    model: str
    fallback_model: str | None = None

    @property
    def model_env(self) -> str:
        return f"{self.name.upper()}_MODEL"


PROVIDERS: dict[str, Provider] = {
    p.name: p
    for p in [
        Provider(
            "groq",
            "Groq",
            "GROQ_API_KEY",
            "https://api.groq.com/openai/v1",
            "openai/gpt-oss-20b",
            "openai/gpt-oss-120b",
        ),
        Provider(
            "gemini",
            "Google Gemini",
            "GEMINI_API_KEY",
            "https://generativelanguage.googleapis.com/v1beta/openai/",
            "gemini-2.5-flash",
            "gemini-2.5-flash-lite",
        ),
        Provider(
            "cerebras", "Cerebras", "CEREBRAS_API_KEY", "https://api.cerebras.ai/v1", "gpt-oss-120b", "llama3.1-8b"
        ),
        Provider(
            "mistral",
            "Mistral",
            "MISTRAL_API_KEY",
            "https://api.mistral.ai/v1",
            "mistral-small-latest",
            "open-mistral-nemo",
        ),
        Provider(
            "openrouter",
            "OpenRouter",
            "OPENROUTER_API_KEY",
            "https://openrouter.ai/api/v1",
            "openai/gpt-oss-20b:free",
            "meta-llama/llama-3.3-70b-instruct:free",
        ),
        Provider(
            "github",
            "GitHub Models",
            "GITHUB_MODELS_TOKEN",
            "https://models.github.ai/inference",
            "openai/gpt-4.1-mini",
        ),
    ]
}

LLM_PROVIDERS_ENV = "LLM_PROVIDERS"
LLM_PROVIDER_ENV = "LLM_PROVIDER"
DEFAULT_PROVIDER_ORDER = ["groq", "gemini", "cerebras", "mistral", "openrouter", "github"]

LLM_CLIENT_FACTORY: Callable[..., Any] | None = None

RESEARCH_SYSTEM = (
    "You are the research analyst on TETHER, a market-neutral pairs desk. "
    "Answer only from ACTIVE_BOOK_CONTEXT_JSON. If a number is missing, say it is not in the book. "
    "Do not invent trades, p-values, dates, or PnL, and do not give investment advice. "
    "A BROKEN tether is correlation without a tradable mean-reverting spread. "
    "If half_life is above 60 days, on-time exits are not evidence the relationship is tradable. "
    "Write 2 short paragraphs. Every claim names the pair and cites the figure that supports it "
    "(p-value, half-life in days, tether score, Sharpe, return, or a blotter row with its dates and PnL). "
    "Percents to one decimal. No preamble and no restatement of the question."
)

DESK_BRIEFING_SYSTEM = (
    "You are the PM on TETHER. Write a briefing from ACTIVE_BOOK_CONTEXT_JSON only. "
    "Return JSON with keys headline, verdict, drivers, risks. No markdown and no code fence. "
    "headline: one line naming the pair, the tether state, and the single most important number. "
    "verdict: 3 or 4 sentences. Lead with BROKEN or a half-life above 60 days when either is true. "
    "Include the tether score, the cointegration p-value, the half-life in days, and the book return "
    "next to the pair buy-and-hold and the benchmark. "
    "drivers: the two or three figures that explain the score and the PnL. "
    "risks: the specific result that would invalidate the read (the out-of-sample state, a stop, "
    "or a half-life too slow to trade). "
    "Do not invent numbers and do not give investment advice."
)

TASK_SYSTEMS = {"briefing": DESK_BRIEFING_SYSTEM, "research": RESEARCH_SYSTEM}


@dataclass
class LLMResult:
    ok: bool
    content: str
    error: str | None = None
    provider: str | None = None
    model: str | None = None


def _load_project_dotenv() -> None:
    load_dotenv(dotenv_path=Path.cwd() / ".env", override=False)
    load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / ".env", override=False)


def json_safe(value: Any):
    if value is None:
        return None
    if isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        if not np.isfinite(value):
            return None
        return float(value)
    if isinstance(value, (np.floating, np.integer)):
        value = value.item()
        if isinstance(value, float) and not np.isfinite(value):
            return None
        return value
    return str(value)


def build_book_context(result: dict) -> dict:
    metrics = {key: json_safe(value) for key, value in result.get("metrics", {}).items()}
    blotter = result.get("blotter", pd.DataFrame())
    records = []
    if isinstance(blotter, pd.DataFrame) and not blotter.empty:
        view = blotter.copy()
        for column in ("entry", "exit"):
            if column in view.columns:
                view[column] = pd.to_datetime(view[column]).dt.strftime("%Y-%m-%d")
        records = json.loads(view.to_json(orient="records"))

    rolling = result.get("rolling_p_value", pd.Series(dtype="float64"))
    rolling_last = None
    rolling_max = None
    if isinstance(rolling, pd.Series) and len(rolling):
        rolling_last = json_safe(float(rolling.iloc[-1]))
        rolling_max = json_safe(float(rolling.max()))

    return {
        "pair": f"{result['asset_a']}/{result['asset_b']}",
        "asset_a": result["asset_a"],
        "asset_b": result["asset_b"],
        "benchmark": result.get("benchmark"),
        "hedge_method": result.get("hedge_method"),
        "sizing_method": result.get("sizing_method"),
        "metrics": metrics,
        "blotter": records,
        "rolling_p_value_last": rolling_last,
        "rolling_p_value_max": rolling_max,
        "desk_rules": {
            "tradable_half_life_days": [8, 40],
            "max_tethered_half_life_days": MAX_TRADABLE_HALF_LIFE,
            "on_time_metric_valid_if_half_life_days": [5, 60],
            "cointegration_p_cutoff": 0.05,
        },
    }


def serialize_book_context(context: dict) -> str:
    return json.dumps(context, default=str, indent=2)


def parse_desk_briefing(text: str) -> dict | None:
    if not text:
        return None
    cleaned = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, flags=re.DOTALL)
    if fenced:
        cleaned = fenced.group(1)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        payload = json.loads(cleaned[start : end + 1])
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    return {
        "headline": str(payload.get("headline") or "").strip(),
        "verdict": str(payload.get("verdict") or "").strip(),
        "drivers": str(payload.get("drivers") or "").strip(),
        "risks": str(payload.get("risks") or "").strip(),
    }


def resolve_api_key(
    env_var: str,
    secrets_getter: Callable[[str], object | None] | None = None,
) -> str | None:
    _load_project_dotenv()
    api_key = (os.getenv(env_var) or "").strip().strip('"')
    if api_key:
        return api_key
    if secrets_getter is not None:
        try:
            secret_value = secrets_getter(env_var)
        except Exception:
            secret_value = None
        if secret_value:
            return str(secret_value)
    return None


def provider_order() -> tuple[list[Provider], list[str]]:
    """Providers to try, in order, plus any unknown names from the environment."""
    _load_project_dotenv()
    raw = os.getenv(LLM_PROVIDERS_ENV) or ""
    if raw.strip():
        names = [n.strip().lower() for n in raw.split(",") if n.strip()]
    else:
        preferred = (os.getenv(LLM_PROVIDER_ENV) or "").strip().lower()
        names = ([preferred] if preferred else []) + [n for n in DEFAULT_PROVIDER_ORDER if n != preferred]
    unknown = [n for n in names if n not in PROVIDERS]
    return [PROVIDERS[n] for n in names if n in PROVIDERS], unknown


def configured_providers() -> list[tuple[Provider, str]]:
    ordered, _ = provider_order()
    pairs = []
    for provider in ordered:
        key = resolve_api_key(provider.key_env)
        if key:
            pairs.append((provider, key))
    return pairs


def missing_key_message() -> str:
    keys = ", ".join(PROVIDERS[n].key_env for n in DEFAULT_PROVIDER_ORDER)
    return f"No AI key found. Add one of {keys} to .env."


def call_llm(
    messages: list[dict[str, str]],
    context: str | dict[str, object],
    task: str = "research",
    system: str | None = None,
    context_label: str = "ACTIVE_BOOK_CONTEXT_JSON",
) -> LLMResult:
    _, unknown = provider_order()
    if unknown:
        choices = ", ".join(PROVIDERS)
        return LLMResult(False, "", f"Unsupported LLM provider {', '.join(unknown)}. Choose from: {choices}.")
    available = configured_providers()
    if not available:
        return LLMResult(False, "", missing_key_message())

    context_text = context if isinstance(context, str) else serialize_book_context(context)
    system_text = system or TASK_SYSTEMS.get(task, RESEARCH_SYSTEM)
    input_messages = [
        {"role": "system", "content": f"{system_text}\n\n{context_label}:\n{context_text}"},
        *messages,
    ]
    errors = []
    for provider, api_key in available:
        result = _call_provider(provider, api_key, input_messages)
        if result.ok:
            return result
        errors.append(f"{provider.label}: {result.error}")
    if len(errors) == 1:
        return LLMResult(False, "", errors[0])
    return LLMResult(False, "", "All AI providers failed. " + " | ".join(errors))


def _call_provider(provider: Provider, api_key: str, input_messages: list[dict[str, str]]) -> LLMResult:
    primary = os.getenv(provider.model_env) or provider.model
    models = [primary] + ([provider.fallback_model] if provider.fallback_model not in (None, primary) else [])
    client = _get_client(provider, api_key)
    for i, model in enumerate(models):
        try:
            response = client.chat.completions.create(model=model, messages=input_messages, temperature=0.2)
        except Exception as exc:
            if i + 1 < len(models) and (_is_model_not_found(exc) or _is_transient(exc)):
                continue
            return LLMResult(False, "", f"request failed: {_describe_error(exc)}", provider.label, model)
        content = response.choices[0].message.content
        if not content:
            return LLMResult(False, "", "returned an empty response.", provider.label, model)
        return LLMResult(True, str(content), None, provider.label, model)
    return LLMResult(False, "", "no model available.", provider.label)


def generate_desk_briefing(context: dict) -> dict:
    result = call_llm(
        [{"role": "user", "content": "Write the desk briefing JSON for this book."}],
        context,
        task="briefing",
    )
    if not result.ok:
        return {"headline": "", "verdict": "", "drivers": "", "risks": "", "error": result.error, "source": None}
    parsed = parse_desk_briefing(result.content)
    source = f"{result.provider} · {result.model}"
    if not parsed or not parsed.get("verdict"):
        return {
            "headline": "",
            "verdict": result.content.strip(),
            "drivers": "",
            "risks": "",
            "error": None,
            "source": source,
        }
    parsed["error"] = None
    parsed["source"] = source
    return parsed


def _status_code(exc: Exception) -> int | None:
    status_code = getattr(exc, "status_code", None)
    response = getattr(exc, "response", None)
    if status_code is None and response is not None:
        status_code = getattr(response, "status_code", None)
    return status_code


def _describe_error(exc: Exception) -> str:
    """One short phrase for the UI; provider error bodies can run to several hundred characters."""
    code = _status_code(exc)
    if code == 429:
        return "rate limited (free-tier quota), try again in a minute"
    if code in (401, 403):
        return "API key rejected"
    if code is not None and code >= 500:
        return f"provider error (HTTP {code})"
    if code is not None:
        return f"HTTP {code}"
    message = str(exc).strip() or type(exc).__name__
    return message if len(message) <= 120 else message[:117] + "..."


def _is_transient(exc: Exception) -> bool:
    """Rate limits and provider outages are worth one try on the fallback model."""
    code = _status_code(exc)
    return code in (429, 500, 502, 503, 504)


def _is_model_not_found(exc: Exception) -> bool:
    status_code = _status_code(exc)
    if status_code not in (400, 404):
        return False
    message = str(exc).lower()
    return "model" in message and (
        "not found" in message
        or "does not exist" in message
        or "model_not_found" in message
        or "not a valid model" in message
        or "invalid model" in message
        or "no endpoints" in message
    )


def _get_client(provider: Provider, api_key: str):
    if LLM_CLIENT_FACTORY is not None:
        return LLM_CLIENT_FACTORY(provider=provider, api_key=api_key)
    from openai import OpenAI

    return OpenAI(api_key=api_key, base_url=provider.base_url, max_retries=1, timeout=45)
