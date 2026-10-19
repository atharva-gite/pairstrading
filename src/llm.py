"""Groq desk briefing and research chat, grounded in the book context."""

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

LLM_PROVIDER_ENV = "LLM_PROVIDER"


LLM_MODEL_ENV = "GROQ_MODEL"


DEFAULT_LLM_PROVIDER = "groq"


GROQ_MODEL_PRIMARY = "openai/gpt-oss-20b"


GROQ_MODEL_FALLBACK = "openai/gpt-oss-120b"


GROQ_CLIENT_FACTORY = None


RESEARCH_SYSTEM = (
    "You are the research analyst on TETHER, a market-neutral pairs desk. "
    "Answer only from ACTIVE_BOOK_CONTEXT_JSON. If a number is missing, say so. "
    "Do not invent trades, p-values, or PnL. "
    "A BROKEN tether is usually correlation without a tradable mean-reverting spread. "
    "If half_life is above 60 days, do not treat on-time exits as success. "
    "Be specific, concise, and use the actual pair and numbers from context."
)


DESK_BRIEFING_SYSTEM = (
    "You are the PM on TETHER. Write a briefing from ACTIVE_BOOK_CONTEXT_JSON only. "
    "Return JSON with keys headline, verdict, drivers, risks. No markdown. "
    "headline: one line. verdict: 3-5 sentences a trader can act on. "
    "drivers: why the tether score and PnL look like this. "
    "risks: what would invalidate the read. "
    "If tether_state is BROKEN, lead with that. "
    "If half_life > 60, say the OU clock is too slow to use on-time exits as evidence."
)


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


@dataclass
class LLMResult:
    ok: bool
    content: str
    error: str | None = None


def resolve_groq_api_key(
    secrets_getter: Callable[[str], object | None] | None = None,
) -> tuple[str | None, str]:
    _load_project_dotenv()
    api_key = os.getenv("GROQ_API_KEY")
    if api_key:
        return api_key.strip().strip('"'), ".env"

    if secrets_getter is not None:
        try:
            secret_value = secrets_getter("GROQ_API_KEY")
        except Exception:
            secret_value = None
        if secret_value:
            return str(secret_value), "st.secrets"
    return None, "not found"


def call_llm(
    messages: list[dict[str, str]],
    context: str | dict[str, object],
    task: str = "research",
) -> LLMResult:
    _load_project_dotenv()
    provider = os.getenv(LLM_PROVIDER_ENV, DEFAULT_LLM_PROVIDER).lower()
    if provider != "groq":
        return LLMResult(False, "", f"Unsupported LLM_PROVIDER '{provider}'. Set LLM_PROVIDER=groq.")
    api_key, key_source = resolve_groq_api_key()
    if not api_key:
        return LLMResult(False, "", "GROQ_API_KEY is missing. Add it to .env.")
    if key_source == "st.secrets":
        os.environ.setdefault("GROQ_API_KEY", api_key)
    model = os.getenv(LLM_MODEL_ENV, GROQ_MODEL_PRIMARY)
    context_text = context if isinstance(context, str) else serialize_book_context(context)
    system = DESK_BRIEFING_SYSTEM if task == "briefing" else RESEARCH_SYSTEM
    input_messages = [
        {"role": "system", "content": f"{system}\n\nACTIVE_BOOK_CONTEXT_JSON:\n{context_text}"},
        *messages,
    ]
    try:
        client = _get_groq_client(api_key)
        response = _create_groq_chat_completion(client, model, input_messages)
        content = response.choices[0].message.content
        if not content:
            return LLMResult(False, "", "Groq returned an empty response.")
        return LLMResult(True, str(content))
    except Exception as exc:
        if _is_groq_model_not_found(exc) and model != GROQ_MODEL_FALLBACK:
            try:
                response = _create_groq_chat_completion(client, GROQ_MODEL_FALLBACK, input_messages)
                content = response.choices[0].message.content
                if not content:
                    return LLMResult(False, "", "Groq returned an empty response.")
                return LLMResult(True, str(content))
            except Exception as fallback_exc:
                return LLMResult(False, "", f"Groq request failed: {fallback_exc}")
        return LLMResult(False, "", f"Groq request failed: {exc}")


def generate_desk_briefing(context: dict) -> dict:
    result = call_llm(
        [{"role": "user", "content": "Write the desk briefing JSON for this book."}],
        context,
        task="briefing",
    )
    if not result.ok:
        return {"headline": "", "verdict": "", "drivers": "", "risks": "", "error": result.error}
    parsed = parse_desk_briefing(result.content)
    if not parsed or not parsed.get("verdict"):
        return {
            "headline": "",
            "verdict": result.content.strip(),
            "drivers": "",
            "risks": "",
            "error": None,
        }
    parsed["error"] = None
    return parsed


def _create_groq_chat_completion(client, model: str, input_messages: list[dict[str, str]]):
    return client.chat.completions.create(model=model, messages=input_messages, temperature=0.2)


def _is_groq_model_not_found(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)
    response = getattr(exc, "response", None)
    if status_code is None and response is not None:
        status_code = getattr(response, "status_code", None)
    if status_code != 404:
        return False
    message = str(exc).lower()
    return "model" in message and (
        "not found" in message or "does not exist" in message or "model_not_found" in message
    )


def _get_groq_client(api_key: str):
    if GROQ_CLIENT_FACTORY is not None:
        return GROQ_CLIENT_FACTORY(api_key=api_key)
    from groq import Groq

    return Groq(api_key=api_key)
