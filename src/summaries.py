"""Plain-English "What's happening" summaries for the Universe and Tape views."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd

from src.llm import call_llm, json_safe

SUMMARY_SYSTEM = (
    "You write the AI Report on TETHER, a market-neutral pairs-trading research desk. "
    "The audience includes hiring managers who are not quants, so be formal and precise, and define a term "
    "the first time you use it. Use only facts and numbers in VIEW_CONTEXT_JSON. If a figure is missing, omit it. "
    "Write two short paragraphs, no headings, bullets, or markdown. "
    "Paragraph one states the current position, the statistical evidence, and the most important number. "
    "Paragraph two covers recent trades, how the book compares with holding both stocks and with the index, "
    "and the main caveat (a failed relationship, a slow half-life, or a result that did not hold after the split). "
    "Do not give investment advice and do not invent numbers."
)

VIEW_FOCUS = {
    "universe": (
        "Summarize this sector scan: how many pairs were tested, how many survive the false-discovery (FDR) "
        "correction, which pair looks strongest and why, and which pairs looked tied together in the discovery "
        "period but came apart after the split date."
    ),
    "tape": (
        "Describe what this pair is doing right now: whether the book is long, short or flat the spread, how "
        "stretched the spread is (the z-score) against the entry, exit and stop bands, how the last few trades "
        "ended, and how the book has done against simply holding both stocks or the index."
    ),
}


def context_fingerprint(context: dict) -> str:
    return hashlib.sha1(json.dumps(context, sort_keys=True, default=str).encode()).hexdigest()


def build_universe_context(screen_table: pd.DataFrame, sector: str, split_date: str | None) -> dict:
    columns = [
        "pair",
        "tether_state",
        "tether_score",
        "p_value",
        "p_value_fdr",
        "fdr_significant",
        "half_life",
        "oos_tether_state",
        "tether_died",
    ]
    rows = [
        {col: json_safe(row[col]) for col in columns if col in screen_table.columns}
        for _, row in screen_table.iterrows()
    ]
    survives = screen_table["fdr_significant"] if "fdr_significant" in screen_table else pd.Series(dtype=bool)
    died = screen_table["tether_died"] if "tether_died" in screen_table else pd.Series(dtype=bool)
    return {
        "sector": sector,
        "split_date": split_date,
        "pairs_tested": int(len(screen_table)),
        "pairs_surviving_fdr": int(survives.sum()),
        "pairs_tethered": int((screen_table["tether_state"] == "TETHERED").sum()) if len(screen_table) else 0,
        "pairs_died_after_split": int(died.fillna(False).astype(bool).sum()),
        "pairs": rows,
    }


def _current_run(positions: pd.Series) -> tuple[float, int]:
    if positions.empty:
        return 0.0, 0
    values = positions.to_numpy()
    side = float(values[-1])
    different = np.flatnonzero(values != side)
    run = len(values) if different.size == 0 else len(values) - int(different[-1]) - 1
    return side, run


def build_tape_context(result: dict, entry_z: float, exit_z: float, stop_loss: float) -> dict:
    positions = result["positions"].fillna(0.0)
    zscore = result["zscore"].dropna()
    side, days_in_state = _current_run(positions)
    strategy_returns = result["strategy_returns"].fillna(0.0)
    blotter = result.get("blotter", pd.DataFrame())
    recent = []
    if isinstance(blotter, pd.DataFrame) and not blotter.empty:
        for _, row in blotter.tail(3).iterrows():
            recent.append(
                {
                    "side": row["side"],
                    "entry": str(pd.Timestamp(row["entry"]).date()),
                    "exit": str(pd.Timestamp(row["exit"]).date()),
                    "days": int(row["days"]),
                    "pnl": json_safe(float(row["pnl"])),
                    "how": row["reason"],
                }
            )
    last_20 = strategy_returns.tail(20)
    return {
        "pair": f"{result['asset_a']}/{result['asset_b']}",
        "benchmark": result.get("benchmark"),
        "as_of": str(positions.index[-1].date()) if len(positions) else None,
        "position_now": {1.0: "long the spread", -1.0: "short the spread"}.get(side, "flat"),
        "days_in_current_state": days_in_state,
        "zscore_now": json_safe(float(zscore.iloc[-1])) if len(zscore) else None,
        "bands": {"entry": entry_z, "exit": exit_z, "stop": stop_loss},
        "recent_trades": recent,
        "book_return_total": json_safe(float(np.exp(strategy_returns.sum()) - 1.0)),
        "book_return_last_20d": json_safe(float(np.exp(last_20.sum()) - 1.0)),
        "pair_buy_hold_return": json_safe(float(result["pair_buy_hold_cumulative"].iloc[-1])),
        "benchmark_return": json_safe(float(result["benchmark_cumulative"].iloc[-1])),
    }


def _pct(value) -> str:
    return "n/a" if value is None else f"{value:.1%}"


def rule_based_summary(view: str, context: dict) -> str:
    if view == "universe":
        n, fdr, died = context["pairs_tested"], context["pairs_surviving_fdr"], context["pairs_died_after_split"]
        lead = context["pairs"][0] if context["pairs"] else None
        text = (
            f"{fdr} of {n} {context['sector']} pairs still look statistically tied together after correcting "
            f"for testing many pairs at once (FDR)."
        )
        if lead:
            hl = lead.get("half_life")
            hl_text = f", and its gap closes halfway in about {hl:.0f} days" if hl else ""
            text += f" The strongest is {lead['pair']} ({lead['tether_state']}, score {lead['tether_score'] or 0:.0f}{hl_text})."
        if context["split_date"]:
            text += f" {died} pair(s) that looked tied before {context['split_date']} came apart afterwards."
        return text

    z = context.get("zscore_now")
    bands = context["bands"]
    if z is None:
        stretch = "The spread does not have enough history for a z-score yet."
    elif abs(z) >= bands["entry"]:
        stretch = f"The spread is stretched (z-score {z:+.2f}), past the ±{bands['entry']:g} entry band."
    elif abs(z) <= bands["exit"]:
        stretch = f"The spread is close to its normal level (z-score {z:+.2f})."
    else:
        stretch = f"The spread is moderately stretched (z-score {z:+.2f}), inside the ±{bands['entry']:g} entry band."
    text = (
        f"As of {context['as_of']}, the book is {context['position_now']} on {context['pair']} "
        f"and has been for {context['days_in_current_state']} trading days. {stretch}"
    )
    if context["recent_trades"]:
        last = context["recent_trades"][-1]
        text += f" The last trade ({last['side'].lower()}, {last['days']} days) ended by {last['how']} for ${last['pnl']:,.0f}."
    text += (
        f" Overall the book returned {_pct(context['book_return_total'])}, versus "
        f"{_pct(context['pair_buy_hold_return'])} for holding both stocks and "
        f"{_pct(context['benchmark_return'])} for {context['benchmark']}."
    )
    return text


def generate_view_summary(view: str, context: dict) -> dict:
    result = call_llm(
        [{"role": "user", "content": VIEW_FOCUS[view]}],
        context,
        system=SUMMARY_SYSTEM,
        context_label="VIEW_CONTEXT_JSON",
    )
    fingerprint = context_fingerprint(context)
    if result.ok and result.content.strip():
        return {
            "text": result.content.strip(),
            "source": f"{result.provider} · {result.model}",
            "error": None,
            "fingerprint": fingerprint,
        }
    return {
        "text": rule_based_summary(view, context),
        "source": None,
        "error": result.error,
        "fingerprint": fingerprint,
    }
