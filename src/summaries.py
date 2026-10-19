"""Plain-English "What's happening" summaries for the Universe and Tape views."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd

from src.llm import call_llm, json_safe

SUMMARY_SYSTEM = (
    "You write the AI Report on TETHER, a market-neutral pairs-trading research desk. "
    "The reader may not be a quant, so name each statistic in plain words the first time, then give the number. "
    "Use only VIEW_CONTEXT_JSON. If a figure is absent, leave it out. Never round a number that is not in the context, "
    "never invent a trade, and never give investment advice. "
    "Write exactly two paragraphs, no headings, bullets, or markdown. "
    "Paragraph one: what is true right now, then the single most important number, then the evidence for it. "
    "Paragraph two: the comparison the reader needs (against the other pairs, or against holding the stocks and the index) "
    "and one caveat, stated as a fact from the context. "
    "Prefer fewer claims that each carry a number over a general description."
)

VIEW_FOCUS = {
    "universe": (
        "Report this sector scan. State pairs tested, pairs still significant after the false-discovery correction "
        "(FDR), and pairs that were tied before the split date and came apart after it. "
        "Name the leading pair with its tether state, tether score, p-value, and half-life in days. "
        "If a pair died after the split, name it. Do not rank pairs the context does not rank."
    ),
    "tape": (
        "Report this pair as of the context date. State the position (long the spread, short the spread, or flat), "
        "how many trading days it has been held, and the z-score against the entry, exit, and stop bands. "
        "A z-score is how many normal wiggles the gap is from its average. "
        "Then the last trade: side, days held, how it ended, and its profit or loss. "
        "Close with three returns side by side: the book, holding both stocks, and the benchmark, each as a percent."
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
