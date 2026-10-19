"""Signals, dollar sizing and limits, blotter, performance."""

from __future__ import annotations

import numpy as np
import pandas as pd

DESK_GROSS_LIMIT = 10_000_000.0


DEFAULT_MAX_LEG_PCT = 0.75


def build_signals(
    zscore: pd.Series,
    entry_z: float = 2.0,
    exit_z: float = 0.5,
    stop_loss: float = 3.0,
) -> pd.Series:
    position = 0
    signals = pd.Series(index=zscore.index, dtype="float64")

    for idx, value in zscore.items():
        if np.isnan(value):
            signals.loc[idx] = position
            continue

        if position == 0:
            if value > entry_z:
                position = -1
            elif value < -entry_z:
                position = 1
        elif position == 1:
            if value >= -exit_z or value <= -stop_loss:
                position = 0
            elif value > entry_z:
                position = -1
        elif position == -1:
            if value <= exit_z or value >= stop_loss:
                position = 0
            elif value < -entry_z:
                position = 1

        signals.loc[idx] = position

    return signals


def compute_performance_metrics(returns: pd.Series, positions: pd.Series) -> dict:
    returns = returns.fillna(0.0)
    trading_days = 252.0
    mean_daily = returns.mean()
    std_daily = returns.std(ddof=0)

    if std_daily > 0:
        sharpe = float(np.sqrt(trading_days) * mean_daily / std_daily)
    else:
        sharpe = float("nan")

    growth = np.exp(returns.cumsum())
    drawdown = growth / growth.cummax() - 1.0
    max_dd = float(drawdown.min())

    active = positions.shift(1).fillna(0) != 0
    trades = returns[active]
    win_rate = float((trades > 0).sum() / trades.shape[0]) if trades.shape[0] else float("nan")

    return {
        "sharpe_ratio": sharpe,
        "max_drawdown": max_dd,
        "win_rate": win_rate,
        "total_return": float(np.exp(returns.sum()) - 1.0),
    }


def build_trade_blotter(
    positions: pd.Series,
    zscore: pd.Series,
    dollar_pnl: pd.Series,
    half_life: float,
    stop_loss: float,
) -> pd.DataFrame:
    pos = positions.fillna(0.0)
    rows = []
    i = 0
    n = len(pos)
    while i < n:
        side = float(pos.iloc[i])
        if side == 0:
            i += 1
            continue
        start = i
        j = i + 1
        while j < n and float(pos.iloc[j]) == side:
            j += 1
        open_trade = j == n
        pnl_end = j if j < n else n - 1
        pnl = float(dollar_pnl.iloc[start + 1 : pnl_end + 1].sum()) if start + 1 < n else 0.0
        days = max(j - start, 1)
        entry_z = float(zscore.iloc[start]) if pd.notna(zscore.iloc[start]) else float("nan")
        exit_z = float(zscore.iloc[min(j, n - 1)]) if pd.notna(zscore.iloc[min(j, n - 1)]) else float("nan")
        if open_trade:
            reason = "open"
        elif pd.notna(exit_z) and abs(exit_z) >= stop_loss - 1e-9:
            reason = "stop"
        else:
            reason = "revert"
        hl_mult = float(days / half_life) if half_life and np.isfinite(half_life) and half_life > 0 else float("nan")
        rows.append(
            {
                "side": "Long spread" if side > 0 else "Short spread",
                "entry": pos.index[start],
                "exit": pos.index[min(j, n - 1)],
                "days": int(days),
                "entry_z": entry_z,
                "exit_z": exit_z,
                "half_life_mult": hl_mult,
                "pnl": pnl,
                "reason": reason,
            }
        )
        i = j
    return pd.DataFrame(rows)


def dollar_notionals(
    beta: pd.Series,
    capital: float,
    sizing_method: str = "Hedge Ratio",
    max_leg_notional: float | None = None,
) -> tuple[pd.Series, pd.Series]:
    """Map a dollar book to legs A and B.

    Hedge Ratio: gross |A| + |B| = capital, with |B| = |beta| * |A|.
    Equal Dollar: |A| = |B| = capital / 2, still short B when beta > 0.
    max_leg_notional scales both legs down together, so the hedge ratio holds and gross falls below capital.
    """
    abs_beta = beta.abs()
    if sizing_method == "Equal Dollar":
        notional_a = pd.Series(capital / 2.0, index=beta.index)
        notional_b = np.sign(beta.replace(0.0, 1.0)) * (capital / 2.0)
    else:
        denom = 1.0 + abs_beta
        denom = denom.replace(0.0, np.nan)
        notional_a = capital / denom
        notional_b = notional_a * beta
    notional_a, notional_b = notional_a.fillna(0.0), notional_b.fillna(0.0)
    if max_leg_notional is not None:
        largest = np.maximum(notional_a.abs(), notional_b.abs())
        scale = (max_leg_notional / largest.where(largest > 0)).clip(upper=1.0).fillna(1.0)
        notional_a, notional_b = notional_a * scale, notional_b * scale
    return notional_a, notional_b
