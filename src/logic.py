from __future__ import annotations

import copy
import json
import os
import re
from collections.abc import Callable
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
import yfinance as yf
from dotenv import load_dotenv
from statsmodels.stats.multitest import multipletests
from statsmodels.tsa.stattools import adfuller, coint

SECTOR_UNIVERSES = {
    "Energy": ["XOM", "CVX", "COP", "BP", "SHEL"],
    "Consumer Staples": ["KO", "PEP", "PG", "WMT", "COST"],
    "Technology": ["AAPL", "MSFT", "GOOGL", "META", "NVDA", "AMD"],
    "Financials": ["JPM", "BAC", "WFC", "GS", "MS"],
    "Healthcare": ["JNJ", "PFE", "UNH", "ABBV", "MRK"],
}

SCREEN_COLUMNS = [
    "asset_a",
    "asset_b",
    "pair",
    "p_value",
    "adf_p_value",
    "hedge_ratio",
    "half_life",
    "tether_score",
    "tether_state",
    "tether_reason",
    "cointegrated",
    "p_value_fdr",
    "p_value_bonferroni",
    "fdr_significant",
    "bonferroni_significant",
    "oos_tether_state",
    "tether_died",
]

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


def download_price_data(tickers, start, end):
    closes = yf.download(tickers, start=start, end=end, progress=False)["Close"]
    if isinstance(closes, pd.Series):
        closes = closes.to_frame(name=tickers)
    return closes.dropna()


def log_price(series: pd.Series) -> pd.Series:
    return np.log(series)


def check_cointegration(series_a: pd.Series, series_b: pd.Series) -> dict:
    a = log_price(series_a).dropna()
    a.name = series_a.name or "asset_a"
    b = log_price(series_b).reindex(a.index).dropna()
    b.name = series_b.name or "asset_b"
    a = a.reindex(b.index)

    x = sm.add_constant(b)
    model = sm.OLS(a, x).fit()

    hedge_ratio = float(model.params[b.name])
    intercept = float(model.params["const"])
    spread = a - (hedge_ratio * b + intercept)

    score, p_value, _ = coint(a, b)
    adf_result = adfuller(spread.dropna())
    adf_stat = float(adf_result[0])
    adf_p = float(adf_result[1])

    return {
        "t_statistic": float(score),
        "p_value": float(p_value),
        "hedge_ratio": hedge_ratio,
        "intercept": intercept,
        "adf_statistic": adf_stat,
        "adf_p_value": adf_p,
        "spread": spread,
    }


def apply_kalman_filter(
    y: pd.Series,
    x: pd.Series,
    delta: float = 1e-5,
    obs_cov: float = 1e-3,
) -> tuple[pd.Series, pd.Series]:
    y = y.dropna()
    x = x.reindex(y.index).dropna()
    x = x.reindex(y.index)

    n = len(y)
    state_means = np.zeros((n, 2))
    current_state_mean = np.zeros(2)
    current_state_cov = np.eye(2)
    trans_cov = delta / (1.0 - delta) * np.eye(2)

    obs_mat = np.column_stack([x.values, np.ones(n)])

    for t in range(n):
        obs_val = y.values[t]
        obs_matrix = obs_mat[t]

        innovation = obs_val - np.dot(obs_matrix, current_state_mean)
        innovation_cov = np.dot(obs_matrix, np.dot(current_state_cov, obs_matrix.T)) + obs_cov
        kalman_gain = np.dot(current_state_cov, obs_matrix.T) / innovation_cov

        current_state_mean = current_state_mean + (kalman_gain * innovation).flatten()
        current_state_cov = current_state_cov - np.outer(kalman_gain, np.dot(obs_matrix, current_state_cov))
        current_state_cov = current_state_cov + trans_cov

        state_means[t] = current_state_mean

    beta = pd.Series(state_means[:, 0], index=y.index, name="beta")
    alpha = pd.Series(state_means[:, 1], index=y.index, name="alpha")
    return beta, alpha


def rolling_ols_hedge_ratio(
    y: pd.Series,
    x: pd.Series,
    window: int = 30,
) -> tuple[pd.Series, pd.Series]:
    y = y.dropna()
    x = x.reindex(y.index).dropna()
    x = x.reindex(y.index)
    x.name = x.name or "x"

    betas = [np.nan] * len(y)
    alphas = [np.nan] * len(y)

    for i in range(window, len(y) + 1):
        y_slice = y.iloc[i - window : i]
        x_slice = x.iloc[i - window : i]
        model = sm.OLS(y_slice, sm.add_constant(x_slice)).fit()
        betas[i - 1] = float(model.params[x.name])
        alphas[i - 1] = float(model.params["const"])

    beta_series = pd.Series(betas, index=y.index, name="beta")
    alpha_series = pd.Series(alphas, index=y.index, name="alpha")
    return beta_series, alpha_series


def calculate_half_life(spread: pd.Series) -> float:
    spread = spread.dropna()
    if len(spread) < 10:
        return float("nan")

    delta_spread = spread.diff().dropna()
    lagged_spread = spread.shift(1).dropna()
    df = pd.concat([delta_spread, lagged_spread], axis=1).dropna()
    df.columns = ["delta", "lag"]

    model = sm.OLS(df["delta"], sm.add_constant(df["lag"])).fit()
    k = float(model.params["lag"])
    if np.isclose(k, 0) or k >= 0:
        return float("nan")

    return float(-np.log(2) / k)


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


def compute_beta_ratio(
    asset_a_returns: pd.Series,
    asset_b_returns: pd.Series,
    benchmark_returns: pd.Series,
) -> dict:
    combined_a = pd.concat([asset_a_returns, benchmark_returns], axis=1).dropna()
    combined_b = pd.concat([asset_b_returns, benchmark_returns], axis=1).dropna()

    beta_a = float(sm.OLS(combined_a.iloc[:, 0], sm.add_constant(combined_a.iloc[:, 1])).fit().params.iloc[-1])
    beta_b = float(sm.OLS(combined_b.iloc[:, 0], sm.add_constant(combined_b.iloc[:, 1])).fit().params.iloc[-1])
    ratio = float(beta_a / beta_b) if not np.isclose(beta_b, 0.0) else float("nan")

    return {
        "beta_a": beta_a,
        "beta_b": beta_b,
        "beta_ratio": ratio,
    }


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


def compute_tether_profile(
    p_value: float,
    adf_p_value: float,
    half_life: float,
    beta: pd.Series | None = None,
) -> dict:
    """Score whether a pair is still tradable as a mean-reverting tether.

    Desks lose money when the long-run relationship breaks, not when z-scores
    look noisy. The score blends Engle-Granger evidence, half-life, and hedge stability.
    """
    p_score = float(np.clip(1.0 - (p_value / 0.10), 0.0, 1.0))
    adf_score = float(np.clip(1.0 - (adf_p_value / 0.10), 0.0, 1.0))

    if half_life is None or not np.isfinite(half_life) or half_life <= 0:
        hl_score = 0.0
    elif 8.0 <= half_life <= 40.0:
        hl_score = 1.0
    elif 5.0 <= half_life <= 60.0:
        hl_score = 0.75
    elif half_life < 5.0:
        hl_score = 0.35
    else:
        hl_score = float(np.clip(1.0 - (half_life - 60.0) / 120.0, 0.0, 0.4))

    if beta is None or len(beta.dropna()) < 8:
        stability = 0.80
    else:
        b = beta.dropna()
        cv = float(b.std(ddof=0) / (abs(float(b.mean())) + 1e-8))
        stability = float(np.clip(1.0 / (1.0 + cv), 0.0, 1.0))

    score = 100.0 * (0.40 * p_score + 0.20 * adf_score + 0.25 * hl_score + 0.15 * stability)
    if p_value >= 0.10:
        reason = "no_cointegration"
    elif half_life is None or not np.isfinite(half_life) or half_life <= 0:
        reason = "no_half_life"
    elif half_life > 60:
        reason = "too_slow"
    elif half_life < 5:
        reason = "too_fast"
    elif score >= 70 and p_value < 0.05:
        reason = "tradable"
    else:
        reason = "weak_evidence"

    if score >= 70 and p_value < 0.05:
        state = "TETHERED"
    elif score >= 45 and p_value < 0.10:
        state = "WEAK"
    else:
        state = "BROKEN"

    return {
        "tether_score": float(score),
        "tether_state": state,
        "tether_reason": reason,
        "p_score": p_score,
        "half_life_score": hl_score,
        "stability_score": stability,
    }


def rolling_cointegration_pvalues(
    series_a: pd.Series,
    series_b: pd.Series,
    window: int = 126,
    step: int = 10,
) -> pd.Series:
    a = series_a.dropna()
    b = series_b.reindex(a.index).dropna()
    a = a.reindex(b.index)
    if len(a) <= window:
        return pd.Series(dtype="float64", name="rolling_p_value")

    dates = []
    values = []
    for i in range(window, len(a) + 1, step):
        try:
            result = check_cointegration(a.iloc[i - window : i], b.iloc[i - window : i])
            dates.append(a.index[i - 1])
            values.append(result["p_value"])
        except (ValueError, np.linalg.LinAlgError, KeyError):
            continue
    return pd.Series(values, index=dates, name="rolling_p_value")


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


def split_masks(index, split_date: str) -> tuple[np.ndarray, pd.DatetimeIndex]:
    idx = pd.DatetimeIndex(index)
    ts = pd.Timestamp(split_date)
    if idx.tz is not None:
        ts = ts.tz_localize(idx.tz) if ts.tzinfo is None else ts.tz_convert(idx.tz)
    else:
        if ts.tzinfo is not None:
            ts = ts.tz_localize(None)
    mask = np.asarray(idx < ts)
    return mask, ts


def apply_multiple_testing(ranked: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    ranked = ranked.copy()
    if ranked.empty:
        ranked["p_value_fdr"] = []
        ranked["p_value_bonferroni"] = []
        ranked["fdr_significant"] = []
        ranked["bonferroni_significant"] = []
        return ranked
    pvals = ranked["p_value"].fillna(1.0).to_numpy(dtype=float)
    reject_fdr, p_fdr, _, _ = multipletests(pvals, alpha=alpha, method="fdr_bh")
    reject_bonf, p_bonf, _, _ = multipletests(pvals, alpha=alpha, method="bonferroni")
    ranked["p_value_fdr"] = p_fdr
    ranked["p_value_bonferroni"] = p_bonf
    ranked["fdr_significant"] = reject_fdr
    ranked["bonferroni_significant"] = reject_bonf
    return ranked


def tether_from_prices(series_a: pd.Series, series_b: pd.Series) -> dict:
    if len(series_a.dropna()) < 40 or len(series_b.dropna()) < 40:
        return {
            "p_value": float("nan"),
            "adf_p_value": float("nan"),
            "hedge_ratio": float("nan"),
            "half_life": float("nan"),
            "tether_score": float("nan"),
            "tether_state": "BROKEN",
            "tether_reason": "too_few_obs",
        }
    coint = check_cointegration(series_a, series_b)
    half_life = calculate_half_life(coint["spread"])
    tether = compute_tether_profile(coint["p_value"], coint["adf_p_value"], half_life)
    return {
        "p_value": coint["p_value"],
        "adf_p_value": coint["adf_p_value"],
        "hedge_ratio": coint["hedge_ratio"],
        "intercept": coint["intercept"],
        "half_life": half_life,
        **tether,
    }


def dollar_notionals(
    beta: pd.Series,
    capital: float,
    sizing_method: str = "Hedge Ratio",
) -> tuple[pd.Series, pd.Series]:
    """Map a dollar book to legs A and B.

    Hedge Ratio: gross |A| + |B| = capital, with |B| = |beta| * |A|.
    Equal Dollar: |A| = |B| = capital / 2, still short B when beta > 0.
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
    return notional_a.fillna(0.0), notional_b.fillna(0.0)


def pair_pvalue_matrix(screen_table: pd.DataFrame, value_col: str = "p_value") -> pd.DataFrame:
    if screen_table.empty:
        return pd.DataFrame()
    tickers = sorted(set(screen_table["asset_a"]) | set(screen_table["asset_b"]))
    matrix = pd.DataFrame(np.nan, index=tickers, columns=tickers)
    col = value_col if value_col in screen_table.columns else "p_value"
    for _, row in screen_table.iterrows():
        matrix.loc[row["asset_a"], row["asset_b"]] = row[col]
        matrix.loc[row["asset_b"], row["asset_a"]] = row[col]
    for ticker in tickers:
        matrix.loc[ticker, ticker] = 0.0
    return matrix


def screen_sector_pairs(
    prices: pd.DataFrame,
    p_value_threshold: float = 0.05,
    split_date: str | None = None,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Rank pairs on the discovery sample, then FDR-correct and optionally score live death."""
    if split_date:
        is_mask, _ = split_masks(prices.index, split_date)
        discovery = prices.loc[is_mask]
        live = prices.loc[~is_mask]
    else:
        discovery = prices
        live = pd.DataFrame()

    tickers = [col for col in discovery.columns if discovery[col].notna().sum() > 30]
    rows = []
    for asset_a, asset_b in combinations(tickers, 2):
        try:
            discovered = tether_from_prices(discovery[asset_a], discovery[asset_b])
            oos_state = None
            died = False
            if not live.empty and asset_a in live.columns and asset_b in live.columns:
                live_tether = tether_from_prices(live[asset_a], live[asset_b])
                oos_state = live_tether["tether_state"]
                died = discovered["tether_state"] == "TETHERED" and oos_state != "TETHERED"
            rows.append(
                {
                    "asset_a": asset_a,
                    "asset_b": asset_b,
                    "pair": f"{asset_a}/{asset_b}",
                    "p_value": discovered["p_value"],
                    "adf_p_value": discovered["adf_p_value"],
                    "hedge_ratio": discovered["hedge_ratio"],
                    "half_life": discovered["half_life"],
                    "tether_score": discovered["tether_score"],
                    "tether_state": discovered["tether_state"],
                    "tether_reason": discovered["tether_reason"],
                    "cointegrated": bool(
                        np.isfinite(discovered["p_value"]) and discovered["p_value"] < p_value_threshold
                    ),
                    "oos_tether_state": oos_state,
                    "tether_died": died,
                }
            )
        except (ValueError, np.linalg.LinAlgError, KeyError):
            continue

    if not rows:
        return pd.DataFrame(columns=SCREEN_COLUMNS)

    ranked = apply_multiple_testing(pd.DataFrame(rows), alpha=alpha)
    return ranked.sort_values(
        ["fdr_significant", "tether_score", "p_value"],
        ascending=[False, False, True],
    ).reset_index(drop=True)


class PairsTradingEngine:
    def __init__(
        self,
        asset_a: str,
        asset_b: str,
        benchmark: str,
        start: str,
        end: str,
        lookback: int = 30,
        entry_z: float = 2.0,
        exit_z: float = 0.5,
        stop_loss: float = 3.0,
        slippage: float = 0.001,
        commission: float = 0.0005,
        hedge_method: str = "Static OLS",
        capital: float = 100_000.0,
        sizing_method: str = "Hedge Ratio",
        walk_forward: bool = True,
        split_date: str = "2024-01-01",
        charge_borrow: bool = True,
        borrow_bps: float = 50.0,
    ):
        self.asset_a = asset_a
        self.asset_b = asset_b
        self.benchmark = benchmark
        self.start = start
        self.end = end
        self.lookback = lookback
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.stop_loss = stop_loss
        self.slippage = slippage
        self.commission = commission
        self.hedge_method = hedge_method
        self.capital = float(capital)
        self.sizing_method = sizing_method
        self.walk_forward = walk_forward
        self.split_date = split_date
        self.charge_borrow = charge_borrow
        self.borrow_bps = float(borrow_bps)
        self.data = pd.DataFrame()

    def fetch(self) -> PairsTradingEngine:
        tickers = [self.asset_a, self.asset_b]
        if self.benchmark not in tickers:
            tickers.append(self.benchmark)
        self.data = download_price_data(tickers, self.start, self.end)
        return self

    def run_backtest(self) -> dict:
        closes = self.data[[self.asset_a, self.asset_b]].copy()
        benchmark = self.data[self.benchmark].copy()

        log_a = log_price(closes[self.asset_a])
        log_b = log_price(closes[self.asset_b])
        log_benchmark = log_price(benchmark)

        is_mask, split_ts = split_masks(closes.index, self.split_date)
        oos_mask = ~is_mask
        walk_forward_applied = bool(self.walk_forward and is_mask.sum() >= 60 and oos_mask.sum() >= 40)
        discovery_closes = closes.loc[is_mask] if walk_forward_applied else closes
        live_closes = closes.loc[oos_mask] if walk_forward_applied else pd.DataFrame()

        discovery = tether_from_prices(
            discovery_closes[self.asset_a],
            discovery_closes[self.asset_b],
        )
        live_tether = (
            tether_from_prices(live_closes[self.asset_a], live_closes[self.asset_b])
            if walk_forward_applied
            else discovery
        )
        cointegration = check_cointegration(
            discovery_closes[self.asset_a],
            discovery_closes[self.asset_b],
        )

        if self.hedge_method == "Kalman Filter":
            beta, alpha = apply_kalman_filter(log_a, log_b)
        elif self.hedge_method == "Rolling OLS":
            beta, alpha = rolling_ols_hedge_ratio(log_a, log_b, window=self.lookback)
        else:
            frozen_beta = discovery.get("hedge_ratio", cointegration["hedge_ratio"])
            frozen_alpha = discovery.get("intercept", cointegration["intercept"])
            if not np.isfinite(frozen_beta):
                frozen_beta = cointegration["hedge_ratio"]
            if not np.isfinite(frozen_alpha):
                frozen_alpha = cointegration["intercept"]
            beta = pd.Series(np.full(len(log_a), frozen_beta), index=log_a.index)
            alpha = pd.Series(np.full(len(log_a), frozen_alpha), index=log_a.index)

        spread = log_a - (beta * log_b + alpha)
        rolling_mean = spread.rolling(self.lookback).mean()
        rolling_std = spread.rolling(self.lookback).std()
        zscore = (spread - rolling_mean) / rolling_std

        positions = build_signals(zscore, self.entry_z, self.exit_z, self.stop_loss)

        returns_a = log_a.diff().fillna(0.0)
        returns_b = log_b.diff().fillna(0.0)
        returns_benchmark = log_benchmark.diff().fillna(0.0)

        factor_source_a = returns_a.loc[is_mask] if walk_forward_applied else returns_a
        factor_source_b = returns_b.loc[is_mask] if walk_forward_applied else returns_b
        factor_source_m = returns_benchmark.loc[is_mask] if walk_forward_applied else returns_benchmark
        beta_ratio_info = compute_beta_ratio(factor_source_a, factor_source_b, factor_source_m)

        beta_lag = beta.shift(1).ffill().fillna(0.0)
        notional_a, notional_b = dollar_notionals(beta_lag, self.capital, self.sizing_method)
        pos_lag = positions.shift(1).fillna(0.0)

        held_a = pos_lag * notional_a
        held_b = -pos_lag * notional_b
        dollar_pnl = held_a * returns_a + held_b * returns_b

        turnover = held_a.diff().abs().fillna(held_a.abs()) + held_b.diff().abs().fillna(held_b.abs())
        transaction_costs = turnover * (self.commission + self.slippage)
        dollar_pnl = (dollar_pnl - transaction_costs).fillna(0.0)

        short_notional = held_a.clip(upper=0.0).abs() + held_b.clip(upper=0.0).abs()
        borrow_cost = short_notional * (self.borrow_bps / 10_000.0) / 252.0
        if not self.charge_borrow:
            borrow_cost = borrow_cost * 0.0
        dollar_pnl = (dollar_pnl - borrow_cost).fillna(0.0)

        strategy_returns = dollar_pnl / self.capital
        equity = self.capital + dollar_pnl.cumsum()

        net_beta = (held_a * beta_ratio_info["beta_a"] + held_b * beta_ratio_info["beta_b"]) / self.capital

        benchmark_returns = returns_benchmark
        benchmark_cum = np.exp(benchmark_returns.cumsum()) - 1.0

        pair_buy_hold = (returns_a + returns_b) / 2.0
        pair_buy_hold_cum = np.exp(pair_buy_hold.cumsum()) - 1.0

        performance = compute_performance_metrics(strategy_returns, positions)
        is_perf = (
            compute_performance_metrics(strategy_returns.loc[is_mask], positions.loc[is_mask])
            if walk_forward_applied
            else performance
        )
        oos_perf = (
            compute_performance_metrics(strategy_returns.loc[oos_mask], positions.loc[oos_mask])
            if walk_forward_applied
            else {"sharpe_ratio": float("nan"), "total_return": float("nan"), "max_drawdown": float("nan")}
        )

        half_life = discovery.get("half_life", calculate_half_life(spread))
        tether = discovery
        rolling_p = rolling_cointegration_pvalues(closes[self.asset_a], closes[self.asset_b])
        blotter = build_trade_blotter(
            positions,
            zscore,
            dollar_pnl,
            half_life,
            self.stop_loss,
        )
        if not blotter.empty and np.isfinite(half_life) and 5.0 <= half_life <= 60.0:
            reverted_on_time = float((blotter["half_life_mult"] <= 1.5).mean())
        else:
            reverted_on_time = float("nan")
        cointegration_status = "Validated" if cointegration["p_value"] < 0.05 else "Not Validated"
        tether_died = (
            walk_forward_applied
            and tether.get("tether_state") == "TETHERED"
            and live_tether.get("tether_state") != "TETHERED"
        )
        active_net = net_beta[pos_lag != 0]

        return {
            "asset_a": self.asset_a,
            "asset_b": self.asset_b,
            "benchmark": self.benchmark,
            "hedge_method": self.hedge_method,
            "sizing_method": self.sizing_method,
            "data": closes,
            "benchmark_series": benchmark,
            "log_prices": pd.DataFrame(
                {
                    self.asset_a: log_a,
                    self.asset_b: log_b,
                    self.benchmark: log_benchmark,
                }
            ),
            "spread": spread,
            "spread_mean": rolling_mean,
            "zscore": zscore,
            "positions": positions,
            "notional_a": notional_a,
            "notional_b": notional_b,
            "dollar_pnl": dollar_pnl,
            "equity": equity,
            "strategy_returns": strategy_returns,
            "benchmark_returns": benchmark_returns,
            "benchmark_cumulative": benchmark_cum,
            "pair_buy_hold_cumulative": pair_buy_hold_cum,
            "rolling_p_value": rolling_p,
            "net_beta": net_beta,
            "borrow_cost": borrow_cost,
            "split_date": str(split_ts.date()) if walk_forward_applied else None,
            "blotter": blotter,
            "metrics": {
                **performance,
                "capital": self.capital,
                "sizing_method": self.sizing_method,
                "total_pnl": float(dollar_pnl.sum()),
                "ending_equity": float(equity.iloc[-1]) if len(equity) else self.capital,
                "half_life": half_life,
                "beta_ratio": beta_ratio_info["beta_ratio"],
                "benchmark_beta_a": beta_ratio_info["beta_a"],
                "benchmark_beta_b": beta_ratio_info["beta_b"],
                "hedge_ratio": cointegration["hedge_ratio"],
                "cointegration_p_value": cointegration["p_value"],
                "cointegration_status": cointegration_status,
                "adf_p_value": cointegration["adf_p_value"],
                "tether_score": tether["tether_score"],
                "tether_state": tether["tether_state"],
                "tether_reason": tether["tether_reason"],
                "stability_score": tether.get("stability_score", float("nan")),
                "trade_count": int(len(blotter)),
                "reverted_on_time": reverted_on_time,
                "walk_forward": walk_forward_applied,
                "split_date": str(split_ts.date()) if walk_forward_applied else None,
                "is_sharpe": is_perf["sharpe_ratio"],
                "oos_sharpe": oos_perf["sharpe_ratio"],
                "is_return": is_perf["total_return"],
                "oos_return": oos_perf["total_return"],
                "oos_tether_state": live_tether.get("tether_state"),
                "oos_tether_score": live_tether.get("tether_score"),
                "tether_died": bool(tether_died),
                "charge_borrow": self.charge_borrow,
                "borrow_bps": self.borrow_bps,
                "borrow_cost_total": float(borrow_cost.sum()),
                "net_beta_mean": float(active_net.mean()) if len(active_net) else 0.0,
                "net_beta_abs_mean": float(active_net.abs().mean()) if len(active_net) else 0.0,
            },
        }


COST_GRID_SLIPPAGE_BPS = (0.0, 5.0, 15.0)


def cost_sensitivity(
    engine: PairsTradingEngine,
    slippage_bps: tuple[float, ...] = COST_GRID_SLIPPAGE_BPS,
) -> pd.DataFrame:
    """Rerun the loaded book at each slippage level; signals do not depend on costs, only PnL does."""
    rows = []
    for bps in slippage_bps:
        variant = copy.copy(engine)
        variant.slippage = bps / 10_000.0
        metrics = variant.run_backtest()["metrics"]
        rows.append(
            {
                "slippage_bps": float(bps),
                "commission_bps": engine.commission * 10_000.0,
                "total_pnl": metrics["total_pnl"],
                "sharpe_ratio": metrics["sharpe_ratio"],
                "oos_sharpe": metrics["oos_sharpe"],
            }
        )
    return pd.DataFrame(rows)


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
