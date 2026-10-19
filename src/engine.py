"""Walk-forward pairs backtest and cost sensitivity."""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import yfinance as yf

from src.book import (
    DEFAULT_MAX_LEG_PCT,
    DESK_GROSS_LIMIT,
    build_signals,
    build_trade_blotter,
    compute_performance_metrics,
    dollar_notionals,
)
from src.stats import (
    apply_kalman_filter,
    calculate_half_life,
    check_cointegration,
    compute_beta_ratio,
    log_price,
    rolling_cointegration_pvalues,
    rolling_ols_hedge_ratio,
    split_masks,
    tether_from_prices,
)


def download_price_data(tickers, start, end):
    closes = yf.download(tickers, start=start, end=end, progress=False)["Close"]
    if isinstance(closes, pd.Series):
        closes = closes.to_frame(name=tickers)
    return closes.dropna()


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
        max_leg_pct: float = DEFAULT_MAX_LEG_PCT,
        gross_limit: float = DESK_GROSS_LIMIT,
    ):
        if not 0.0 < float(capital) <= float(gross_limit):
            raise ValueError(f"Gross capital ${capital:,.0f} is outside the desk limit (0, ${gross_limit:,.0f}].")
        if not 0.5 <= float(max_leg_pct) <= 1.0:
            raise ValueError(f"max_leg_pct must be between 0.5 and 1.0, got {max_leg_pct}.")
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
        self.max_leg_pct = float(max_leg_pct)
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
        max_leg = self.max_leg_pct * self.capital
        notional_a, notional_b = dollar_notionals(beta_lag, self.capital, self.sizing_method, max_leg_notional=max_leg)
        uncapped_a, uncapped_b = dollar_notionals(beta_lag, self.capital, self.sizing_method)
        leg_cap_binding = (np.maximum(uncapped_a.abs(), uncapped_b.abs()) > max_leg + 1e-9) & (
            positions.shift(1).fillna(0.0) != 0
        )
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
                "max_leg_pct": self.max_leg_pct,
                "leg_cap_binding_days": int(leg_cap_binding.sum()),
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
