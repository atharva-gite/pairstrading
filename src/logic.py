from itertools import combinations

import numpy as np
import pandas as pd
import yfinance as yf
import statsmodels.api as sm
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
    "cointegrated",
]


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

    beta_a = float(
        sm.OLS(combined_a.iloc[:, 0], sm.add_constant(combined_a.iloc[:, 1])).fit().params.iloc[-1]
    )
    beta_b = float(
        sm.OLS(combined_b.iloc[:, 0], sm.add_constant(combined_b.iloc[:, 1])).fit().params.iloc[-1]
    )
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


def screen_sector_pairs(
    prices: pd.DataFrame,
    p_value_threshold: float = 0.05,
) -> pd.DataFrame:
    """Rank every unique pair in a price panel by Engle-Granger p-value."""
    tickers = [col for col in prices.columns if prices[col].notna().sum() > 30]
    rows = []
    for asset_a, asset_b in combinations(tickers, 2):
        try:
            result = check_cointegration(prices[asset_a], prices[asset_b])
            half_life = calculate_half_life(result["spread"])
            rows.append(
                {
                    "asset_a": asset_a,
                    "asset_b": asset_b,
                    "pair": f"{asset_a}/{asset_b}",
                    "p_value": result["p_value"],
                    "adf_p_value": result["adf_p_value"],
                    "hedge_ratio": result["hedge_ratio"],
                    "half_life": half_life,
                    "cointegrated": bool(result["p_value"] < p_value_threshold),
                }
            )
        except (ValueError, np.linalg.LinAlgError, KeyError):
            continue

    if not rows:
        return pd.DataFrame(columns=SCREEN_COLUMNS)

    ranked = pd.DataFrame(rows)
    return ranked.sort_values(
        ["cointegrated", "p_value", "half_life"],
        ascending=[False, True, True],
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
        self.data = pd.DataFrame()

    def fetch(self) -> "PairsTradingEngine":
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

        cointegration = check_cointegration(closes[self.asset_a], closes[self.asset_b])

        if self.hedge_method == "Kalman Filter":
            beta, alpha = apply_kalman_filter(log_a, log_b)
        elif self.hedge_method == "Rolling OLS":
            beta, alpha = rolling_ols_hedge_ratio(log_a, log_b, window=self.lookback)
        else:
            beta = pd.Series(np.full(len(log_a), cointegration["hedge_ratio"]), index=log_a.index)
            alpha = pd.Series(np.full(len(log_a), cointegration["intercept"]), index=log_a.index)

        spread = log_a - (beta * log_b + alpha)
        rolling_mean = spread.rolling(self.lookback).mean()
        rolling_std = spread.rolling(self.lookback).std()
        zscore = (spread - rolling_mean) / rolling_std

        positions = build_signals(zscore, self.entry_z, self.exit_z, self.stop_loss)

        returns_a = log_a.diff().fillna(0.0)
        returns_b = log_b.diff().fillna(0.0)
        returns_benchmark = log_benchmark.diff().fillna(0.0)

        beta_lag = beta.shift(1).ffill().fillna(0.0)
        notional_a, notional_b = dollar_notionals(beta_lag, self.capital, self.sizing_method)
        pos_lag = positions.shift(1).fillna(0.0)

        held_a = pos_lag * notional_a
        held_b = -pos_lag * notional_b
        dollar_pnl = held_a * returns_a + held_b * returns_b

        turnover = held_a.diff().abs().fillna(held_a.abs()) + held_b.diff().abs().fillna(held_b.abs())
        transaction_costs = turnover * (self.commission + self.slippage)
        dollar_pnl = (dollar_pnl - transaction_costs).fillna(0.0)

        strategy_returns = dollar_pnl / self.capital
        equity = self.capital + dollar_pnl.cumsum()

        benchmark_returns = returns_benchmark
        benchmark_cum = np.exp(benchmark_returns.cumsum()) - 1.0

        pair_buy_hold = (returns_a + returns_b) / 2.0
        pair_buy_hold_cum = np.exp(pair_buy_hold.cumsum()) - 1.0

        beta_ratio_info = compute_beta_ratio(returns_a, returns_b, benchmark_returns)
        performance = compute_performance_metrics(strategy_returns, positions)
        half_life = calculate_half_life(spread)
        cointegration_status = "Validated" if cointegration["p_value"] < 0.05 else "Not Validated"

        return {
            "asset_a": self.asset_a,
            "asset_b": self.asset_b,
            "benchmark": self.benchmark,
            "hedge_method": self.hedge_method,
            "sizing_method": self.sizing_method,
            "data": closes,
            "benchmark_series": benchmark,
            "log_prices": pd.DataFrame({
                self.asset_a: log_a,
                self.asset_b: log_b,
                self.benchmark: log_benchmark,
            }),
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
            },
        }
