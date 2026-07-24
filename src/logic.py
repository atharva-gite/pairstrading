import numpy as np
import pandas as pd
import yfinance as yf
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, coint


def download_price_data(tickers, start, end):
    closes = yf.download(tickers, start=start, end=end, progress=False)["Close"]
    if isinstance(closes, pd.Series):
        closes = closes.to_frame(name=tickers)
    return closes.dropna()


def log_price(series: pd.Series) -> pd.Series:
    return np.log(series)


def check_cointegration(series_a: pd.Series, series_b: pd.Series) -> dict:
    a = log_price(series_a).dropna()
    b = log_price(series_b).reindex(a.index).dropna()
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

    obs_mat = np.vstack([x.values, np.ones(n)]).T[:, :, None]

    for t in range(n):
        obs_val = y.values[t]
        obs_matrix = obs_mat[t]

        innovation = obs_val - np.dot(obs_matrix, current_state_mean)
        innovation_cov = np.dot(obs_matrix, np.dot(current_state_cov, obs_matrix.T)) + obs_cov
        kalman_gain = np.dot(current_state_cov, obs_matrix.T) / innovation_cov

        current_state_mean = current_state_mean + (kalman_gain * innovation).flatten()
        current_state_cov = current_state_cov - np.dot(kalman_gain, np.dot(obs_matrix, current_state_cov))
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
        sm.OLS(combined_a.iloc[:, 0], sm.add_constant(combined_a.iloc[:, 1])).fit().params[1]
    )
    beta_b = float(
        sm.OLS(combined_b.iloc[:, 0], sm.add_constant(combined_b.iloc[:, 1])).fit().params[1]
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

        cointegration = check_cointegration(log_a, log_b)

        if self.hedge_method == "Kalman Filter":
            beta, alpha = apply_kalman_filter(log_a, log_b)
        elif self.hedge_method == "Rolling OLS":
            beta, alpha = rolling_ols_hedge_ratio(log_a, log_b, window=self.lookback)
        else:
            beta = pd.Series(np.full(len(log_a), cointegration["hedge_ratio"]), index=log_a.index)
            alpha = pd.Series(np.full(len(log_a), cointegration["intercept"]), index=log_a.index)

        spread = log_a - (beta * log_b + alpha)
        zscore = (spread - spread.rolling(self.lookback).mean()) / spread.rolling(self.lookback).std()
        zscore = zscore.fillna(0.0)

        positions = build_signals(zscore, self.entry_z, self.exit_z, self.stop_loss)
        trades = positions.diff().abs().fillna(0.0)
        transaction_costs = trades * (self.commission + self.slippage)

        returns_a = log_a.diff().fillna(0.0)
        returns_b = log_b.diff().fillna(0.0)
        returns_benchmark = log_benchmark.diff().fillna(0.0)

        beta_prev = beta.shift(1).fillna(method="ffill").fillna(0.0)
        base_returns = returns_a - beta_prev * returns_b
        strategy_returns = positions.shift(1).fillna(0.0) * base_returns - transaction_costs
        strategy_returns = strategy_returns.fillna(0.0)

        benchmark_returns = returns_benchmark
        benchmark_cum = np.exp(benchmark_returns.cumsum()) - 1.0

        pair_buy_hold = (returns_a + returns_b) / 2.0
        pair_buy_hold_cum = np.exp(pair_buy_hold.cumsum()) - 1.0

        beta_ratio_info = compute_beta_ratio(returns_a, returns_b, benchmark_returns)
        performance = compute_performance_metrics(strategy_returns, positions)
        half_life = calculate_half_life(spread)

        cointegration = check_cointegration(closes[self.asset_a], closes[self.asset_b])
        cointegration_status = "Validated" if cointegration["p_value"] < 0.05 else "Not Validated"

        return {
            "asset_a": self.asset_a,
            "asset_b": self.asset_b,
            "benchmark": self.benchmark,
            "hedge_method": self.hedge_method,
            "data": closes,
            "benchmark_series": benchmark,
            "log_prices": pd.DataFrame({
                self.asset_a: log_a,
                self.asset_b: log_b,
                self.benchmark: log_benchmark,
            }),
            "spread": spread,
            "zscore": zscore,
            "positions": positions,
            "strategy_returns": strategy_returns,
            "benchmark_returns": benchmark_returns,
            "benchmark_cumulative": benchmark_cum,
            "pair_buy_hold_cumulative": pair_buy_hold_cum,
            "metrics": {
                **performance,
                "half_life": half_life,
                "beta_ratio": beta_ratio_info["beta_ratio"],
                "benchmark_beta_a": beta_ratio_info["beta_a"],
                "benchmark_beta_b": beta_ratio_info["beta_b"],
                "cointegration_p_value": cointegration["p_value"],
                "cointegration_status": cointegration_status,
            },
        }
