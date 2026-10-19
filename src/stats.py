"""Cointegration, half-life, tether scoring, FDR screening."""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
import statsmodels.api as sm
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


MAX_TRADABLE_HALF_LIFE = 60.0


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
    elif half_life > MAX_TRADABLE_HALF_LIFE:
        reason = "too_slow"
    elif half_life < 5:
        reason = "too_fast"
    elif score >= 70 and p_value < 0.05:
        reason = "tradable"
    else:
        reason = "weak_evidence"

    too_slow = half_life is not None and np.isfinite(half_life) and half_life > MAX_TRADABLE_HALF_LIFE
    if score >= 70 and p_value < 0.05 and not too_slow:
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
