import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from datetime import date, timedelta
from src.logic import PairsTradingEngine
import numpy as np

TICKER_NAMES = {
    "AAPL": "Apple",
    "MSFT": "Microsoft",
    "GOOGL": "Alphabet",
    "META": "Meta Platforms",
    "NVDA": "NVIDIA",
    "AMD": "Advanced Micro Devices",
    "TSLA": "Tesla",
    "XOM": "Exxon Mobil",
    "JNJ": "Johnson & Johnson",
    "JPM": "JPMorgan Chase",
}
DEFAULT_TICKERS = list(TICKER_NAMES.keys())
BENCHMARKS = ["SPY", "QQQ", "DIA", "IWM", "XLK"]
HEDGE_METHODS = ["Static OLS", "Rolling OLS", "Kalman Filter"]


def build_price_chart(log_prices: pd.DataFrame, asset_a: str, asset_b: str) -> go.Figure:
    normalized = np.exp(log_prices.subtract(log_prices.iloc[0]))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=normalized.index, y=normalized[asset_a], name=asset_a, line=dict(width=2)))
    fig.add_trace(go.Scatter(x=normalized.index, y=normalized[asset_b], name=asset_b, line=dict(width=2)))
    fig.update_layout(
        title="Normalized Price Comparison (Log Scale)",
        yaxis_title="Normalized Price",
        yaxis_type="log",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template="plotly_white",
    )
    return fig


def build_zscore_chart(zscore: pd.Series, entry_z: float, exit_z: float, stop_loss: float) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=zscore.index, y=zscore, name="Rolling Z-Score", line=dict(color="#0055AA", width=2)))
    fig.add_hline(y=entry_z, line_dash="dash", line_color="red", annotation_text="Entry", annotation_position="top left")
    fig.add_hline(y=-entry_z, line_dash="dash", line_color="green", annotation_text="Entry", annotation_position="bottom left")
    fig.add_hline(y=exit_z, line_dash="dot", line_color="gray", annotation_text="Exit", annotation_position="top left")
    fig.add_hline(y=-exit_z, line_dash="dot", line_color="gray", annotation_text="Exit", annotation_position="bottom left")
    fig.add_hline(y=stop_loss, line_dash="dashdot", line_color="darkred", annotation_text="Stop Loss", annotation_position="top left")
    fig.add_hline(y=-stop_loss, line_dash="dashdot", line_color="darkgreen", annotation_text="Stop Loss", annotation_position="bottom left")
    fig.update_layout(
        title="Rolling Z-Score with Trade Bands",
        yaxis_title="Z-Score",
        template="plotly_white",
    )
    return fig


def build_returns_chart(strategy_returns: pd.Series, benchmark_cumulative: pd.Series, benchmark_name: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=strategy_returns.index, y=np.exp(strategy_returns.cumsum()) - 1.0, name="Strategy", line=dict(color="#1F77B4")))
    fig.add_trace(go.Scatter(x=benchmark_cumulative.index, y=benchmark_cumulative, name=f"{benchmark_name} Buy & Hold", line=dict(color="#FF7F0E")))
    fig.update_layout(
        title="Cumulative Strategy Returns vs Benchmark",
        yaxis_title="Cumulative Return",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template="plotly_white",
    )
    return fig


st.set_page_config(page_title="Pairs Trading Statistical Arbitrage Engine", layout="wide")

st.title("Pairs Trading Statistical Arbitrage Engine")
st.markdown(
    "This professional fintech dashboard combines Engle-Granger cointegration validation, factor beta neutrality, and dynamic hedge ratio estimation for pairs trading analysis."
)

with st.sidebar:
    st.header("Model Controls")
    st.markdown("Configure the trading pair, benchmark, and risk parameters. Hover the info icons to see how each input affects the model.")

    asset_choices = [f"{ticker} ({TICKER_NAMES[ticker]})" for ticker in DEFAULT_TICKERS]
    asset_a_label = st.selectbox(
        "Asset A",
        asset_choices,
        index=0,
        help="Select the primary asset for the trading pair.",
    )
    asset_b_label = st.selectbox(
        "Asset B",
        asset_choices,
        index=1,
        help="Select the secondary asset for the trading pair.",
    )
    asset_a = asset_a_label.split(" ")[0]
    asset_b = asset_b_label.split(" ")[0]
    benchmark = st.selectbox(
        "Benchmark",
        BENCHMARKS,
        index=0,
        help="Choose a market benchmark to compute factor beta exposure and compare performance.",
    )

    st.markdown("---")
    st.subheader("Backtest Parameters")
    start_date = st.date_input(
        "Start Date",
        date.today() - timedelta(days=365 * 2),
        help="The earliest date included in the historical analysis.",
    )
    end_date = st.date_input(
        "End Date",
        date.today(),
        help="The final date included in the historical analysis.",
    )
    lookback = st.slider(
        "Lookback Period (formation)",
        min_value=20,
        max_value=180,
        value=60,
        step=10,
        help="Window length used to compute the rolling spread mean and volatility.",
    )
    entry_z = st.slider(
        "Entry Z-Score",
        min_value=1.0,
        max_value=4.0,
        value=2.0,
        step=0.1,
        help="Opening threshold for mean-reversion entry signals.",
    )
    exit_z = st.slider(
        "Exit Z-Score",
        min_value=0.1,
        max_value=2.0,
        value=0.5,
        step=0.1,
        help="Closing threshold for neutralizing the position.",
    )
    stop_loss = st.slider(
        "Stop-Loss Z-Score",
        min_value=1.0,
        max_value=5.0,
        value=3.0,
        step=0.1,
        help="The z-score threshold to exit a trade and limit losses.",
    )
    hedge_method = st.selectbox(
        "Hedge Ratio Method",
        HEDGE_METHODS,
        help="Choose static OLS, rolling regression, or Kalman filter for estimating the hedge ratio.",
    )

    st.markdown("---")
    st.write("**Execution assumptions**")
    st.caption("Slippage = 0.1%, commission = 0.05% per side, log-price returns.")
    run_button = st.button("Run Backtest")

if asset_a == asset_b:
    st.sidebar.error("Asset A and Asset B must be different.")

if run_button and asset_a != asset_b and start_date < end_date:
    with st.spinner("Downloading data and running the quant engine..."):
        engine = PairsTradingEngine(
            asset_a=asset_a,
            asset_b=asset_b,
            benchmark=benchmark,
            start=start_date.isoformat(),
            end=end_date.isoformat(),
            lookback=lookback,
            entry_z=entry_z,
            exit_z=exit_z,
            stop_loss=stop_loss,
            slippage=0.001,
            commission=0.0005,
            hedge_method=hedge_method,
        )
        result = engine.fetch().run_backtest()

    if result["data"].empty:
        st.error("No historical data was loaded for the selected assets and date range.")
    else:
        metrics = result["metrics"]
        st.subheader("Key Performance Metrics")
        metric_cols = st.columns(4)
        metric_cols[0].metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}")
        metric_cols[1].metric("Max Drawdown", f"{metrics['max_drawdown']:.2%}")
        metric_cols[2].metric("Win Rate", f"{metrics['win_rate']:.2%}")
        metric_cols[3].metric("Half-Life", f"{metrics['half_life']:.1f} days")

        status_cols = st.columns(3)
        status_cols[0].metric("Cointegration", metrics["cointegration_status"], f"p = {metrics['cointegration_p_value']:.4f}")
        status_cols[1].metric("Beta Ratio", f"{metrics['beta_ratio']:.3f}", f"A: {metrics['benchmark_beta_a']:.3f}, B: {metrics['benchmark_beta_b']:.3f}")
        status_cols[2].metric("Hedge Method", hedge_method)

        st.markdown("---")
        st.subheader("Analytics Dashboard")

        overview_tab, details_tab, raw_tab = st.tabs(["Overview", "Diagnostics", "Raw Data"])

        with overview_tab:
            chart_cols = st.columns([2, 1])
            with chart_cols[0]:
                st.plotly_chart(build_price_chart(result["log_prices"], asset_a, asset_b), use_container_width=True)
                st.plotly_chart(build_returns_chart(result["strategy_returns"], result["benchmark_cumulative"], benchmark), use_container_width=True)
            with chart_cols[1]:
                st.plotly_chart(build_zscore_chart(result["zscore"], entry_z, exit_z, stop_loss), use_container_width=True)
                st.markdown("#### Strategy Summary")
                st.write(f"**Pair:** {asset_a} / {asset_b}")
                st.write(f"**Benchmark:** {benchmark}")
                st.write(f"**Lookback:** {lookback} days")
                st.write(f"**Positioning:** Entry ±{entry_z}, Exit ±{exit_z}, Stop-Loss ±{stop_loss}")
                st.write(f"**Sample Period:** {start_date.isoformat()} to {end_date.isoformat()}")

        with details_tab:
            st.markdown("#### Cointegration & Stationarity")
            st.write(f"- Engle-Granger p-value: **{metrics['cointegration_p_value']:.4f}**")
            st.write(f"- Cointegration status: **{metrics['cointegration_status']}**")
            st.write(f"- Beta A vs benchmark: **{metrics['benchmark_beta_a']:.3f}**")
            st.write(f"- Beta B vs benchmark: **{metrics['benchmark_beta_b']:.3f}**")
            st.write(f"- Beta ratio (A/B): **{metrics['beta_ratio']:.3f}**")
            st.markdown("#### Hedge Ratio Method")
            st.write(f"Selected: **{hedge_method}**")

        with raw_tab:
            st.write("### Raw Data")
            st.dataframe(result["data"].tail(20))
            st.write("### Strategy Signals")
            st.dataframe(
                pd.DataFrame(
                    {
                        "Spread": result["spread"],
                        "Z-Score": result["zscore"],
                        "Position": result["positions"],
                        "Strategy Returns": result["strategy_returns"],
                    }
                ).tail(20)
            )
else:
    st.info("Set the pair and backtest parameters in the sidebar, then click Run Backtest.")
