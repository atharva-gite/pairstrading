import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from datetime import date, timedelta
from src.logic import PairsTradingEngine
import numpy as np

DEFAULT_TICKERS = [
    "AAPL",
    "MSFT",
    "GOOGL",
    "META",
    "NVDA",
    "AMD",
    "TSLA",
    "XOM",
    "JNJ",
    "JPM",
]
BENCHMARKS = ["SPY", "QQQ", "DIA", "IWM", "XLK"]
HEDGE_METHODS = ["Static OLS", "Rolling OLS", "Kalman Filter"]


st.set_page_config(
    page_title="Pairs Trading Statistical Arbitrage Engine",
    layout="wide",
)

st.title("Pairs Trading Statistical Arbitrage Engine")
st.markdown(
    "Use Engle-Granger validation, beta-neutral factor exposure, and dynamic hedge ratio estimation to evaluate a pairs trading strategy."
)

with st.sidebar:
    st.header("Configuration")
    asset_a = st.selectbox("Asset A", DEFAULT_TICKERS, index=0)
    asset_b = st.selectbox("Asset B", DEFAULT_TICKERS, index=1)
    benchmark = st.selectbox("Benchmark", BENCHMARKS, index=0)

    st.subheader("Backtest Parameters")
    start_date = st.date_input("Start Date", date.today() - timedelta(days=365 * 2))
    end_date = st.date_input("End Date", date.today())
    lookback = st.slider("Lookback Period (formation)", min_value=20, max_value=180, value=60, step=10)
    entry_z = st.slider("Entry Z-Score", min_value=1.0, max_value=4.0, value=2.0, step=0.1)
    exit_z = st.slider("Exit Z-Score", min_value=0.1, max_value=2.0, value=0.5, step=0.1)
    stop_loss = st.slider("Stop-Loss Z-Score", min_value=1.0, max_value=5.0, value=3.0, step=0.1)
    hedge_method = st.selectbox("Hedge Ratio Method", HEDGE_METHODS)

    if asset_a == asset_b:
        st.error("Please choose two different assets for the pair.")

run_button = st.sidebar.button("Run Backtest")

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

    metrics = result["metrics"]
    st.subheader("Strategy Summary")
    cols = st.columns(4)
    cols[0].metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}")
    cols[1].metric("Max Drawdown", f"{metrics['max_drawdown']:.2%}")
    cols[2].metric("Win Rate", f"{metrics['win_rate']:.2%}")
    cols[3].metric("Half-Life", f"{metrics['half_life']:.1f} days")

    st.markdown(
        f"**Cointegration p-value:** {metrics['cointegration_p_value']:.4f} — **Status:** {metrics['cointegration_status']}"
    )
    st.markdown(
        f"**Benchmark Beta Ratio**: {metrics['beta_ratio']:.3f} (A vs B)"
    )

    log_prices = result["log_prices"].copy()
    normalized = log_prices.subtract(log_prices.iloc[0]).apply(np.exp)
    normalized = normalized / normalized.iloc[0]

    fig_price = go.Figure()
    fig_price.add_trace(go.Scatter(x=normalized.index, y=normalized[asset_a], name=asset_a))
    fig_price.add_trace(go.Scatter(x=normalized.index, y=normalized[asset_b], name=asset_b))
    fig_price.update_layout(
        title="Normalized Price Comparison (Log Scale)",
        yaxis_title="Normalized Price",
        yaxis_type="log",
        template="plotly_white",
    )

    zscore = result["zscore"].rename("Z-Score")
    fig_z = go.Figure()
    fig_z.add_trace(go.Scatter(x=zscore.index, y=zscore, name="Rolling Z-Score", line=dict(color="#0066CC")))
    fig_z.add_hline(y=entry_z, line_dash="dash", line_color="red")
    fig_z.add_hline(y=-entry_z, line_dash="dash", line_color="green")
    fig_z.add_hline(y=exit_z, line_dash="dot", line_color="gray")
    fig_z.add_hline(y=-exit_z, line_dash="dot", line_color="gray")
    fig_z.add_hline(y=stop_loss, line_dash="dashdot", line_color="darkred")
    fig_z.add_hline(y=-stop_loss, line_dash="dashdot", line_color="darkgreen")
    fig_z.update_layout(
        title="Rolling Z-Score with Trade Bands",
        yaxis_title="Z-Score",
        template="plotly_white",
    )

    fig_returns = go.Figure()
    fig_returns.add_trace(
        go.Scatter(
            x=result["strategy_returns"].index,
            y=(result["strategy_returns"].cumsum().apply(np.exp) - 1.0),
            name="Strategy Returns",
            line=dict(color="#1F77B4"),
        )
    )
    fig_returns.add_trace(
        go.Scatter(
            x=result["benchmark_cumulative"].index,
            y=result["benchmark_cumulative"],
            name=f"{benchmark} Buy & Hold",
            line=dict(color="#FF7F0E"),
        )
    )
    fig_returns.update_layout(
        title="Cumulative Strategy Returns vs. Benchmark",
        yaxis_title="Cumulative Return",
        template="plotly_white",
    )

    st.plotly_chart(fig_price, use_container_width=True)
    st.plotly_chart(fig_z, use_container_width=True)
    st.plotly_chart(fig_returns, use_container_width=True)

    with st.expander("View raw result data"):
        st.dataframe(
            pd.DataFrame(
                {
                    "Spread": result["spread"],
                    "Z-Score": result["zscore"],
                    "Position": result["positions"],
                    "Strategy Returns": result["strategy_returns"],
                }
            )
        )
else:
    st.info("Adjust the sidebar parameters and click Run Backtest to generate the dashboard.")
