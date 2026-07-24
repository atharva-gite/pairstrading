import os
import json
from datetime import date, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

from src.logic import PairsTradingEngine

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

HEDGE_METHOD_DESCRIPTIONS = {
    "Static OLS": "Uses a fixed historical hedge ratio for the sample period, making execution simpler for stable cointegrated pairs.",
    "Rolling OLS": "Recomputes the hedge ratio over each lookback window to adapt to changing pair dynamics and regime shifts.",
    "Kalman Filter": "Estimates the hedge ratio dynamically with state-space filtering, providing smoother adaptation to market noise.",
}

BENCHMARK_DESCRIPTIONS = {
    "SPY": "S&P 500 exposure shows how the pair behaves relative to the broad market, helping separate stock-specific alpha from factor risk.",
    "QQQ": "Tech-heavy benchmark exposure highlights whether the pair is driven by growth sector swings rather than pair-specific mean reversion.",
    "DIA": "Dow exposure helps assess whether the pair is sensitive to large-cap industrial and financial market moves.",
    "IWM": "Small-cap benchmark exposure measures whether the pair's risk profile is aligned with more volatile, smaller-growth stocks.",
    "XLK": "Technology sector exposure shows how much the pair is influenced by the tech factor versus pure cross-sectional relative value.",
}


def build_contextual_system_prompt(metrics: dict | None) -> str:
    default_metrics = {
        "cointegration_p_value": None,
        "sharpe_ratio": None,
        "half_life": None,
        "beta_ratio": None,
    }
    if metrics:
        default_metrics.update(metrics)

    p_value = default_metrics.get("cointegration_p_value")
    sharpe = default_metrics.get("sharpe_ratio")
    half_life = default_metrics.get("half_life")
    beta_ratio = default_metrics.get("beta_ratio")

    p_value_text = "not available" if p_value is None else f"{p_value:.4f}"
    sharpe_text = "not available" if sharpe is None else f"{sharpe:.2f}"
    half_life_text = "not available" if half_life is None else f"{half_life:.1f}"
    beta_text = "not available" if beta_ratio is None else f"{beta_ratio:.3f}"

    return (
        "You are an AI Quantitative Assistant for a pairs trading strategy. "
        "You act like a calm Quant Mentor who explains results plainly and with practical nuance. "
        "Focus on statistical interpretation, risk, and decision quality. "
        f"Latest backtest metrics: cointegration p-value={p_value_text}, Sharpe ratio={sharpe_text}, half-life={half_life_text}, beta ratio={beta_text}. "
        "If the p-value is high, explain that it may signal weaker mean reversion, a regime shift, or an asymmetric risk event. "
        "If the Sharpe is low, explain that the strategy may be under-earning after costs. "
        "Use quantitative analogies such as the 'Drunken Man and his Dog' to explain cointegration in simple language. "
        "Keep the tone expert, educational, and concise."
    )


def get_ai_reply(user_message: str, history: list[dict], system_prompt: str) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return fallback_ai_reply(user_message, system_prompt)

    try:
        payload = {
            "model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            "messages": [{"role": "system", "content": system_prompt}] + history + [{"role": "user", "content": user_message}],
            "temperature": 0.35,
        }
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            data=json.dumps(payload),
            timeout=30,
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"].strip()
        return content or fallback_ai_reply(user_message, system_prompt)
    except Exception:
        return fallback_ai_reply(user_message, system_prompt)


def fallback_ai_reply(user_message: str, system_prompt: str) -> str:
    prompt = user_message.lower()
    if "cointegration" in prompt or "co-integr" in prompt:
        return (
            "Cointegration is like the 'Drunken Man and his Dog' analogy: each one wanders randomly, but their relationship stays stable over time. "
            "In pairs trading, that means the spread can drift and revert even when each stock moves independently."
        )
    if "p-value" in prompt or "p value" in prompt:
        return (
            "A high p-value usually suggests the spread may not be strongly stationary or mean-reverting. "
            "That can happen after a regime shift, a liquidity shock, or an asymmetric risk event that breaks the old relationship."
        )
    if "sharpe" in prompt:
        return "A lower Sharpe ratio usually means the strategy is not generating enough excess return per unit of risk after costs and slippage."
    if "half-life" in prompt:
        return "A shorter half-life often means the spread reverts more quickly, which can be attractive for short-horizon mean reversion strategies."
    if "beta" in prompt:
        return "Beta ratio helps you understand how much of the spread exposure is tied to the benchmark versus the pair-specific relationship."
    if "correlation" in prompt and "cointegration" in prompt:
        return "Correlation measures co-movement, while cointegration measures a stable long-run relationship. Two assets can be correlated without being cointegrated."
    return (
        f"I’m using your latest backtest context to guide the answer. {system_prompt.split('Latest backtest metrics:')[1].split('. ')[0] if 'Latest backtest metrics:' in system_prompt else ''} "
        "Ask me about cointegration, p-values, Sharpe, beta, or risk controls and I’ll explain it in a quantitative way."
    )


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

st.markdown("# Pairs Trading Statistical Arbitrage Engine")
st.markdown(
    "A professional fintech tool for Engle-Granger cointegration analysis, factor beta-neutrality assessment, and dynamic hedge ratio execution."
)

with st.sidebar:
    st.header("Engine Configuration")
    st.markdown(
        "Configure the pair, benchmark, risk thresholds, and hedge ratio method. Hover over each control for guidance."
    )

    with st.expander("Common Quant Questions", expanded=True):
        st.markdown(
            "- What is the difference between correlation and cointegration?\n"
            "- Why does a high p-value matter for mean reversion?\n"
            "- How should I interpret a low Sharpe ratio?\n"
            "- What does a short half-life imply for spread reversion?"
        )
        st.caption("These answers are surfaced in the AI Quant Assistant tab below using the latest backtest context.")

    asset_labels = [f"{ticker} ({TICKER_NAMES[ticker]})" for ticker in DEFAULT_TICKERS]
    asset_a_label = st.selectbox(
        "Asset A",
        asset_labels,
        index=0,
        key="asset_a",
        help="Select the first asset in the pair. The app will use the ticker symbol for quant calculations.",
    )
    asset_b_label = st.selectbox(
        "Asset B",
        asset_labels,
        index=1,
        key="asset_b",
        help="Select the second asset in the pair. Choose a correlated instrument for mean-reversion analysis.",
    )
    asset_a = asset_a_label.split(" ")[0]
    asset_b = asset_b_label.split(" ")[0]

    benchmark = st.selectbox(
        "Benchmark",
        BENCHMARKS,
        index=0,
        key="benchmark",
        help="Select a market benchmark used for factor beta ratio estimation.",
    )
    st.caption(BENCHMARK_DESCRIPTIONS.get(benchmark, ""))

    st.markdown("---")
    st.subheader("Risk & Signal Parameters")
    start_date = st.date_input(
        "Start Date",
        date.today() - timedelta(days=365 * 2),
        key="start_date",
        help="Choose the start of the historical sample used for the backtest.",
    )
    end_date = st.date_input(
        "End Date",
        date.today(),
        key="end_date",
        help="Choose the end of the historical sample used for the backtest.",
    )
    lookback = st.slider(
        "Lookback Period (days)",
        min_value=20,
        max_value=180,
        value=60,
        step=10,
        key="lookback",
        help="Size of the rolling window used to compute spread mean, volatility, and rolling hedge ratios.",
    )
    entry_z = st.slider(
        "Entry Z-Score",
        min_value=1.0,
        max_value=4.0,
        value=2.0,
        step=0.1,
        key="entry_z",
        help="The threshold at which the pair trade is entered when the spread becomes extended.",
    )
    exit_z = st.slider(
        "Exit Z-Score",
        min_value=0.1,
        max_value=2.0,
        value=0.5,
        step=0.1,
        key="exit_z",
        help="The threshold at which the position is closed as the spread reverts.",
    )
    stop_loss = st.slider(
        "Stop-Loss Z-Score",
        min_value=1.0,
        max_value=5.0,
        value=3.0,
        step=0.1,
        key="stop_loss",
        help="Hard stop for z-score deviations to limit downside risk.",
    )
    hedge_method = st.selectbox(
        "Hedge Ratio Method",
        HEDGE_METHODS,
        key="hedge_method",
        help="Choose static OLS, rolling regression, or Kalman filtering for hedge ratio estimation.",
    )
    st.caption(HEDGE_METHOD_DESCRIPTIONS.get(hedge_method, ""))

    st.markdown("---")
    st.write("**Execution assumptions**")
    st.caption("Slippage = 0.1%, commission = 0.05% per side, based on log-price returns.")
    run_button = st.button("Run Backtest")

if "ai_messages" not in st.session_state:
    st.session_state.ai_messages = []

if asset_a == asset_b:
    st.sidebar.error("Asset A and Asset B must be different.")

result = None
metrics = None
show_results = False

if run_button and asset_a != asset_b and start_date < end_date:
    with st.spinner("Running the quant engine..."):
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

    st.session_state["latest_backtest_metrics"] = result["metrics"]
    st.session_state["assistant_system_prompt"] = build_contextual_system_prompt(result["metrics"])

    if result["data"].empty:
        st.error("No historical data loaded for the selected symbols and date range.")
    else:
        metrics = result["metrics"]
        show_results = True

st.markdown("---")
main_tabs = st.tabs(["Strategy Setup", "Performance", "Diagnostics", "Charts", "AI Quant Assistant"])

with main_tabs[0]:
    st.markdown("### Strategy Setup")
    if show_results:
        st.success("Backtest completed. Review the insights in the performance, diagnostics, charts, and assistant tabs.")
    else:
        st.markdown(
            "This tool is designed to feel engineered, not just plotted. Configure the pair, benchmark, and risk profile, then run the backtest to see a professional-grade results interface."
        )
        st.markdown(
            "- Engle-Granger two-step cointegration testing\n- Beta-neutral factor exposure analysis\n- Static OLS, Rolling OLS, and Kalman Filter hedge ratio methods\n- Tabbed results for overview, diagnostics, and raw outputs"
        )
        hero_cols = st.columns([2, 1])
        with hero_cols[0]:
            st.info("Fill in the sidebar and click Run Backtest to execute the quant engine.")
        with hero_cols[1]:
            st.metric("Selected Pair", f"{asset_a} / {asset_b}")
            st.metric("Benchmark", benchmark)
            st.metric("Hedge Method", hedge_method)
            st.metric("Lookback", f"{lookback} days")

with main_tabs[1]:
    if show_results:
        st.markdown("## Performance Summary")
        perf_cols = st.columns([1, 1, 1, 1])
        perf_cols[0].metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}")
        perf_cols[1].metric("Max Drawdown", f"{metrics['max_drawdown']:.2%}")
        perf_cols[2].metric("Win Rate", f"{metrics['win_rate']:.2%}")
        perf_cols[3].metric("Half-Life", f"{metrics['half_life']:.1f} days")
    else:
        st.info("Run a backtest to see performance metrics.")

with main_tabs[2]:
    if show_results:
        st.markdown("## Model Diagnostics")
        diag_cols = st.columns([1, 1, 1])
        diag_cols[0].metric("Cointegration", metrics["cointegration_status"], f"p = {metrics['cointegration_p_value']:.4f}")
        diag_cols[1].metric("Beta Ratio", f"{metrics['beta_ratio']:.3f}", f"A: {metrics['benchmark_beta_a']:.3f} / B: {metrics['benchmark_beta_b']:.3f}")
        diag_cols[2].metric("Hedge Method", hedge_method)
        st.markdown("### Statistical Details")
        st.write("- Engle-Granger cointegration validates long-term equilibrium between the log-price series.")
        st.write("- Beta-neutrality compares each asset's market exposure to the benchmark.")
        st.write("- A low beta ratio indicates closer factor neutrality between the pair and benchmark.")
        st.write(f"- Cointegration p-value: **{metrics['cointegration_p_value']:.4f}**")
        st.write(f"- Cointegration status: **{metrics['cointegration_status']}**")
        st.write(f"- Beta A vs benchmark: **{metrics['benchmark_beta_a']:.3f}**")
        st.write(f"- Beta B vs benchmark: **{metrics['benchmark_beta_b']:.3f}**")
        st.write(f"- Factor beta ratio: **{metrics['beta_ratio']:.3f}**")
        st.write(f"- Mean reversion half-life: **{metrics['half_life']:.1f} days**")
    else:
        st.info("Run a backtest to inspect the model diagnostics.")

with main_tabs[3]:
    if show_results:
        st.markdown("### Overview Chart")
        overview_cols = st.columns([2, 1])
        with overview_cols[0]:
            st.plotly_chart(build_price_chart(result["log_prices"], asset_a, asset_b), use_container_width=True)
            st.plotly_chart(build_returns_chart(result["strategy_returns"], result["benchmark_cumulative"], benchmark), use_container_width=True)
        with overview_cols[1]:
            st.plotly_chart(build_zscore_chart(result["zscore"], entry_z, exit_z, stop_loss), use_container_width=True)
            st.markdown("#### Strategy Configuration")
            st.write(f"**Pair:** {asset_a} / {asset_b}")
            st.write(f"**Benchmark:** {benchmark}")
            st.write(f"**Benchmark Significance:** {BENCHMARK_DESCRIPTIONS.get(benchmark, '')}")
            st.write(f"**Lookback:** {lookback} days")
            st.write(f"**Entry:** {entry_z} | Exit: {exit_z} | Stop-Loss: {stop_loss}")
            st.write(f"**Hedge Method:** {hedge_method}")
            st.write(f"**Hedge Method Significance:** {HEDGE_METHOD_DESCRIPTIONS.get(hedge_method, '')}")
    else:
        st.info("Run a backtest to visualize the pair chart and trade bands.")

with main_tabs[4]:
    if "assistant_system_prompt" not in st.session_state:
        st.session_state["assistant_system_prompt"] = build_contextual_system_prompt(st.session_state.get("latest_backtest_metrics"))

    st.info("The assistant uses your latest backtest metrics as context and remembers your conversation while this session is active.")
    for message in st.session_state.ai_messages:
        if message["role"] == "user":
            with st.chat_message("user"):
                st.write(message["content"])
        elif message["role"] == "assistant":
            with st.chat_message("assistant"):
                st.write(message["content"])

    if prompt := st.chat_input("Ask about cointegration, p-values, Sharpe, beta, or risk controls..."):
        st.session_state.ai_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        assistant_reply = get_ai_reply(
            prompt,
            st.session_state.ai_messages[:-1],
            st.session_state["assistant_system_prompt"],
        )
        st.session_state.ai_messages.append({"role": "assistant", "content": assistant_reply})
        with st.chat_message("assistant"):
            st.write(assistant_reply)

    if st.button("Clear Conversation"):
        st.session_state.ai_messages = []
        st.rerun()
