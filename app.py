import os
import json
from datetime import date, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

from src.logic import PairsTradingEngine, SECTOR_UNIVERSES, download_price_data, screen_sector_pairs

TICKER_NAMES = {
    "AAPL": "Apple",
    "MSFT": "Microsoft",
    "GOOGL": "Alphabet",
    "META": "Meta Platforms",
    "NVDA": "NVIDIA",
    "AMD": "Advanced Micro Devices",
    "TSLA": "Tesla",
    "XOM": "Exxon Mobil",
    "CVX": "Chevron",
    "COP": "ConocoPhillips",
    "BP": "BP",
    "SHEL": "Shell",
    "KO": "Coca-Cola",
    "PEP": "PepsiCo",
    "PG": "Procter & Gamble",
    "WMT": "Walmart",
    "COST": "Costco",
    "JNJ": "Johnson & Johnson",
    "PFE": "Pfizer",
    "UNH": "UnitedHealth",
    "ABBV": "AbbVie",
    "MRK": "Merck",
    "JPM": "JPMorgan Chase",
    "BAC": "Bank of America",
    "WFC": "Wells Fargo",
    "GS": "Goldman Sachs",
    "MS": "Morgan Stanley",
}
DEFAULT_TICKERS = list(TICKER_NAMES.keys())
BENCHMARKS = ["SPY", "QQQ", "DIA", "IWM", "XLK"]
HEDGE_METHODS = ["Static OLS", "Rolling OLS", "Kalman Filter"]
SIZING_METHODS = ["Hedge Ratio", "Equal Dollar"]
SECTOR_NAMES = list(SECTOR_UNIVERSES.keys())

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
    half_life_text = "not available" if half_life is None or pd.isna(half_life) else f"{half_life:.1f}"
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


def ticker_label(ticker: str) -> str:
    name = TICKER_NAMES.get(ticker, ticker)
    return f"{ticker} ({name})" if name != ticker else ticker


def ticker_from_label(label: str) -> str:
    return label.split(" ")[0]


def trade_markers(positions: pd.Series) -> dict[str, pd.Series]:
    previous = positions.shift(1).fillna(0.0)
    return {
        "long_entry": positions[(positions == 1) & (previous != 1)],
        "short_entry": positions[(positions == -1) & (previous != -1)],
        "exit": positions[(positions == 0) & (previous != 0)],
    }


def build_price_chart(
    log_prices: pd.DataFrame,
    asset_a: str,
    asset_b: str,
    positions: pd.Series | None = None,
) -> go.Figure:
    normalized = np.exp(log_prices.subtract(log_prices.iloc[0]))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=normalized.index, y=normalized[asset_a], name=asset_a, line=dict(width=2)))
    fig.add_trace(go.Scatter(x=normalized.index, y=normalized[asset_b], name=asset_b, line=dict(width=2)))
    if positions is not None and not positions.empty:
        marks = trade_markers(positions)
        price_a = normalized[asset_a]
        marker_specs = [
            ("long_entry", "triangle-up", "green", "Long spread"),
            ("short_entry", "triangle-down", "red", "Short spread"),
            ("exit", "x", "gray", "Exit"),
        ]
        for key, symbol, color, label in marker_specs:
            idx = marks[key].index.intersection(price_a.index)
            if len(idx) == 0:
                continue
            fig.add_trace(
                go.Scatter(
                    x=idx,
                    y=price_a.loc[idx],
                    mode="markers",
                    name=label,
                    marker=dict(symbol=symbol, size=11, color=color),
                )
            )
    fig.update_layout(
        title="Normalized Prices with Trade Entries and Exits",
        yaxis_title="Normalized Price",
        yaxis_type="log",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template="plotly_white",
    )
    return fig


def build_spread_chart(spread: pd.Series, spread_mean: pd.Series) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=spread.index, y=spread, name="Spread", line=dict(color="#0055AA", width=2)))
    fig.add_trace(go.Scatter(x=spread_mean.index, y=spread_mean, name="Rolling Mean", line=dict(color="#FF7F0E", width=1.5)))
    fig.update_layout(
        title="Mean-Reverting Spread",
        yaxis_title="Log-Price Spread",
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


def build_returns_chart(
    strategy_returns: pd.Series,
    benchmark_cumulative: pd.Series,
    pair_buy_hold_cumulative: pd.Series,
    benchmark_name: str,
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=strategy_returns.index,
            y=np.exp(strategy_returns.cumsum()) - 1.0,
            name="Pairs Strategy",
            line=dict(color="#1F77B4"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=pair_buy_hold_cumulative.index,
            y=pair_buy_hold_cumulative,
            name="Pair Buy & Hold",
            line=dict(color="#2CA02C"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=benchmark_cumulative.index,
            y=benchmark_cumulative,
            name=f"{benchmark_name} Buy & Hold",
            line=dict(color="#FF7F0E"),
        )
    )
    fig.update_layout(
        title="Cumulative PnL vs Pair Buy-and-Hold and Benchmark",
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

    asset_labels = [ticker_label(ticker) for ticker in DEFAULT_TICKERS]
    asset_a_label = st.selectbox(
        "Asset A",
        asset_labels,
        index=0,
        key="asset_a",
        help="Select the first asset in the pair. Load a candidate from the Pair Screener tab to populate this.",
    )
    asset_b_label = st.selectbox(
        "Asset B",
        asset_labels,
        index=1,
        key="asset_b",
        help="Select the second asset in the pair. Choose a cointegrated instrument from the same sector when possible.",
    )
    asset_a = ticker_from_label(asset_a_label)
    asset_b = ticker_from_label(asset_b_label)

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

    capital = st.number_input(
        "Dollar Book ($)",
        min_value=10_000.0,
        max_value=10_000_000.0,
        value=100_000.0,
        step=10_000.0,
        key="capital",
        help="Gross dollar capital allocated to the pair. Hedge-ratio sizing splits this as $A + $|beta| A = capital.",
    )
    sizing_method = st.selectbox(
        "Position Sizing",
        SIZING_METHODS,
        key="sizing_method",
        help="Hedge Ratio sizes B to the OLS/Kalman beta. Equal Dollar puts the same notional on each leg.",
    )

    st.markdown("---")
    st.write("**Execution assumptions**")
    st.caption("Slippage = 0.1%, commission = 0.05% of traded dollar notional per side.")
    run_button = st.button("Run Backtest")

if "ai_messages" not in st.session_state:
    st.session_state.ai_messages = []

if asset_a == asset_b:
    st.sidebar.error("Asset A and Asset B must be different.")

result = st.session_state.get("latest_backtest")
metrics = None if result is None else result.get("metrics")
show_results = bool(result) and not result["data"].empty

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
            capital=float(capital),
            sizing_method=sizing_method,
        )
        result = engine.fetch().run_backtest()

    st.session_state["latest_backtest"] = result
    st.session_state["latest_backtest_metrics"] = result["metrics"]
    st.session_state["assistant_system_prompt"] = build_contextual_system_prompt(result["metrics"])

    if result["data"].empty:
        st.error("No historical data loaded for the selected symbols and date range.")
        show_results = False
        metrics = None
    else:
        metrics = result["metrics"]
        show_results = True

st.markdown("---")
main_tabs = st.tabs(["Pair Screener", "Strategy Setup", "Performance", "Diagnostics", "Charts", "AI Quant Assistant"])

with main_tabs[0]:
    st.markdown("### Sector Pair Screener")
    st.markdown(
        "Download a sector universe from Yahoo Finance, run Engle-Granger on every unique pair, "
        "and rank by cointegration p-value and half-life. Load a candidate into the sidebar, then run the backtest."
    )
    screen_cols = st.columns([2, 1, 1])
    with screen_cols[0]:
        sector = st.selectbox("Sector Universe", SECTOR_NAMES, key="screen_sector")
    with screen_cols[1]:
        p_threshold = st.number_input(
            "p-value cutoff",
            min_value=0.01,
            max_value=0.20,
            value=0.05,
            step=0.01,
            key="screen_p_value",
        )
    with screen_cols[2]:
        st.write("")
        st.write("")
        run_screen = st.button("Screen Sector", type="primary")

    st.caption("Universe: " + ", ".join(SECTOR_UNIVERSES[sector]))

    if run_screen:
        with st.spinner(f"Screening {sector} pairs..."):
            sector_prices = download_price_data(
                SECTOR_UNIVERSES[sector],
                start_date.isoformat(),
                end_date.isoformat(),
            )
            screen_table = screen_sector_pairs(sector_prices, p_value_threshold=float(p_threshold))
        st.session_state["screen_table"] = screen_table
        st.session_state["screen_sector_used"] = sector

    screen_table = st.session_state.get("screen_table")
    if screen_table is None:
        st.info("Choose a sector and click Screen Sector. Classic examples: Energy (XOM/CVX), Consumer Staples (KO/PEP).")
    elif screen_table.empty:
        st.warning("No pairs could be tested for this universe and date range.")
    else:
        n_coint = int(screen_table["cointegrated"].sum())
        st.success(f"{len(screen_table)} pairs tested, {n_coint} below the p-value cutoff.")
        display = screen_table.copy()
        display["p_value"] = display["p_value"].map(lambda x: f"{x:.4f}")
        display["adf_p_value"] = display["adf_p_value"].map(lambda x: f"{x:.4f}")
        display["hedge_ratio"] = display["hedge_ratio"].map(lambda x: f"{x:.3f}")
        display["half_life"] = display["half_life"].map(lambda x: "n/a" if pd.isna(x) else f"{x:.1f}")
        st.dataframe(
            display[["pair", "p_value", "adf_p_value", "hedge_ratio", "half_life", "cointegrated"]],
            use_container_width=True,
            hide_index=True,
        )
        pair_options = screen_table["pair"].tolist()
        chosen_pair = st.selectbox("Load candidate into the engine", pair_options, key="screen_pair_choice")
        if st.button("Use Selected Pair"):
            chosen_a, chosen_b = chosen_pair.split("/")
            st.session_state["asset_a"] = ticker_label(chosen_a)
            st.session_state["asset_b"] = ticker_label(chosen_b)
            st.rerun()

with main_tabs[1]:
    st.markdown("### Strategy Setup")
    if show_results:
        st.success("Backtest completed. Review the insights in the performance, diagnostics, charts, and assistant tabs.")
    else:
        st.markdown(
            "Screen a sector, load a cointegrated pair, then run a dollar-book long/short backtest with z-score entry and exit rules."
        )
        st.markdown(
            "- Sector pair screening with Engle-Granger tests\n"
            "- Mean-reverting spread and rolling z-score signals\n"
            "- Hedge-ratio or equal-dollar position sizing\n"
            "- Transaction costs, stop-loss, and buy-and-hold comparison"
        )
        hero_cols = st.columns([2, 1])
        with hero_cols[0]:
            st.info("Fill in the sidebar and click Run Backtest, or load a pair from the screener first.")
        with hero_cols[1]:
            st.metric("Selected Pair", f"{asset_a} / {asset_b}")
            st.metric("Benchmark", benchmark)
            st.metric("Hedge Method", hedge_method)
            st.metric("Dollar Book", f"${capital:,.0f}")

with main_tabs[2]:
    if show_results:
        st.markdown("## Performance Summary")
        perf_cols = st.columns([1, 1, 1, 1])
        perf_cols[0].metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}")
        perf_cols[1].metric("Max Drawdown", f"{metrics['max_drawdown']:.2%}")
        perf_cols[2].metric("Total Return", f"{metrics['total_return']:.2%}")
        perf_cols[3].metric("Win Rate", f"{metrics['win_rate']:.2%}")
        money_cols = st.columns([1, 1, 1, 1])
        money_cols[0].metric("Starting Capital", f"${metrics['capital']:,.0f}")
        money_cols[1].metric("Total PnL", f"${metrics['total_pnl']:,.0f}")
        money_cols[2].metric("Ending Equity", f"${metrics['ending_equity']:,.0f}")
        money_cols[3].metric(
            "Half-Life",
            "n/a" if pd.isna(metrics["half_life"]) else f"{metrics['half_life']:.1f} days",
        )
        st.caption(f"Sizing method: {metrics.get('sizing_method', sizing_method)}. Returns are dollar PnL divided by the dollar book.")
    else:
        st.info("Run a backtest to see performance metrics.")

with main_tabs[3]:
    if show_results:
        st.markdown("## Model Diagnostics")
        diag_cols = st.columns([1, 1, 1])
        diag_cols[0].metric("Cointegration", metrics["cointegration_status"], f"p = {metrics['cointegration_p_value']:.4f}")
        diag_cols[1].metric("Beta Ratio", f"{metrics['beta_ratio']:.3f}", f"A: {metrics['benchmark_beta_a']:.3f} / B: {metrics['benchmark_beta_b']:.3f}")
        diag_cols[2].metric("Hedge Ratio", f"{metrics['hedge_ratio']:.3f}")
        st.markdown("### Statistical Details")
        st.write("- Engle-Granger cointegration validates long-term equilibrium between the log-price series.")
        st.write("- Hedge-ratio sizing holds $A and $β of B so the book stays close to spread-neutral.")
        st.write(f"- Cointegration p-value: **{metrics['cointegration_p_value']:.4f}**")
        st.write(f"- Cointegration status: **{metrics['cointegration_status']}**")
        st.write(f"- Beta A vs benchmark: **{metrics['benchmark_beta_a']:.3f}**")
        st.write(f"- Beta B vs benchmark: **{metrics['benchmark_beta_b']:.3f}**")
        st.write(f"- Factor beta ratio: **{metrics['beta_ratio']:.3f}**")
        half_life_text = "n/a" if pd.isna(metrics["half_life"]) else f"{metrics['half_life']:.1f} days"
        st.write(f"- Mean reversion half-life: **{half_life_text}**")
    else:
        st.info("Run a backtest to inspect the model diagnostics.")

with main_tabs[4]:
    if show_results:
        st.markdown("### Overview Chart")
        chart_a = result["asset_a"]
        chart_b = result["asset_b"]
        overview_cols = st.columns([2, 1])
        with overview_cols[0]:
            st.plotly_chart(
                build_price_chart(result["log_prices"], chart_a, chart_b, result["positions"]),
                use_container_width=True,
            )
            st.plotly_chart(
                build_spread_chart(result["spread"], result["spread_mean"]),
                use_container_width=True,
            )
            st.plotly_chart(
                build_returns_chart(
                    result["strategy_returns"],
                    result["benchmark_cumulative"],
                    result["pair_buy_hold_cumulative"],
                    result["benchmark"],
                ),
                use_container_width=True,
            )
        with overview_cols[1]:
            st.plotly_chart(build_zscore_chart(result["zscore"], entry_z, exit_z, stop_loss), use_container_width=True)
            st.markdown("#### Strategy Configuration")
            st.write(f"**Pair:** {chart_a} / {chart_b}")
            st.write(f"**Benchmark:** {result['benchmark']}")
            st.write(f"**Lookback:** {lookback} days")
            st.write(f"**Entry:** {entry_z} | Exit: {exit_z} | Stop-Loss: {stop_loss}")
            st.write(f"**Hedge Method:** {result['hedge_method']}")
            st.write(f"**Sizing:** {result['sizing_method']} on a ${metrics['capital']:,.0f} book")
            st.write("Green triangles: long the spread. Red triangles: short the spread. Gray X: exit.")
    else:
        st.info("Run a backtest to visualize spread, z-score, trade markers, and cumulative PnL.")

with main_tabs[5]:
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
