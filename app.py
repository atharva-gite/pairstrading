from datetime import date
from html import escape

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

from src.logic import (
    SECTOR_UNIVERSES,
    PairsTradingEngine,
    build_book_context,
    call_llm,
    cost_sensitivity,
    download_price_data,
    generate_desk_briefing,
    pair_pvalue_matrix,
    screen_sector_pairs,
)

load_dotenv()

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
    "Static OLS": "Fixed hedge for stable tethers. Best when the relationship is not drifting.",
    "Rolling OLS": "Re-estimates beta each lookback window when the pair is rotating regimes.",
    "Kalman Filter": "Smooths hedge-ratio drift. Use when the spread is tradable but beta is not static.",
}

COPPER = "#c4a35a"
MINT = "#7dce9a"
ROSE = "#e07a5f"
INK = "#e8e4d6"
MUTED = "#9aa394"
PLOT = "#121612"
GRID = "#2a332c"

STATE_COPY = {
    "TETHERED": "The long-run relationship is intact. Mean reversion is statistically supported.",
    "WEAK": "The tether is fraying. Trade smaller, or wait for the rolling p-value to settle.",
    "BROKEN": "Do not treat this as a pairs book. The spread may be a directional residual.",
}


def inject_css() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@600;700&family=IBM+Plex+Mono:wght@400;600&family=Outfit:wght@400;500;600&display=swap');

        :root {
            --bg: #0c0f0d;
            --panel: #151a16;
            --line: #2a332c;
            --text: #e8e4d6;
            --muted: #9aa394;
            --copper: #c4a35a;
            --mint: #7dce9a;
            --rose: #e07a5f;
        }
        html, body, .stApp, [data-testid="stAppViewContainer"] {
            background: var(--bg);
            color: var(--text);
            font-family: Outfit, system-ui, sans-serif;
        }
        [data-testid="stHeader"] { background: var(--bg); }
        [data-testid="stSidebar"] { background: #101410; border-right: 1px solid var(--line); }
        [data-testid="stSidebar"] * { color: var(--text); }
        #MainMenu, footer, [data-testid="stDeployButton"] { visibility: hidden; }
        h1, h2, h3 { font-family: Fraunces, Georgia, serif; letter-spacing: -0.02em; }
        [data-testid="stMetricValue"] { font-family: "IBM Plex Mono", monospace; color: var(--text); }
        [data-testid="stMetricLabel"] { color: var(--muted); text-transform: uppercase; letter-spacing: 0.06em; font-size: 0.72rem; }
        div[data-testid="stTabs"] button { font-family: Outfit, sans-serif; }
        .brand-wrap { display: flex; justify-content: space-between; gap: 24px; align-items: flex-end; margin: 4px 0 18px; }
        .brand-mark { font-family: Fraunces, serif; font-size: 2.1rem; font-weight: 700; color: var(--text); line-height: 1; }
        .brand-mark span { color: var(--copper); }
        .brand-kicker { color: var(--muted); font-size: 0.92rem; margin-top: 8px; max-width: 620px; }
        .desk-chip {
            font-family: "IBM Plex Mono", monospace;
            font-size: 0.78rem;
            border: 1px solid var(--line);
            padding: 8px 12px;
            border-radius: 999px;
            color: var(--muted);
            white-space: nowrap;
        }
        .status-bar {
            display: flex; flex-wrap: wrap; gap: 10px; align-items: center;
            border: 1px solid var(--line); background: var(--panel);
            border-radius: 12px; padding: 12px 14px; margin-bottom: 16px;
        }
        .pill { font-family: "IBM Plex Mono", monospace; font-size: 0.78rem; font-weight: 600; padding: 4px 10px; border-radius: 999px; }
        .pill-tethered { background: rgba(125,206,154,0.16); color: var(--mint); }
        .pill-weak { background: rgba(196,163,90,0.16); color: var(--copper); }
        .pill-broken { background: rgba(224,122,95,0.16); color: var(--rose); }
        .status-meta { color: var(--muted); font-size: 0.86rem; }
        .metric-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px; margin: 8px 0 16px; }
        .metric-card { background: var(--panel); border: 1px solid var(--line); border-radius: 12px; padding: 12px; min-height: 92px; }
        .metric-label { color: var(--muted); font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.05em; font-weight: 600; }
        .metric-value { font-family: "IBM Plex Mono", monospace; font-size: 1.28rem; font-weight: 600; margin-top: 8px; }
        .metric-sub { color: var(--muted); font-size: 0.8rem; margin-top: 6px; }
        .tone-mint .metric-value { color: var(--mint); }
        .tone-rose .metric-value { color: var(--rose); }
        .tone-copper .metric-value { color: var(--copper); }
        .verdict {
            border-left: 3px solid var(--copper);
            padding: 8px 0 8px 14px;
            color: var(--text);
            margin: 4px 0 18px;
        }
        .briefing-headline { font-family: Fraunces, serif; font-size: 1.35rem; margin: 0 0 8px; }
        .briefing-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin: 0 0 16px; }
        .briefing-card { background: var(--panel); border: 1px solid var(--line); border-radius: 12px; padding: 12px 14px; }
        .briefing-card h4 { margin: 0 0 8px; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.05em; color: var(--muted); font-family: Outfit, sans-serif; }
        .briefing-card p { margin: 0; color: var(--text); font-size: 0.92rem; line-height: 1.45; }
        @media (max-width: 900px) { .briefing-grid { grid-template-columns: 1fr; } }
        .empty-panel {
            border: 1px dashed var(--line);
            border-radius: 12px;
            padding: 28px 18px;
            color: var(--muted);
            background: #101410;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def metric_card(label: str, value: str, sub: str = "", tone: str = "") -> str:
    cls = f"metric-card {tone}".strip()
    return (
        f'<div class="{cls}">'
        f'<div class="metric-label">{escape(label)}</div>'
        f'<div class="metric-value">{value}</div>'
        f'<div class="metric-sub">{escape(sub)}</div>'
        f"</div>"
    )


def apply_chart_layout(fig: go.Figure, title: str, y_title: str) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, font=dict(size=15, color=INK)),
        yaxis_title=y_title,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=PLOT,
        font=dict(color=INK, family="Outfit"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=11)),
        margin=dict(l=40, r=16, t=56, b=36),
        xaxis=dict(gridcolor=GRID, zeroline=False),
        yaxis=dict(gridcolor=GRID, zeroline=False),
    )
    return fig

    return fig


def ticker_label(ticker: str) -> str:
    name = TICKER_NAMES.get(ticker, ticker)
    return f"{ticker} ({name})" if name != ticker else ticker


def ticker_from_label(label: str) -> str:
    return label.split(" ")[0]


def fmt_pct(value: float) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{value:.1%}"


def fmt_num(value: float, digits: int = 2) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{value:.{digits}f}"


def trade_markers(positions: pd.Series) -> dict[str, pd.Series]:
    previous = positions.shift(1).fillna(0.0)
    return {
        "long_entry": positions[(positions == 1) & (previous != 1)],
        "short_entry": positions[(positions == -1) & (previous != -1)],
        "exit": positions[(positions == 0) & (previous != 0)],
    }


def build_price_chart(
    log_prices: pd.DataFrame, asset_a: str, asset_b: str, positions: pd.Series | None = None
) -> go.Figure:
    normalized = np.exp(log_prices.subtract(log_prices.iloc[0]))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=normalized.index, y=normalized[asset_a], name=asset_a, line=dict(width=2, color=COPPER)))
    fig.add_trace(go.Scatter(x=normalized.index, y=normalized[asset_b], name=asset_b, line=dict(width=2, color=MINT)))
    if positions is not None and not positions.empty:
        marks = trade_markers(positions)
        price_a = normalized[asset_a]
        for key, symbol, color, label in [
            ("long_entry", "triangle-up", MINT, "Long spread"),
            ("short_entry", "triangle-down", ROSE, "Short spread"),
            ("exit", "x", MUTED, "Exit"),
        ]:
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
    fig.update_yaxes(type="log")
    return apply_chart_layout(fig, "Normalized prices · entries and exits", "Price")


def build_spread_chart(spread: pd.Series, spread_mean: pd.Series) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=spread.index, y=spread, name="Spread", line=dict(color=COPPER, width=2)))
    fig.add_trace(go.Scatter(x=spread_mean.index, y=spread_mean, name="Equilibrium", line=dict(color=MINT, width=1.5)))
    return apply_chart_layout(fig, "Spread versus estimated equilibrium", "Log spread")


def build_zscore_chart(zscore: pd.Series, entry_z: float, exit_z: float, stop_loss: float) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=zscore.index, y=zscore, name="Z-score", line=dict(color=COPPER, width=2)))
    fig.add_hline(y=entry_z, line_dash="dash", line_color=ROSE)
    fig.add_hline(y=-entry_z, line_dash="dash", line_color=MINT)
    fig.add_hline(y=exit_z, line_dash="dot", line_color=MUTED)
    fig.add_hline(y=-exit_z, line_dash="dot", line_color=MUTED)
    fig.add_hline(y=stop_loss, line_dash="dashdot", line_color=ROSE)
    fig.add_hline(y=-stop_loss, line_dash="dashdot", line_color=ROSE)
    return apply_chart_layout(fig, "Z-score with entry, exit, and stop bands", "Z")


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
            name="TETHER book",
            line=dict(color=COPPER, width=2.2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=pair_buy_hold_cumulative.index,
            y=pair_buy_hold_cumulative,
            name="Pair buy & hold",
            line=dict(color=MINT, width=1.6),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=benchmark_cumulative.index,
            y=benchmark_cumulative,
            name=f"{benchmark_name} long-only",
            line=dict(color=MUTED, width=1.4),
        )
    )
    return apply_chart_layout(fig, "Book versus holding the pair and the index", "Return")


def build_net_beta_chart(net_beta: pd.Series) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=net_beta.index, y=net_beta, name="Net book β", line=dict(color=COPPER, width=2)))
    fig.add_hline(y=0, line_dash="dot", line_color=MUTED)
    return apply_chart_layout(fig, "Net market beta of the dollar book", "β vs sleeve")


def build_tether_chart(rolling_p: pd.Series) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=rolling_p.index, y=rolling_p, name="Rolling p-value", line=dict(color=COPPER, width=2)))
    fig.add_hline(y=0.05, line_dash="dash", line_color=MINT, annotation_text="5% line", annotation_font_color=MINT)
    fig.add_hline(y=0.10, line_dash="dot", line_color=ROSE, annotation_text="break", annotation_font_color=ROSE)
    return apply_chart_layout(fig, "Tether health · rolling Engle-Granger p-value", "p-value")


def build_heatmap(matrix: pd.DataFrame) -> go.Figure:
    fig = go.Figure(
        data=go.Heatmap(
            z=matrix.values,
            x=list(matrix.columns),
            y=list(matrix.index),
            colorscale=[[0, MINT], [0.5, COPPER], [1, ROSE]],
            zmin=0,
            zmax=0.2,
            colorbar=dict(title="p-value"),
            hovertemplate="%{y} / %{x}<br>p=%{z:.3f}<extra></extra>",
        )
    )
    return apply_chart_layout(fig, "Universe map · FDR-adjusted p-value (mint = tighter)", "Ticker")


def desk_verdict(metrics: dict, pair: str) -> str:
    state = metrics.get("tether_state", "WEAK")
    reason = metrics.get("tether_reason")
    score = metrics.get("tether_score")
    hl = metrics.get("half_life")
    p_value = metrics.get("cointegration_p_value")
    on_time = metrics.get("reverted_on_time")
    trades = metrics.get("trade_count", 0)

    if reason == "no_cointegration":
        core = (
            f"{pair} is not a tradable tether. The names can still move together through a shared factor "
            f"(Engle-Granger p={fmt_num(p_value, 3)}). That is correlation, not a mean-reverting spread."
        )
    elif reason == "too_slow":
        core = (
            f"{pair} reverts too slowly for this book "
            f"({hl:.0f}-day half-life; the desk wants 8–40 days). Inventory would sit for months."
        )
    elif reason == "too_fast":
        core = f"{pair} mean-reverts too quickly to be distinguished from noise."
    elif reason == "no_half_life":
        core = f"{pair} does not show a usable mean-reversion clock."
    else:
        core = f"{pair} is {state} (score {fmt_num(score, 0)}). {STATE_COPY.get(state, '')}"

    if trades and on_time is not None and not pd.isna(on_time) and hl is not None and 5 <= hl <= 60:
        clock = f" {on_time:.0%} of round-trips closed inside 1.5× the {hl:.0f}-day clock."
    elif trades and hl is not None and not pd.isna(hl) and hl > 60:
        clock = f" On-time rate is not a green light here: 1.5× a {hl:.0f}-day clock is ~{hl * 1.5:.0f} days, so almost any trade looks punctual."
    else:
        clock = ""
    return core + clock


st.set_page_config(page_title="TETHER · RelVal Desk", layout="wide", page_icon="⚓")
inject_css()

st.markdown(
    """
    <div class="brand-wrap">
      <div>
        <div class="brand-mark">TE<span>THER</span></div>
        <div class="brand-kicker">Market-neutral pairs desk. You do not trade two stocks — you trade whether their tether is still holding.</div>
      </div>
      <div class="desk-chip">RELVAL · STAT ARB</div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("**Ticket**")
    asset_labels = [ticker_label(ticker) for ticker in DEFAULT_TICKERS]
    asset_a_label = st.selectbox(
        "Asset A",
        asset_labels,
        index=DEFAULT_TICKERS.index("KO"),
        key="asset_a",
    )
    asset_b_label = st.selectbox(
        "Asset B",
        asset_labels,
        index=DEFAULT_TICKERS.index("PEP"),
        key="asset_b",
    )
    asset_a = ticker_from_label(asset_a_label)
    asset_b = ticker_from_label(asset_b_label)
    benchmark = st.selectbox("Factor sleeve", BENCHMARKS, index=0, key="benchmark")

    st.markdown("**Window**")
    start_date = st.date_input("Start", date(2019, 1, 1), key="start_date")
    end_date = st.date_input("End", date.today(), key="end_date")
    walk_forward = st.checkbox("Walk-forward (no peeking)", value=True, key="walk_forward")
    split_date = st.date_input("Discover / trade split", date(2024, 1, 1), key="split_date")
    st.caption("Discover hedge and tether on or before the split. Trade only after it.")

    st.markdown("**Signals**")
    lookback = st.slider("Lookback (days)", min_value=20, max_value=180, value=60, step=10, key="lookback")
    entry_z = st.slider("Entry |z|", min_value=1.0, max_value=4.0, value=2.0, step=0.1, key="entry_z")
    exit_z = st.slider("Exit |z|", min_value=0.1, max_value=2.0, value=0.5, step=0.1, key="exit_z")
    stop_loss = st.slider("Stop |z|", min_value=1.0, max_value=5.0, value=3.0, step=0.1, key="stop_loss")

    st.markdown("**Book**")
    hedge_method = st.selectbox("Hedge", HEDGE_METHODS, key="hedge_method")
    st.caption(HEDGE_METHOD_DESCRIPTIONS.get(hedge_method, ""))
    capital = st.number_input(
        "Gross capital ($)", min_value=10_000.0, max_value=10_000_000.0, value=100_000.0, step=10_000.0, key="capital"
    )
    sizing_method = st.selectbox("Sizing", SIZING_METHODS, key="sizing_method")
    max_leg_pct = st.number_input(
        "Max per leg (% of gross)", min_value=50, max_value=100, value=75, step=5, key="max_leg_pct"
    )
    charge_borrow = st.checkbox("Charge stock-borrow on shorts", value=True, key="charge_borrow")
    borrow_bps = st.number_input(
        "Borrow (bps / year)",
        min_value=0.0,
        max_value=500.0,
        value=50.0,
        step=10.0,
        key="borrow_bps",
        disabled=not charge_borrow,
    )
    st.caption("10 bps slippage + 5 bps commission on traded notional. Borrow is an annualized rate on short dollars.")
    run_button = st.button("Run the book", type="primary", use_container_width=True)

if "ai_messages" not in st.session_state:
    st.session_state.ai_messages = []

if asset_a == asset_b:
    st.sidebar.error("Choose two different names.")

result = st.session_state.get("latest_backtest")
metrics = None if result is None else result.get("metrics")
show_results = bool(result) and not result["data"].empty
if show_results and not st.session_state.get("book_context"):
    st.session_state["book_context"] = build_book_context(result)

if run_button and asset_a != asset_b and start_date < end_date:
    with st.spinner("Estimating the tether and running the book..."):
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
            walk_forward=walk_forward,
            split_date=split_date.isoformat(),
            charge_borrow=charge_borrow,
            borrow_bps=float(borrow_bps),
            max_leg_pct=max_leg_pct / 100.0,
        )
        result = engine.fetch().run_backtest()
        st.session_state["cost_grid"] = cost_sensitivity(engine) if not result["data"].empty else None

    st.session_state["latest_backtest"] = result
    st.session_state["latest_backtest_metrics"] = result["metrics"]
    st.session_state["book_context"] = build_book_context(result)
    st.session_state.ai_messages = []
    if not result["data"].empty:
        with st.spinner("Groq is writing the desk briefing from this book..."):
            st.session_state["desk_briefing"] = generate_desk_briefing(st.session_state["book_context"])
    else:
        st.session_state["desk_briefing"] = None

    if result["data"].empty:
        st.error("No prices for that ticket and window.")
        show_results = False
        metrics = None
    else:
        metrics = result["metrics"]
        show_results = True

pair_name = f"{asset_a}/{asset_b}"
if show_results:
    state = metrics["tether_state"]
    pill_class = {"TETHERED": "pill-tethered", "WEAK": "pill-weak", "BROKEN": "pill-broken"}.get(state, "pill-weak")
    st.markdown(
        f"""
        <div class="status-bar">
          <span class="pill {pill_class}">{escape(state)}</span>
          <span class="status-meta"><b>{escape(result["asset_a"])}/{escape(result["asset_b"])}</b> · score {fmt_num(metrics["tether_score"], 0)} · p={metrics["cointegration_p_value"]:.3f} · {escape(result["hedge_method"])} · ${metrics["capital"]:,.0f}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

main_tabs = st.tabs(["Universe", "Desk", "Tape", "Research"])

with main_tabs[0]:
    st.markdown("Find a pair whose prices are still tied together, then load it as the ticket.")
    screen_cols = st.columns([2, 1, 1])
    with screen_cols[0]:
        sector = st.selectbox("Sector", SECTOR_NAMES, key="screen_sector")
    with screen_cols[1]:
        p_threshold = st.number_input(
            "p-value line", min_value=0.01, max_value=0.20, value=0.05, step=0.01, key="screen_p_value"
        )
    with screen_cols[2]:
        st.write("")
        st.write("")
        run_screen = st.button("Scan sector", type="primary")

    st.caption("Universe: " + " · ".join(SECTOR_UNIVERSES[sector]))

    if run_screen:
        with st.spinner(f"Scoring {sector} tethers..."):
            sector_prices = download_price_data(SECTOR_UNIVERSES[sector], start_date.isoformat(), end_date.isoformat())
            screen_table = screen_sector_pairs(
                sector_prices,
                p_value_threshold=float(p_threshold),
                split_date=split_date.isoformat() if walk_forward else None,
            )
        st.session_state["screen_table"] = screen_table
        st.session_state["screen_sector_used"] = sector

    screen_table = st.session_state.get("screen_table")
    if screen_table is None:
        st.markdown(
            '<div class="empty-panel">Scan Energy for XOM/CVX or Consumer Staples for KO/PEP. Rankings use Tether Score, not raw correlation.</div>',
            unsafe_allow_html=True,
        )
    elif screen_table.empty:
        st.warning("No pairs could be tested in that window.")
    else:
        n_coint = int(screen_table["cointegrated"].sum())
        n_fdr = int(screen_table["fdr_significant"].sum()) if "fdr_significant" in screen_table else 0
        n_died = int(screen_table["tether_died"].sum()) if "tether_died" in screen_table else 0
        n_tethered = int((screen_table["tether_state"] == "TETHERED").sum())
        st.caption(
            f"{n_fdr} of {len(screen_table)} pairs survive FDR. "
            f"{n_coint} raw p-values sit under the uncorrected line. "
            f"{n_died} of {n_tethered} TETHERED names die live after the split."
        )
        heat = pair_pvalue_matrix(
            screen_table, value_col="p_value_fdr" if "p_value_fdr" in screen_table.columns else "p_value"
        )
        if not heat.empty:
            st.plotly_chart(build_heatmap(heat), use_container_width=True)
        display = screen_table.copy()
        display["p_value"] = display["p_value"].map(lambda x: f"{x:.4f}")
        if "p_value_fdr" in display.columns:
            display["p_value_fdr"] = display["p_value_fdr"].map(lambda x: "—" if pd.isna(x) else f"{x:.4f}")
        display["tether_score"] = display["tether_score"].map(lambda x: f"{x:.0f}" if pd.notna(x) else "—")
        display["hedge_ratio"] = display["hedge_ratio"].map(lambda x: f"{x:.3f}" if pd.notna(x) else "—")
        display["half_life"] = display["half_life"].map(lambda x: "—" if pd.isna(x) else f"{x:.1f}d")
        cols = ["pair", "tether_state", "tether_score", "p_value", "p_value_fdr", "fdr_significant", "half_life"]
        if "oos_tether_state" in display.columns:
            cols.extend(["oos_tether_state", "tether_died"])
        st.dataframe(display[[c for c in cols if c in display.columns]], use_container_width=True, hide_index=True)
        pair_options = screen_table["pair"].tolist()
        chosen_pair = st.selectbox("Load into the ticket", pair_options, key="screen_pair_choice")
        if st.button("Load pair"):
            chosen_a, chosen_b = chosen_pair.split("/")
            st.session_state["asset_a"] = ticker_label(chosen_a)
            st.session_state["asset_b"] = ticker_label(chosen_b)
            st.rerun()

with main_tabs[1]:
    if show_results:
        ticket = f"{result['asset_a']}/{result['asset_b']}"
        briefing = st.session_state.get("desk_briefing") or {}
        groq_verdict = (briefing.get("verdict") or "").strip()
        if briefing.get("headline"):
            st.markdown(f'<div class="briefing-headline">{escape(briefing["headline"])}</div>', unsafe_allow_html=True)
        if groq_verdict:
            st.markdown(f'<div class="verdict">{escape(groq_verdict)}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="verdict">{escape(desk_verdict(metrics, ticket))}</div>', unsafe_allow_html=True)
        if briefing.get("drivers") or briefing.get("risks"):
            st.markdown(
                f"""
                <div class="briefing-grid">
                  <div class="briefing-card"><h4>What the book is saying</h4><p>{escape(briefing.get("drivers") or "—")}</p></div>
                  <div class="briefing-card"><h4>What would change the read</h4><p>{escape(briefing.get("risks") or "—")}</p></div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        if briefing.get("error"):
            st.caption(f"Groq briefing unavailable: {briefing['error']}. Showing the rule-based verdict.")
        pnl_tone = "tone-mint" if metrics["total_pnl"] >= 0 else "tone-rose"
        state_tone = {"TETHERED": "tone-mint", "WEAK": "tone-copper", "BROKEN": "tone-rose"}.get(
            metrics["tether_state"], ""
        )
        cards = "".join(
            [
                metric_card("Tether score", fmt_num(metrics["tether_score"], 0), metrics["tether_state"], state_tone),
                metric_card("Sharpe", fmt_num(metrics["sharpe_ratio"]), "after costs"),
                metric_card("Max drawdown", fmt_pct(metrics["max_drawdown"]), "", "tone-rose"),
                metric_card("Book PnL", f"${metrics['total_pnl']:,.0f}", f"on ${metrics['capital']:,.0f}", pnl_tone),
                metric_card(
                    "Half-life", "—" if pd.isna(metrics["half_life"]) else f"{metrics['half_life']:.0f}d", "OU clock"
                ),
                metric_card(
                    "On-time exits",
                    fmt_pct(metrics.get("reverted_on_time")),
                    "only scored if half-life is 5–60d"
                    if pd.isna(metrics.get("reverted_on_time"))
                    else f"{metrics['trade_count']} round-trips",
                ),
            ]
        )
        st.markdown(f'<div class="metric-grid">{cards}</div>', unsafe_allow_html=True)

        screen_table = st.session_state.get("screen_table")
        if screen_table is not None and not screen_table.empty and "fdr_significant" in screen_table.columns:
            n_fdr = int(screen_table["fdr_significant"].sum())
            n_died = int(screen_table["tether_died"].sum()) if "tether_died" in screen_table else 0
            n_tethered = int((screen_table["tether_state"] == "TETHERED").sum())
            st.caption(
                f"{n_fdr} of {len(screen_table)} sector pairs survive FDR. "
                f"{n_died} of {n_tethered} TETHERED names die after the walk-forward split."
            )

        if metrics.get("walk_forward"):
            wf = pd.DataFrame(
                {
                    "Sample": ["In-sample (discover)", "Out-of-sample (live)"],
                    "Sharpe": [metrics.get("is_sharpe"), metrics.get("oos_sharpe")],
                    "Return": [metrics.get("is_return"), metrics.get("oos_return")],
                    "Tether": [metrics.get("tether_state"), metrics.get("oos_tether_state")],
                }
            )
            st.markdown("**Walk-forward**")
            wf_view = wf.copy()
            wf_view["Sharpe"] = wf_view["Sharpe"].map(lambda v: "—" if pd.isna(v) else f"{v:.2f}")
            wf_view["Return"] = wf_view["Return"].map(lambda v: "—" if pd.isna(v) else f"{v:.1%}")
            st.dataframe(wf_view, hide_index=True, use_container_width=True)
            is_s, oos_s = metrics.get("is_sharpe"), metrics.get("oos_sharpe")
            if pd.notna(is_s) and pd.notna(oos_s) and oos_s < is_s:
                st.caption(
                    "Out-of-sample Sharpe is worse than discovery. That is the point of walk-forward — leave it on the desk."
                )
            if metrics.get("tether_died"):
                st.caption("This pair was TETHERED in discovery and is not TETHERED live.")
            st.caption(
                f"Frozen hedge from {metrics.get('split_date')}. "
                f"Net book β while on: mean {fmt_num(metrics.get('net_beta_mean'), 3)}, "
                f"|β| {fmt_num(metrics.get('net_beta_abs_mean'), 3)}. "
                f"Borrow charged: {metrics.get('borrow_bps'):.0f} bps (${metrics.get('borrow_cost_total', 0):,.0f}). "
                f"Leg cap {metrics.get('max_leg_pct', 1.0):.0%} of gross bound on "
                f"{metrics.get('leg_cap_binding_days', 0)} days in the book."
            )
        elif metrics.get("walk_forward") is False:
            st.caption("Walk-forward needs enough bars on both sides of the split. Widen the window.")

        cost_grid = st.session_state.get("cost_grid")
        if cost_grid is not None and not cost_grid.empty:
            st.markdown("**Cost sensitivity**")
            grid_view = pd.DataFrame(
                {
                    "Slippage": cost_grid["slippage_bps"].map(lambda v: f"{v:.0f} bps"),
                    "Commission": cost_grid["commission_bps"].map(lambda v: f"{v:.0f} bps"),
                    "Book PnL": cost_grid["total_pnl"].map(lambda v: f"${v:,.0f}"),
                    "Sharpe": cost_grid["sharpe_ratio"].map(lambda v: fmt_num(v)),
                    "OOS Sharpe": cost_grid["oos_sharpe"].map(lambda v: fmt_num(v)),
                }
            )
            st.dataframe(grid_view, hide_index=True, use_container_width=True)
            st.caption("Same signals, different slippage. If the edge disappears at 5 bps, it was never there.")

        left, right = st.columns([1.15, 1])
        with left:
            rolling_p = result.get("rolling_p_value", pd.Series(dtype=float))
            if rolling_p is not None and len(rolling_p):
                st.plotly_chart(build_tether_chart(rolling_p), use_container_width=True)
            net_beta = result.get("net_beta")
            if net_beta is not None and len(net_beta):
                st.plotly_chart(build_net_beta_chart(net_beta), use_container_width=True)
            else:
                st.caption("Need a longer window to plot rolling tether health.")
            st.caption(
                f"Engle-Granger p={metrics['cointegration_p_value']:.4f} · ADF p={metrics.get('adf_p_value', float('nan')):.4f} · "
                f"hedge β={metrics['hedge_ratio']:.3f} · factor β ratio={fmt_num(metrics['beta_ratio'], 3)}"
            )
        with right:
            st.markdown("**Blotter**")
            blotter = result.get("blotter", pd.DataFrame())
            if blotter is None or blotter.empty:
                st.caption("No round-trips in this window. The z-score never crossed entry.")
            else:
                view = blotter.copy()
                view["entry"] = pd.to_datetime(view["entry"]).dt.strftime("%Y-%m-%d")
                view["exit"] = pd.to_datetime(view["exit"]).dt.strftime("%Y-%m-%d")
                view["entry_z"] = view["entry_z"].map(lambda x: "—" if pd.isna(x) else f"{x:.2f}")
                view["exit_z"] = view["exit_z"].map(lambda x: "—" if pd.isna(x) else f"{x:.2f}")
                view["half_life_mult"] = view["half_life_mult"].map(lambda x: "—" if pd.isna(x) else f"{x:.1f}×")
                view["pnl"] = view["pnl"].map(lambda x: f"${x:,.0f}")
                view = view.rename(
                    columns={
                        "side": "Side",
                        "entry": "Entry",
                        "exit": "Cover",
                        "days": "Days",
                        "entry_z": "Entry z",
                        "exit_z": "Exit z",
                        "half_life_mult": "vs HL",
                        "pnl": "PnL",
                        "reason": "How",
                    }
                )
                st.dataframe(view, use_container_width=True, hide_index=True)
                st.caption(
                    "vs HL is holding time divided by the spread half-life. Stops are relationship failures, not noise."
                )
    else:
        st.markdown(
            '<div class="empty-panel">Load a pair from Universe, then run the book. Desk shows tether health, dollar PnL, and whether exits arrived on the half-life clock.</div>',
            unsafe_allow_html=True,
        )
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Ticket", pair_name)
        c2.metric("Sleeve", benchmark)
        c3.metric("Hedge", hedge_method)
        c4.metric("Capital", f"${capital:,.0f}")

with main_tabs[2]:
    if show_results:
        chart_a = result["asset_a"]
        chart_b = result["asset_b"]
        c1, c2 = st.columns([1.4, 1])
        with c1:
            st.plotly_chart(
                build_price_chart(result["log_prices"], chart_a, chart_b, result["positions"]), use_container_width=True
            )
            st.plotly_chart(build_spread_chart(result["spread"], result["spread_mean"]), use_container_width=True)
            st.plotly_chart(
                build_returns_chart(
                    result["strategy_returns"],
                    result["benchmark_cumulative"],
                    result["pair_buy_hold_cumulative"],
                    result["benchmark"],
                ),
                use_container_width=True,
            )
        with c2:
            st.plotly_chart(build_zscore_chart(result["zscore"], entry_z, exit_z, stop_loss), use_container_width=True)
            st.markdown(
                f"**{chart_a}/{chart_b}** · {result['sizing_method']} · lookback {lookback}d · entry {entry_z:g} / exit {exit_z:g} / stop {stop_loss:g}"
            )
            st.caption("Mint triangles long the spread. Rose triangles short it. Marks sit on asset A.")
    else:
        st.markdown(
            '<div class="empty-panel">Run the book to see spread, z-score bands, and the PnL tape.</div>',
            unsafe_allow_html=True,
        )

with main_tabs[3]:
    context = st.session_state.get("book_context")
    if not show_results or not context:
        st.markdown(
            '<div class="empty-panel">Research is grounded in the last book. Run a ticket first, then ask why the tether scored the way it did.</div>',
            unsafe_allow_html=True,
        )
    else:
        st.caption(
            f"Grounded on {context['pair']} · {context['metrics'].get('tether_state')} · "
            f"score {context['metrics'].get('tether_score')} · {len(context.get('blotter') or [])} round-trips. "
            "Groq only sees this book JSON, not the open web."
        )
        for message in st.session_state.ai_messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])
        if prompt := st.chat_input("Ask about this book: tether, half-life, blotter, PnL, stops..."):
            st.session_state.ai_messages.append({"role": "user", "content": prompt})
            with st.spinner("Groq is reading the active book context..."):
                llm = call_llm(st.session_state.ai_messages, context, task="research")
            if llm.ok:
                st.session_state.ai_messages.append({"role": "assistant", "content": llm.content})
            else:
                st.session_state.ai_messages.append(
                    {"role": "assistant", "content": llm.error or "Groq request failed."}
                )
            st.rerun()
        if st.button("Clear research"):
            st.session_state.ai_messages = []
            st.rerun()
