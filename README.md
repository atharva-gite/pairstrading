# TETHER

Market-neutral pairs desk. Screen a sector, score whether the cointegrating tether is still holding, then backtest a dollar-book long/short.

- `app.py` - RelVal desk UI: universe heatmap, tether health, trade blotter, tape.
- `src/logic.py` - Cointegration, Tether Score, rolling p-values, hedge ratios, signals, costs.
- `requirements.txt` - Python dependencies.
- `tests/test_logic.py` - Unit tests for the core quant logic.

## Installation

1. Create a Python virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

## Run the App

Start the Streamlit app from the repository root:

```bash
streamlit run app.py
```

Open the URL shown in the terminal to use the web dashboard.

## Usage

- Select Asset A and Asset B from the sidebar, or load a pair from the Pair Screener tab.
- Choose a sector universe (Energy, Consumer Staples, Technology, Financials, Healthcare) and screen every pair with Engle-Granger.
- Choose a benchmark such as `SPY` or `XLK`.
- Set the dollar book and hedge-ratio vs equal-dollar sizing.
- Adjust lookback period, entry/exit z-score, and stop-loss values.
- Run the backtest to view metrics, cointegration status, spread/z-score charts, trade markers, and PnL vs buy-and-hold.

## Notes

- The app calculates all spread and return values using log prices.
- Hedge ratio methods include `Static OLS`, `Rolling OLS`, and `Kalman Filter`.
- Hedge-ratio sizing splits gross capital as $|A| + |\beta| \times |A| = \text{capital}$. Equal-dollar puts the same notional on each leg.
- Trade execution charges 0.1% slippage and 0.05% commission on traded dollar notional.
