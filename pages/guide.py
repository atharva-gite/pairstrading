"""Recruiter-facing guide. Opened from the Help button on the desk."""

from pathlib import Path

import streamlit as st

ASSETS = Path(__file__).resolve().parents[1] / "docs" / "assets" / "guide"

st.set_page_config(page_title="How TETHER works", layout="wide", page_icon="⚓")
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@600;700&family=Outfit:wght@400;500;600&display=swap');
    :root { --bg:#0c0f0d; --text:#e8e4d6; --muted:#9aa394; --copper:#c4a35a; --line:#2a332c; }
    html, body, .stApp { background: var(--bg); color: var(--text); font-family: Outfit, system-ui, sans-serif; }
    h1, h2, h3 { font-family: Fraunces, Georgia, serif; }
    [data-testid="stSidebarNav"], #MainMenu, footer { visibility: hidden; height: 0; }
    .lede { color: var(--muted); font-size: 1.05rem; max-width: 760px; }
    </style>
    """,
    unsafe_allow_html=True,
)

if st.button("← Back to the desk"):
    st.switch_page("app.py")

st.title("How TETHER works")
st.markdown(
    '<p class="lede">A short tour for someone who has never traded a pair. '
    "The pictures are a real Coca-Cola / PepsiCo run from the app, not a mock-up.</p>",
    unsafe_allow_html=True,
)

st.header("What the app is doing")
st.markdown(
    """
Most investing apps ask one question: will this stock go up? TETHER asks a different one.

Coca-Cola and PepsiCo sell the same kind of product to the same kind of customer. Their share prices
tend to stay near each other for years. The app does not bet on either company. It watches the **gap**
between them. When that gap gets unusually wide, it buys the one that looks cheap and sells short the one
that looks expensive, and waits for the gap to close.

Because one position profits when its stock rises and the other profits when its stock falls, a rising
or falling stock market should mostly cancel out. That is what "market-neutral" means here.

The difficult part is knowing the two prices are **still** tied together. A link that used to exist can
break (a company changes its business, or the old pattern was a coincidence). TETHER's job is to test
that link, and to refuse to trade it when the test fails.
    """
)

st.header("The four screens")
st.markdown(
    """
**Universe** searches a whole sector (energy, banks, consumer staples) and ranks every pair. A high score
means the prices still look tied together after a correction for the fact that many pairs were tested at once.
That correction is called FDR. Without it, some pairs would look special purely by chance.

**Desk** is the verdict for the pair you chose. It shows whether the link is healthy, how much money the
book made or lost, and whether that result survived an honesty check called a walk-forward test: the app
learns the relationship on older data, freezes the rules, and only then "trades" the later data. If the
later period is worse, that worse number stays on screen.

**Tape** is the pictures below: the two prices, the gap, how stretched the gap is, and the profit chart.

**Research** is a conversation about the book you just ran. The model is only allowed to use that book's
numbers. It cannot browse the web or invent a profit figure. The **AI Report** at the top of Universe and
Tape is the same idea in a written paragraph.
    """
)

st.header("Reading the charts")
st.markdown("These four pictures are Coca-Cola against PepsiCo, January 2019 through the latest saved prices.")

st.image(str(ASSETS / "prices.png"), width="stretch")
st.markdown(
    """
**The two prices.** Both lines are scaled so they start at 1, which makes the gap easier to see than raw
dollar prices would. The coral dots are days a trade was opened. A trade here is never "buy Coca-Cola."
It is "buy one and short the other," so the book is betting the gap shrinks, whichever direction that takes.
    """
)

st.image(str(ASSETS / "spread.png"), width="stretch")
st.markdown(
    """
**The gap.** This is the thing being traded. The white line is how far apart the two prices are, after a
hedge ratio has scaled them so one share of the expensive stock is comparable to several shares of the
cheaper one. The gold line is the gap's recent average. Trades happen when the white line wanders far
from the gold line and are closed when it comes back.
    """
)

st.image(str(ASSETS / "zscore.png"), width="stretch")
st.markdown(
    """
**How stretched the gap is.** A z-score says how unusual today's gap is, measured in "normal wiggles."
A reading of +2 means the gap is about two normal wiggles above its average. The dashed lines are the rules:

- **Green (±2).** Open a trade. The gap is stretched enough to be interesting.
- **Gold (±0.5).** Close the trade. The gap has come back close to normal.
- **Coral (±3).** Cut the trade even at a loss. The gap kept widening, which can mean the relationship broke
  rather than stretched. That loss is the app telling you the idea failed, and it is supposed to be visible.
    """
)

st.image(str(ASSETS / "returns.png"), width="stretch")
st.markdown(
    """
**The result.** Green is the pair book. Gold is what you would have earned by simply holding both stocks.
Grey is the S&P 500, the stand-in for "the market." A good pairs book does not need to beat the stock market
in a bull run. It needs to make its money from the gap, with little dependence on whether the market rose.
On this Coca-Cola / PepsiCo example the book lost money. That is shown on purpose: a research tool that
hides its losing examples is not useful.
    """
)

st.header("Words you will see on the desk")
st.markdown(
    """
**Tether score (0–100).** One number for "how convincing is this link," blending the statistical test,
how fast the gap tends to close, and how stable the hedge is. **TETHERED** means tradable. **WEAK** means
fraying. **BROKEN** means do not trade it as a pair. Apple and Microsoft are the built-in example of BROKEN:
they rise together, and they are not a pair you can trade this way.

**Cointegration.** The formal test for "these two prices share a long-run level they keep returning to."
Two stocks can be highly correlated (they move on the same days) without being cointegrated (the gap between
them comes back). Correlation is not the trade.

**Half-life.** How many days the gap typically takes to close halfway. If that is longer than about 60 trading
days, the app will not treat the pair as tradable: you would be waiting too long for the idea to work.

**Sharpe ratio.** Profit divided by how bumpy the path was. Higher means a smoother result, not a bigger one.

**Drawdown.** The worst peak-to-trough loss along the way.

**Net beta.** How much the book still moves with the stock market after the long and the short are combined.
Near zero is the goal.

**Costs.** Every trade is charged a small commission and slippage, and short positions can be charged a
borrow fee. The Desk also reruns the same trades at higher costs, so a result that only works with free
trading is obvious.
    """
)

st.header("What the AI is, and is not")
st.markdown(
    """
The AI Report, the desk briefing, and Research all receive a packet of numbers from the app (scores, dates,
trade results) and are instructed to use nothing else. If no API key is configured, the Report falls back to
a paragraph assembled from those same numbers by fixed rules, and it says so.


This is a research tool. It is not investment advice, and a green chart is not a recommendation.
    """
)
