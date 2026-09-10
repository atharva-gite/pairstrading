# KO/PEP vs AAPL/MSFT: when correlation is not a book

**TETHER RelVal desk · walk-forward research note**

This note is the recruiter-facing record of what the desk actually tests. The live app screens a sector, FDR-corrects the p-values, freezes the hedge on a discovery sample, and only then trades. AAPL/MSFT is the negative control. KO/PEP is the classic staples tether. Correlation is not the trade.

## 1. What the desk is allowed to know

Walk-forward split: **discover on or before 2024-01-01, trade after**. Static OLS hedge, Engle–Granger p-value, half-life, and Tether Score are estimated on the discovery window only. Rolling z-scores use a backward lookback. Kalman / rolling OLS, when selected, are causal filters — they do not peek at next week’s price.

That is the difference between a pairs tutorial and a book. In-sample Sharpe is the in-sample story. Out-of-sample Sharpe is the live story. If live is worse, the Desk leaves it on the screen.

## 2. Multiple testing

A five-name sector has ten unique pairs. Ten Engle–Granger tests at 5% will produce false “cointegrated” hits. TETHER reports:

- raw Engle–Granger p
- **Benjamini–Hochberg FDR**
- Bonferroni

The sentence the desk prints is the one that matters: *N of K pairs survive FDR. D of T TETHERED names die live after the split.*

## 3. Negative control: AAPL / MSFT

AAPL and MSFT share a tech factor. They often **move together**. That is not a mean-reverting spread.

On a recent full-sample run the desk labeled the pair **BROKEN** (Tether Score ~25) with a ~**114-day** Ornstein–Uhlenbeck half-life. The book wants **8–40 days**. A 114-day clock means you wait a third of a year for half the gap to close. “On-time exits” against 1.5× that clock (~171 days) are not evidence the pair works — almost any round-trip looks punctual.

Rolling Engle–Granger p-values that sit above 0.05 are the tether decaying in public. Treat the residual as **two directional bets**, not relative value.

## 4. Working hypothesis: KO / PEP

Coca-Cola and PepsiCo are the textbook consumer-staples pair: similar cost bases, overlapping distribution, no mega-cap tech beta. The discovery sample is allowed to estimate β and the half-life. Live (2024–) is not allowed to rewrite that hedge.

Read the Desk table:

| Sample | What it is allowed to do |
|---|---|
| In-sample | Estimate cointegration, freeze β, score the tether |
| Out-of-sample | Trade z-score rules with that frozen hedge; re-score the tether |

If KO/PEP’s OOS Sharpe is below IS Sharpe, **that result stays on the Desk**. A pretty equity curve that cannot survive 2024 is not a book.

One blotter line is enough to explain the clock: side, entry z, days held, holding time vs half-life, dollar PnL, exit reason (revert vs stop). Stops are relationship failures, not noise.

## 5. Residual risk the fund actually charges you for

**Borrow.** Short inventory is not free. The toggle charges an annualized borrow rate (default 50 bps) on short dollar notional. A pairs PnL that only works with zero borrow is not a pairs PnL.

**Net book beta.** Leg betas vs SPY are not the book. Dollar holdings times those betas, divided by capital, is the **net market β of the ticket**. When Kalman or rolling β moves, notionals move, and net β drifts. That is the residual a stat-arb PM will ask about. Plot it. If |β| while on is not small, you are sneaking index risk into a “market-neutral” line.

## 6. How to read TETHER in 40 seconds

1. **Universe** — Consumer Staples or Energy. Scan. Count FDR survivors. Load KO/PEP, not AAPL/MSFT.
2. **Desk** — Walk-forward table. If OOS Sharpe is worse, leave it. Ask Research: *why is this BROKEN?*
3. **Tape** — Spread vs equilibrium, z-bands, trade marks, PnL vs pair buy-and-hold.

The engine is in `src/logic.py`. This note is what you send instead of the file.

## 7. What would change the read

- A structural break (product mix, merger, index reconstitution) that the rolling p-value has not yet printed.
- Borrow spikes that the 50 bp assumption understates.
- A Kalman hedge that lets net β wander through a factor crisis.

TETHER’s job is not to promise that pairs always pay. It is to show **when the tether is still holding, and when you are just long tech vs long tech**.
