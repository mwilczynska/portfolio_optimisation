# About this portfolio chart

> **Not investment advice.** This page is an educational illustration built from historical
> simulations. It is not a recommendation to buy or sell anything, and past results do not
> predict future returns.

## What it shows

Every dot on the chart is one **portfolio**: a fixed recipe for splitting a pot of money across
four building-block investments. The page tested **every blend in 5% steps that adds up to 100%**
and plots how each one would have behaved over the full backtest period. You choose what the two
axes measure, so you can compare growth (CAGR) against worst-case loss (max drawdown),
risk-adjusted return (Sharpe) against anything else, and so on.

The default building blocks are an un-levered "all weather" set:

| Code | What it is | Series covers |
|---|---|---|
| **GLSTOCK** | All-world equities in USD with dividends reinvested, similar to VT or MSCI ACWI | 1970 to today, with VT's own returns from 2008-06-27 |
| **GLBOND** | Bonds issued by governments around the world, held in USD with currency swings left in rather than hedged away | 1970 to today, with a BND / BWX fund blend from 2007-10-12 |
| **GOLDPM** | Bullion measured as a total return that already carries fund fees, similar to GLD | 1970 to today, with GLD's own returns from 2004-11-18 |
| **COMM** | A diversified basket of commodity futures spanning energy, metals and agriculture, including roll yield and Treasury-bill collateral, similar to DBC | 1970 to today, with DBC's own returns from 2006-02-07 |

The script can also run a leveraged set (`USLCAP3x`, `LTT3x`, `ITT3x`, `GOLDPM2x`, `COMM`), which
models 2x and 3x daily-reset funds such as UPRO, TMF, TYD and UGL. Those series also run from 1970,
switching to the real funds when they launched around 2008 to 2009. When any of them are in the mix,
the generated page adds the leverage-specific explanation and warnings automatically.

## How to use it

- **Axis dropdowns** pick what the X and Y axes measure, from any numeric column in the results.
- **Hover** a dot to see that portfolio's exact mix and headline numbers. **Click** it to pin the
  label so it stays put, and click again to unpin. A pinned dot turns purple and moves to the front.
- **Highlight dropdown** colours every dot past a threshold you choose, which makes it easy to see,
  say, which blends fell further than 60% at their worst.
- **The table** lists every portfolio tested. Click a column heading to sort by it, type in the
  filter box to search the weights, or build precise metric filters such as "Max Drawdown at most
  60%" or "Sharpe at least 0.5". Filters stack, and a counter shows how many portfolios match.
  Clicking a table row pins its dot on the chart, and clicking a dot highlights its row.

## How the numbers were worked out

This is a **backtest**: a simulation of how each blend would have performed had it existed, run on
daily price history going back to 1970. Every portfolio starts with the same amount of money
($100,000 by default) and is **rebalanced back to its target mix once a year**. Daily returns are
compounded to trace each portfolio's value over time, and every statistic in the table is measured
from that path.

Every building block runs from 1970 to today, but only the most recent stretch of each one is a real
fund. Before that the series is extended backwards using published index data and, where no index
reaches far enough, a documented model. So the recent decades are observed history and the earlier
decades are reconstructions.

## Where each building block comes from

Every series runs from 1970, but no real fund goes back that far. The further back you go, the more
of the data is reconstructed rather than observed.

- **GLSTOCK.** From 1970 to 1989 the series is a U.S. large-cap daily path, rescaled so that each
  calendar year matches MSCI World's published annual return. The first half of 1990 stays a plain
  U.S. proxy, Fama-French developed-market daily returns then run to mid-2008, and VT's own returns
  take over from 2008-06-27.
- **GLBOND.** Everything up to 2007 is reconstructed. The Jorda-Schularick-Taylor macrohistory
  database sets each year's return for a GDP-weighted basket of 16 advanced-economy government bond
  markets, and the path within each year comes from BIS daily exchange rates plus government bond
  yields: daily yields for the U.S., Japan and the U.K., together about 62% of the basket from 1979,
  and monthly yields for everywhere else. A 45% BND / 55% BWX blend, rebalanced daily, takes over
  from 2007-10-12.
- **GOLDPM.** From 1970 until GLD launched, the series is the London afternoon gold price less GLD's
  0.40% a year expense drag, so the whole history is priced the way someone holding the fund would
  have experienced it. GLD's own returns take over from 2004-11-18.
- **COMM.** From 1970 to 1991 the series is anchored to S&P GSCI total-return data, interpolated
  between roughly bi-monthly readings until 1984 and then laid over daily GSCI spot movements. The
  Bloomberg Commodity excess-return index plus Treasury-bill collateral covers 1991 to 2006, and
  DBC's own returns take over from 2006-02-07.

Full methodology for every series, including the calibration checks against the real funds, is
documented in the companion repository:
[mwilczynska/financial_datasets](https://github.com/mwilczynska/financial_datasets).

## What the headline numbers mean

| Metric | Plain-language meaning |
|---|---|
| **CAGR** | Compound annual growth rate: the steady yearly rate that would turn the start value into the end value. |
| **Max Drawdown** | The worst peak-to-trough fall along the way. A value of 70% means it once lost 70% from a high. |
| **Sharpe** | Return per unit of overall volatility. Higher means a smoother ride for the return achieved. |
| **Sortino** | Like Sharpe, but it only counts downside volatility against you. |
| **Calmar** | Growth (CAGR) divided by the worst drawdown. |
| **Ulcer Index** | How deep drawdowns were and how long they lasted. Lower is calmer. |
| **Std Dev** | Annualised volatility of returns. |
| **TWRR** | Time-weighted rate of return. With no contributions or withdrawals it equals CAGR. |
| **Final Value** | What the starting pot grew to by the end of the period. |

## What to watch out for

- **This is not investment advice**, and **past results do not predict future returns.**
- **Hindsight is baked in.** Picking the best-scoring dot means picking whatever happened to suit
  the past, using information nobody had at the time. The top of the ranking is the blend most
  fitted to this particular history.
- **The early decades are reconstructed, not observed.** No real fund covers this period, so the
  further back the backtest runs, the more it rests on models. See the section above for which parts,
  for which asset.
- **Real-world costs are missing.** Taxes, commissions, bid/ask spreads and the practical friction
  of rebalancing are not modelled. Fees charged inside the funds themselves are included.
- **`COMM` is the patchiest series.** It changes benchmark twice, in 1991 and again in 2006, so it
  should not be read as one continuously observed index running back to 1970.
- **`GLBOND` covers advanced-economy government bonds only** in its reconstructed era, with no
  corporate, emerging-market or inflation-linked debt.
- **When the leveraged set is used**, the 2x and 3x blocks can fall very fast. Many blends show
  drawdowns worse than 80%, which few people could hold through in real time.

## Reproducing the data

The chart and the results table are generated by `portfolio_optimizer.py`, which loads local
processed daily price datasets, simulates every weight combination, ranks them, and writes both
`output/portfolio_optimisation_results.csv` and the interactive page
`output/portfolio_optimisation_results.html`. The editable settings, covering assets, weight step,
weight limits, starting value, rebalancing and the ranking metric, live at the top of that script.
