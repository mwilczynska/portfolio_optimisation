# Portfolio Optimisation

A brute-force portfolio grid search over long-run daily asset histories. It tests every weight
combination at a fixed increment, backtests each one from 1970 to the present, and writes a ranked
results table plus a self-contained interactive chart you can open in any browser.

The point is to see the whole trade-off surface at once rather than a single "optimal" answer.
Every blend is a dot, so you can look at what growth actually costs in drawdown, where the
diminishing returns start, and how wide the band of near-equivalent portfolios really is.

```
pip install pandas numpy
python portfolio_optimizer.py
```

Outputs land in `output/`:

| File | What it is |
|---|---|
| `portfolio_optimisation_results.csv` | Every portfolio tested, ranked, with all metrics |
| `portfolio_optimisation_results.html` | Interactive chart and table, no dependencies, no server needed |

A generated example of both is committed to this repo. Download the HTML and open it locally;
GitHub will not render it in the browser.

## Data

Price data comes from the companion repository
[mwilczynska/financial_datasets](https://github.com/mwilczynska/financial_datasets). Every series
there runs from 1970 to the present. No real fund goes back that far, so only the most recent
stretch of each series is a fund's own returns; before that it is reconstructed from published index
data and documented models, with the method and its limits written up per asset. Check the repo out
**as a sibling directory**:

```
your-folder/
├── financial_datasets/
└── portfolio_optimisation/
```

`DATA_ROOT` in `portfolio_optimizer.py` resolves to `../financial_datasets/data/processed`. Point
it somewhere else if your layout differs.

Two asset sets are wired up out of the box. The default is un-levered:

| Code | What it is | Series covers |
|---|---|---|
| `GLSTOCK` | All-world equities in USD, dividends reinvested (VT-like) | 1970 on, VT's own returns from 2008 |
| `GLBOND` | Unhedged global government bonds in USD (BND/BWX blend) | 1970 on, the fund blend from 2007 |
| `GOLDPM` | Gold bullion as a total return (GLD-like) | 1970 on, GLD's own returns from 2004 |
| `COMM` | Broad commodity futures with roll yield and collateral (DBC-like) | 1970 on, DBC's own returns from 2006 |

The alternative, commented out in the config block, is a leveraged set: `USLCAP3x`, `LTT3x`,
`ITT3x`, `GOLDPM2x` and `COMM`, modelling daily-reset funds like UPRO, TMF, TYD and UGL with their
financing costs, fees and volatility decay included. Those series also start in 1970 and switch to
the real funds when they launched, around 2008 to 2009. The generated page adapts its explanations
and warnings to whichever set you actually ran.

## Configuration

Everything editable sits at the top of `portfolio_optimizer.py`:

| Setting | Purpose |
|---|---|
| `asset_list` | Which assets to blend |
| `weight_step` | Grid increment. `0.05` for the full 5% run, `0.20` for a quick test |
| `min_weights` / `max_weights` | Per-asset floors and ceilings |
| `max_drawdown_limit` | Drop portfolios worse than this, or `None` to keep everything |
| `rank_by` / `rank_ascending` | Which metric sorts the table |
| `rebalance_annually` | Annual rebalance, or buy and hold |
| `starting_value` | Opening pot for every portfolio |
| `risk_free_rate_annual` | Used by Sharpe and Sortino |

Grid size grows fast: four assets at 5% steps is 1,771 portfolios, five assets is 10,626.

## Metrics

Final value, CAGR, annual return, standard deviation, TWRR, Sharpe, Sortino, Calmar, max drawdown,
Ulcer Index, average drawdown, longest drawdown in days, daily win rate, best and worst day, month
and year, 95% VaR and CVaR, skew, and excess kurtosis.

Each row also carries an `Existing Script Syntax` column, a ready-to-paste
`tickers, weights = [...],[...]` line for use in other backtesting scripts.

## The interactive page

Written as plain HTML, SVG and JavaScript with no libraries and no build step, so it works from a
local file or any static host.

- Pick what the X and Y axes measure from any numeric column.
- Hover a dot for that portfolio's mix and statistics. Click to pin it, click again to unpin.
- Highlight every dot past a threshold you choose on the current X axis.
- Sort the table by any column, search the weights, or stack numeric filters such as
  "Max Drawdown at most 60%". Loss-type columns are magnitude-aware, so that filter keeps the
  shallower drawdowns rather than matching everything.
- Chart and table are linked. Pinning a dot scrolls to and highlights its row, and vice versa.

The page also carries a plain-language explainer of what it shows, how each data series was built,
and what to be sceptical about. [`ABOUT.md`](ABOUT.md) is a standalone copy of that write-up.

## Caveats

This is a historical simulation, not advice, and past results do not predict future returns.
Hindsight is baked into any ranking of backtests: the top row is the blend most fitted to this
particular history. Taxes, commissions, bid/ask spreads and rebalancing friction are not modelled.
The early decades of every series are reconstructions rather than observed history, since no real
fund reaches back to 1970, and [`ABOUT.md`](ABOUT.md) spells out which parts, for which asset.

## Licence

Code is MIT, see [`LICENSE`](LICENSE).

The datasets are not covered by that licence. Some upstream sources carry their own redistribution
or non-commercial terms, notably the S&P GSCI republication behind the early commodity history and
the Jorda-Schularick-Taylor macrohistory database behind the early bond history. See
`DATA_LICENSE.md` in
[mwilczynska/financial_datasets](https://github.com/mwilczynska/financial_datasets) before reusing
the data itself.
