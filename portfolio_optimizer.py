"""
Portfolio optimisation grid search
----------------------------------
Tests all 5% increment portfolios for:
USLCAP3x, LTT3x, ITT3x, GOLDPM2x, COMM

Outputs:
    output/portfolio_optimisation_results.csv
    output/portfolio_optimisation_results.html

Edit the USER CONFIG section below, then run:
    python portfolio_optimizer.py
"""

import itertools
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------
# 1) User configurable inputs
# --------------------------------------------------------------------------
starting_value = 100000
start_date, end_date = "1970-01-01", "2026-12-31"
rebalance_annually = True

# asset_list = ["USLCAP3x", "LTT3x", "ITT3x", "GOLDPM2x", "COMM"]

# Un-levered All Weather asset list
asset_list = ["GLSTOCK", "GLBOND", "GOLDPM", "COMM"]
# weight_step = 0.20  # quick test grid
weight_step = 0.05  # full grid; uncomment for the 5% increment run

# Optional minimum floors and maximum roofs for each asset.
# Values are decimals: 0.10 means 10%. Leave at 0.00 / 1.00 for unrestricted.
# min_weights = {
#     "USLCAP3x": 0.00,
#     "LTT3x": 0.00,
#     "ITT3x": 0.00,
#     "GOLDPM2x": 0.00,
#     "COMM": 0.00,
# }

# max_weights = {
#     "USLCAP3x": 1.00,
#     "LTT3x": 1.00,
#     "ITT3x": 1.00,
#     "GOLDPM2x": 1.00,
#     "COMM": 1.00,
# }

# Un-levered All Weather assets
min_weights = {
    "GOLDPM": 0.00,
    "COMM": 0.00,
    "GLSTOCK": 0.00,
    "GLBOND": 0.00,
}

max_weights = {
    "GOLDPM": 1.00,
    "COMM": 1.00,
    "GLSTOCK": 1.00,
    "GLBOND": 1.00,
}

# Optional drawdown filter. Example: -0.60 keeps portfolios with max drawdown
# no worse than -60%. Set to None to keep all portfolios.
max_drawdown_limit = None

# Ranking column and direction. Examples: "CAGR", "Sharpe", "Max Drawdown".
rank_by = "CAGR"
rank_ascending = False

# Risk-free rate used for Sharpe/Sortino. Keep at 0.0 for pure portfolio ranking.
risk_free_rate_annual = 0.0

output_dir = Path("output")
portfolio_chunk_size = 500


# --------------------------------------------------------------------------
# 2) Local dataset configuration
# --------------------------------------------------------------------------
DATA_ROOT = Path(__file__).resolve().parent.parent / "financial_datasets" / "data" / "processed"

CUSTOM_DATASETS = {
    "USLCAP3x": {
        "path": DATA_ROOT / "us_large_cap_3x_sp500.csv",
        "date_col": "Date",
        "price_col": "Adj Close",
    },
    "LTT3x": {
        "path": DATA_ROOT / "long_term_us_treasury_3x.csv",
        "date_col": "Date",
        "price_col": "Adj Close",
    },
    "ITT3x": {
        "path": DATA_ROOT / "intermediate_term_us_treasury_3x.csv",
        "date_col": "Date",
        "price_col": "Adj Close",
    },
    "GOLDPM2x": {
        "path": DATA_ROOT / "gold_2x.csv",
        "date_col": "Date",
        "price_col": "Adj Close",
    },
    "GOLDPM": {
        "path": DATA_ROOT / "gold.csv",
        "date_col": "Date",
        "price_col": "Adj Close",
    },
    "COMM": {
        "path": DATA_ROOT / "broad_commodities.csv",
        "date_col": "Date",
        "price_col": "Adj Close",
    },
    "GLSTOCK": {
        "path": DATA_ROOT / "global_stocks.csv",
        "date_col": "Date",
        "price_col": "Adj Close",
    },
    "GLBOND": {
        "path": DATA_ROOT / "global_bonds.csv",
        "date_col": "Date",
        "price_col": "Adj Close",
    },
}


def load_custom_price_series(ticker, start, end):
    """Load a local dataset as a one-column close-price frame."""
    config = CUSTOM_DATASETS[ticker]
    path = config["path"]
    if not path.exists():
        raise FileNotFoundError(f"Custom dataset for {ticker} not found: {path}")

    df = pd.read_csv(path, usecols=[config["date_col"], config["price_col"]])
    df[config["date_col"]] = pd.to_datetime(df[config["date_col"]], errors="raise").dt.normalize()
    df[config["price_col"]] = pd.to_numeric(df[config["price_col"]], errors="coerce")
    df = df.dropna(subset=[config["price_col"]]).drop_duplicates(subset=[config["date_col"]], keep="last")
    df = df.set_index(config["date_col"]).sort_index()

    start_ts = pd.to_datetime(start).normalize()
    end_ts = min(pd.to_datetime(end).normalize(), pd.Timestamp.today().normalize())
    series = df.loc[(df.index >= start_ts) & (df.index <= end_ts), config["price_col"]].rename(ticker)
    if series.empty:
        raise ValueError(f"Custom dataset {ticker} has no rows between {start_ts.date()} and {end_ts.date()}")
    return series.to_frame()


def load_prices(tickers, start, end):
    frames = [load_custom_price_series(ticker, start, end) for ticker in tickers]
    prices = pd.concat(frames, axis=1, sort=False).sort_index()
    prices = prices.dropna(how="any")
    if len(prices) < 2:
        raise ValueError("Not enough overlapping price history after dropping missing values.")
    return prices


def generate_weight_grid(tickers, step, floors, roofs):
    """Generate all long-only weights that sum to 100%, filtered by floors/roofs."""
    scale = int(round(1 / step))
    floor_units = [int(round(floors[ticker] * scale)) for ticker in tickers]
    roof_units = [int(round(roofs[ticker] * scale)) for ticker in tickers]

    if sum(floor_units) > scale:
        raise ValueError("Minimum weight floors add to more than 100%.")
    if sum(roof_units) < scale:
        raise ValueError("Maximum weight roofs add to less than 100%.")

    rows = []
    for combo in itertools.product(range(scale + 1), repeat=len(tickers) - 1):
        last = scale - sum(combo)
        if last < 0:
            continue
        units = list(combo) + [last]
        if all(floor_units[i] <= units[i] <= roof_units[i] for i in range(len(tickers))):
            rows.append([unit / scale for unit in units])

    if not rows:
        raise ValueError("No portfolios remain after applying min/max weight constraints.")
    return np.asarray(rows, dtype=float)


def annual_rebalanced_values(returns, weights, initial_value):
    """Build daily portfolio values for many portfolios, resetting weights each calendar year."""
    dates = returns.index
    asset_returns = returns.to_numpy(dtype=float)
    portfolio_count = weights.shape[0]
    out = np.empty((len(dates), portfolio_count), dtype=float)
    start_values = np.full(portfolio_count, float(initial_value), dtype=float)

    years = pd.Index(dates.year)
    for year in years.unique():
        idx = np.flatnonzero(years == year)
        block = asset_returns[idx]
        growth = np.cumprod(1.0 + block, axis=0)
        weighted_growth = growth @ weights.T
        out[idx] = weighted_growth * start_values
        start_values = out[idx[-1]]

    return out


def buy_and_hold_values(returns, weights, initial_value):
    asset_growth = np.cumprod(1.0 + returns.to_numpy(dtype=float), axis=0)
    return (asset_growth @ weights.T) * float(initial_value)


def longest_underwater_days(values, dates):
    peak = np.maximum.accumulate(values, axis=0)
    underwater = values < peak
    longest = np.zeros(values.shape[1], dtype=int)

    for col in range(values.shape[1]):
        max_run = 0
        run_start = None
        for row, is_underwater in enumerate(underwater[:, col]):
            if is_underwater and run_start is None:
                run_start = row
            elif not is_underwater and run_start is not None:
                max_run = max(max_run, int((dates[row - 1] - dates[run_start]).days))
                run_start = None
        if run_start is not None:
            max_run = max(max_run, int((dates[-1] - dates[run_start]).days))
        longest[col] = max_run

    return longest


def summarise_portfolios(values, weights, tickers, dates, rf_annual):
    daily_returns = values[1:] / values[:-1] - 1.0
    n_years = (dates[-1] - dates[0]).days / 365.25
    rf_daily = (1 + rf_annual) ** (1 / 252) - 1

    final_value = values[-1]
    cagr = (final_value / values[0]) ** (1 / n_years) - 1
    annual_return = np.nanmean(daily_returns, axis=0) * 252
    std_dev = np.nanstd(daily_returns, axis=0, ddof=1) * math.sqrt(252)
    downside = np.where(daily_returns < rf_daily, daily_returns - rf_daily, 0.0)
    downside_dev = np.sqrt(np.nanmean(downside ** 2, axis=0)) * math.sqrt(252)

    excess_cagr = cagr - rf_annual
    sharpe = np.divide(excess_cagr, std_dev, out=np.full_like(cagr, np.nan), where=std_dev != 0)
    sortino = np.divide(excess_cagr, downside_dev, out=np.full_like(cagr, np.nan), where=downside_dev != 0)

    peak = np.maximum.accumulate(values, axis=0)
    drawdown = values / peak - 1.0
    max_drawdown = np.nanmin(drawdown, axis=0)
    ulcer_index = np.sqrt(np.nanmean(np.square(np.minimum(drawdown, 0.0) * 100.0), axis=0))
    avg_drawdown = np.nanmean(np.minimum(drawdown, 0.0), axis=0)
    calmar = np.divide(cagr, np.abs(max_drawdown), out=np.full_like(cagr, np.nan), where=max_drawdown != 0)

    centered = daily_returns - np.nanmean(daily_returns, axis=0)
    daily_std = np.nanstd(daily_returns, axis=0, ddof=1)
    skew = np.nanmean(centered ** 3, axis=0) / np.where(daily_std == 0, np.nan, daily_std ** 3)
    kurtosis = np.nanmean(centered ** 4, axis=0) / np.where(daily_std == 0, np.nan, daily_std ** 4) - 3.0

    var_95 = np.nanpercentile(daily_returns, 5, axis=0)
    cvar_95 = np.array([
        np.nanmean(daily_returns[:, i][daily_returns[:, i] <= var_95[i]])
        for i in range(daily_returns.shape[1])
    ])

    daily_win_rate = np.nanmean(daily_returns > 0, axis=0)
    best_day = np.nanmax(daily_returns, axis=0)
    worst_day = np.nanmin(daily_returns, axis=0)

    monthly_values = pd.DataFrame(values, index=dates).resample("ME").last()
    monthly_returns = monthly_values.pct_change(fill_method=None).dropna().to_numpy(dtype=float)
    yearly_values = pd.DataFrame(values, index=dates).resample("YE").last()
    yearly_returns = yearly_values.pct_change(fill_method=None).dropna().to_numpy(dtype=float)

    results = pd.DataFrame(weights, columns=[f"{ticker} Weight" for ticker in tickers])
    results["Final Value"] = final_value
    results["CAGR"] = cagr
    results["Annual Return"] = annual_return
    results["Std Dev"] = std_dev
    results["TWRR"] = cagr
    results["Sharpe"] = sharpe
    results["Sortino"] = sortino
    results["Calmar"] = calmar
    results["Max Drawdown"] = max_drawdown
    results["Ulcer Index"] = ulcer_index
    results["Average Drawdown"] = avg_drawdown
    results["Longest Drawdown Days"] = longest_underwater_days(values, dates)
    results["Daily Win Rate"] = daily_win_rate
    results["Best Day"] = best_day
    results["Worst Day"] = worst_day
    results["Best Month"] = np.nanmax(monthly_returns, axis=0)
    results["Worst Month"] = np.nanmin(monthly_returns, axis=0)
    results["Best Year"] = np.nanmax(yearly_returns, axis=0)
    results["Worst Year"] = np.nanmin(yearly_returns, axis=0)
    results["VaR 95 Daily"] = var_95
    results["CVaR 95 Daily"] = cvar_95
    results["Skew"] = skew
    results["Excess Kurtosis"] = kurtosis
    return results


def format_weight_label(row, tickers):
    return ", ".join(f"{ticker}: {row[f'{ticker} Weight']:.0%}" for ticker in tickers)


def format_existing_script_syntax(row, tickers):
    tickers_text = "[" + ",".join(f"'{ticker}'" for ticker in tickers) + "]"
    weights_text = "[" + ",".join(f"{row[f'{ticker} Weight']:.2g}" for ticker in tickers) + "]"
    return f"tickers, weights = {tickers_text},{weights_text}"


# Plain-language descriptions of the building-block assets, used by the public
# "About" section. Facts are sourced from the dataset methodology docs in the
# companion repository: https://github.com/mwilczynska/financial_datasets
#
# Every series runs from 1970-01-02. Only the most recent stretch of each one is
# a real fund; everything earlier is reconstructed from index data and documented
# models. The copy below has to make that obvious, so ``coverage`` always states
# the full span alongside the fund handover date.
#
# Each entry carries:
#   name     - short human name
#   detail   - what the exposure actually is
#   coverage - the span the series covers, and when the real fund takes over
#   history  - how the pre-fund history was built, in chronological order
#   caveat   - the limitation a reader should know about
#   levered  - whether the series is a daily-reset leveraged model
#
# Unknown tickers degrade gracefully via ``asset_info`` below.
ASSET_INFO = {
    "GLSTOCK": {
        "name": "Global stocks",
        "detail": "all-world equities in USD with dividends reinvested, similar to VT or MSCI ACWI",
        "coverage": "1970 to today, with VT's own returns from 2008-06-27",
        "history": "From 1970 to 1989 the series is a U.S. large-cap daily path, rescaled so that each "
                   "calendar year matches MSCI World's published annual return. The first half of 1990 stays "
                   "a plain U.S. proxy, Fama-French developed-market daily returns then run to mid-2008, and "
                   "VT's own returns take over from 2008-06-27.",
        "caveat": "Anything before 1990 is a U.S. proxy shaped to global annual returns rather than real "
                  "daily global history. The 1990 to 2008 stretch covers developed markets only, so emerging "
                  "markets enter just once VT takes over.",
        "levered": False,
    },
    "GLBOND": {
        "name": "Global government bonds",
        "detail": "issued by governments around the world, held in USD with currency swings left in rather "
                  "than hedged away",
        "coverage": "1970 to today, with a BND / BWX fund blend from 2007-10-12",
        "history": "Everything up to 2007 is reconstructed. The Jorda-Schularick-Taylor macrohistory database "
                   "sets each year's return for a GDP-weighted basket of 16 advanced-economy government bond "
                   "markets, and the path within each year comes from BIS daily exchange rates plus government "
                   "bond yields: daily yields for the U.S., Japan and the U.K., together about 62% of the "
                   "basket from 1979, and monthly yields for everywhere else. A 45% BND / 55% BWX blend, "
                   "rebalanced daily, takes over from 2007-10-12.",
        "caveat": "The reconstructed era covers government bonds in advanced economies only, with no corporate, "
                  "emerging-market or inflation-linked debt. Outside the U.S., Japan and the U.K. its daily "
                  "moves are interpolated from monthly readings, so short-term wobbles are smoother than they "
                  "really were.",
        "levered": False,
    },
    "GOLDPM": {
        "name": "Gold",
        "detail": "bullion measured as a total return that already carries fund fees, similar to GLD",
        "coverage": "1970 to today, with GLD's own returns from 2004-11-18",
        "history": "From 1970 until GLD launched, the series is the London afternoon gold price less GLD's "
                   "0.40% a year expense drag, so the whole history is priced the way someone holding the fund "
                   "would have experienced it. GLD's own returns take over from 2004-11-18.",
        "caveat": "Prices before 2004 are struck at the London afternoon fix, around 10am in New York, rather "
                  "than at the U.S. close. That loosens day-to-day alignment with U.S.-listed assets, though it "
                  "does not affect the long-run path.",
        "levered": False,
    },
    "COMM": {
        "name": "Broad commodities",
        "detail": "a diversified basket of commodity futures spanning energy, metals and agriculture, "
                  "including roll yield and Treasury-bill collateral, similar to DBC",
        "coverage": "1970 to today, with DBC's own returns from 2006-02-07",
        "history": "From 1970 to 1991 the series is anchored to S&amp;P GSCI total-return data, interpolated "
                   "between roughly bi-monthly readings until 1984 and then laid over daily GSCI spot movements. "
                   "The Bloomberg Commodity excess-return index plus Treasury-bill collateral covers 1991 to "
                   "2006, and DBC's own returns take over from 2006-02-07.",
        "caveat": "This is the patchiest series on the page. It changes benchmark twice, in 1991 and again in "
                  "2006, so it should not be read as one continuously observed index running back to 1970. Its "
                  "daily swings before 1984 are also smoothed by interpolation.",
        "levered": False,
    },
    "USLCAP3x": {
        "name": "3x U.S. large-cap stocks",
        "detail": "three times the daily move of the S&amp;P 500 total return, reset every day, similar to UPRO",
        "coverage": "1970 to today, with UPRO's own returns from 2009-06-25",
        "history": "From 1970 until UPRO launched, the series is modelled as three times the daily index "
                   "return, less borrowing costs (the Treasury-bill rate plus a 0.65% spread) and a 0.91% "
                   "annual expense ratio. Measured against UPRO's live record, the model matches its daily "
                   "returns with a correlation of 0.998. UPRO's own returns take over from 2009-06-25.",
        "caveat": "Because the leverage resets daily, this is not the same thing as 3x the S&amp;P 500 over the "
                  "period. Choppy markets erode it. Before 1988 the underlying index is a broad large-cap proxy "
                  "rather than the official S&amp;P 500 total return.",
        "levered": True,
    },
    "LTT3x": {
        "name": "3x long-term Treasuries",
        "detail": "three times the daily move of 20+ year U.S. Treasuries, reset every day, similar to TMF",
        "coverage": "1970 to today, with TMF's own returns from 2009-04-16",
        "history": "From 1970 until TMF launched, the series is modelled as three times the daily bond return, "
                   "less borrowing costs (the Treasury-bill rate plus a 0.53% spread) and a 1.06% annual "
                   "expense ratio, matching TMF's daily returns with a correlation of 0.997. TMF's own returns "
                   "take over from 2009-04-16.",
        "caveat": "The underlying bond history only matches TMF's 20+ year benchmark from 2002; earlier years "
                  "use a 25-year constant-maturity model. Decay bites hard here. Through the rising rates of "
                  "2009 to 2026, TMF lost far more than long Treasuries themselves did.",
        "levered": True,
    },
    "ITT3x": {
        "name": "3x intermediate Treasuries",
        "detail": "three times the daily move of 7 to 10 year U.S. Treasuries, reset every day, similar to TYD",
        "coverage": "1970 to today, with TYD's own returns from 2009-04-16",
        "history": "From 1970 until TYD launched, the series is modelled as three times the daily bond return, "
                   "less borrowing costs (the Treasury-bill rate plus a 0.19% spread, which partly reflects "
                   "TYD's fee waivers) and a 1.09% annual expense ratio. TYD's own returns take over from "
                   "2009-04-16.",
        "caveat": "TYD is small and thinly traded, and its published prices go stale between 2014 and 2018, so "
                  "day-to-day agreement with the model is poor across those years. That is a problem with the "
                  "fund's market data rather than with the model. Before 2002 the underlying is an 8.5-year par "
                  "bond model.",
        "levered": True,
    },
    "GOLDPM2x": {
        "name": "2x gold",
        "detail": "twice the daily move of gold, reset every day, similar to UGL",
        "coverage": "1970 to today, with UGL's own returns from 2008-12-03",
        "history": "From 1970 until UGL launched, the series is modelled as twice the daily spot gold return, "
                   "less borrowing costs (the Treasury-bill rate plus a 0.93% spread) and a 0.95% annual "
                   "expense ratio. UGL's own returns take over from 2008-12-03.",
        "caveat": "The model prices gold at the London afternoon fix while UGL closes at 4pm in New York, so "
                  "daily agreement is loose (a correlation of about 0.67) even though cumulative growth tracks "
                  "closely. UGL also benchmarks a futures index rather than spot bullion.",
        "levered": True,
    },
}


def asset_info(ticker):
    """Return the description record for ``ticker``, or a neutral placeholder."""
    return ASSET_INFO.get(ticker) or {
        "name": ticker,
        "detail": "a building-block asset in this backtest",
        "coverage": "",
        "history": "",
        "caveat": "",
        "levered": False,
    }


def build_about_html(tickers, n_portfolios, meta):
    """Build the public-facing 'About this chart' section as an HTML string.

    ``meta`` is an optional dict describing the run (period, rebalancing,
    weight step, starting value, weight constraints). Missing values degrade
    gracefully so the section is still readable when called without it.

    The wording adapts to whichever assets were actually run: the leverage
    material only appears when a leveraged series is in the mix, and the
    caveats are assembled from the per-asset records in ``ASSET_INFO``.
    """
    meta = meta or {}
    period_start = meta.get("period_start")
    period_end = meta.get("period_end")
    trading_days = meta.get("trading_days")
    years = meta.get("years")
    weight_step = meta.get("weight_step")
    starting_value = meta.get("starting_value")
    rebalance_annually = meta.get("rebalance_annually", True)

    period = f"{period_start} to {period_end}" if period_start and period_end else "the full available history"
    years_text = f", about {years:.0f} years of daily data" if years else ""
    step_text = f"{weight_step:.0%}" if weight_step else "fixed"
    start_value_text = f"${starting_value:,.0f}" if starting_value else "a fixed starting amount"
    rebalance_text = (
        "rebalanced back to its target mix once a year"
        if rebalance_annually
        else "bought once and held for the whole period without rebalancing"
    )
    days_text = f"{trading_days:,}" if trading_days else "many"

    infos = [(ticker, asset_info(ticker)) for ticker in tickers]
    has_leverage = any(info["levered"] for _, info in infos)

    # Building blocks. The third column states the full span each series covers,
    # so nobody reads the fund handover date as the start of the data.
    block_rows = "".join(
        f"<tr><td><code>{ticker}</code></td><td><strong>{info['name']}</strong>, {info['detail']}</td>"
        f"<td>{info['coverage'] or 'n/a'}</td></tr>"
        for ticker, info in infos
    )
    blocks_table = (
        "<table class=\"asset-table\">"
        "<thead><tr><th>Code</th><th>What it is</th><th>Series covers</th></tr></thead>"
        f"<tbody>{block_rows}</tbody></table>"
    )

    # Per-asset provenance, so the reconstructed portions of each history are explicit.
    history_items = "".join(
        f"<li><strong>{ticker}</strong>: {info['history']}</li>"
        for ticker, info in infos
        if info["history"]
    )
    history_html = (
        "<h3>Where each building block comes from</h3>"
        "<p>Every series runs from 1970, but no real fund goes back that far. The further back you go, the "
        "more of the data is reconstructed rather than observed. Here is what each one is made of:</p>"
        f"<ul>{history_items}</ul>"
    ) if history_items else ""

    if has_leverage:
        construction_html = (
            "<p>Every building block runs from 1970 to today, but only the most recent stretch of each one is "
            "a real fund. The 2x and 3x blocks are daily-reset models of exchange-traded funds such as UPRO, "
            "TMF, TYD and UGL, and they already carry the costs that make leverage expensive in practice: the "
            "daily interest on borrowed money, the fund's own fees, and the decay that comes from resetting the "
            "leverage every single day. From each fund's launch, around 2008 to 2009, the series switches to "
            "that fund's own returns.</p>"
        )
    else:
        construction_html = (
            "<p>Every building block runs from 1970 to today, but only the most recent stretch of each one is "
            "a real fund. Before that the series is extended backwards using published index data and, where no "
            "index reaches far enough, a documented model. So the recent decades are observed history and the "
            "earlier decades are reconstructions.</p>"
        )

    # Human-readable weight constraints (only if any floor/roof is non-trivial).
    constraint_bits = []
    for ticker in tickers:
        floor = (meta.get("min_weights") or {}).get(ticker, 0.0)
        roof = (meta.get("max_weights") or {}).get(ticker, 1.0)
        if floor and floor > 0:
            constraint_bits.append(f"{ticker} at least {floor:.0%}")
        if roof is not None and roof < 1.0:
            constraint_bits.append(f"{ticker} at most {roof:.0%}")
    constraints_html = (
        f"<p><strong>Limits applied to this run:</strong> {', '.join(constraint_bits)}.</p>"
        if constraint_bits else ""
    )

    # Caveats: the universal ones, then anything specific to the assets in play.
    caveats = [
        "<strong>This is not investment advice.</strong> It is an educational illustration, not a recommendation "
        "to buy or sell anything.",
        "<strong>Past results do not predict future returns.</strong> A blend that looked good in history can do "
        "badly from here.",
        "<strong>Hindsight is baked in.</strong> Picking the best-scoring dot means picking whatever happened to "
        "suit the past, using information nobody had at the time.",
        "<strong>The early decades are reconstructed, not observed.</strong> No real fund covers this period, so "
        "the further back the backtest runs, the more it rests on models. The section above says which parts, "
        "for which asset.",
    ]
    if has_leverage:
        caveats.append(
            "<strong>Leverage is risky.</strong> The 2x and 3x blocks can fall very fast. Many blends on this page "
            "show drawdowns worse than 80%, which few people could hold through in real time."
        )
    caveats.append(
        "<strong>Real-world costs are missing.</strong> Taxes, commissions, bid/ask spreads and the practical "
        "friction of rebalancing are not modelled. Fees charged inside the funds themselves are included."
    )
    for ticker, info in infos:
        if info["caveat"]:
            caveats.append(f"<strong>{ticker}.</strong> {info['caveat']}")
    caveats_html = "".join(f"<li>{item}</li>" for item in caveats)

    return f"""<section class="about">
<h2>About this chart</h2>

<h3>What it shows</h3>
<p>Every dot is one portfolio: a fixed recipe for splitting a pot of money across
{len(tickers)} building-block investments. This page tested <strong>{n_portfolios:,} recipes</strong>, which is
every blend in {step_text} steps that adds up to 100%, and plots how each one would have behaved over
{period}{years_text}. The dropdowns above the chart control what the axes measure, so you can compare any two
results against each other.</p>
<p>The building blocks are:</p>
{blocks_table}

<h3>How to use it</h3>
<ul>
<li>Pick what each axis measures with the <strong>X Axis</strong> and <strong>Y Axis</strong> dropdowns.</li>
<li><strong>Hover</strong> a dot to see that portfolio's mix and headline numbers. <strong>Click</strong> it to pin the label in place, and click again to unpin.</li>
<li>The <strong>Highlight</strong> dropdown colours every dot past a threshold you choose, which makes it easy to see, say, which blends fell further than 60% at their worst.</li>
<li>The table below lists every portfolio tested. Click a column heading to sort by it, type in the filter box to search the weights, or add precise metric filters such as &ldquo;Max Drawdown at most 60%&rdquo;. Clicking a table row pins its dot on the chart, and clicking a dot highlights its row.</li>
</ul>

<h3>How the numbers were worked out</h3>
<p>These results are a backtest, meaning a simulation of how each blend would have performed had it existed.
It runs on {days_text} days of daily price history covering {period}. Every portfolio starts at {start_value_text}
and is {rebalance_text}. Daily returns are then compounded to trace its value over time, and every statistic in
the table is measured from that path. CAGR is the annual growth rate that would take the starting pot to the
finishing one, and max drawdown is the worst peak-to-trough fall along the way.</p>
{construction_html}
{constraints_html}
{history_html}

<h3>What to watch out for</h3>
<ul>
{caveats_html}
</ul>

<p class="about-foot">Backtest period {period}. {days_text} trading days, {n_portfolios:,} portfolios tested.
Built by <code>portfolio_optimizer.py</code> from the daily datasets in
<a href="https://github.com/mwilczynska/financial_datasets">mwilczynska/financial_datasets</a>, where the full
methodology for every series is documented.</p>
</section>"""


def write_interactive_scatter(results, tickers, path, meta=None):
    """Write a dependency-free HTML/SVG scatter with selectable axes."""
    width, height = 1100, 560
    left, right, top, bottom = 90, 40, 35, 90
    plot_w, plot_h = width - left - right, height - top - bottom

    numeric_columns = [
        column for column in results.columns
        if pd.api.types.is_numeric_dtype(results[column])
    ]
    percent_columns = [
        column for column in numeric_columns
        if column.endswith("Weight") or column in {
            "CAGR", "Annual Return", "Std Dev", "TWRR", "Max Drawdown",
            "Average Drawdown", "Daily Win Rate", "Best Day", "Worst Day",
            "Best Month", "Worst Month", "Best Year", "Worst Year",
            "VaR 95 Daily", "CVaR 95 Daily",
        }
    ]
    chart_rows = results.astype(object).where(pd.notna(results), None).to_dict("records")
    chart_data = json.dumps(chart_rows, allow_nan=False)
    column_data = json.dumps(numeric_columns)
    percent_column_data = json.dumps(percent_columns)
    weight_column_data = json.dumps([f"{ticker} Weight" for ticker in tickers])
    ticker_data = json.dumps(list(tickers))
    about_section = build_about_html(tickers, len(results), meta)

    doc = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Portfolio Optimisation Results</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 24px; color: #111827; }}
h1 {{ font-size: 22px; margin: 0 0 4px; }}
p {{ margin: 0 0 16px; color: #4b5563; }}
.controls {{ display: flex; gap: 16px; align-items: end; flex-wrap: wrap; margin: 0 0 14px; }}
.control {{ display: grid; gap: 5px; }}
label {{ font-size: 12px; color: #4b5563; font-weight: 700; text-transform: uppercase; }}
select {{
    min-width: 220px;
    padding: 7px 9px;
    border: 1px solid #9ca3af;
    border-radius: 6px;
    background: white;
    color: #111827;
    font-size: 14px;
}}
svg {{ display: block; margin: 0 auto; width: 100%; max-width: 1520px; height: auto; border: 1px solid #d1d5db; background: white; }}
circle:hover {{ opacity: 1; }}
#tooltip, #pinned-tooltip {{
    position: fixed;
    display: none;
    max-width: 460px;
    padding: 10px 12px;
    border: 1px solid #9ca3af;
    border-radius: 6px;
    background: #ffffff;
    box-shadow: 0 8px 24px rgba(17, 24, 39, 0.18);
    color: #111827;
    font-size: 16px;
    line-height: 1.45;
    white-space: pre-line;
    pointer-events: none;
    z-index: 10;
}}
#pinned-tooltip {{
    position: absolute;
    border-color: #7c3aed;
    box-shadow: 0 8px 24px rgba(124, 58, 237, 0.22);
    z-index: 9;
}}
.table-controls {{ display: flex; gap: 16px; align-items: center; flex-wrap: wrap; margin: 26px 0 10px; }}
.table-controls input[type="text"] {{
    min-width: 280px;
    padding: 7px 9px;
    border: 1px solid #9ca3af;
    border-radius: 6px;
    font-size: 14px;
}}
.table-controls .toggle {{ display: flex; align-items: center; gap: 6px; font-size: 13px; color: #374151; }}
.table-controls .toggle label {{ font-size: 13px; text-transform: none; font-weight: 400; color: #374151; }}
#row-count {{ font-size: 13px; color: #6b7280; margin-left: auto; }}
.filter-builder {{ display: flex; gap: 8px; align-items: center; flex-wrap: wrap; margin: 0 0 10px; }}
.filter-builder .fb-label {{ font-size: 12px; font-weight: 700; text-transform: uppercase; color: #4b5563; }}
.filter-builder select, .filter-builder input {{
    padding: 6px 8px;
    border: 1px solid #9ca3af;
    border-radius: 6px;
    font-size: 13px;
    background: white;
}}
.filter-builder #filter-column {{ min-width: 170px; }}
.filter-builder #filter-op {{ min-width: 52px; }}
.filter-builder #filter-value {{ width: 110px; }}
.filter-builder .fb-hint {{ font-size: 12px; color: #6b7280; }}
.filter-builder button {{
    padding: 6px 12px;
    border: 1px solid #2563eb;
    border-radius: 6px;
    background: #2563eb;
    color: white;
    font-size: 13px;
    cursor: pointer;
}}
.filter-builder button:hover {{ background: #1d4ed8; }}
.filter-builder .fb-clear {{ background: white; color: #4b5563; border-color: #9ca3af; }}
.filter-builder .fb-clear:hover {{ background: #f3f4f6; }}
.filter-chips {{ display: flex; gap: 8px; flex-wrap: wrap; margin: 0 0 10px; }}
.filter-chips:empty {{ margin: 0; }}
.chip {{
    display: inline-flex;
    align-items: center;
    gap: 7px;
    padding: 4px 6px 4px 10px;
    border: 1px solid #c7d2fe;
    border-radius: 14px;
    background: #eef2ff;
    color: #3730a3;
    font-size: 12px;
}}
.chip button {{
    border: none;
    background: #c7d2fe;
    color: #3730a3;
    border-radius: 50%;
    width: 18px;
    height: 18px;
    line-height: 1;
    cursor: pointer;
    font-size: 12px;
}}
.chip button:hover {{ background: #a5b4fc; }}
.about {{ max-width: 820px; margin: 36px auto 8px; color: #1f2937; line-height: 1.6; }}
.about h2 {{ font-size: 20px; margin: 0 0 6px; }}
.about h3 {{ font-size: 15px; margin: 22px 0 6px; color: #111827; }}
.about p {{ margin: 0 0 12px; color: #374151; }}
.about ul {{ margin: 0 0 12px; padding-left: 22px; }}
.about li {{ margin: 0 0 6px; color: #374151; }}
.about code {{ background: #f3f4f6; padding: 1px 5px; border-radius: 4px; font-size: 13px; }}
.about .about-foot {{ font-size: 12px; color: #6b7280; border-top: 1px solid #e5e7eb; padding-top: 12px; margin-top: 18px; }}
.about a {{ color: #4338ca; }}
.about .asset-table {{ width: 100%; border-collapse: collapse; font-size: 13px; margin: 0 0 14px; }}
.about .asset-table th, .about .asset-table td {{
    border-bottom: 1px solid #e5e7eb;
    padding: 7px 10px;
    text-align: left;
    vertical-align: top;
    white-space: normal;
}}
.about .asset-table th {{
    background: #f9fafb;
    color: #374151;
    font-weight: 600;
    position: static;
    cursor: default;
}}
.about .asset-table th:hover {{ background: #f9fafb; }}
.about .asset-table td:first-child {{ white-space: nowrap; }}
.about .asset-table tbody tr {{ cursor: default; }}
.about .asset-table tbody tr:hover {{ background: transparent; }}
.about .asset-table tbody tr:nth-child(even) {{ background: #fafafa; }}
#table-wrap {{ max-height: 460px; overflow: auto; border: 1px solid #d1d5db; border-radius: 6px; }}
table {{ border-collapse: collapse; font-size: 13px; width: 100%; }}
th, td {{ border-bottom: 1px solid #e5e7eb; padding: 7px 10px; text-align: right; white-space: nowrap; }}
th:first-child, td:first-child {{ text-align: left; }}
th {{
    background: #f3f4f6;
    position: sticky;
    top: 0;
    cursor: pointer;
    user-select: none;
    z-index: 1;
}}
th:hover {{ background: #e5e7eb; }}
th .arrow {{ color: #2563eb; font-size: 11px; }}
tbody tr {{ cursor: pointer; }}
tbody tr:nth-child(even) {{ background: #fafafa; }}
tbody tr:hover {{ background: #eef2ff; }}
tbody tr.row-selected, tbody tr.row-selected:hover {{ background: #ede9fe; box-shadow: inset 3px 0 0 #7c3aed; }}
</style>
</head>
<body>
<h1>Portfolio Optimisation Results</h1>
<p>Pick any numeric result columns for the axes. Hover a dot for its weights and core statistics, or click to pin it. Pinned dots highlight their row in the table below; click a table row to pin its dot.</p>
<div class="controls">
    <div class="control">
        <label for="x-select">X Axis</label>
        <select id="x-select"></select>
    </div>
    <div class="control">
        <label for="y-select">Y Axis</label>
        <select id="y-select"></select>
    </div>
    <div class="control">
        <label id="threshold-label" for="threshold-select">Highlight Max Drawdown</label>
        <select id="threshold-select"></select>
    </div>
</div>
<div id="tooltip"></div>
<div id="pinned-tooltip"></div>
<svg viewBox="0 0 {width} {height}" role="img" aria-label="Portfolio metric scatter plot with selectable axes">
<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#111827"/>
<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#111827"/>
<g id="grid-layer"></g>
<g id="dots-layer"></g>
<text id="x-axis-label" x="{left + plot_w / 2:.0f}" y="{height - 30}" text-anchor="middle">Max Drawdown</text>
<text id="y-axis-label" x="22" y="{top + plot_h / 2:.0f}" transform="rotate(-90 22 {top + plot_h / 2:.0f})" text-anchor="middle">CAGR</text>
</svg>
<div class="table-controls">
    <input type="text" id="table-filter" placeholder="Filter portfolios… (e.g. 60%)" autocomplete="off">
    <span class="toggle"><input type="checkbox" id="show-all-cols"><label for="show-all-cols">Show all columns</label></span>
    <span id="row-count"></span>
</div>
<div class="filter-builder">
    <span class="fb-label">Add metric filter</span>
    <select id="filter-column"></select>
    <select id="filter-op">
        <option value="ge">&ge;</option>
        <option value="le">&le;</option>
    </select>
    <input type="number" id="filter-value" step="any" placeholder="value">
    <span id="filter-hint" class="fb-hint"></span>
    <button id="filter-add" type="button">Add filter</button>
    <button id="filter-clear" type="button" class="fb-clear">Clear all</button>
</div>
<div id="filter-chips" class="filter-chips"></div>
<div id="table-wrap">
<table id="results-table">
<thead><tr id="table-head-row"></tr></thead>
<tbody id="table-body"></tbody>
</table>
</div>
{about_section}
<script>
(() => {{
const chartData = {chart_data};
const numericColumns = {column_data};
const percentColumns = new Set({percent_column_data});
const weightColumns = {weight_column_data};
const tickers = {ticker_data};
const width = {width};
const height = {height};
const left = {left};
const top = {top};
const plotW = {plot_w};
const plotH = {plot_h};
const xSelect = document.getElementById("x-select");
const ySelect = document.getElementById("y-select");
const thresholdSelect = document.getElementById("threshold-select");
const thresholdLabel = document.getElementById("threshold-label");
const gridLayer = document.getElementById("grid-layer");
const dotsLayer = document.getElementById("dots-layer");
const xAxisLabel = document.getElementById("x-axis-label");
const yAxisLabel = document.getElementById("y-axis-label");
const hoverTooltip = document.getElementById("tooltip");
const pinnedTooltip = document.getElementById("pinned-tooltip");
const tableFilter = document.getElementById("table-filter");
const showAllCols = document.getElementById("show-all-cols");
const rowCount = document.getElementById("row-count");
const tableHeadRow = document.getElementById("table-head-row");
const tableBody = document.getElementById("table-body");
const filterColumn = document.getElementById("filter-column");
const filterOp = document.getElementById("filter-op");
const filterValue = document.getElementById("filter-value");
const filterHint = document.getElementById("filter-hint");
const filterAdd = document.getElementById("filter-add");
const filterClear = document.getElementById("filter-clear");
const filterChips = document.getElementById("filter-chips");
let pinnedDot = null;
let pinnedRank = null;
const activeFilters = [];

const coreColumns = ["Rank"].concat(weightColumns,
    ["CAGR", "Max Drawdown", "Sharpe", "Sortino", "Calmar", "Ulcer Index", "Final Value"]);
const allColumns = ["Rank", "Portfolio"].concat(
    numericColumns.filter((column) => column !== "Rank"));
let sortColumn = "Rank";
let sortAscending = true;

numericColumns.forEach((column) => {{
    xSelect.add(new Option(column, column));
    ySelect.add(new Option(column, column));
}});
xSelect.value = "Max Drawdown";
ySelect.value = "CAGR";
xSelect.addEventListener("change", () => {{
    updateThresholdOptions();
    renderChart();
}});
ySelect.addEventListener("change", renderChart);
thresholdSelect.addEventListener("change", renderChart);
tableFilter.addEventListener("input", renderTable);
showAllCols.addEventListener("change", renderTable);

numericColumns.forEach((column) => filterColumn.add(new Option(column, column)));
if (numericColumns.includes("Max Drawdown")) filterColumn.value = "Max Drawdown";
filterColumn.addEventListener("change", updateFilterHint);
filterValue.addEventListener("keydown", (event) => {{ if (event.key === "Enter") addFilter(); }});
filterAdd.addEventListener("click", addFilter);
filterClear.addEventListener("click", () => {{
    activeFilters.length = 0;
    renderFilters();
    renderTable();
}});
updateFilterHint();

updateThresholdOptions();
renderChart();
renderTable();

function renderChart() {{
    const xColumn = xSelect.value;
    const yColumn = ySelect.value;
    const xThreshold = thresholdSelect.value === "" ? null : Number(thresholdSelect.value);
    const rows = chartData.filter((row) => Number.isFinite(row[xColumn]) && Number.isFinite(row[yColumn]));
    if (!rows.length) return;

    const xValues = rows.map((row) => row[xColumn]);
    const yValues = rows.map((row) => row[yColumn]);
    let xDomain = paddedDomain(xValues);
    let yDomain = paddedDomain(yValues);
    const xTicks = makeTicks(xDomain[0], xDomain[1], 9);
    const yTicks = makeTicks(yDomain[0], yDomain[1], 9);
    xDomain = [xTicks[0], xTicks[xTicks.length - 1]];
    yDomain = [yTicks[0], yTicks[yTicks.length - 1]];

    xAxisLabel.textContent = xColumn;
    yAxisLabel.textContent = yColumn;
    gridLayer.innerHTML = "";
    dotsLayer.innerHTML = "";
    pinnedDot = null;
    hideHoverTooltip();
    hidePinnedTooltip();

    for (const tick of xTicks) {{
        const x = scaleX(tick, xDomain);
        gridLayer.insertAdjacentHTML("beforeend", `<line x1="${{x.toFixed(2)}}" y1="${{top}}" x2="${{x.toFixed(2)}}" y2="${{top + plotH}}" stroke="#e5e7eb"/>`);
        gridLayer.insertAdjacentHTML("beforeend", `<text x="${{x.toFixed(2)}}" y="${{height - 62}}" text-anchor="middle">${{formatAxisValue(tick, xColumn)}}</text>`);
    }}
    for (const tick of yTicks) {{
        const y = scaleY(tick, yDomain);
        gridLayer.insertAdjacentHTML("beforeend", `<line x1="${{left}}" y1="${{y.toFixed(2)}}" x2="${{left + plotW}}" y2="${{y.toFixed(2)}}" stroke="#e5e7eb"/>`);
        gridLayer.insertAdjacentHTML("beforeend", `<text x="${{left - 12}}" y="${{(y + 4).toFixed(2)}}" text-anchor="end">${{formatAxisValue(tick, yColumn)}}</text>`);
    }}

    for (const row of rows) {{
        const x = scaleX(row[xColumn], xDomain);
        const y = scaleY(row[yColumn], yDomain);
        const tooltipText = buildTooltipText(row, xColumn, yColumn);
        const dot = document.createElementNS("http://www.w3.org/2000/svg", "circle");
        const baseColor = xThreshold !== null && row[xColumn] <= xThreshold ? "#dc2626" : "#2563eb";
        dot.setAttribute("cx", x.toFixed(2));
        dot.setAttribute("cy", y.toFixed(2));
        dot.setAttribute("r", "4");
        dot.setAttribute("fill", baseColor);
        dot.setAttribute("opacity", "0.62");
        dot.dataset.baseColor = baseColor;
        dot.dataset.tooltip = tooltipText;
        dot.dataset.rank = row.Rank;
        dot.addEventListener("mouseenter", showHoverTooltip);
        dot.addEventListener("mousemove", moveHoverTooltip);
        dot.addEventListener("mouseleave", hideHoverTooltip);
        dot.addEventListener("click", togglePinnedTooltip);
        dotsLayer.appendChild(dot);
    }}

    if (pinnedRank !== null) {{
        const dot = dotsLayer.querySelector(`circle[data-rank="${{pinnedRank}}"]`);
        if (dot) {{
            const r = dot.getBoundingClientRect();
            setPin(dot, r.left + r.width / 2, r.top, false);
        }} else {{
            hidePinnedTooltip();
        }}
    }}
}}

function updateThresholdOptions() {{
    const xColumn = xSelect.value;
    const xValues = chartData
        .map((row) => row[xColumn])
        .filter((value) => Number.isFinite(value));
    const min = Math.min(...xValues);
    const max = Math.max(...xValues);
    const thresholds = makeTicks(min, max, 9);

    thresholdLabel.textContent = `Highlight ${{xColumn}}`;
    thresholdSelect.innerHTML = "";
    thresholdSelect.add(new Option("No highlight", ""));
    thresholds.forEach((threshold) => {{
        thresholdSelect.add(new Option(`<= ${{formatTooltipValue(threshold, xColumn)}}`, String(threshold)));
    }});
}}

function paddedDomain(values) {{
    let min = Math.min(...values);
    let max = Math.max(...values);
    if (min === max) {{
        const pad = Math.abs(min || 1) * 0.05;
        return [min - pad, max + pad];
    }}
    const pad = (max - min) * 0.05;
    return [min - pad, max + pad];
}}

function makeTicks(min, max, count) {{
    if (!Number.isFinite(min) || !Number.isFinite(max)) return [];
    if (min === max) return [min];

    const rawStep = Math.abs(max - min) / Math.max(1, count - 1);
    const magnitude = 10 ** Math.floor(Math.log10(rawStep));
    const residual = rawStep / magnitude;
    let niceResidual;
    if (residual <= 1) niceResidual = 1;
    else if (residual <= 2) niceResidual = 2;
    else if (residual <= 2.5) niceResidual = 2.5;
    else if (residual <= 5) niceResidual = 5;
    else niceResidual = 10;

    const step = niceResidual * magnitude;
    const niceMin = Math.floor(min / step) * step;
    const niceMax = Math.ceil(max / step) * step;
    const ticks = [];
    for (let value = niceMin; value <= niceMax + step * 0.5; value += step) {{
        ticks.push(Number(value.toPrecision(12)));
    }}
    return ticks;
}}

function scaleX(value, domain) {{
    return left + (value - domain[0]) / (domain[1] - domain[0]) * plotW;
}}

function scaleY(value, domain) {{
    return top + plotH - (value - domain[0]) / (domain[1] - domain[0]) * plotH;
}}

function formatAxisValue(value, column) {{
    if (percentColumns.has(column)) return `${{(value * 100).toFixed(0)}}%`;
    if (Math.abs(value) >= 1000000) return value.toExponential(1);
    if (Math.abs(value) >= 1000) return value.toFixed(0);
    if (Math.abs(value) >= 10) return value.toFixed(1);
    return value.toFixed(2);
}}

function formatTooltipValue(value, column) {{
    if (!Number.isFinite(value)) return "";
    if (percentColumns.has(column)) return `${{(value * 100).toFixed(2)}}%`;
    if (column === "Final Value") return `$${{Math.round(value).toLocaleString()}}`;
    if (column === "Rank" || column === "Longest Drawdown Days") return Math.round(value).toLocaleString();
    return value.toFixed(3);
}}

function buildTooltipText(row, xColumn, yColumn) {{
    const lines = [
        row.Portfolio,
        `${{xColumn}}: ${{formatTooltipValue(row[xColumn], xColumn)}}`,
    ];
    if (yColumn !== xColumn) {{
        lines.push(`${{yColumn}}: ${{formatTooltipValue(row[yColumn], yColumn)}}`);
    }}

    for (const column of ["CAGR", "Max Drawdown", "Sharpe", "Ulcer Index"]) {{
        if (column !== xColumn && column !== yColumn) {{
            lines.push(`${{column}}: ${{formatTooltipValue(row[column], column)}}`);
        }}
    }}
    return lines.join("\\n");
}}

function showHoverTooltip(event) {{
    if (event.currentTarget === pinnedDot) return;
    hoverTooltip.textContent = event.currentTarget.dataset.tooltip;
    hoverTooltip.style.display = "block";
    positionTooltip(hoverTooltip, event);
}}

function moveHoverTooltip(event) {{
    if (event.currentTarget === pinnedDot) return;
    positionTooltip(hoverTooltip, event);
}}

function hideHoverTooltip() {{
    hoverTooltip.style.display = "none";
}}

function hidePinnedTooltip() {{
    pinnedTooltip.style.display = "none";
}}

function togglePinnedTooltip(event) {{
    const dot = event.currentTarget;
    if (pinnedDot === dot) {{
        clearPin();
        return;
    }}
    setPin(dot, event.clientX, event.clientY, true);
}}

function setPin(dot, clientX, clientY, scrollTable) {{
    if (pinnedDot && pinnedDot !== dot) restoreDot(pinnedDot);
    markDotPinned(dot);
    hideHoverTooltip();
    pinnedTooltip.textContent = dot.dataset.tooltip;
    pinnedTooltip.style.display = "block";
    positionTooltip(pinnedTooltip, clientX, clientY);
    highlightRow(dot.dataset.rank, scrollTable);
}}

function markDotPinned(dot) {{
    pinnedDot = dot;
    pinnedRank = dot.dataset.rank;
    dotsLayer.appendChild(dot);
    dot.setAttribute("fill", "#7c3aed");
    dot.setAttribute("opacity", "1");
}}

function restoreDot(dot) {{
    dot.setAttribute("fill", dot.dataset.baseColor);
    dot.setAttribute("opacity", "0.62");
}}

function clearPin() {{
    if (pinnedDot) restoreDot(pinnedDot);
    pinnedDot = null;
    pinnedRank = null;
    hidePinnedTooltip();
    highlightRow(null, false);
}}

function positionTooltip(tooltipEl, clientX, clientY) {{
    const pad = 14;
    const rect = tooltipEl.getBoundingClientRect();
    let posLeft = clientX + pad;
    let posTop = clientY + pad;
    if (posLeft + rect.width > window.innerWidth) {{
        posLeft = clientX - rect.width - pad;
    }}
    if (posTop + rect.height > window.innerHeight) {{
        posTop = clientY - rect.height - pad;
    }}
    posLeft = Math.max(8, posLeft);
    posTop = Math.max(8, posTop);
    // The pinned tooltip is absolutely positioned, so anchor it to the document
    // (page coordinates) — that keeps it next to its dot when the page scrolls.
    // The hover tooltip is fixed and tracks the cursor, so it stays in viewport coords.
    if (tooltipEl.style.position === "absolute" || tooltipEl.id === "pinned-tooltip") {{
        posLeft += window.scrollX;
        posTop += window.scrollY;
    }}
    tooltipEl.style.left = posLeft + "px";
    tooltipEl.style.top = posTop + "px";
}}

function headerLabel(column) {{
    const ticker = tickers.find((name) => column === `${{name}} Weight`);
    return ticker !== undefined ? ticker : column;
}}

function compareValues(a, b) {{
    const aMissing = a === null || a === undefined || a === "";
    const bMissing = b === null || b === undefined || b === "";
    if (aMissing && bMissing) return 0;
    if (aMissing) return 1;
    if (bMissing) return -1;
    if (typeof a === "number" && typeof b === "number") return a - b;
    return String(a).localeCompare(String(b));
}}

function columnRange(column) {{
    const values = chartData
        .map((row) => row[column])
        .filter((value) => Number.isFinite(value));
    if (!values.length) return null;
    return [Math.min(...values), Math.max(...values)];
}}

// Loss-type metrics (drawdowns, worst periods, VaR) are stored as non-positive
// numbers. Users think of them as positive magnitudes ("drawdown <= 60%"), so for
// any column whose values are all <= 0 we filter on the absolute value.
function isMagnitudeColumn(column) {{
    const range = columnRange(column);
    return range !== null && range[1] <= 0 && range[0] < 0;
}}

function updateFilterHint() {{
    const column = filterColumn.value;
    const range = columnRange(column);
    if (!range) {{
        filterHint.textContent = "";
        filterValue.placeholder = "value";
        return;
    }}
    const isPercent = percentColumns.has(column);
    filterValue.placeholder = isPercent ? "%" : "value";
    if (isMagnitudeColumn(column)) {{
        const lo = formatTooltipValue(Math.min(Math.abs(range[0]), Math.abs(range[1])), column);
        const hi = formatTooltipValue(Math.max(Math.abs(range[0]), Math.abs(range[1])), column);
        filterHint.textContent = `magnitude ${{lo}} to ${{hi}}`
            + (isPercent ? " (enter as %, e.g. 60)" : "");
    }} else {{
        filterHint.textContent = `range: ${{formatTooltipValue(range[0], column)}} to ${{formatTooltipValue(range[1], column)}}`
            + (isPercent ? " (enter as %)" : "");
    }}
}}

function addFilter() {{
    const column = filterColumn.value;
    const raw = Number(filterValue.value);
    if (filterValue.value.trim() === "" || !Number.isFinite(raw)) return;
    const magnitude = isMagnitudeColumn(column);
    // Magnitude columns compare on |value|, so the stored threshold is always positive.
    const scaled = magnitude ? Math.abs(raw) : raw;
    // Inputs for percent columns are entered as percentages; data is stored as a fraction.
    const value = percentColumns.has(column) ? scaled / 100 : scaled;
    activeFilters.push({{ column, op: filterOp.value, value, magnitude }});
    filterValue.value = "";
    renderFilters();
    renderTable();
}}

function renderFilters() {{
    filterChips.innerHTML = "";
    activeFilters.forEach((filter, index) => {{
        const chip = document.createElement("span");
        chip.className = "chip";
        const symbol = filter.op === "le" ? "≤" : "≥";
        const label = document.createElement("span");
        label.textContent = `${{filter.column}} ${{symbol}} ${{formatTooltipValue(filter.value, filter.column)}}`;
        const remove = document.createElement("button");
        remove.type = "button";
        remove.textContent = "×";
        remove.title = "Remove filter";
        remove.addEventListener("click", () => {{
            activeFilters.splice(index, 1);
            renderFilters();
            renderTable();
        }});
        chip.appendChild(label);
        chip.appendChild(remove);
        filterChips.appendChild(chip);
    }});
}}

function renderTable() {{
    const columns = showAllCols.checked ? allColumns : coreColumns;
    const query = tableFilter.value.trim().toLowerCase();
    let rows = chartData;
    if (query) {{
        rows = rows.filter((row) =>
            String(row.Portfolio || "").toLowerCase().includes(query));
    }}
    for (const filter of activeFilters) {{
        rows = rows.filter((row) => {{
            let value = row[filter.column];
            if (!Number.isFinite(value)) return false;
            if (filter.magnitude) value = Math.abs(value);
            return filter.op === "le" ? value <= filter.value : value >= filter.value;
        }});
    }}
    if (!columns.includes(sortColumn)) sortColumn = "Rank";
    rows = rows.slice().sort((rowA, rowB) => {{
        const result = compareValues(rowA[sortColumn], rowB[sortColumn]);
        return sortAscending ? result : -result;
    }});

    tableHeadRow.innerHTML = "";
    for (const column of columns) {{
        const th = document.createElement("th");
        const arrow = column === sortColumn ? (sortAscending ? " ▲" : " ▼") : "";
        th.innerHTML = `${{headerLabel(column)}}<span class="arrow">${{arrow}}</span>`;
        th.title = column;
        th.addEventListener("click", () => {{
            if (sortColumn === column) {{
                sortAscending = !sortAscending;
            }} else {{
                sortColumn = column;
                sortAscending = column === "Rank" || column === "Portfolio";
            }}
            renderTable();
        }});
        tableHeadRow.appendChild(th);
    }}

    tableBody.innerHTML = "";
    for (const row of rows) {{
        const tr = document.createElement("tr");
        tr.dataset.rank = row.Rank;
        if (String(row.Rank) === String(pinnedRank)) tr.classList.add("row-selected");
        for (const column of columns) {{
            const td = document.createElement("td");
            td.textContent = column === "Portfolio"
                ? (row.Portfolio || "")
                : formatTooltipValue(row[column], column);
            tr.appendChild(td);
        }}
        tr.addEventListener("click", () => onRowClick(row.Rank));
        tableBody.appendChild(tr);
    }}

    rowCount.textContent = `Showing ${{rows.length.toLocaleString()}} of ${{chartData.length.toLocaleString()}} portfolios`;
}}

function onRowClick(rank) {{
    const dot = dotsLayer.querySelector(`circle[data-rank="${{rank}}"]`);
    if (dot) {{
        const r = dot.getBoundingClientRect();
        if (pinnedDot === dot) {{
            clearPin();
        }} else {{
            setPin(dot, r.left + r.width / 2, r.top, false);
        }}
    }} else {{
        // Portfolio is not plotted on the current axes; just toggle the row highlight.
        highlightRow(String(rank) === String(pinnedRank) ? null : rank, false);
        pinnedRank = String(rank) === String(pinnedRank) ? null : String(rank);
    }}
}}

function highlightRow(rank, scrollIntoView) {{
    for (const tr of tableBody.querySelectorAll("tr.row-selected")) {{
        tr.classList.remove("row-selected");
    }}
    if (rank === null || rank === undefined) return;
    const tr = tableBody.querySelector(`tr[data-rank="${{rank}}"]`);
    if (tr) {{
        tr.classList.add("row-selected");
        if (scrollIntoView) tr.scrollIntoView({{ block: "nearest" }});
    }}
}}
}})();
</script>
</body>
</html>
"""
    path.write_text(doc, encoding="utf-8")


def main():
    output_dir.mkdir(parents=True, exist_ok=True)

    prices = load_prices(asset_list, start_date, end_date)
    returns = prices.pct_change(fill_method=None).dropna()
    dates = returns.index

    weights = generate_weight_grid(asset_list, weight_step, min_weights, max_weights)
    print(f"Loaded {len(dates):,} daily return rows from {dates[0].date()} to {dates[-1].date()}.")
    print(f"Testing {len(weights):,} portfolios at {weight_step:.0%} increments.")

    result_chunks = []
    for start_idx in range(0, len(weights), portfolio_chunk_size):
        end_idx = min(start_idx + portfolio_chunk_size, len(weights))
        weight_chunk = weights[start_idx:end_idx]
        print(f"Processing portfolios {start_idx + 1:,}-{end_idx:,} of {len(weights):,}...", flush=True)

        values = (
            annual_rebalanced_values(returns, weight_chunk, starting_value)
            if rebalance_annually
            else buy_and_hold_values(returns, weight_chunk, starting_value)
        )
        result_chunks.append(summarise_portfolios(values, weight_chunk, asset_list, dates, risk_free_rate_annual))

    results = pd.concat(result_chunks, ignore_index=True)

    if max_drawdown_limit is not None:
        results = results.loc[results["Max Drawdown"] >= max_drawdown_limit].copy()
        print(f"{len(results):,} portfolios remain after max drawdown filter {max_drawdown_limit:.0%}.")

    if rank_by not in results.columns:
        raise ValueError(f"rank_by column not found: {rank_by}. Available columns: {list(results.columns)}")

    results = results.sort_values(rank_by, ascending=rank_ascending).reset_index(drop=True)
    results.insert(0, "Rank", np.arange(1, len(results) + 1))
    results["Portfolio"] = results.apply(lambda row: format_weight_label(row, asset_list), axis=1)
    results["Existing Script Syntax"] = results.apply(lambda row: format_existing_script_syntax(row, asset_list), axis=1)

    csv_path = output_dir / "portfolio_optimisation_results.csv"
    html_path = output_dir / "portfolio_optimisation_results.html"
    results.to_csv(csv_path, index=False)
    meta = {
        "period_start": str(dates[0].date()),
        "period_end": str(dates[-1].date()),
        "trading_days": len(dates),
        "years": (dates[-1] - dates[0]).days / 365.25,
        "weight_step": weight_step,
        "starting_value": starting_value,
        "rebalance_annually": rebalance_annually,
        "min_weights": min_weights,
        "max_weights": max_weights,
    }
    write_interactive_scatter(results, asset_list, html_path, meta=meta)

    display_cols = ["Rank", "Portfolio", "CAGR", "Std Dev", "Sharpe", "Max Drawdown", "Ulcer Index", "Final Value"]
    print("\nTop portfolios:")
    print(results[display_cols].head(20).to_string(index=False, formatters={
        "CAGR": "{:.2%}".format,
        "Std Dev": "{:.2%}".format,
        "Max Drawdown": "{:.2%}".format,
        "Final Value": "${:,.0f}".format,
    }))
    print(f"\nSaved results: {csv_path}")
    print(f"Saved plot:    {html_path}")


if __name__ == "__main__":
    main()
