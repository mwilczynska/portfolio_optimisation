"""
Portfolio optimisation grid search
----------------------------------
Tests all 5% increment portfolios for:
USLCAP3x, LTT3x, ITT3x, GOLDPM2x, COMM

Outputs:
    output/portfolio_optimisation_results.csv
    output/cagr_vs_max_drawdown.html

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

asset_list = ["USLCAP3x", "LTT3x", "ITT3x", "GOLDPM2x", "COMM"]
# weight_step = 0.20  # quick test grid
weight_step = 0.05  # full grid; uncomment for the 5% increment run

# Optional minimum floors and maximum roofs for each asset.
# Values are decimals: 0.10 means 10%. Leave at 0.00 / 1.00 for unrestricted.
min_weights = {
    "USLCAP3x": 0.00,
    "LTT3x": 0.00,
    "ITT3x": 0.00,
    "GOLDPM2x": 0.20,
    "COMM": 0.00,
}

max_weights = {
    "USLCAP3x": 1.00,
    "LTT3x": 1.00,
    "ITT3x": 1.00,
    "GOLDPM2x": 1.00,
    "COMM": 1.00,
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
CUSTOM_DATASETS = {
    "USLCAP3x": {
        "path": Path(r"C:\Datascience\projects\finance\financial_datasets\data\processed\us_large_cap_3x_sp500.csv"),
        "date_col": "Date",
        "price_col": "Adj Close",
    },
    "LTT3x": {
        "path": Path(r"C:\Datascience\projects\finance\financial_datasets\data\processed\long_term_us_treasury_3x.csv"),
        "date_col": "Date",
        "price_col": "Adj Close",
    },
    "ITT3x": {
        "path": Path(r"C:\Datascience\projects\finance\financial_datasets\data\processed\intermediate_term_us_treasury_3x.csv"),
        "date_col": "Date",
        "price_col": "Adj Close",
    },
    "GOLDPM2x": {
        "path": Path(r"C:\Datascience\projects\finance\financial_datasets\data\processed\gold_2x.csv"),
        "date_col": "Date",
        "price_col": "Adj Close",
    },
    "COMM": {
        "path": Path(r"C:\Datascience\projects\finance\financial_datasets\data\processed\broad_commodities.csv"),
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


def write_interactive_scatter(results, tickers, path):
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
    <input type="text" id="table-filter" placeholder="Filter portfolios… (e.g. GLD, 60%)" autocomplete="off">
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
    html_path = output_dir / "cagr_vs_max_drawdown.html"
    results.to_csv(csv_path, index=False)
    write_interactive_scatter(results, asset_list, html_path)

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
