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

import html
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
    "GOLDPM2x": 0.00,
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
    width, height = 1100, 640
    left, right, top, bottom = 90, 40, 35, 100
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

    top_rows = results.head(10).copy()
    top_rows["Portfolio"] = top_rows.apply(lambda row: format_weight_label(row, tickers), axis=1)
    table_rows = []
    for _, row in top_rows.iterrows():
        table_rows.append(
            "<tr>"
            f"<td>{html.escape(row['Portfolio'])}</td>"
            f"<td>{row['CAGR']:.2%}</td>"
            f"<td>{row['Max Drawdown']:.2%}</td>"
            f"<td>{row['Sharpe']:.2f}</td>"
            f"<td>{row['Ulcer Index']:.2f}</td>"
            "</tr>"
        )

    doc = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>CAGR vs Max Drawdown</title>
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
svg {{ max-width: 100%; height: auto; border: 1px solid #d1d5db; background: white; }}
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
    border-color: #7c3aed;
    box-shadow: 0 8px 24px rgba(124, 58, 237, 0.22);
    z-index: 9;
}}
table {{ border-collapse: collapse; margin-top: 22px; font-size: 13px; }}
th, td {{ border: 1px solid #d1d5db; padding: 7px 9px; text-align: right; }}
th:first-child, td:first-child {{ text-align: left; }}
th {{ background: #f3f4f6; }}
</style>
</head>
<body>
<h1>CAGR vs Max Drawdown</h1>
<p>Pick any numeric result columns for the axes. Hover a dot to identify the portfolio weights and core statistics.</p>
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
<svg viewBox="0 0 {width} {height}" role="img" aria-label="CAGR versus max drawdown scatter plot">
<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#111827"/>
<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#111827"/>
<g id="grid-layer"></g>
<g id="dots-layer"></g>
<text id="x-axis-label" x="{left + plot_w / 2:.0f}" y="{height - 30}" text-anchor="middle">Max Drawdown</text>
<text id="y-axis-label" x="22" y="{top + plot_h / 2:.0f}" transform="rotate(-90 22 {top + plot_h / 2:.0f})" text-anchor="middle">CAGR</text>
</svg>
<table>
<thead><tr><th>Portfolio</th><th>CAGR</th><th>Max Drawdown</th><th>Sharpe</th><th>Ulcer Index</th></tr></thead>
<tbody>{''.join(table_rows)}</tbody>
</table>
<script>
(() => {{
const chartData = {chart_data};
const numericColumns = {column_data};
const percentColumns = new Set({percent_column_data});
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
let pinnedDot = null;

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
updateThresholdOptions();
renderChart();

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
        dot.addEventListener("mouseenter", showHoverTooltip);
        dot.addEventListener("mousemove", moveHoverTooltip);
        dot.addEventListener("mouseleave", hideHoverTooltip);
        dot.addEventListener("click", togglePinnedTooltip);
        dotsLayer.appendChild(dot);
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
        pinnedDot.setAttribute("fill", pinnedDot.dataset.baseColor);
        pinnedDot.setAttribute("opacity", "0.62");
        pinnedDot = null;
        hidePinnedTooltip();
        return;
    }}

    if (pinnedDot) {{
        pinnedDot.setAttribute("fill", pinnedDot.dataset.baseColor);
        pinnedDot.setAttribute("opacity", "0.62");
    }}
    pinnedDot = dot;
    dotsLayer.appendChild(pinnedDot);
    pinnedDot.setAttribute("fill", "#7c3aed");
    pinnedDot.setAttribute("opacity", "1");
    hideHoverTooltip();
    pinnedTooltip.textContent = pinnedDot.dataset.tooltip;
    pinnedTooltip.style.display = "block";
    positionTooltip(pinnedTooltip, event);
}}

function positionTooltip(tooltipEl, event) {{
    const pad = 14;
    const rect = tooltipEl.getBoundingClientRect();
    let left = event.clientX + pad;
    let top = event.clientY + pad;
    if (left + rect.width > window.innerWidth) {{
        left = event.clientX - rect.width - pad;
    }}
    if (top + rect.height > window.innerHeight) {{
        top = event.clientY - rect.height - pad;
    }}
    tooltipEl.style.left = Math.max(8, left) + "px";
    tooltipEl.style.top = Math.max(8, top) + "px";
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
