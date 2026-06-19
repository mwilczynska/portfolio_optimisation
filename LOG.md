# Log

## 2026-06-19 (UX polish)

- Centred the scatter plot (`svg` now `display: block; margin: 0 auto`).
- Confirmed the X/Y axis titles already update live with the dropdown selections (verified CAGR -> Final Value in-browser); no code change needed.
- Fixed the pinned (click) tooltip drifting on scroll: changed `#pinned-tooltip` from `position: fixed` to `position: absolute` and anchored it in document/page coordinates (add `window.scrollX/scrollY` in `positionTooltip`). It now stays beside its dot when the page scrolls. The hover tooltip stays `fixed` and tracks the cursor.
- Renamed the page `<title>`, the `<h1>`, and the SVG aria-label from "CAGR vs Max Drawdown" to "Portfolio Optimisation Results" (axes are user-selectable, so the old fixed title was misleading).

## 2026-06-19 (later)

- Added numeric range filtering to the HTML table: a filter builder (metric dropdown + ≥/≤ + value, with a live "range: …" hint) lets you add filters such as `Max Drawdown ≥ -65%`. Active filters show as removable chips, plus a "Clear all" button; filters combine with the text filter and update the row count. Percent columns are entered as percentages and converted to fractions internally.
- Made the metric filter magnitude-aware for loss-type columns: any column whose values are all <= 0 (Max Drawdown, Average Drawdown, Worst Day/Month/Year, VaR/CVaR) now filters on the absolute value, so `Max Drawdown <= 60%` keeps drawdowns no deeper than 60% (the intuitive reading) instead of matching every row. The value is entered as a positive magnitude; the hint shows "magnitude X to Y (enter as %, e.g. 60)" and the chip shows the positive magnitude. Non-loss columns keep signed behaviour.
- Reduced the scatter plot size: viewBox height 640 -> 560, and capped the rendered SVG at `max-width: 760px` so the chart no longer dominates the page above the table.
- Verified live in-browser (served over `http://127.0.0.1`, since `file://` was mangled by the extension): the `Max Drawdown ≥ -65%` filter narrowed 969 -> 53 rows, all with drawdowns better than -65%.

## 2026-06-19

- Reviewed the rendered HTML chart (headless Chrome screenshot) for UX issues. Main weakness: the results table was a static top-10 with a verbose `Portfolio` label, no sorting/filtering, only 5 of ~25 metrics, and no link to the chart.
- Replaced the static table in `write_interactive_scatter` with a dependency-free interactive table driven by the data already embedded for the chart:
  - Sortable column headers (click to toggle asc/desc, arrow indicator).
  - A text filter box that matches against the portfolio weight label, with a live "Showing X of N portfolios" count.
  - Capped-height (460px) scroll area with a sticky header.
  - Compact per-asset weight columns (headers show the ticker only) plus a core metric set; a "Show all columns" checkbox switches to every numeric column.
  - Chart <-> table linking: clicking/pinning a dot highlights and scrolls to its row; clicking a row pins its dot. Pin state survives axis changes via the row Rank.
- Refactored the pin/tooltip helpers (`setPin`/`clearPin`/`markDotPinned`/`restoreDot`, coordinate-based `positionTooltip`) so pinning can be triggered from either the chart or the table.
- Removed the now-unused `html` import and the Python-side static table builder.
- Regenerated `output/cagr_vs_max_drawdown.html` from the existing 969-row results CSV (HTML is a pure function of the CSV, so the simulation was not re-run).

## 2026-06-17

- Read `PLAN.md`.
- Inspected the prior portfolio backtesting script at `C:\Datascience\python\investment_tracker\portfolio_plot15.py`.
- Confirmed the required processed datasets exist under `C:\Datascience\projects\finance\financial_datasets\data\processed`.
- Created initial `portfolio_optimizer.py`:
  - Tests all 5% increment long-only portfolios for `USLCAP3x`, `LTT3x`, `ITT3x`, `GOLDPM2x`, and `COMM`.
  - Supports user-editable min/max asset weights.
  - Supports an optional max drawdown filter.
  - Calculates CAGR, TWRR, annual return, standard deviation, Sharpe, Sortino, Calmar, max drawdown, Ulcer Index, average drawdown, longest drawdown duration, win rate, best/worst day, best/worst month, best/worst year, VaR, CVaR, skew, and excess kurtosis.
  - Writes ranked results to `output/portfolio_optimisation_results.csv`.
  - Writes a dependency-free hoverable HTML scatter plot to `output/cagr_vs_max_drawdown.html`.
- Created `AGENTS.md`, `LOG.md`, and `HANDOVER.md`.
- Added `Existing Script Syntax` to the results table so top portfolios can be pasted into the prior backtesting script.
- Changed the default optimisation grid to 20% increments for faster test runs; left the 5% grid setting commented in `portfolio_optimizer.py`.
- Increased the HTML scatter plot bottom margin so the x-axis tick labels and axis title are not clipped.
- Replaced delayed native SVG hover titles with an instant custom tooltip and larger popup text.
- Reduced the HTML scatter plot height and added X/Y axis dropdowns for numeric CSV result columns.
- Added a max drawdown highlight picker to the HTML chart; selected thresholds color matching/worse drawdowns red without filtering points.
- Changed the highlight picker to follow the selected X-axis metric instead of always referring to max drawdown.
- Wrapped the generated HTML chart JavaScript in a private scope so dropdown population does not collide with browser globals; defaults remain X `Max Drawdown`, Y `CAGR`.
- Changed HTML chart grid ticks to use more frequent rounded intervals instead of sparse decimal-spaced intervals.
- Added click-to-pin chart tooltips with purple selected dots, and removed duplicate CAGR/max-drawdown tooltip lines when those metrics are already selected as axes.
- Split hover and pinned chart tooltips so hover remains transient, clicked selections stay visible, selected dots remain the same size, and clicking the selected dot again clears the pin.
- Moved pinned chart dots to the foreground and made them fully opaque so selected portfolios are not hidden behind overlapping dots.
