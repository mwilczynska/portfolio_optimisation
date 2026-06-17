# Log

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
