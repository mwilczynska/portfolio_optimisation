# Handover

## Status

Initial optimizer implementation is in `portfolio_optimizer.py`.

Run with:

```powershell
python portfolio_optimizer.py
```

Expected outputs:

- `output/portfolio_optimisation_results.csv`
- `output/cagr_vs_max_drawdown.html`

## Notes For Next Agent

- The workspace was not a Git repository when first inspected (`git status` failed with "not a git repository"). If commits/pushes are required, initialize or clone the GitHub repo before committing.
- The active Python environment has `pandas` and `numpy`, but did not have `matplotlib`, `plotly`, or `openpyxl` at initial inspection. The optimizer therefore avoids non-core plotting dependencies and writes a standalone HTML/SVG scatter plot.
- `AGENTS.md` and `CLAUDE.md` must be hard linked. `AGENTS.md` has been created; verify/create the hard link before committing.
- The current optimizer assumes long-only portfolios summing to 100%. It does not yet model contributions, cash flows, leverage beyond the leveraged source assets, taxes, transaction costs, or inflation adjustment.
- Metrics currently use `risk_free_rate_annual = 0.0` unless edited in the script.
- Review whether `TWRR = CAGR` is acceptable for this optimizer. With no contributions, they are equivalent.
- Results include an `Existing Script Syntax` column in the form `tickers, weights = ['USLCAP3x','LTT3x',...],[0.35,0.05,...]` for direct copy/paste into the legacy script.
- The active `weight_step` is currently `0.05` for the full grid run. In `portfolio_optimizer.py`, switch to the adjacent `weight_step = 0.20` line for faster test runs.
- The HTML scatter plot is dependency-free and interactive:
  - X/Y selectors are populated from numeric result columns and default to X `Max Drawdown`, Y `CAGR`.
  - The highlight selector follows the current X-axis metric and colors points red when their X value is at or below the selected threshold.
  - Hover shows a transient tooltip.
  - Clicking a dot pins a separate tooltip, turns the dot solid purple, and moves it to the foreground.
  - Clicking the same purple dot again clears the pinned tooltip and restores the dot.
  - Tooltip text suppresses duplicate core metrics already displayed by the selected axes.
- The results table below the chart is now a full interactive table (not a static top-10):
  - Sortable headers, a text filter, a sticky-header scroll area, and a live row count.
  - Defaults to a core column set (Rank, per-asset weights, CAGR, Max Drawdown, Sharpe, Sortino, Calmar, Ulcer Index, Final Value); the "Show all columns" checkbox switches to every numeric column.
  - It reads the same embedded `chartData` as the chart, so no extra data is written; the whole HTML is still a pure function of the results CSV (regenerate via `write_interactive_scatter` without re-running the simulation).
  - Chart and table are linked by row `Rank`: pinning a dot highlights/scrolls to its row and vice versa.
  - Numeric range filters: a filter builder adds constraints like `Max Drawdown <= 60%` (shown as removable chips, combinable, "Clear all" to reset). Percent columns are typed as percentages and stored as fractions.
  - Loss-type columns (all values <= 0: drawdowns, worst periods, VaR/CVaR) are magnitude-aware via `isMagnitudeColumn` — the user enters a positive magnitude and the filter compares on `|value|`, so `Max Drawdown <= 60%` keeps drawdowns shallower than 60%. The hint reads "magnitude X to Y". Other columns use signed comparison.
  - The scatter SVG is intentionally compact (viewBox 1100x560, capped at `max-width: 760px`) so the table stays visible.
  - To regenerate just the HTML from an existing CSV: `import portfolio_optimizer as po; po.write_interactive_scatter(pd.read_csv(csv), po.asset_list, html_path)`.

## Follow-Up Ideas

- Add command-line arguments while preserving the editable config block.
- Add optional benchmark comparison against the user-selected legacy portfolio format.
- Add saved chart presets for common comparisons such as Sharpe vs. max drawdown or CAGR vs. Ulcer Index.
- Add tests for weight-grid generation and core metric calculations.
