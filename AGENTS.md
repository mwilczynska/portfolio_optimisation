# Agent Notes

This project builds a Python portfolio optimisation grid search using local financial datasets.

## Current Objective

Test all 5% increment portfolios across:

- `USLCAP3x`
- `LTT3x`
- `ITT3x`
- `GOLDPM2x`
- `COMM`

The first implementation is in `portfolio_optimizer.py`. It loads local processed CSV datasets, simulates either annually rebalanced or buy-and-hold portfolios, ranks results by a user-editable metric, writes a CSV table, and creates a dependency-free interactive HTML scatter plot.

The HTML chart supports selectable X/Y axes from numeric result columns, an X-axis threshold highlighter, hover tooltips, and click-to-pin tooltips. Pinned dots are solid purple, moved to the foreground, and can be unpinned with a second click.

## Important Paths

- Workspace: `C:\Datascience\projects\finance\portfolio_optimisation`
- Processed datasets: `C:\Datascience\projects\finance\financial_datasets\data\processed`
- Dataset methodology docs: `C:\Datascience\projects\finance\financial_datasets\docs`
- Prior style reference: `C:\Datascience\python\investment_tracker\portfolio_plot15.py`

## Working Conventions

- Keep the user-editable configuration near the top of scripts.
- Prefer local processed datasets over live web data.
- Keep outputs under `output/`.
- Update `LOG.md` and `HANDOVER.md` after meaningful work.
- `AGENTS.md` and `CLAUDE.md` should be hard linked so edits to either file are reflected in the other.
