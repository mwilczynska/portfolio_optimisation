# Agent Notes

This project builds a Python portfolio optimisation grid search using local financial datasets.

## Current Objective

Grid-search every 5% increment portfolio across the configured `asset_list`.

The default set is the un-levered "all weather" one: `GLSTOCK`, `GLBOND`, `GOLDPM`, `COMM`. The
leveraged alternative (`USLCAP3x`, `LTT3x`, `ITT3x`, `GOLDPM2x`, `COMM`) is kept commented out in
the config block.

The implementation is in `portfolio_optimizer.py`. It loads local processed CSV datasets, simulates either annually rebalanced or buy-and-hold portfolios, ranks results by a user-editable metric, writes a CSV table, and creates a dependency-free interactive HTML scatter plot.

The HTML chart supports selectable X/Y axes from numeric result columns, an X-axis threshold highlighter, hover tooltips, and click-to-pin tooltips. Pinned dots are solid purple, moved to the foreground, and can be unpinned with a second click. Below the table it renders a public-facing "About this chart" section built from `ASSET_INFO`, which adapts its wording to whichever assets were actually run.

## Important Paths

Paths are relative to this repository. The datasets repo is expected as a sibling checkout of
https://github.com/mwilczynska/financial_datasets.

- Processed datasets: `../financial_datasets/data/processed` (resolved by `DATA_ROOT`)
- Dataset methodology docs: `../financial_datasets/docs`
- Outputs: `output/`

## Working Conventions

- Keep the user-editable configuration near the top of scripts.
- Prefer local processed datasets over live web data.
- Keep outputs under `output/`. Only the demo pair `portfolio_optimisation_results.{csv,html}` is tracked; other scenario runs stay local.
- Update `LOG.md` and `HANDOVER.md` after meaningful work. These are local working notes and are deliberately gitignored, so they never reach the public repo.
- This repo is public. Keep absolute machine paths, personal email addresses and anything beyond the `mwilczynska` handle out of tracked files and commit metadata.
- Asset descriptions for the public "About" section live in `ASSET_INFO` in `portfolio_optimizer.py`. Source any new facts from the dataset methodology docs rather than inventing them.
- Prose aimed at readers should stay plain: no em-dashes, no inflated phrasing.
- `AGENTS.md` and `CLAUDE.md` should be hard linked so edits to either file are reflected in the other.
