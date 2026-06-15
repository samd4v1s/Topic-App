# TLG Project Timelines

Probabilistic project duration forecasting using Monte Carlo simulation with Bayesian updating. Takes two-point time estimates (optimistic, pessimistic) from subject matter experts, imputes a most-likely midpoint, and produces P50/P90 confidence intervals corrected for systemic estimation bias using historical actuals.

## Setup

Requires Python 3.10+ and [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

## Data

Place CSV files in the `data_synthetic/` directory. Two formats are used:

- **Wide format** (`Raw-data-wide.csv`) — one row per respondent, with columns like `Phase 1 Minimum`, `Phase 1 Maximum`, etc. Used by the Monte Carlo simulation.
- **Long format** (`Raw-data-long.csv`) — one row per respondent/scenario combination, with columns `Team`, `Scenario`, `Phase 1`...`Phase 5`. Used by the descriptive analysis.

## Running the pipeline

A `Makefile` is provided to run both scripts with their dependencies tracked. Make will only rerun a step if its inputs have changed.

```bash
# Run everything (simulation + descriptive analysis)
make

# Run just the Monte Carlo simulation
make simulation

# Run just the descriptive analysis
make descriptive

# Run the test suite
make test

# Remove all generated outputs
make clean
```

Outputs are saved to `output/simulation/` and `output/descriptive/`.

### Running scripts individually

You can also run each script directly:

```bash
uv run python MC_simulation.py data_synthetic/Raw-data-wide.csv output/simulation
uv run python Descriptive.py data_synthetic/Raw-data-long.csv output/descriptive
```

## Scripts

### `MC_simulation.py`

Runs 10,000 Monte Carlo simulations using PERT-Beta distributions with an AR(1) copula correlation structure across stages. Calibrates estimates against historical actuals via Normal-Inverse-Gamma Bayesian updating and produces three plots:

- Systemic error multiplier KDEs (prior vs posterior)
- Project duration histograms (baseline vs bias-corrected)
- Task-level posterior distributions vs historical observations

### `Descriptive.py`

Aggregates survey responses by team and scenario, produces a stacked bar chart comparing Max vs Min timelines, and writes an HTML summary table.

### `colour_palette.py`

Centralised TLG brand colour definitions and `apply_theme()` function used across both scripts for consistent matplotlib theming.
