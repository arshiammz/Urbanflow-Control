# Phase 2 implementation status

## Delivered in source

- Archived ECMWF IFS run ingestion with a six-hour as-of availability boundary.
- TLC shapefile centroid builder and geometry-only zone graph.
- Graph-neighbour lag features and borough-level weather features joined to each zone.
- Global XGBoost P10/P50/P90 training with April-only model-complexity selection.
- Monotonic quantile correction, pinball loss, empirical coverage, interval width, and June evaluation metrics.
- Train-only multi-step quantile horizon projection.
- All-fit-eligible-zone simulation input configuration.
- Optional graph-edge relocation constraints and P50/P90 risk-aversion blend in the simulator policy.
- Phase 2 runbook, configs, and unit tests.

## Validation performed while assembling this package

- Python `compileall` completed successfully for `src`, `scripts`, and `tests`.
- YAML parsing completed successfully for all eight new Phase 2 configs.
- `pyproject.toml` parsed successfully.
- 25 focused unit tests passed, including graph, quantile, decision-time projection, and simulator adjacency tests.

## Intentionally deferred to the final practical test

The full local data pipeline, Open-Meteo archive download, all-zone Parquet build,
XGBoost quantile training, and all-zone simulation were not executed in this
package-building environment. The Phase 1 project archive does not include the
local processed Parquet data, and this environment does not have Polars/PyArrow
installed. Per the project workflow, run the end-to-end commands in
[`README.md`](README.md) when the final practical test is scheduled.

No Phase 2 model-performance numbers are claimed before that run.
