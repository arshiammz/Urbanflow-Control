# Phase 3 implementation status

## Delivered in source

- A pure NumPy successive-shortest-augmenting-path min-cost transportation
  solver with residual reverse edges and explicit unserved-demand arcs.
- Forecast-aware relocation action construction with current reserve, future
  target inventory, in-transit supply credit, a global relocation budget,
  maximum travel steps, graph adjacency, and borough-level fairness costs.
- Candidate-based receding-horizon MPC with a no-move candidate, progressively
  longer min-cost-flow candidates, expected passenger/relocation arrival
  rollouts, a Greedy tail policy, and first-action-only execution.
- A four-policy benchmark runner comparing No Rebalancing, Phase 2 Greedy,
  Min-Cost Flow, and MPC on identical demand, fleet sizing, warmup, network, and
  cost coefficients.
- Decision-time P10/P50/P90 horizon loader with dense-grid, timestamp,
  split, provenance, zone-order, non-negativity, and quantile-monotonicity
  checks.
- Zone and borough/region fairness summaries, decision latency, relative
  comparisons, and simulation invariants.
- Unit tests for global min-cost-flow rerouting, unserved-demand economics,
  fairness preference, graph/travel/budget constraints, MPC candidate rollout,
  zero-demand behavior, and simulator pending-arrival visibility.

## Validation performed while assembling this package

- Python compilation completed for the Phase 3 source and scripts.
- Focused tests passed in the assembly environment: 26 passed across
  `tests/unit/optimization`, `tests/unit/simulation/test_fleet_simulator.py`,
  `tests/unit/geo/test_zone_graph.py`, and
  `tests/unit/simulation/test_decision_time_forecasting.py`.
- Decision-time Parquet loader integration tests were added. They were skipped
  in the assembly environment because Polars is not installed here; they will
  run in the project's local environment.
- Full Parquet integration and all-zone benchmark execution were not performed
  in this assembly environment because Polars/PyArrow and the local Phase 2
  processed Parquet artifacts are not available here.
- Ruff was not available in this assembly environment. The documented Ruff
  command must be run in the project's local virtual environment.

## Final practical run required on the project machine

The following command consumes the Phase 2 all-zone inputs and writes the
benchmark report and Parquet tables:

```bash
python scripts/run_optimization_benchmark.py \
  --config configs/simulation/optimization_benchmark.yaml
```

Do not mark Phase 3 as empirically benchmarked until this command completes and
the report invariants all pass. No operational performance numbers are claimed
by this source package.
