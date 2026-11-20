# Phase 3 — Constrained Fleet Optimization

Phase 3 adds a decision-optimization benchmark on top of the Phase 2 forecasting
artifacts and the existing fleet simulator. It does not retrain or replace the
Phase 2 forecasting engine.

The benchmark compares four policies under the same initial fleet, historical
demand sequence, warmup period, OD network, travel-time matrices, graph, and
operating-cost coefficients:

1. **No Rebalancing** — no empty-vehicle relocation.
2. **Greedy** — the existing Phase 2 forecast-guided greedy policy.
3. **Min-Cost Flow** — a global transportation optimization with relocation
   costs, unmet-demand penalties, region fairness costs, and explicit
   unserved-demand arcs.
4. **MPC** — candidate-based receding-horizon control. It evaluates first-action
   candidates over the Phase 2 forecast horizon, rolls each candidate forward
   using expected demand and the training-fitted OD/travel-time model, and
   applies only the selected first action.

## 1. Phase 2 inputs consumed

The default configuration is
`configs/simulation/optimization_benchmark.yaml`.

| Input | Purpose |
|---|---|
| `configs/simulation/real_episode_baseline_all_zones.yaml` | Same fleet sizing, historical demand, OD matrix, costs, and evaluation date as Phase 2 |
| `configs/simulation/forecast_guided_heuristic_all_zones.yaml` | Reuses the Phase 2 Greedy policy controls and P50/P90 risk-aversion setting |
| `data/processed/full/decision_time_quantile_horizon_15m_2024-01_2024-06.parquet` | Dense, leakage-safe P10/P50/P90 forecasts for each decision time and horizon step |
| `data/processed/spatial/nyc_taxi_zone_graph_all_eligible.json` | Geometry-only graph used to restrict relocation edges |
| `data/reference/taxi_zone_lookup.csv` | Maps zones to TLC borough/region labels for fairness accounting |

The forecast loader fails closed unless the decision-time grid is dense, the
zone ordering matches the simulation network, the temporal split matches the
evaluation split, the target timestamps match their decision-time horizon,
quantiles are finite and monotonic, and the provenance is
`quantile_anchor_plus_train_only_weekly_profile`.

The policy signal is the Phase 2 risk-adjusted forecast:

```text
risk_adjusted = (1 - risk_aversion) * P50 + risk_aversion * P90
```

No future realized demand or future offline one-step predictions are passed to
the optimization policies.

## 2. Min-Cost Flow formulation

For each decision interval, the optimizer derives:

- origin supply: available vehicles above the current-interval reserve;
- destination deficit: forecast target inventory minus available and credited
  in-transit supply;
- feasible arcs: graph neighbours whose relocation travel time does not exceed
  the configured maximum;
- a global relocation budget: a fixed fraction of currently available fleet.

The transportation network has origin-to-destination relocation arcs and an
explicit dummy source for unserved destination demand. Each unit is therefore
assigned either to a real relocation or to an unserved-demand arc. The solver
uses successive shortest augmenting paths with reduced-cost Dijkstra and
residual reverse edges, so it can reroute earlier assignments instead of using
a cheapest-edge greedy approximation.

The per-unit relocation objective includes:

```text
travel_steps * relocation_cost_per_vehicle_step
+ fairness_cost_weight * historical_region_service_rate
```

The unserved-demand arc includes the unmet-demand penalty and a fairness credit
for regions with lower historical service rates. The historical rate is
computed only from demand and service observed before the current decision.

## 3. MPC formulation

The MPC controller uses a compact candidate search rather than claiming a
large-scale exact mixed-integer optimum:

1. Generate a no-relocation candidate.
2. Generate Min-Cost Flow first-action candidates for progressively longer
   forecast windows (1 through `H-1`).
3. Roll each candidate through the complete expected-demand horizon.
4. Project passenger arrivals using the training-fit OD probabilities and
   passenger travel steps; project relocation arrivals using relocation travel
   steps.
5. At later simulated steps, use the Phase 2 Greedy policy as the tail policy.
6. Score projected operating cost plus a regional service-fairness penalty.
7. Apply only the first action of the best candidate to the real simulator.

This is a deterministic expected-value, candidate-based receding-horizon MPC
approximation. It is intentionally described as such; it is not a global joint
MILP/MINLP solver. Replanning happens at every real 15-minute decision point.

## 4. Shared constraints and fairness

Every action is checked before it reaches the simulator:

- no negative relocation counts or self-relocations;
- origin outflow cannot exceed vehicles available at the decision boundary;
- total relocations cannot exceed the configured fleet fraction;
- relocation edges must exist in the Phase 2 zone graph;
- relocation travel time must respect the configured step limit;
- the simulator must conserve the complete fleet, including vehicles in transit.

Fairness is measured at both zone and TLC region/borough level. The benchmark
reports minimum and 10th-percentile service rates, service-rate gap, and Jain's
fairness index. The optimization objective also gives a soft priority to
historically under-served regions; it does not silently change the simulator's
reported operating-cost definition.

## 5. Reproducible run

Run from the repository root with the project's virtual environment active.

```bash
# Install the project and development tools if not already installed.
pip install -e '.[dev]'

# Pure optimization and simulator-contract tests.
pytest -q tests/unit/optimization tests/unit/simulation/test_fleet_simulator.py

# Lint the Phase 3 files.
ruff check \
  src/urbanflow/optimization \
  src/urbanflow/simulation/optimization_benchmark.py \
  src/urbanflow/simulation/fleet_simulator.py \
  scripts/run_optimization_benchmark.py \
  tests/unit/optimization

# Run the four-policy benchmark.
python scripts/run_optimization_benchmark.py \
  --config configs/simulation/optimization_benchmark.yaml
```

The default run evaluates **2024-05-15** using the validation split and a
one-day shared warmup, matching the existing Phase 2 all-zone episode
configuration. June remains the development-test period and is not used for
policy selection in this benchmark. Change the baseline configuration only
when intentionally defining a new evaluation protocol.

## 6. Outputs

| Artifact | Contents |
|---|---|
| `reports/simulation/optimization_benchmark/2024-05-15-step-metrics.parquet` | One row per policy and evaluation interval |
| `reports/simulation/optimization_benchmark/2024-05-15-interval-zone-metrics.parquet` | Forecast, demand, service, and relocation by policy/interval/zone |
| `reports/simulation/optimization_benchmark/2024-05-15-zone-summary.parquet` | Cumulative zone service and relocation metrics |
| `reports/simulation/optimization_benchmark/2024-05-15-region-summary.parquet` | Cumulative borough/region service metrics |
| `reports/simulation/2024-05-15-optimization-benchmark.json` | Policy comparison, fairness, latency, MPC diagnostics, and invariants |

The JSON report includes comparisons against both No Rebalancing and Greedy:

- unmet-demand reduction;
- service-rate improvement in percentage points;
- operating-cost reduction;
- worst-region service-rate change;
- regional Jain-fairness change.

It also reports mean decision latency. Do not interpret simulated service rate
as a measured real-world passenger wait-time improvement; the simulator uses a
historical OD/travel-time approximation.

## 7. Interpretation and limitations

- The Min-Cost Flow objective is a one-step allocation over the near-term
  arrival window; it does not optimize a complete future action sequence.
- MPC evaluates a finite set of first-action candidates and uses Greedy as its
  future tail policy. It is scalable and auditable, but not a global exact
  horizon optimizer.
- Forecasts beyond the first interval use the Phase 2 train-only weekly
  projection anchored to the decision-time quantile forecast. They are not
  independently weather-refreshed at each future step.
- Relocation feasibility is based on the Phase 2 centroid-derived graph and
  historical travel-step matrix. It is not live road routing.
- Borough fairness is a proxy for geographic service equity; it does not
  establish individual passenger-level fairness.
- The full benchmark must be executed on the project machine where the Phase 2
  Parquet forecasts, all-zone simulation inputs, graph, and TLC lookup are
  present. Source compilation and pure unit tests alone are not full data-run
  evidence.

Phase 3 establishes the optimization benchmark that the later RL policy must
compete against. It does not implement or claim an RL policy.
