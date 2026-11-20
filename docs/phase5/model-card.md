# Model Card — UrbanFlow Demand Forecasting and Fleet Rebalancing

## Intended use

Research and portfolio demonstration of forecast-aware fleet rebalancing on NYC TLC for-hire vehicle demand. The forecast model produces zone-level P10/P50/P90 demand estimates. Policies use these forecasts to propose vehicle relocations in a deterministic fleet simulator.

## Components

- Quantile demand forecaster: XGBoost quantile models with graph-derived zone context and decision-time-safe weather forecast features.
- Multi-step forecast: one-step quantile anchor plus train-only weekly/temporal profile projection.
- Rebalancing policies: forecast-guided Greedy, Min-Cost Flow, candidate-based receding-horizon MPC and constrained PPO.
- Hard action constraints: zone graph edges, available origin supply, relocation travel-time limit, global relocation budget and nonnegative integer moves.

## Evaluation protocol

Model complexity selection uses pre-holdout validation data. PPO checkpoint/seed selection uses the Phase 4 train-only protocol. July 2024 is reserved for final evaluation and must not influence model, seed, reward or policy selection. The final report is created by `scripts/run_final_holdout.py`.

## Metrics

Demand: MAE/pinball loss and quantile coverage. Simulation: served/unmet demand, service rate, operating cost, relocations, zone/region fairness, decision latency, daily bootstrap intervals and CVaR.

## Known limitations

- Forecast quality is tied to TLC records and the eligible-zone set; it is not a real-time demand guarantee.
- Graph-derived context is not a learned message-passing GNN.
- Future multi-step quantiles use a train-only temporal profile, not recursive re-inference of a fully dynamic model at every future step.
- Simulator outputs are not direct measurements of real passenger wait times, driver earnings or actual fleet availability.
- Fairness is partly represented through soft objective/reward terms; hard constraints guarantee action feasibility, not equal service rates.
- A graph or OD travel-time update invalidates a PPO checkpoint unless its stored graph-contract signature still matches.

## Operational controls

The API verifies PPO checkpoint provenance and graph/constraint compatibility before inference. Production model promotion writes SHA-256 hashes for all model files and graph artifacts. Monitoring uses train-only demand reference statistics and batch PSI.
