# Evaluation Protocol

## Development and model selection

Use historical training/validation splits only for feature decisions, estimator selection, PPO seed selection, reward tuning and optimizer tuning. Record configs, source hashes and model/checkpoint hashes. Do not select policy settings using the final July test period.

## Final holdout

The final holdout is July 1–31, 2024. The simulator uses June 30 as a shared no-rebalancing warmup and then carries fleet state continuously across all 31 July days. All five policies use identical demand, fleet, network and cost coefficients.

The final report includes aggregate demand/service/cost, daily metrics, bootstrap confidence intervals, service-rate CVaR, cost CVaR, worst day, fairness, latency and process memory. Bootstrap resampling is over daily metrics; because the fleet state carries forward, daily samples are dependent and the intervals are descriptive rather than independent-day inferential guarantees.

## Stress tests

Stress tests alter demand, forecasts, fleet reserve, travel times and graph edges. Every scenario must preserve simulator invariants. When a graph or OD travel-time contract changes, the frozen PPO checkpoint is not re-used unless it is retrained and re-verified against the new contract.

## Reporting

Report both aggregate and segment results. Do not present simulated service rate or operating cost as observed real-world performance. Keep test metrics separate from model/policy selection artifacts.
