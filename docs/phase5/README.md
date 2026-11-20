# Phase 5 — Final Evaluation & Production Delivery

Phase 5 packages the Phase 1–4 forecasting, simulation, optimization and constrained-PPO work into a reproducible evaluation and serving workflow.

## Scope

- Untouched July 2024 holdout, evaluated as one continuous 31-day simulator episode after a single June 30 no-rebalancing warmup.
- Five-policy comparison: no rebalancing, forecast-guided greedy, Min-Cost Flow, MPC and frozen Constrained PPO.
- Daily metrics, deterministic bootstrap confidence intervals, lower-tail service-rate CVaR, upper-tail operating-cost CVaR, worst-day reporting and runtime/memory profiling.
- Demand, forecast, fleet, travel-time and graph stress scenarios for the four Phase 3 optimization policies.
- FastAPI serving API, Streamlit operational dashboard, Docker Compose deployment, Prometheus metrics, MLflow report logging, drift monitoring, file-based model manifest and rollback.
- GitHub Actions quality and Docker workflows, deployment and governance documentation.

## Important evaluation boundary

The July period is the final untouched evaluation window. It must not be used to select XGBoost/quantile model complexity, PPO seeds, reward coefficients, optimizer settings or policy hyperparameters. The forecasting configuration selects model complexity on May–June and refits only on pre-July data. PPO remains the train-only checkpoint selected in Phase 4. July actual demand is used only by the final simulator evaluation.

The final evaluation uses one continuous simulator state across July. June 30 is a single no-rebalancing warmup day; fleet state is not reset between July dates. The bootstrap confidence intervals are computed over daily aggregates, but days are temporally dependent because the simulator state carries forward. The report states this caveat explicitly.

## Build July inputs and forecasts

Run from the repository root after the Phase 4 source and dependencies are installed:

```bash
python -m pip install -e '.[dev,rl,api,dashboard,mlops]'
make build-july
```

`make build-july` performs the following in order:

1. Ensure the TLC zone lookup exists.
2. Download/validate/aggregate January–July TLC trip data and create the combined 15-minute demand panel.
3. Build leakage-safe temporal features with January–April as train, May–June as validation, and July as test.
4. Build the simulation network and historical demand panel. The OD fit window intentionally remains January–April to preserve the exact graph/travel-time contract used by the frozen Phase 4 PPO checkpoint. Training fleet sizing also remains train-only.
5. Build the official zone centroids and all-eligible-zone graph.
6. Fetch archived as-of weather forecast runs through July 31. The weather feature join uses forecast runs available at each decision time; future observations are not used.
7. Build graph/weather features, select quantile model complexity on May–June, refit on pre-July data and generate July P10/P50/P90 predictions.
8. Build the decision-time quantile horizon using the train-only weekly profile for future horizon steps.

The TLC monthly downloader may fetch several large Parquet files. Allow sufficient disk space. Open-Meteo archive availability and API limits can affect weather retrieval; the weather config permits missing model runs, but the downstream feature report must be inspected for coverage before accepting the forecast artifact.

## Run the untouched holdout

```bash
python scripts/run_final_holdout.py \
  --config configs/evaluation/final_holdout_july.yaml
```

The script fails closed unless:

- Every July date is present with the configured dense 15-minute zone panel.
- June 30 warmup data is complete.
- Decision-time forecasts cover every July interval, zone and horizon step.
- P10 ≤ P50 ≤ P90 and all forecast values are finite/nonnegative.
- The quantile-model report proves the July prediction period was excluded from model selection and final training.
- The frozen PPO checkpoint hash matches the Phase 4 train-only promotion report.
- Phase 3 and PPO replay use identical fleet, network, cost and forecast contracts.

Outputs:

- `reports/evaluation/2024-07-final-holdout.json`
- `reports/evaluation/2024-07-final-holdout-daily-metrics.parquet`
- `reports/evaluation/final_holdout_july/optimization-benchmark.json`
- `reports/evaluation/final_holdout_july/rl-benchmark.json`
- Step, zone, region and action-level Parquet artifacts under `reports/evaluation/final_holdout_july/`.

## Stress scenarios

After the final-holdout runner generates the frozen benchmark config:

```bash
python scripts/run_stress_scenarios.py \
  --config configs/evaluation/stress_scenarios.yaml
```

The scenarios are demand surge (+25%), forecast underestimation (−25%), fleet shortage (−20%), travel-time disruption (+50%) and graph edge loss (15%, while preserving at least one neighbour per zone). This stress suite compares the four Phase 3 optimization policies. PPO stress replay is intentionally outside this suite; any graph/OD perturbation requires a retrained checkpoint and matching graph-contract verification. The final July holdout comparison includes the frozen PPO policy.

## Production API

The API consumes the generated July decision-time forecast file and matching graph/network configs. The generated runtime configs are written by the final-holdout script to `artifacts/generated_configs/final_holdout_july/`.

```bash
cp .env.example .env
# Set a long random URBANFLOW_API_KEY in .env.
docker compose up --build -d
```

Endpoints:

- `GET /health` — liveness and artifact status.
- `GET /ready` — load and validate all serving artifacts.
- `GET /metrics` — Prometheus metrics.
- `GET /v1/models/current` — promoted model manifest and hash verification.
- `GET /v1/zones` — eligible zones and centroid coordinates.
- `GET /v1/zones/{zone_id}/status` — quantile demand forecast for one zone.
- `POST /v1/demand/forecast` — P10/P50/P90 for a decision time and horizon.
- `POST /v1/rebalancing/plan` — Greedy, Min-Cost Flow, MPC or Constrained PPO plan.
- `POST /v1/scenarios/simulate` — one-step simulator preview.

Protected endpoints accept `X-API-Key` when `URBANFLOW_API_KEY` is set. Do not expose this demo API directly to the public internet without TLS, authentication, rate limiting and a proper secrets manager. The API uses a request-supplied fleet snapshot; it is not connected to a live vehicle telemetry system.

The one-step simulation endpoint initializes the submitted available fleet as on-hand supply. It is a scenario preview, not a continuation of an active multi-step fleet session. MPC requires a pending-arrival schedule whenever in-transit supply is nonzero. PPO inference requires the optional RL dependency group and a checkpoint whose graph/constraint signature matches the serving artifacts.

## Monitoring and MLflow

Build a train-only demand reference:

```bash
python scripts/build_monitoring_reference.py
```

Compare a new batch against the reference:

```bash
python scripts/run_monitoring_report.py --split test
```

PSI thresholds default to 0.10 (warning) and 0.25 (alert). They are operational heuristics, not statistical hypothesis tests. Monitor forecast residuals, zone/borough service rates and action distributions alongside PSI before taking action.

Log a report to MLflow:

```bash
python scripts/log_experiment_to_mlflow.py \
  --report reports/evaluation/2024-07-final-holdout.json
```

Promote and verify the model bundle after the final evaluation is accepted:

```bash
python scripts/create_production_manifest.py
python scripts/verify_production_manifest.py
```

Rollback the local manifest to its previous version:

```bash
python scripts/rollback_production_manifest.py
python scripts/verify_production_manifest.py
```

The file-based manifest stores SHA-256 hashes for all quantile model files, PPO checkpoint, metadata and zone graph. It is a lightweight local registry; use a managed model registry and object storage for multi-user production.

## CI and local commands

```bash
make lint
make test
python scripts/check_production_readiness.py --static-only
```

GitHub Actions runs Ruff, unit tests, static production preflight and a Docker build. It does not download TLC data or run the July holdout in CI. Full data evaluation is an explicit, reproducible local workflow.

## Limitations and interpretation

- All service, cost and fairness metrics are simulator estimates based on historical TLC demand and OD profiles; they are not measured passenger wait times or observed driver outcomes.
- The network and OD profile are frozen to the Phase 4 contract so the frozen PPO checkpoint remains valid. A future network/OD refresh requires a new PPO checkpoint and graph-contract verification.
- The API accepts a fleet snapshot from its caller; no live dispatch, driver app or real-time telemetry integration is included.
- Streamlit dashboard uses zone centroids for visual context, not taxi-zone polygon boundaries.
- MLflow and Prometheus are provided as local operational infrastructure. Configure durable storage, authentication and retention policies before shared deployment.
