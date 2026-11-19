# Phase 2 — Advanced Spatiotemporal Forecasting

Phase 2 extends the Phase 1 leakage-safe forecasting and simulation system with
weather-vintage features, a geometry-derived taxi-zone graph, probabilistic
P10/P50/P90 demand forecasts, and an all-eligible-zone simulation network.

The implementation is deliberately a **graph-enhanced global XGBoost quantile
forecaster**: zone adjacency is represented through weighted neighbour-lag
features. It is not presented as a Graph Neural Network (GNN). This keeps the
model compatible with the existing XGBoost stack while exposing a graph
contract that later optimization/RL policies can consume.

## 1. What is added

| Component | Implementation | Main artifact |
|---|---|---|
| Zone geometry | Official TLC taxi-zone shapefile → WGS84 centroid per zone | `data/reference/taxi_zone_centroids.csv` |
| Zone graph | Symmetric 8-nearest-neighbour graph with distance-decayed edge weights | `data/processed/spatial/nyc_taxi_zone_graph_all_eligible.json` |
| Weather vintages | Open-Meteo ECMWF IFS individual runs; `available_at = run_time + 6h` | `data/processed/weather/nyc_2024-03-14_2024-06-30_asof.parquet` |
| Spatiotemporal features | Graph neighbour lags + as-of weather features | `data/processed/full/spatiotemporal_features_15m_2024-01_2024-06.parquet` |
| Probabilistic forecast | Three XGBoost quantile models: P10, P50, P90 | `artifacts/models/urbanflow-quantile-graph-weather/` |
| Decision-time horizon | Train-only seasonal projection of quantile anchors | `data/processed/full/decision_time_quantile_horizon_15m_2024-01_2024-06.parquet` |
| All-zone simulation | Every zone with at least one pickup in the Jan–Apr fit window | `data/processed/simulation/nyc_2024-01_2024-06_all_zones/` |
| Spatial policy constraints | Relocations are restricted to graph edges; P50/P90 can be blended by risk aversion | `configs/simulation/forecast_guided_heuristic_all_zones.yaml` |

## 2. Chronology and leakage controls

- The weather archive starts on **2024-03-14**, the beginning of the archived
  ECMWF IFS HRES individual-run data used here. Earlier rows remain present and
  carry `weather_missing = 1`; observed/reanalysis weather is not substituted.
- Runs are requested at 00, 06, 12, and 18 UTC. A run is not treated as
  available at initialization. The pipeline uses a conservative six-hour
  publication delay and selects only the latest run satisfying
  `available_at <= decision_time`.
- For a target interval beginning at `t`, the simulator's decision time is
  `t`. Weather is joined for the target hour using only a run that was already
  available by that decision time, matching the Phase 1 decision-time contract.
- The zone graph is built from zone geometry, not from future demand or OD
  labels. Graph lags aggregate only historical demand lag/rolling features.
- Quantile model complexity is selected on April. Final models are refit on
  Jan–Apr and produce May/June forecasts. May and June labels do not participate
  in estimator selection or final training; June is evaluation-only in the
  generated report. June is a development-test period already inspected in
  earlier project work, not a pristine never-touched holdout.
- P10/P50/P90 are clipped to non-negative demand and monotonically reordered
  per row if quantile crossing occurs. Crossing counts are reported.
- Multi-step decision forecasts use the one-step quantile anchor plus a
  train-only weekly profile. They do not read future actual demand or future
  offline one-step predictions. Weather is used in the one-step anchor; the
  current horizon extension does not re-query weather separately at each
  future step.

## 3. Build order

Run these commands from the repository root with the project virtual
environment active.

```bash
# Install the new geospatial dependencies into the project environment.
pip install -e '.[dev]'

# 1. Build WGS84 centroids from the official TLC zone shapefile.
python scripts/build_zone_centroids.py

# 2. Rebuild simulation inputs for all fit-eligible TLC zones.
python scripts/build_real_simulation_inputs.py \
  --config configs/simulation/real_data_inputs_all_zones.yaml

# 3. Build the graph in the exact all-zone simulation order.
python scripts/build_zone_graph.py

# 4. Fetch archived, as-of-safe ECMWF IFS runs.
python scripts/fetch_weather_forecast_archive.py \
  --config configs/weather/open_meteo_asof.yaml

# 5. Add graph and weather features to the full forecast panel.
python scripts/build_spatiotemporal_features.py \
  --config configs/forecasting/spatiotemporal_features.yaml

# 6. Select quantile tree counts on April, refit Jan–Apr, predict May/June.
python scripts/train_probabilistic_forecaster.py \
  --config configs/forecasting/probabilistic_graph_weather.yaml

# 7. Build four-step decision-time P10/P50/P90 inputs for optimization/RL.
python scripts/build_decision_time_quantile_forecasts.py \
  --config configs/forecasting/decision_time_quantile_horizon.yaml
```

Optional integration run using the expanded network, uncertainty-aware demand
signal, and graph-constrained relocations:

```bash
python scripts/run_forecast_guided_validation.py \
  --config configs/simulation/forecast_guided_validation_all_zones.yaml
```

## 4. Model and feature contracts

### Graph features

The graph is an undirected k-nearest-neighbour graph built from WGS84 taxi-zone
centroids. Edge weights decay exponentially with great-circle distance. The
following features are added for each zone and 15-minute interval:

- `graph_neighbor_lag_1`
- `graph_neighbor_lag_4`
- `graph_neighbor_rolling_mean_4`
- `graph_neighbor_count`
- `zone_latitude`, `zone_longitude`

The stored graph is symmetric and has no self-loops. Message passing uses a
row-normalized adjacency matrix.

### Weather features

- `weather_temperature_2m`
- `weather_relative_humidity_2m`
- `weather_precipitation_probability`
- `weather_precipitation`
- `weather_wind_speed_10m`
- `weather_code`
- `weather_forecast_lead_hours`
- `weather_forecast_age_hours`
- `weather_missing`

Weather is fetched at borough-level centroids and mapped to zones by borough.
Unknown/N/A boroughs use a citywide centroid. The raw archive preserves
`run_time`, `available_at`, and `valid_time` for auditability.

### Quantile output

One-step predictions are written with:

- `p10_demand`
- `p50_demand`
- `p90_demand`
- `decision_time`
- `forecast_provenance`

The horizon table adds `horizon_step`, `target_interval_start`, and
`forecast_provenance`. It is shaped for optimization and RL consumers without
exposing the realized target. For steps 2+, the current implementation uses
the train-only weekly profile anchored by the quantile forecast; it does not
re-run the quantile models with a separate future-weather vintage for every
horizon.

### Simulation constraints

The all-zone policy uses P50/P90 blending controlled by `risk_aversion`:

`risk_adjusted = (1 - risk_aversion) * P50 + risk_aversion * P90`

A relocation action is rejected if it traverses a non-edge in the zone graph.
The existing OD travel-time and maximum relocation-step constraints still
apply, so graph adjacency is an additional spatial feasibility constraint,
not a replacement for travel time.

## 5. Tests

The new pure tests cover:

- graph symmetry, no self-loops, deterministic neighbour aggregation;
- P10/P50/P90 monotonicity and pinball/coverage metrics;
- quantile horizon shape and non-negative outputs;
- simulator rejection of relocations outside the graph.

Run:

```bash
ruff check src/urbanflow scripts tests
pytest -q
```

The full data build and model training require the local TLC Parquet files,
Polars/PyArrow, XGBoost, and network access for the public weather archive.

## 6. External data references

- NYC TLC taxi-zone shapefile and lookup resources: <https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page>
- Open-Meteo Single Runs API: <https://open-meteo.com/en/docs/single-runs-api>

The Open-Meteo Single Runs API preserves individual model runs. Its documentation
states that ECMWF IFS HRES 9 km runs are archived from 2024-03-14, and that global
runs typically become available 4–6 hours after initialization. This pipeline
uses six hours as a conservative availability boundary.
