# Data Card — NYC TLC Demand and Weather Inputs

## Sources

- NYC Taxi & Limousine Commission For-Hire Vehicle trip records, monthly Parquet files.
- TLC taxi-zone lookup and official zone shapefile/centroids.
- Archived ECMWF IFS weather forecast runs retrieved through the Open-Meteo single-runs archive endpoint.

## Time and geography

Demand is aggregated to 15-minute intervals in `America/New_York` and mapped to TLC taxi zones. The simulation network uses all zones meeting the configured historical fit-period eligibility rule. Zone index order is stored and treated as a model contract.

## Temporal separation

- Forecast selection training: January–April 2024.
- Forecast selection validation: May–June 2024.
- Final forecasting refit: pre-July data only.
- Untouched final holdout: July 1–31, 2024 (`test` split).
- Simulator warmup: June 30, 2024, no rebalancing.
- OD/zone network fit window remains January–April to preserve the frozen Phase 4 PPO graph/travel-time signature.

The July test labels are used only for final evaluation. Weather feature joins use archived forecast runs whose availability time is no later than the decision time; observed future weather is not used as a feature.

## Quality controls

Monthly trip validation, dense zone-time panel checks, duplicate/null/negative checks, split chronology, quantile monotonicity, forecast key uniqueness, graph connectivity and fleet-conservation invariants are enforced by the upstream pipelines and final-holdout preflight.

## Limitations

TLC trip records represent recorded completed trips and have reporting/coverage limitations. Forecast archive runs can be missing for some issue times. The simulation uses smoothed historical OD probabilities and estimated travel-time steps; these are not live traffic measurements. Weather archives and data files are generated locally and are not bundled in the source release.
