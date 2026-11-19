# UrbanFlow Phase 1: Reproducible Leakage-Safe May Policy Selection

This package contains **full source files**, not diffs. Keep your `.git`
folder and existing local `data/processed/**/*.parquet` files unchanged.

## Why the old May numbers are no longer the benchmark

The previous greedy evaluator used offline one-step predictions for *future*
intervals. Those predictions included lag features derived from realized
demand later than the decision point. Also, the constrained calibration and
XGBoost early-stopping selection used May labels. Earlier multi-day policy
results must be considered preliminary and must be re-evaluated.

## Required source and local input

- `data/processed/full/forecast_features_15m_2024-01_2024-06.parquet`
- `data/processed/simulation/nyc_2024-01_2024-06/{zone_index,od_profile,historical_demand}.parquet`
- Existing source and configs, XGBoost 3.3, Polars, NumPy, PyYAML, etc.

The shared project ZIP intentionally did **not** contain these generated
Parquet files. Never fabricate metrics or claim a champion was selected
until the commands below finish on the machine holding the actual data.

## One-time execution, in order

From the repository root:

```bash
source .venv/bin/activate

# Static and unit checks
ruff check \
  src/urbanflow/simulation/decision_time_forecasting.py \
  src/urbanflow/simulation/forecast_guided_heuristic.py \
  src/urbanflow/simulation/forecast_guided_validation.py \
  src/urbanflow/simulation/forecast_guided_tuning.py \
  scripts/build_asof_validation_forecasts.py \
  scripts/run_forecast_guided_validation.py \
  scripts/run_forecast_guided_tuning.py \
  scripts/promote_forecast_guided_champion.py \
  scripts/verify_frozen_champion.py \
  tests/unit/simulation/test_decision_time_forecasting.py \
  tests/unit/simulation/test_forecast_selection_contract.py

pytest tests/unit/simulation -q

# Important: train Jan-Mar, internally select on April, refit Jan-Apr,
# then predict May without using May labels for training/selection.
python scripts/build_asof_validation_forecasts.py

# Re-run the old and continuous May baselines with corrected information access.
python scripts/run_forecast_guided_heuristic.py
python scripts/run_forecast_guided_validation.py

# Select from 108 candidates on May only.
python scripts/run_forecast_guided_tuning.py

# Lock the winner and verify that May replays exactly.
python scripts/promote_forecast_guided_champion.py
python scripts/verify_frozen_champion.py
```

**Do not run `promote` before `tuning` succeeds**. Freeze scripts refuse to
overwrite an existing champion; version your experiment instead.

## Expected files

- `data/processed/simulation/nyc_2024-01_2024-06/asof_validation_predictions.parquet`
- `artifacts/models/xgboost-asof-may/model.ubj`
- `reports/simulation/asof-validation-forecast-metadata.json`
- `reports/simulation/2024-05-02_2024-05-30-forecast-guided-tuning.json`
- `reports/simulation/forecast_guided_tuning/*-candidates.parquet`
- `reports/simulation/forecast_guided_tuning/*-daily-results.parquet`
- `configs/simulation/forecast_guided_champion.yaml` (generated after selection)
- `configs/simulation/forecast_guided_champion_validation.yaml` (generated)
- `reports/simulation/forecast-guided-champion-freeze.json` (generated)
- `reports/simulation/frozen-champion-may-validation.json` (after verification)

Only the small JSON/YAML configs/reports and source should be committed;
keep models/large parquet outputs out of Git. Since `.gitignore` ignores most
simulation JSON reports, use `git add -f` for the ones you explicitly want.

## Interpretation

- `asof_prediction(t)` is one-step prediction for time `t`, available prior
  to realized demand at `t`.
- `t+1..t+H` values are a **training-weekly-profile projection anchored to
  `asof_prediction(t)`**, not unrelated future offline predictions.
- The best May policy is selected with feasibility and objective criteria;
  its May results are **tuning/validation performance**, not an unbiased
  holdout estimate.
- June is a development-test period (already inspected historically).
  Reserve July as an untouched independent temporal holdout for final claims.
