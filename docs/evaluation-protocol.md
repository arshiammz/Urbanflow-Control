# UrbanFlow: Decision-Time Forecasting & Policy Evaluation Protocol

## Phase 1 scope

- **Selection period:** May 2–30, 2024, with May 1 warmup.
- **Fitted OD network / zone choice / fleet:** Jan–Apr 2024 only.
- **Model validation forecast:** generated with `scripts/build_asof_validation_forecasts.py`: Jan–Mar early-stopping training, **April internal selection**, Jan–Apr refit, May forecast. Neither May labels nor June data influence the fitted model or estimator-count selection.
- **Realized passenger demand:** consumed only by `FleetSimulator.step`, *after* the relocation action for that interval is chosen.
- **No June/test data:** May-only validation and tuning do not select parameters using June.
- **July:** reserved for a future untouched final holdout; June has been inspected during development.

## Previously discovered leakage risk

The earlier heuristic constructed `future_forecasts` for step `t` by loading
offline one-step predictions for `t+1, t+2, ...`. Their features (e.g. lag-1)
were built from realized demand **after** time `t`. They were valid as future
one-step predictions at their own decision times, but **not** valid forecasts
available to the policy at time `t`. Earlier simulated performance numbers
(May 15 and May 2–30) are preliminary and **must be re-run**.

## Corrected information boundary

At time `t` the decision policy can see:

1. Current `FleetSnapshot` (available vehicles, in-transit vehicles).
2. **Pre-May refitted** one-step XGBoost forecast `P(t)` for the interval just
   beginning, whose demand lags end no later than `t-1`. The estimator count
   was fixed using April, not May.
3. Calendar timestamps for planned future intervals (known in advance).
4. **Training-only** (Jan–Apr) zone × weekday × 15-minute-slot demand profiles.

The policy **cannot** see:

- Actual demand at `t` before `simulator.step`.
- Actual demand in any future interval.
- Offline one-step forecasts whose timestamp exceeds `t`.
- Any test-period labels, XGBoost predictions, or model scores for tuning.

A transparent short-horizon projection is used:

```text
ProjectedDemand(t+h | t)
    = CurrentOneStepForecast(t)
      × clip(
          TrainWeeklyProfile(zone, slot(t+h))
          / TrainWeeklyProfile(zone, slot(t)),
          0.5,
          2.0,
        )
```

The profile is fitted exclusively on `split == train` rows, with a per-zone
mean shrinkage prior of strength 8 for sparse week slots. Future calendar
features are known, so this is decision-time-safe. It is a **seasonal projection
baseline**, not a recursively fitted XGBoost multi-horizon forecaster.
The 4-step (60-minute) setting can still be compared with 2/8-step horizons.

## Selection and freeze

The fixed 108-candidate search runs May only with:

- Forecast horizon `[2, 4, 8]` intervals.
- Buffer `[1.00, 1.05, 1.10]`.
- Zone reserve `[0, 2]`.
- Max relocation duration `[1, 2]` intervals.
- Max relocation fraction `[0.10, 0.15, 0.20]`.
- Transit supply credit `0.25`.

Eligible candidates must improve unmet demand **and** total operating cost on
every evaluated day. Among eligible candidates choose:

1. Lowest aggregate May operating cost.
2. Higher *worst-day* service-rate improvement.
3. Fewer relocation vehicle-steps.
4. Smaller deterministic candidate ID.

The reference config is included exactly once but need not win or even pass
all constraints. All policies share identical demand, training-derived initial
fleet, network, cost coefficients, and warmup. The simulator is restarted for
each candidate, but **not** between days within a candidate.

Build the genuinely pre-May forecasts **first**:

```bash
python scripts/build_asof_validation_forecasts.py
python scripts/run_forecast_guided_validation.py
```

Then select and freeze:

```bash
python scripts/run_forecast_guided_tuning.py
python scripts/promote_forecast_guided_champion.py
python scripts/verify_frozen_champion.py
```

Promotion writes immutable `configs/simulation/forecast_guided_champion.yaml`,
`configs/simulation/forecast_guided_champion_validation.yaml`, and a JSON
manifest containing source/config/data SHA-256 hashes. Verification reruns May
under the exact frozen config and checks aggregate cost, unmet demand, all
simulation invariants, and hashes. Promotion refuses to overwrite an existing
freeze; a changed experiment needs a *new* versioned freeze.

## Reproducibility and claims

The ZIP source snapshot does **not** include generated Parquet demand, OD or
forecast files. Those must be recreated locally. No new May tuning result
should be reported until the corrected runner has actually executed on those
artifacts. The old `47.92%` reduction is not a leakage-corrected result.

The stage is complete when the May tuning report and frozen validation replay
exist and pass all invariants. **Do not begin June evaluation before this point.**
