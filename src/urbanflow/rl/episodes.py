"""Causal training episodes and decision-time evaluation episodes.

Training uses ONLY train-split observed demand, plus a train-fitted seasonal
profile and a deliberately approximate uncertainty proxy. Evaluation uses the
configured split's archived decision-time quantile forecasts. Holdout labels
never enter training replay construction or policy gradient updates.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from urbanflow.simulation.real_episode_baseline import NetworkInputs, RealEpisodeBaselineConfig


class RLEpisodeError(ValueError):
    """Training/evaluation episode violates a temporal or network contract."""


@dataclass(frozen=True)
class Episode:
    timestamps: tuple[datetime, ...]
    demand: NDArray[np.int64]
    p10: NDArray[np.float64]
    p50: NDArray[np.float64]
    p90: NDArray[np.float64]
    warmup_steps: int
    split: str
    forecast_provenance: str

    def validate(self, *, horizon: int, zones: int, interval_minutes: int) -> None:
        t = len(self.timestamps)
        if t <= self.warmup_steps or self.warmup_steps < 0:
            raise RLEpisodeError("Episode needs a nonempty evaluation period.")
        if self.demand.shape != (t, zones):
            raise RLEpisodeError("Realized demand grid is not dense.")
        if (self.demand < 0).any() or not np.issubdtype(self.demand.dtype, np.integer):
            raise RLEpisodeError("Realized demand must be nonnegative integers.")
        for q in (self.p10, self.p50, self.p90):
            if q.shape != (t, horizon, zones) or not np.isfinite(q).all() or (q < 0).any():
                raise RLEpisodeError("Quantile tensors must be dense, finite and nonnegative.")
        if ((self.p10 > self.p50) | (self.p50 > self.p90)).any():
            raise RLEpisodeError("Forecast quantiles must be ordered.")
        if any(ts.tzinfo is None for ts in self.timestamps):
            raise RLEpisodeError("All timestamps must be timezone-aware.")
        expected = timedelta(minutes=interval_minutes)
        for prev, next_ts in zip(self.timestamps, self.timestamps[1:], strict=False):
            if next_ts.astimezone(UTC) - prev.astimezone(UTC) != expected:
                raise RLEpisodeError("Episode timestamps must be contiguous in UTC.")


@dataclass(frozen=True)
class EpisodeBundle:
    network: NetworkInputs
    initial_vehicles: NDArray[np.int64]
    allowed_edges: NDArray[np.bool_]
    region_index: NDArray[np.int64]
    region_names: tuple[str, ...]
    baseline: RealEpisodeBaselineConfig
    episodes: tuple[Episode, ...]
    forecast_horizon_steps: int
    source: str

    def validate(self) -> None:
        n = len(self.network.zone_ids)
        if n <= 1 or len(self.episodes) == 0:
            raise RLEpisodeError("At least two zones and one episode are required.")
        if self.initial_vehicles.shape != (n,) or (self.initial_vehicles < 0).any():
            raise RLEpisodeError("Invalid initial fleet.")
        if self.allowed_edges.shape != (n, n) or self.region_index.shape != (n,):
            raise RLEpisodeError("Graph/region shape disagrees with zones.")
        for episode in self.episodes:
            episode.validate(
                horizon=self.forecast_horizon_steps,
                zones=n,
                interval_minutes=self.baseline.interval_minutes,
            )


def _inputs(config_path: Path, *, training_only: bool = False):
    """Reuse the Phase 3 fleet/graph contracts without reading holdout labels in training."""
    from urbanflow.simulation.optimization_benchmark import (
        _load_graph_mask, _load_region_index, read_config as read_benchmark_config,
    )
    from urbanflow.simulation.real_episode_baseline import (
        NetworkInputs, estimate_initial_fleet, load_historical_demand, load_od_matrices,
        load_zone_index, read_config as read_baseline_config,
    )
    config = read_benchmark_config(config_path)
    baseline = read_baseline_config(config.baseline_config_path)
    _, zone_ids, shares = load_zone_index(baseline)
    n = len(zone_ids)
    probabilities, passenger_steps, relocation_steps = load_od_matrices(baseline, zone_count=n)
    network = NetworkInputs(
        zone_ids=zone_ids, fit_pickup_shares=shares,
        destination_probabilities=probabilities,
        passenger_travel_steps=passenger_steps,
        relocation_travel_steps=relocation_steps,
    )
    if training_only:
        import polars as pl
        # Scan-filter before materialization: May and June labels are never
        # materialized into the training process, not merely ignored later.
        historical = (
            pl.scan_parquet(baseline.historical_demand_path)
            .filter(pl.col("split") == "train")
            .collect()
        )
        required = {
            "interval_start", "service_date", "zone_index", "zone_id",
            "split", "actual_demand",
        }
        if not required.issubset(historical.columns) or historical.height == 0:
            raise RLEpisodeError("Train-only historical demand is missing required rows/columns.")
        if (historical.select(sorted(required)).null_count().to_numpy() > 0).any():
            raise RLEpisodeError("Nulls in training demand.")
        if historical.filter(pl.col("actual_demand") < 0).height:
            raise RLEpisodeError("Negative training demand.")
        duplicates = (
            historical.group_by("interval_start", "zone_index").len()
            .filter(pl.col("len") > 1).height
        )
        if duplicates:
            raise RLEpisodeError("Duplicate interval/zone keys in training demand.")
    else:
        historical = load_historical_demand(baseline, zone_count=n)
    fleet = estimate_initial_fleet(historical, network=network, config=baseline)
    _, region_index, region_names = _load_region_index(config.zone_lookup_path, zone_ids)
    allowed_edges = _load_graph_mask(config.zone_graph_path, zone_ids)
    return config, baseline, network, historical, fleet.initial_vehicles, allowed_edges, region_index, region_names


def _weekly_profile(train, *, n: int, interval_minutes: int) -> NDArray[np.float64]:
    """Smoothed train-only calendar profile for causal training proxies."""
    import polars as pl
    slots_per_day = 1440 // interval_minutes
    prepared = train.select(
        pl.col("zone_index").cast(pl.Int64).alias("z"),
        ((pl.col("interval_start").dt.weekday() - 1) * slots_per_day
         + (pl.col("interval_start").dt.hour() * 60
            + pl.col("interval_start").dt.minute()) // interval_minutes)
        .cast(pl.Int64).alias("slot"),
        pl.col("actual_demand").cast(pl.Float64).alias("y"),
    )
    zone = prepared.get_column("z").to_numpy()
    slot = prepared.get_column("slot").to_numpy()
    y = prepared.get_column("y").to_numpy()
    if (zone < 0).any() or (zone >= n).any() or (slot >= 7 * slots_per_day).any():
        raise RLEpisodeError("Invalid training zone or calendar slot.")
    total = np.bincount(zone, weights=y, minlength=n)
    count = np.bincount(zone, minlength=n)
    if (count == 0).any():
        raise RLEpisodeError("Missing train observations for a simulation zone.")
    zone_mean = total / count
    sums = np.zeros((n, 7 * slots_per_day), dtype=np.float64)
    counts = np.zeros_like(sums)
    np.add.at(sums, (zone, slot), y)
    np.add.at(counts, (zone, slot), 1.0)
    return np.maximum((sums + 8.0 * zone_mean[:, None]) / (counts + 8.0), 1e-3)


def _make_train_proxy(
    timestamps: tuple[datetime, ...], demand: NDArray[np.int64],
    profile: NDArray[np.float64], *, interval_minutes: int, horizon: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Each decision uses only the last OBSERVED demand plus train-only profile.

    +/-20% intervals are intentionally UNCALIBRATED training augmentation,
    never presented as fitted P10/P90 or used for external validation.
    """
    t, n = demand.shape
    p50 = np.zeros((t, horizon, n), dtype=np.float64)
    slots_per_day = 1440 // interval_minutes

    def slot(ts: datetime) -> int:
        return ts.weekday() * slots_per_day + (ts.hour * 60 + ts.minute) // interval_minutes

    for index, now in enumerate(timestamps):
        if index == 0:
            continue
        previous = timestamps[index - 1]
        anchor = demand[index - 1].astype(np.float64)
        reference = profile[:, slot(previous)]
        for h in range(horizon):
            target = (now.astimezone(UTC) + timedelta(minutes=h * interval_minutes)).astimezone(now.tzinfo)
            seasonal_ratio = np.clip(profile[:, slot(target)] / reference, 0.5, 2.0)
            p50[index, h] = anchor * seasonal_ratio
    return np.maximum(0.0, p50 * 0.8), p50, p50 * 1.2


def load_training_bundle(
    phase3_config: Path, *, start: date, end: date, horizon: int,
) -> EpisodeBundle:
    """Load Feb-Apr train replay with a strictly earlier January forecast profile."""
    if not start <= end or not (date(2024, 1, 9) <= start <= end <= date(2024, 4, 30)):
        raise RLEpisodeError("RL training dates must lie within 2024-01-09 to 2024-04-30.")
    import polars as pl
    from urbanflow.simulation.real_episode_baseline import extract_episode
    from dataclasses import replace
    (config, baseline, network, historical, initial, graph,
     region_index, region_names) = _inputs(phase3_config, training_only=True)
    train = historical.filter(pl.col("split") == "train")
    if train.height == 0 or set(train.get_column("split").unique().to_list()) != {"train"}:
        raise RLEpisodeError("Train-only time series unavailable.")
    # Fit the profile ONLY from dates preceding the first RL replay date.
    # This avoids within-train future-information leakage: e.g. February
    # decisions cannot depend on profile estimates using March/April labels.
    profile_source = train.filter(pl.col("service_date") < start)
    profile_days = profile_source.get_column("service_date").n_unique()
    if profile_days < 7:
        raise RLEpisodeError(
            "At least seven earlier TRAIN calendar days are required for the seasonal profile."
        )
    profile = _weekly_profile(
        profile_source, n=len(network.zone_ids),
        interval_minutes=baseline.interval_minutes,
    )
    episodes: list[Episode] = []
    for offset in range((end - start).days + 1):
        service_date = start + timedelta(days=offset)
        episode_config = replace(baseline, service_date=service_date, evaluation_split="train")
        try:
            ts, demand = extract_episode(historical, zone_count=len(network.zone_ids), config=episode_config)
        except Exception as error:
            raise RLEpisodeError(f"Could not build training day {service_date}: {error}") from error
        warmup = sum(timestamp.date() < service_date for timestamp in ts)
        if warmup <= 0 or sum(timestamp.date() == service_date for timestamp in ts) <= 0:
            raise RLEpisodeError("Training episode lacks a full warmup/evaluation day.")
        p10, p50, p90 = _make_train_proxy(
            tuple(ts), demand, profile, interval_minutes=baseline.interval_minutes,
            horizon=horizon,
        )
        episodes.append(Episode(
            timestamps=tuple(ts), demand=demand, p10=p10, p50=p50, p90=p90,
            warmup_steps=warmup, split="train",
            forecast_provenance="past_train_profile_lag1_proxy_uncalibrated",
        ))
    result = EpisodeBundle(
        network=network, initial_vehicles=initial, allowed_edges=graph,
        region_index=region_index, region_names=region_names, baseline=baseline,
        episodes=tuple(episodes), forecast_horizon_steps=horizon, source="train",
    )
    result.validate()
    return result


def load_evaluation_bundle(
    phase3_config: Path,
    *,
    evaluation_start_date: date | None = None,
    evaluation_end_date: date | None = None,
) -> EpisodeBundle:
    """Load one frozen-policy evaluation window using decision-time forecasts.

    When a date range is configured, simulator state is continuous across the
    entire range and the shared no-rebalancing warmup is applied once. Forecasts
    are loaded only for evaluation timestamps; warmup forecast slots are zero
    because the policy is not permitted to act during warmup.
    """
    from urbanflow.simulation.optimization_benchmark import _load_decision_time_quantiles
    from urbanflow.simulation.real_episode_baseline import (
        extract_episode,
        extract_episode_range,
    )

    (config, baseline, network, historical, initial, graph,
     region_index, region_names) = _inputs(phase3_config)
    if (evaluation_start_date is None) != (evaluation_end_date is None):
        raise RLEpisodeError("Evaluation start and end dates must be configured together.")
    if evaluation_start_date is not None and evaluation_end_date is not None:
        if evaluation_end_date < evaluation_start_date:
            raise RLEpisodeError("Evaluation date range is invalid.")
        ts, demand = extract_episode_range(
            historical,
            zone_count=len(network.zone_ids),
            config=baseline,
            evaluation_start_date=evaluation_start_date,
            evaluation_end_date=evaluation_end_date,
        )
        start_date = evaluation_start_date
        end_date = evaluation_end_date
    else:
        ts, demand = extract_episode(
            historical,
            zone_count=len(network.zone_ids),
            config=baseline,
        )
        start_date = baseline.service_date
        end_date = baseline.service_date

    evaluation_mask = np.asarray(
        [start_date <= timestamp.date() <= end_date for timestamp in ts],
        dtype=bool,
    )
    evaluation_timestamps = [
        timestamp
        for timestamp, selected in zip(ts, evaluation_mask, strict=True)
        if selected
    ]
    if not evaluation_timestamps:
        raise RLEpisodeError("Evaluation window contains no timestamps.")
    eval_p10, eval_p50, eval_p90, provenance = _load_decision_time_quantiles(
        config.decision_time_forecasts_path,
        episode_timestamps=evaluation_timestamps,
        zone_ids=network.zone_ids,
        expected_split=baseline.evaluation_split,
        timezone=baseline.timezone,
        interval_minutes=baseline.interval_minutes,
        horizon_steps=config.forecast_horizon_steps,
    )
    warmup = int((~evaluation_mask).sum())
    shape = (len(ts), config.forecast_horizon_steps, len(network.zone_ids))
    p10 = np.zeros(shape, dtype=np.float64)
    p50 = np.zeros(shape, dtype=np.float64)
    p90 = np.zeros(shape, dtype=np.float64)
    p10[evaluation_mask] = eval_p10
    p50[evaluation_mask] = eval_p50
    p90[evaluation_mask] = eval_p90
    result = EpisodeBundle(
        network=network, initial_vehicles=initial, allowed_edges=graph,
        region_index=region_index, region_names=region_names, baseline=baseline,
        episodes=(Episode(
            timestamps=tuple(ts), demand=demand, p10=p10, p50=p50, p90=p90,
            warmup_steps=warmup, split=baseline.evaluation_split,
            forecast_provenance=provenance,
        ),),
        forecast_horizon_steps=config.forecast_horizon_steps,
        source=baseline.evaluation_split,
    )
    result.validate()
    return result


def load_validation_bundle(phase3_config: Path) -> EpisodeBundle:
    """Keep the Phase 4 May-validation contract explicit and fail closed."""
    from urbanflow.simulation.optimization_benchmark import read_config as read_benchmark_config
    from urbanflow.simulation.real_episode_baseline import read_config as read_baseline_config

    benchmark = read_benchmark_config(phase3_config)
    baseline = read_baseline_config(benchmark.baseline_config_path)
    if baseline.evaluation_split != "validation" or not (
        date(2024, 5, 1) <= baseline.service_date <= date(2024, 5, 31)
    ):
        raise RLEpisodeError("Phase 4 selection/evaluation must be May validation only.")
    return load_evaluation_bundle(phase3_config)
