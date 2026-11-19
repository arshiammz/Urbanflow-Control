"""Leakage-safe, training-fitted short-horizon demand projection.

A validation XGBoost prediction for interval t uses only features available
before t. It is NOT valid to read the offline one-step prediction for t+h
at decision time t: that later prediction may use actual demand at t+h-1.

This module anchors every future estimate to the ONE forecast for t,
then projects it using a train-only zone/time-of-week demand profile.
It is a transparent seasonal projection baseline, not a multi-horizon XGBoost
model. No future observed demand or future one-step prediction is accessed.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import numpy as np
from numpy.typing import NDArray


class DecisionTimeForecastError(ValueError):
    """Invalid information boundary for a decision-time projection."""


@dataclass(frozen=True)
class DecisionTimeProjector:
    """Zone-by-week-slot profile fitted exclusively from the training split."""

    zone_ids: tuple[int, ...]
    profile: NDArray[np.float64]
    interval_minutes: int = 15
    minimum_ratio: float = 0.5
    maximum_ratio: float = 2.0
    prior_strength: float = 8.0

    @property
    def slots_per_day(self) -> int:
        return 1440 // self.interval_minutes

    @property
    def method(self) -> str:
        return "asof_single_step_anchor_train_only_weekly_profile_v1"

    def slot(self, timestamp: datetime) -> int:
        """Calendar feature known at decision time (DST folds share a slot)."""
        if timestamp.tzinfo is None:
            raise DecisionTimeForecastError("Timestamps must be timezone-aware.")
        if timestamp.minute % self.interval_minutes != 0 or timestamp.second != 0:
            raise DecisionTimeForecastError("Timestamp is not interval-aligned.")
        return (
            timestamp.weekday() * self.slots_per_day
            + (timestamp.hour * 60 + timestamp.minute) // self.interval_minutes
        )

    @classmethod
    def fit(
        cls,
        *,
        zone_ids: tuple[int, ...],
        timestamps: list[datetime],
        zone_indices: NDArray[np.int64],
        demand: NDArray[np.int64],
        split_labels: list[str],
        interval_minutes: int = 15,
        training_split: str = "train",
        prior_strength: float = 8.0,
        minimum_ratio: float = 0.5,
        maximum_ratio: float = 2.0,
    ) -> DecisionTimeProjector:
        """Fit only from rows explicitly labelled training; reject all other rows.

        Caller must supply a training-only table. Failing closed here avoids
        silently fitting seasonal profiles on validation/test demand.
        """
        n = len(timestamps)
        if not zone_ids or n == 0:
            raise DecisionTimeForecastError("Empty zones or training observations.")
        if not (len(zone_indices) == len(demand) == len(split_labels) == n):
            raise DecisionTimeForecastError("Training columns have mismatched lengths.")
        if set(split_labels) != {training_split}:
            raise DecisionTimeForecastError("Profile fitting accepts training rows only.")
        if interval_minutes <= 0 or 1440 % interval_minutes != 0:
            raise DecisionTimeForecastError("interval_minutes must divide a day.")
        if not np.isfinite([prior_strength, minimum_ratio, maximum_ratio]).all():
            raise DecisionTimeForecastError("Projection controls must be finite.")
        if prior_strength < 0 or not (0 < minimum_ratio <= 1 <= maximum_ratio):
            raise DecisionTimeForecastError("Invalid projection controls.")
        if np.any(zone_indices < 0) or np.any(zone_indices >= len(zone_ids)):
            raise DecisionTimeForecastError("Training zone index is invalid.")
        if np.any(demand < 0):
            raise DecisionTimeForecastError("Training demand must be nonnegative.")

        slots_per_day = 1440 // interval_minutes
        slot = np.empty(n, dtype=np.int64)
        for i, ts in enumerate(timestamps):
            if ts.tzinfo is None or ts.minute % interval_minutes or ts.second:
                raise DecisionTimeForecastError("Training timestamp is not aligned/aware.")
            slot[i] = ts.weekday() * slots_per_day + (
                ts.hour * 60 + ts.minute
            ) // interval_minutes

        zone_count = len(zone_ids)
        total = np.bincount(
            zone_indices, weights=demand.astype(np.float64), minlength=zone_count
        )
        counts = np.bincount(zone_indices, minlength=zone_count)
        if np.any(counts == 0):
            raise DecisionTimeForecastError("Every zone needs training observations.")
        zone_mean = total / counts
        sums = np.zeros((zone_count, 7 * slots_per_day), dtype=np.float64)
        observations = np.zeros_like(sums)
        np.add.at(sums, (zone_indices, slot), demand)
        np.add.at(observations, (zone_indices, slot), 1)
        profile = (sums + prior_strength * zone_mean[:, None]) / (
            observations + prior_strength
        )
        # A small positive floor avoids blow-up for very sparse zones.
        profile = np.maximum(profile, 1e-3)
        profile.setflags(write=False)
        return cls(
            zone_ids=zone_ids,
            profile=profile,
            interval_minutes=interval_minutes,
            prior_strength=prior_strength,
            minimum_ratio=minimum_ratio,
            maximum_ratio=maximum_ratio,
        )

    def project(
        self,
        *,
        decision_time: datetime,
        current_prediction: NDArray[np.float64],
        horizon_steps: int,
    ) -> NDArray[np.float64]:
        """Only consult current one-step prediction and train-only profile.

        Crucially this function does not accept or read predictions/realized
        demand for times after `decision_time`.
        """
        current = np.asarray(current_prediction, dtype=np.float64)
        if current.shape != (len(self.zone_ids),):
            raise DecisionTimeForecastError("Current prediction zone shape mismatch.")
        if np.any(~np.isfinite(current)) or np.any(current < 0):
            raise DecisionTimeForecastError("Current predictions must be finite >= 0.")
        if not isinstance(horizon_steps, int) or horizon_steps < 1:
            raise DecisionTimeForecastError("horizon_steps must be positive.")
        present_slot = self.slot(decision_time)
        denom = self.profile[:, present_slot]
        result = np.empty((horizon_steps, len(self.zone_ids)), dtype=np.float64)
        for step in range(1, horizon_steps + 1):
            future_time = (
                decision_time.astimezone(UTC)
                + timedelta(minutes=self.interval_minutes * step)
            ).astimezone(decision_time.tzinfo)
            target_slot = self.slot(future_time)
            ratio = np.clip(
                self.profile[:, target_slot] / denom,
                self.minimum_ratio,
                self.maximum_ratio,
            )
            result[step - 1] = current * ratio
        return result


def fit_train_only_projector(
    historical_demand: object,
    *,
    zone_ids: tuple[int, ...],
    interval_minutes: int,
    training_split: str = "train",
) -> DecisionTimeProjector:
    """Adapter for the project's historical_demand Polars DataFrame.

    This adapter first filters to training rows. `DecisionTimeProjector.fit`
    additionally rejects non-training rows if callers violate its contract.
    """
    import polars as pl

    training = historical_demand.filter(pl.col("split") == training_split)
    if training.height == 0:
        raise DecisionTimeForecastError("Training history for profile is empty.")
    rows = training.select("interval_start", "zone_index", "actual_demand", "split")
    return DecisionTimeProjector.fit(
        zone_ids=zone_ids,
        timestamps=rows.get_column("interval_start").to_list(),
        zone_indices=rows.get_column("zone_index").to_numpy().astype(np.int64),
        demand=rows.get_column("actual_demand").to_numpy().astype(np.int64),
        split_labels=rows.get_column("split").to_list(),
        interval_minutes=interval_minutes,
        training_split=training_split,
    )
