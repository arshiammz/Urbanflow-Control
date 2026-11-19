"""Decision-time boundary tests (pure NumPy; no NYC dataset required)."""

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pytest

from urbanflow.simulation.decision_time_forecasting import (
    DecisionTimeForecastError,
    DecisionTimeProjector,
)


NY = ZoneInfo("America/New_York")


def projector() -> DecisionTimeProjector:
    # Two training zones for several weeks; known periodic profile.
    start = datetime(2024, 3, 25, tzinfo=NY)
    timestamps = [start + timedelta(minutes=15 * t) for t in range(96 * 14)]
    size = len(timestamps)
    return DecisionTimeProjector.fit(
        zone_ids=(101, 102),
        timestamps=timestamps * 2,
        zone_indices=np.concatenate([
            np.zeros(size, dtype=np.int64), np.ones(size, dtype=np.int64)
        ]),
        demand=np.concatenate([
            np.array([30 if 8 <= t.hour < 10 else 10 for t in timestamps], dtype=np.int64),
            np.array([4 if 8 <= t.hour < 10 else 12 for t in timestamps], dtype=np.int64),
        ]),
        split_labels=["train"] * (size * 2),
    )


def test_prediction_is_deterministic_nonnegative_with_correct_shape() -> None:
    p = projector()
    now = datetime(2024, 5, 2, 7, 45, tzinfo=NY)
    nowcast = np.array([10.0, 20.0])
    f = p.project(decision_time=now, current_prediction=nowcast, horizon_steps=12)
    assert f.shape == (12, 2)
    assert np.array_equal(f, p.project(
        decision_time=now, current_prediction=nowcast, horizon_steps=12
    ))
    assert (f >= 0).all()
    assert np.isfinite(f).all()
    assert not np.array_equal(f[0], f[-1])


def test_future_outcome_and_future_offline_predictions_do_not_enter_api() -> None:
    # Decision-time API has no parameters for future truth or future one-step
    # predictions; mutating them must not change a decision-time projection.
    p = projector()
    now = datetime(2024, 5, 2, 8, 15, tzinfo=NY)
    observed_future = np.array([[50, 50], [80, 20], [0, 200]], dtype=np.int64)
    offline_future = np.array([[49., 41.], [99., 3.], [0., 500.]])
    before = p.project(
        decision_time=now, current_prediction=np.array([11., 13.]), horizon_steps=3
    )
    observed_future[:] = -999
    offline_future[:] = 9999
    after = p.project(
        decision_time=now, current_prediction=np.array([11., 13.]), horizon_steps=3
    )
    assert np.array_equal(before, after)


def test_profile_fit_rejects_validation_and_test_rows() -> None:
    with pytest.raises(DecisionTimeForecastError, match="training rows only"):
        DecisionTimeProjector.fit(
            zone_ids=(1,),
            timestamps=[datetime(2024, 5, 2, tzinfo=NY)],
            zone_indices=np.array([0], dtype=np.int64),
            demand=np.array([500], dtype=np.int64),
            split_labels=["validation"],
        )


def test_rejects_naive_timestamp_and_invalid_nowcast() -> None:
    p = projector()
    with pytest.raises(DecisionTimeForecastError, match="timezone-aware"):
        p.project(
            decision_time=datetime(2024, 5, 2, 8),
            current_prediction=np.array([3., 4.]), horizon_steps=2,
        )
    with pytest.raises(DecisionTimeForecastError, match="finite"):
        p.project(
            decision_time=datetime(2024, 5, 2, 8, tzinfo=NY),
            current_prediction=np.array([float("nan"), 4.]), horizon_steps=2,
        )


def test_train_only_profile_is_immutable_and_uses_future_calendar_not_future_truth() -> None:
    p = projector()
    assert not p.profile.flags.writeable
    now = datetime(2024, 5, 2, 7, 45, tzinfo=NY)
    first = p.project(
        decision_time=now,
        current_prediction=np.array([10., 10.]), horizon_steps=8,
    )
    assert (first[:, 0] <= 20).all() and (first[:, 0] >= 5).all()
    assert np.isfinite(first).all()


def test_dst_transition_projects_actual_elapsed_intervals() -> None:
    p = projector()
    # From 01:45 EST the next real 15-minute step is 03:00 EDT.
    start = datetime(2024, 3, 10, 1, 45, tzinfo=NY)
    result = p.project(
        decision_time=start, current_prediction=np.array([10., 12.]), horizon_steps=1
    )
    expected_target = datetime(2024, 3, 10, 3, 0, tzinfo=NY)
    expected_ratio = np.clip(
        p.profile[:, p.slot(expected_target)] / p.profile[:, p.slot(start)],
        p.minimum_ratio,
        p.maximum_ratio,
    )
    assert np.allclose(result[0], np.array([10., 12.]) * expected_ratio)
