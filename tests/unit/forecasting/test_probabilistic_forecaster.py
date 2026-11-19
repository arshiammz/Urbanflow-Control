from __future__ import annotations

import numpy as np
import pytest

from urbanflow.forecasting.probabilistic_forecaster import (
    DecisionTimeQuantileProjector,
    ProbabilisticForecastError,
    enforce_quantile_order,
    pinball_loss,
    quantile_metrics,
)
from urbanflow.simulation.decision_time_forecasting import DecisionTimeProjector
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import xgboost as xgb


def test_quantile_order_clips_negative_and_counts_crossing_rows() -> None:
    raw = np.asarray([[4.0, 2.0, 7.0], [-1.0, 3.0, 2.0]])
    ordered, crossing_rows = enforce_quantile_order(raw)
    assert crossing_rows == 2
    assert np.array_equal(ordered, np.asarray([[2.0, 4.0, 7.0], [0.0, 2.0, 3.0]]))
    assert np.all(np.diff(ordered, axis=1) >= 0)


def test_pinball_loss_and_quantile_metrics_are_finite() -> None:
    actual = np.asarray([0.0, 10.0, 20.0, 30.0])
    predictions = np.asarray(
        [[0.0, 2.0, 5.0], [4.0, 10.0, 15.0], [8.0, 18.0, 22.0], [15.0, 28.0, 35.0]]
    )
    assert pinball_loss(actual, predictions[:, 1], 0.5) >= 0
    metrics = quantile_metrics(actual, predictions)
    assert metrics["row_count"] == 4
    assert metrics["central_80_coverage"] >= 0
    assert np.isfinite(list(metrics.values())).all()


def test_pinball_rejects_invalid_quantile() -> None:
    with pytest.raises(ProbabilisticForecastError, match="between zero and one"):
        pinball_loss(np.asarray([1.0]), np.asarray([1.0]), 1.0)


def test_decision_time_quantile_projector_preserves_order_and_shape() -> None:
    zone = ZoneInfo("America/New_York")
    start = datetime(2024, 3, 25, tzinfo=zone)
    timestamps = [start + timedelta(minutes=15 * step) for step in range(96 * 14)]
    size = len(timestamps)
    projector = DecisionTimeProjector.fit(
        zone_ids=(101, 102),
        timestamps=timestamps * 2,
        zone_indices=np.concatenate(
            [np.zeros(size, dtype=np.int64), np.ones(size, dtype=np.int64)]
        ),
        demand=np.concatenate([
            np.asarray([10 + (t.hour >= 8) * 20 for t in timestamps]),
            np.asarray([8 + (t.hour >= 8) * 5 for t in timestamps]),
        ]),
        split_labels=["train"] * (size * 2),
    )
    quantile_projector = DecisionTimeQuantileProjector(projector)
    result = quantile_projector.project(
        decision_time=datetime(2024, 5, 2, 7, 45, tzinfo=zone),
        current_quantiles=np.asarray([[5.0, 3.0], [10.0, 8.0], [20.0, 15.0]]),
        horizon_steps=4,
    )
    assert result.shape == (4, 3, 2)
    assert np.isfinite(result).all()
    assert (result >= 0).all()
    assert np.all(np.diff(result, axis=1) >= 0)


def test_quantile_projector_rejects_wrong_anchor_shape() -> None:
    zone = ZoneInfo("America/New_York")
    start = datetime(2024, 3, 25, tzinfo=zone)
    timestamps = [start + timedelta(minutes=15 * step) for step in range(96 * 2)]
    projector = DecisionTimeProjector.fit(
        zone_ids=(101, 102),
        timestamps=timestamps * 2,
        zone_indices=np.concatenate(
            [
                np.zeros(len(timestamps), dtype=np.int64),
                np.ones(len(timestamps), dtype=np.int64),
            ]
        ),
        demand=np.ones(len(timestamps) * 2, dtype=np.int64),
        split_labels=["train"] * (len(timestamps) * 2),
    )
    with pytest.raises(ProbabilisticForecastError, match="shape"):
        DecisionTimeQuantileProjector(projector).project(
            decision_time=start,
            current_quantiles=np.asarray([[1.0, 2.0]]),
            horizon_steps=2,
        )


def test_xgboost_quantile_objective_smoke() -> None:
    features = np.arange(120, dtype=np.float32).reshape(-1, 1)
    target = np.maximum(0.0, features[:, 0] * 0.25 + np.sin(features[:, 0]))
    model = xgb.XGBRegressor(
        objective="reg:quantileerror",
        quantile_alpha=0.5,
        tree_method="hist",
        n_estimators=4,
        max_depth=2,
        n_jobs=1,
        random_state=7,
    )
    model.fit(features, target, verbose=False)
    prediction = model.predict(features[:10])
    assert prediction.shape == (10,)
    assert np.isfinite(prediction).all()
