from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

pl = pytest.importorskip("polars")

from urbanflow.simulation.optimization_benchmark import (  # noqa: E402
    OptimizationBenchmarkError,
    _load_decision_time_quantiles,
)


NY = ZoneInfo("America/New_York")
PROVENANCE = "quantile_anchor_plus_train_only_weekly_profile"


def build_forecast_frame() -> pl.DataFrame:
    decision_times = [
        datetime(2024, 5, 14, 0, 0, tzinfo=NY),
        datetime(2024, 5, 14, 0, 15, tzinfo=NY),
    ]
    rows: list[dict[str, object]] = []
    for decision_index, decision_time in enumerate(decision_times):
        for horizon_step in (1, 2):
            target_time = decision_time + timedelta(minutes=15 * (horizon_step - 1))
            for zone_id in (10, 20):
                p50 = float(decision_index * 100 + horizon_step * 10 + zone_id)
                rows.append(
                    {
                        "decision_time": decision_time,
                        "target_interval_start": target_time,
                        "horizon_step": horizon_step,
                        "zone_id": zone_id,
                        "split": "validation",
                        "p10_demand": p50 - 2.0,
                        "p50_demand": p50,
                        "p90_demand": p50 + 2.0,
                        "forecast_provenance": PROVENANCE,
                    }
                )
    return pl.DataFrame(rows)


def test_decision_time_quantile_loader_builds_zone_index_ordered_tensor(
    tmp_path: Path,
) -> None:
    path = tmp_path / "decision-time-forecasts.parquet"
    build_forecast_frame().write_parquet(path)
    decision_times = [
        datetime(2024, 5, 14, 0, 0, tzinfo=NY),
        datetime(2024, 5, 14, 0, 15, tzinfo=NY),
    ]

    p10, p50, p90, provenance = _load_decision_time_quantiles(
        path,
        episode_timestamps=decision_times,
        zone_ids=(20, 10),
        expected_split="validation",
        timezone="America/New_York",
        interval_minutes=15,
        horizon_steps=2,
    )

    assert p10.shape == (2, 2, 2)
    assert p50.shape == (2, 2, 2)
    assert p90.shape == (2, 2, 2)
    assert p50[0, 0].tolist() == [30.0, 20.0]
    assert p50[1, 1].tolist() == [140.0, 130.0]
    assert (p10 <= p50).all()
    assert (p50 <= p90).all()
    assert provenance == PROVENANCE


def test_decision_time_quantile_loader_rejects_non_phase2_provenance(
    tmp_path: Path,
) -> None:
    path = tmp_path / "wrong-provenance.parquet"
    frame = build_forecast_frame().with_columns(
        pl.lit("offline_future_predictions").alias("forecast_provenance")
    )
    frame.write_parquet(path)

    with pytest.raises(
        OptimizationBenchmarkError,
        match="provenance",
    ):
        _load_decision_time_quantiles(
            path,
            episode_timestamps=[
                datetime(2024, 5, 14, 0, 0, tzinfo=NY),
                datetime(2024, 5, 14, 0, 15, tzinfo=NY),
            ],
            zone_ids=(10, 20),
            expected_split="validation",
            timezone="America/New_York",
            interval_minutes=15,
            horizon_steps=2,
        )
