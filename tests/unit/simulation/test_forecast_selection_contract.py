"""Parameter grid and strict forecast-loading guards (Polars required)."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest

pl = pytest.importorskip("polars")

from urbanflow.simulation.forecast_guided_heuristic import (  # noqa: E402
    ForecastGuidedPolicyError,
    load_forecast_matrix,
)
from urbanflow.simulation.forecast_guided_tuning import (  # noqa: E402
    TuningConfig,
    candidate_grid,
    ranking_key,
)


NY = ZoneInfo("America/New_York")


def test_candidate_grid_108_deterministic() -> None:
    config = TuningConfig(
        baseline_config_path=Path("any"),
        heuristic_config_path=Path("any"),
        candidate_results_path=Path("any"),
        daily_results_path=Path("any"),
        report_path=Path("any"),
        validation_split="validation",
        start_date=datetime(2024, 5, 2).date(),
        end_date_inclusive=datetime(2024, 5, 30).date(),
        forecast_horizon_steps=(2, 4, 8),
        demand_buffer_multiplier=(1.0, 1.05, 1.1),
        minimum_zone_reserve=(0, 2),
        maximum_relocation_steps=(1, 2),
        maximum_relocation_fraction=(0.10, 0.15, 0.20),
        in_transit_supply_credit=(0.25,),
        require_every_day_reduce_unmet_demand=True,
        require_every_day_reduce_operating_cost=True,
        compression="zstd",
    )
    grid = candidate_grid(config)
    assert len(grid) == 108
    assert [x["candidate_id"] for x in grid] == list(range(1, 109))
    assert grid == candidate_grid(config)


def test_rank_prefers_cost_then_worst_day_then_relocation() -> None:
    rows = [
        {"candidate_id": 1, "policy_total_operating_cost": 100.,
         "worst_day_service_rate_improvement": 0.4, "relocation_vehicle_steps": 90},
        {"candidate_id": 2, "policy_total_operating_cost": 99.,
         "worst_day_service_rate_improvement": 0.1, "relocation_vehicle_steps": 500},
        {"candidate_id": 3, "policy_total_operating_cost": 100.,
         "worst_day_service_rate_improvement": 0.4, "relocation_vehicle_steps": 70},
    ]
    assert [r["candidate_id"] for r in sorted(rows, key=ranking_key)] == [2, 3, 1]


def build_forecast_data(tmp_path: Path) -> tuple[Path, list[datetime]]:
    times = [datetime(2024, 5, 2, tzinfo=NY) + timedelta(minutes=15*k) for k in range(4)]
    rows = [
        (time, zone, "validation", 2.0 + k + zone)
        for k, time in enumerate(times)
        for zone in (101, 102)
    ]
    # Future one-step predictions are present but MUST NOT be loaded as a horizon.
    rows.extend([
        (times[-1] + timedelta(minutes=15), 101, "validation", 100000.0),
        (times[-1] + timedelta(minutes=15), 102, "validation", 100000.0),
    ])
    df = pl.DataFrame(
        rows,
        schema=["interval_start", "zone_id", "split", "constrained_prediction"],
        orient="row",
    ).with_columns(
        pl.col("interval_start").cast(pl.Datetime("ns", "America/New_York")),
        pl.col("zone_id").cast(pl.Int64),
        pl.col("constrained_prediction").cast(pl.Float64),
    )
    path = tmp_path / "forecasts.parquet"
    df.write_parquet(path)
    return path, times


def policy_config(path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        forecast_predictions_path=path,
        forecast_timestamp_column="interval_start",
        forecast_zone_column="zone_id",
        forecast_prediction_column="constrained_prediction",
        forecast_split_column="split",
    )


def test_loader_does_not_include_future_offline_predictions(tmp_path: Path) -> None:
    path, times = build_forecast_data(tmp_path)
    actual_times, frame, upper = load_forecast_matrix(
        config=policy_config(path),
        baseline_config=SimpleNamespace(timezone="America/New_York"),
        zone_ids=(101, 102), episode_timestamps=times,
        expected_split="validation",
    )
    assert actual_times == times
    assert frame.shape == (4, 2)
    assert upper is None
    assert float(frame.max()) < 100000.0


def test_loader_rejects_wrong_split_and_missing_key(tmp_path: Path) -> None:
    path, times = build_forecast_data(tmp_path)
    df = pl.read_parquet(path)
    df = df.with_columns(
        pl.when((pl.col("zone_id") == 101) & (pl.col("interval_start") == times[0]))
        .then(pl.lit("test")).otherwise(pl.col("split")).alias("split")
    )
    df.write_parquet(path)
    with pytest.raises(ForecastGuidedPolicyError, match="different temporal split"):
        load_forecast_matrix(
            config=policy_config(path),
            baseline_config=SimpleNamespace(timezone="America/New_York"),
            zone_ids=(101, 102), episode_timestamps=times,
        )
    df = df.filter(~((pl.col("zone_id") == 101) & (pl.col("interval_start") == times[0])))
    df.write_parquet(path)
    with pytest.raises(ForecastGuidedPolicyError, match="Missing forecasts"):
        load_forecast_matrix(
            config=policy_config(path),
            baseline_config=SimpleNamespace(timezone="America/New_York"),
            zone_ids=(101, 102), episode_timestamps=times,
        )
