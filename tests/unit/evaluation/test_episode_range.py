from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest


def _dependencies():
    pl = pytest.importorskip("polars")
    from urbanflow.simulation.real_episode_baseline import (
        RealEpisodeBaselineConfig,
        RealEpisodeBaselineError,
        extract_episode_range,
    )

    return pl, RealEpisodeBaselineConfig, RealEpisodeBaselineError, extract_episode_range


def _config(config_type):
    return config_type(
        dataset_name="synthetic-range-test",
        timezone="America/New_York",
        interval_minutes=60,
        zone_index_path=Path("zone-index.parquet"),
        od_profile_path=Path("od.parquet"),
        historical_demand_path=Path("demand.parquet"),
        step_metrics_path=Path("steps.parquet"),
        interval_zone_metrics_path=Path("zone-steps.parquet"),
        zone_summary_path=Path("zone-summary.parquet"),
        report_path=Path("report.json"),
        evaluation_split="test",
        service_date=date(2024, 7, 1),
        warmup_days=1,
        sizing_split="train",
        reserve_multiplier=1.2,
        minimum_vehicles_per_zone=1,
        initial_distribution_column="fit_pickup_share",
        unmet_demand_penalty=10.0,
        relocation_cost_per_vehicle_step=0.5,
        idle_vehicle_cost=0.01,
        policy_name="synthetic",
        compression="zstd",
    )


def _frame(pl):
    tz = ZoneInfo("America/New_York")
    rows = []
    for day_offset in range(3):
        day = date(2024, 6, 30) + timedelta(days=day_offset)
        split = "validation" if day < date(2024, 7, 1) else "test"
        for hour in range(24):
            timestamp = datetime.combine(day, datetime.min.time(), tzinfo=tz) + timedelta(hours=hour)
            for zone_index, zone_id in enumerate((1, 2)):
                rows.append({
                    "interval_start": timestamp,
                    "service_date": day,
                    "split": split,
                    "zone_index": zone_index,
                    "zone_id": zone_id,
                    "actual_demand": day_offset * 100 + hour + zone_index,
                })
    return pl.DataFrame(rows).sort("interval_start", "zone_index")


def test_multi_day_episode_has_one_shared_warmup_and_continuous_state() -> None:
    pl, config_type, _, extract_episode_range = _dependencies()
    timestamps, demand = extract_episode_range(
        _frame(pl),
        zone_count=2,
        config=_config(config_type),
        evaluation_start_date=date(2024, 7, 1),
        evaluation_end_date=date(2024, 7, 2),
    )
    assert len(timestamps) == 72
    assert demand.shape == (72, 2)
    assert sum(value.date() < date(2024, 7, 1) for value in timestamps) == 24
    assert sum(value.date() >= date(2024, 7, 1) for value in timestamps) == 48


def test_multi_day_episode_rejects_wrong_holdout_split() -> None:
    pl, config_type, error_type, extract_episode_range = _dependencies()
    frame = _frame(pl).with_columns(
        pl.when(pl.col("service_date") == date(2024, 7, 2))
        .then(pl.lit("validation"))
        .otherwise(pl.col("split"))
        .alias("split")
    )
    with pytest.raises(error_type, match="does not match configured split"):
        extract_episode_range(
            frame,
            zone_count=2,
            config=_config(config_type),
            evaluation_start_date=date(2024, 7, 1),
            evaluation_end_date=date(2024, 7, 2),
        )
