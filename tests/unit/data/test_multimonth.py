from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import polars as pl
import pytest
import yaml

from urbanflow.data.multimonth import (
    MultiMonthPipelineError,
    MultiMonthSettings,
    build_month_paths,
    collect_combined_summary,
    collect_monthly_summary,
    enforce_combined_invariants,
    generate_month_configs,
    validate_monthly_outputs,
    write_combined_output,
)


TIMEZONE = "America/New_York"
INTERVAL_MINUTES = 15


def write_yaml(
    path: Path,
    content: dict[str, Any],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        yaml.safe_dump(
            content,
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def build_local_intervals(
    *,
    year: int,
    month: int,
) -> list[datetime]:
    timezone = ZoneInfo(TIMEZONE)

    start = datetime(
        year=year,
        month=month,
        day=1,
        tzinfo=timezone,
    )

    if month == 12:
        end = datetime(
            year=year + 1,
            month=1,
            day=1,
            tzinfo=timezone,
        )
    else:
        end = datetime(
            year=year,
            month=month + 1,
            day=1,
            tzinfo=timezone,
        )

    current_utc = start.astimezone(UTC)
    end_utc = end.astimezone(UTC)

    intervals: list[datetime] = []

    while current_utc < end_utc:
        intervals.append(
            current_utc.astimezone(
                timezone
            )
        )

        current_utc += timedelta(
            minutes=INTERVAL_MINUTES
        )

    return intervals


def write_monthly_demand(
    path: Path,
    *,
    year: int,
    month: int,
    zone_ids: tuple[int, ...] = (
        1,
        2,
    ),
) -> dict[str, int]:
    intervals = build_local_intervals(
        year=year,
        month=month,
    )

    time_steps: list[int] = []
    interval_values: list[datetime] = []
    zones: list[int] = []
    pickup_counts: list[int] = []

    for time_step, interval_start in enumerate(
        intervals
    ):
        for zone_id in zone_ids:
            time_steps.append(
                time_step
            )
            interval_values.append(
                interval_start
            )
            zones.append(
                zone_id
            )
            pickup_counts.append(
                (
                    time_step
                    + zone_id
                )
                % 11
            )

    frame = pl.DataFrame(
        {
            "time_step": time_steps,
            "interval_start": pl.Series(
                "interval_start",
                interval_values,
                dtype=pl.Datetime(
                    time_unit="ns",
                    time_zone=TIMEZONE,
                ),
            ),
            "zone_id": zones,
            "pickup_count": pickup_counts,
        },
        schema_overrides={
            "time_step": pl.Int32,
            "zone_id": pl.Int16,
            "pickup_count": pl.Int32,
        },
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    frame.write_parquet(
        path,
        compression="zstd",
        statistics=True,
    )

    return {
        "row_count": frame.height,
        "interval_count": len(
            intervals
        ),
        "zone_count": len(
            zone_ids
        ),
        "total_pickups": int(
            frame.get_column(
                "pickup_count"
            ).sum()
        ),
    }


def build_test_environment(
    tmp_path: Path,
    *,
    start_month: int = 1,
    end_month: int = 2,
) -> tuple[
    Path,
    MultiMonthSettings,
]:
    validation_template = (
        tmp_path
        / "configs"
        / "validation.yaml"
    )

    aggregation_template = (
        tmp_path
        / "configs"
        / "aggregation.yaml"
    )

    zone_lookup = (
        tmp_path
        / "reference"
        / "taxi_zone_lookup.csv"
    )

    zone_lookup.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    zone_lookup.write_text(
        (
            "LocationID,Borough,Zone,service_zone\n"
            "1,Manhattan,Zone One,Yellow Zone\n"
            "2,Queens,Zone Two,Boro Zone\n"
        ),
        encoding="utf-8",
    )

    write_yaml(
        validation_template,
        {
            "dataset": {
                "name": "hvfhv_validation",
                "expected_year": 2024,
                "expected_month": 1,
                "timezone": TIMEZONE,
            },
            "paths": {
                "input": "unused.parquet",
                "zone_lookup": (
                    "unused.csv"
                ),
                "output": (
                    "unused-validated.parquet"
                ),
                "report": (
                    "unused-validation.json"
                ),
            },
            "rules": {
                "preserved_setting": True,
            },
        },
    )

    write_yaml(
        aggregation_template,
        {
            "dataset": {
                "name": "hvfhv_demand",
                "expected_year": 2024,
                "expected_month": 1,
                "timezone": TIMEZONE,
            },
            "paths": {
                "input": (
                    "unused-validated.parquet"
                ),
                "zone_lookup": (
                    "unused.csv"
                ),
                "output": (
                    "unused-demand.parquet"
                ),
                "report": (
                    "unused-demand.json"
                ),
            },
            "aggregation": {
                "interval": "15m",
                "eligibility_column": (
                    "eligible_for_demand_forecasting"
                ),
                "preserved_setting": True,
            },
        },
    )

    config_path = (
        tmp_path
        / "configs"
        / "multimonth.yaml"
    )

    write_yaml(
        config_path,
        {
            "dataset": {
                "dataset": "hvfhv",
                "year": 2024,
                "start_month": (
                    start_month
                ),
                "end_month": (
                    end_month
                ),
                "timezone": TIMEZONE,
            },
            "templates": {
                "validation": str(
                    validation_template
                ),
                "demand_aggregation": str(
                    aggregation_template
                ),
            },
            "paths": {
                "raw_directory": str(
                    tmp_path
                    / "raw"
                ),
                "zone_lookup": str(
                    zone_lookup
                ),
                "interim_directory": str(
                    tmp_path
                    / "interim"
                ),
                "monthly_demand_directory": str(
                    tmp_path
                    / "processed"
                    / "monthly"
                ),
                "generated_config_directory": str(
                    tmp_path
                    / "generated"
                ),
                "combined_output": str(
                    tmp_path
                    / "processed"
                    / "combined.parquet"
                ),
                "report": str(
                    tmp_path
                    / "reports"
                    / "multimonth.json"
                ),
            },
            "execution": {
                "download_missing": True,
                "validate_missing": True,
                "aggregate_missing": True,
                "force_download": False,
                "force_validation": False,
                "force_aggregation": False,
                "force_combine": False,
                "cleanup_interim_after_aggregation": False,
            },
            "output": {
                "compression": "zstd",
            },
        },
    )

    settings = MultiMonthSettings.from_yaml(
        config_path
    )

    return (
        config_path,
        settings,
    )


def test_settings_load_month_range_and_paths(
    tmp_path: Path,
) -> None:
    _, settings = build_test_environment(
        tmp_path
    )

    assert settings.dataset == "hvfhv"
    assert settings.year == 2024
    assert settings.months == (
        1,
        2,
    )
    assert settings.month_count == 2
    assert settings.timezone == TIMEZONE
    assert (
        settings.aggregation_interval
        == "15m"
    )
    assert settings.interval_slug == "15m"


def test_month_paths_are_deterministic(
    tmp_path: Path,
) -> None:
    _, settings = build_test_environment(
        tmp_path
    )

    paths = build_month_paths(
        settings,
        2,
    )

    assert paths.period == "2024-02"

    assert paths.raw_path.name == (
        "fhvhv_tripdata_2024-02.parquet"
    )

    assert paths.validated_path.name == (
        "hvfhv_2024-02_validated.parquet"
    )

    assert paths.demand_path.name == (
        "zone_demand_15m_2024-02.parquet"
    )

    assert (
        paths.validation_config_path.name
        == "2024-02-validation.yaml"
    )

    assert (
        paths.aggregation_config_path.name
        == "2024-02-demand-aggregation.yaml"
    )


def test_generated_configs_override_month_paths(
    tmp_path: Path,
) -> None:
    _, settings = build_test_environment(
        tmp_path
    )

    paths = build_month_paths(
        settings,
        2,
    )

    generate_month_configs(
        settings,
        paths,
    )

    validation_config = yaml.safe_load(
        paths.validation_config_path.read_text(
            encoding="utf-8"
        )
    )

    aggregation_config = yaml.safe_load(
        paths.aggregation_config_path.read_text(
            encoding="utf-8"
        )
    )

    assert (
        validation_config["dataset"][
            "expected_year"
        ]
        == 2024
    )

    assert (
        validation_config["dataset"][
            "expected_month"
        ]
        == 2
    )

    assert (
        validation_config["dataset"][
            "timezone"
        ]
        == TIMEZONE
    )

    assert (
        validation_config["paths"][
            "input"
        ]
        == str(paths.raw_path)
    )

    assert (
        validation_config["paths"][
            "output"
        ]
        == str(paths.validated_path)
    )

    assert (
        validation_config["rules"][
            "preserved_setting"
        ]
        is True
    )

    assert (
        aggregation_config["dataset"][
            "expected_month"
        ]
        == 2
    )

    assert (
        aggregation_config["paths"][
            "input"
        ]
        == str(paths.validated_path)
    )

    assert (
        aggregation_config["paths"][
            "output"
        ]
        == str(paths.demand_path)
    )

    assert (
        aggregation_config[
            "aggregation"
        ]["interval"]
        == "15m"
    )

    assert (
        aggregation_config[
            "aggregation"
        ]["preserved_setting"]
        is True
    )


def test_monthly_summary_validates_dense_panel(
    tmp_path: Path,
) -> None:
    path = (
        tmp_path
        / "january.parquet"
    )

    expected = write_monthly_demand(
        path,
        year=2024,
        month=1,
    )

    summary = collect_monthly_summary(
        path
    )

    assert (
        summary["row_count"]
        == expected["row_count"]
    )

    assert (
        summary["interval_count"]
        == expected["interval_count"]
    )

    assert (
        summary["zone_count"]
        == expected["zone_count"]
    )

    assert (
        summary["total_pickups"]
        == expected["total_pickups"]
    )

    assert (
        summary["duplicate_key_count"]
        == 0
    )

    assert (
        summary["null_value_count"]
        == 0
    )

    assert (
        summary["minimum_pickup_count"]
        == 0
    )

    assert (
        summary["minimum_time_step"]
        == 0
    )

    assert (
        summary["maximum_time_step"]
        == expected["interval_count"] - 1
    )


def test_validate_monthly_outputs_accepts_contiguous_months(
    tmp_path: Path,
) -> None:
    _, settings = build_test_environment(
        tmp_path
    )

    month_paths = [
        build_month_paths(
            settings,
            month,
        )
        for month in settings.months
    ]

    for paths in month_paths:
        write_monthly_demand(
            paths.demand_path,
            year=paths.year,
            month=paths.month,
        )

    summaries, interval_seconds = (
        validate_monthly_outputs(
            settings,
            month_paths,
        )
    )

    assert interval_seconds == 900
    assert len(summaries) == 2

    assert (
        summaries[0]["interval_count"]
        == 31 * 96
    )

    assert (
        summaries[1]["interval_count"]
        == 29 * 96
    )

    assert (
        summaries[0]["zone_ids"]
        == summaries[1]["zone_ids"]
    )


def test_validate_monthly_outputs_handles_dst(
    tmp_path: Path,
) -> None:
    _, settings = build_test_environment(
        tmp_path,
        start_month=3,
        end_month=3,
    )

    paths = build_month_paths(
        settings,
        3,
    )

    write_monthly_demand(
        paths.demand_path,
        year=2024,
        month=3,
        zone_ids=(1,),
    )

    summaries, interval_seconds = (
        validate_monthly_outputs(
            settings,
            [paths],
        )
    )

    assert interval_seconds == 900

    assert (
        summaries[0]["interval_count"]
        == (31 * 96) - 4
    )

    assert (
        summaries[0]["row_count"]
        == (31 * 96) - 4
    )

    assert (
        summaries[0][
            "first_interval"
        ].isoformat()
        == "2024-03-01T00:00:00-05:00"
    )

    assert (
        summaries[0][
            "last_interval"
        ].isoformat()
        == "2024-03-31T23:45:00-04:00"
    )


def test_validate_monthly_outputs_rejects_gap(
    tmp_path: Path,
) -> None:
    _, settings = build_test_environment(
        tmp_path,
        start_month=1,
        end_month=3,
    )

    january = build_month_paths(
        settings,
        1,
    )

    march = build_month_paths(
        settings,
        3,
    )

    write_monthly_demand(
        january.demand_path,
        year=2024,
        month=1,
    )

    write_monthly_demand(
        march.demand_path,
        year=2024,
        month=3,
    )

    with pytest.raises(
        MultiMonthPipelineError,
        match="Gap or overlap",
    ):
        validate_monthly_outputs(
            settings,
            [
                january,
                march,
            ],
        )


def test_combination_reindexes_global_time_step(
    tmp_path: Path,
) -> None:
    _, settings = build_test_environment(
        tmp_path
    )

    month_paths = [
        build_month_paths(
            settings,
            month,
        )
        for month in settings.months
    ]

    expected_total_pickups = 0
    expected_rows = 0
    expected_intervals = 0

    for paths in month_paths:
        expected = write_monthly_demand(
            paths.demand_path,
            year=paths.year,
            month=paths.month,
        )

        expected_total_pickups += (
            expected["total_pickups"]
        )

        expected_rows += (
            expected["row_count"]
        )

        expected_intervals += (
            expected["interval_count"]
        )

    monthly_summaries, _ = (
        validate_monthly_outputs(
            settings,
            month_paths,
        )
    )

    action = write_combined_output(
        settings=settings,
        month_paths=month_paths,
        monthly_summaries=monthly_summaries,
    )

    assert action == "created"

    combined_summary = (
        collect_combined_summary(
            settings.combined_output_path
        )
    )

    enforce_combined_invariants(
        monthly_summaries=(
            monthly_summaries
        ),
        combined_summary=(
            combined_summary
        ),
    )

    assert (
        combined_summary["row_count"]
        == expected_rows
    )

    assert (
        combined_summary[
            "interval_count"
        ]
        == expected_intervals
    )

    assert (
        combined_summary[
            "total_pickups"
        ]
        == expected_total_pickups
    )

    assert (
        combined_summary[
            "minimum_time_step"
        ]
        == 0
    )

    assert (
        combined_summary[
            "maximum_time_step"
        ]
        == expected_intervals - 1
    )

    frame = pl.read_parquet(
        settings.combined_output_path
    )

    february_start = datetime(
        year=2024,
        month=2,
        day=1,
        tzinfo=ZoneInfo(TIMEZONE),
    )

    february_first_time_step = (
        frame.filter(
            pl.col("interval_start")
            == february_start
        )
        .get_column("time_step")
        .unique()
        .item()
    )

    assert (
        february_first_time_step
        == 31 * 96
    )


def test_combination_is_resumable(
    tmp_path: Path,
) -> None:
    _, settings = build_test_environment(
        tmp_path
    )

    month_paths = [
        build_month_paths(
            settings,
            month,
        )
        for month in settings.months
    ]

    for paths in month_paths:
        write_monthly_demand(
            paths.demand_path,
            year=paths.year,
            month=paths.month,
        )

    monthly_summaries, _ = (
        validate_monthly_outputs(
            settings,
            month_paths,
        )
    )

    first_action = write_combined_output(
        settings=settings,
        month_paths=month_paths,
        monthly_summaries=monthly_summaries,
    )

    second_action = write_combined_output(
        settings=settings,
        month_paths=month_paths,
        monthly_summaries=monthly_summaries,
    )

    assert first_action == "created"
    assert second_action == "skipped"


def test_monthly_summary_rejects_negative_demand(
    tmp_path: Path,
) -> None:
    path = (
        tmp_path
        / "negative.parquet"
    )

    write_monthly_demand(
        path,
        year=2024,
        month=1,
    )

    frame = pl.read_parquet(
        path
    )

    frame = frame.with_columns(
        pl.when(
            (
                pl.col("time_step") == 0
            )
            & (
                pl.col("zone_id") == 1
            )
        )
        .then(-1)
        .otherwise(
            pl.col("pickup_count")
        )
        .alias("pickup_count")
    )

    frame.write_parquet(
        path,
        compression="zstd",
    )

    with pytest.raises(
        MultiMonthPipelineError,
        match="negative pickup counts",
    ):
        collect_monthly_summary(
            path
        )


def test_invalid_month_range_is_rejected(
    tmp_path: Path,
) -> None:
    config_path, _ = build_test_environment(
        tmp_path
    )

    config = yaml.safe_load(
        config_path.read_text(
            encoding="utf-8"
        )
    )

    config["dataset"][
        "start_month"
    ] = 6

    config["dataset"][
        "end_month"
    ] = 2

    write_yaml(
        config_path,
        config,
    )

    with pytest.raises(
        MultiMonthPipelineError,
        match=(
            "start_month must not be "
            "after end_month"
        ),
    ):
        MultiMonthSettings.from_yaml(
            config_path
        )


def test_unknown_timezone_is_rejected(
    tmp_path: Path,
) -> None:
    config_path, _ = build_test_environment(
        tmp_path
    )

    config = yaml.safe_load(
        config_path.read_text(
            encoding="utf-8"
        )
    )

    config["dataset"][
        "timezone"
    ] = "Invalid/Timezone"

    write_yaml(
        config_path,
        config,
    )

    with pytest.raises(
        MultiMonthPipelineError,
        match="Unknown timezone",
    ):
        MultiMonthSettings.from_yaml(
            config_path
        )