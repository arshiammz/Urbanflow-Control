from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path
from typing import Any

import polars as pl
import pytest
import yaml

from urbanflow.data.aggregation import (
    DemandAggregationError,
    DemandAggregationSettings,
    build_time_index,
    build_zone_demand,
)


TIMEZONE = "America/New_York"


def write_zone_lookup(path: Path) -> None:
    """Write a minimal zone lookup containing core and non-core zones."""
    zones = pl.DataFrame(
        {
            "LocationID": [
                1,
                2,
                3,
                264,
                265,
            ],
            "Borough": [
                "EWR",
                "Queens",
                "Brooklyn",
                "Unknown",
                "N/A",
            ],
            "Zone": [
                "Newark Airport",
                "Test Queens Zone",
                "Test Brooklyn Zone",
                "N/A",
                "Outside of NYC",
            ],
            "service_zone": [
                "EWR",
                "Boro Zone",
                "Boro Zone",
                "N/A",
                "N/A",
            ],
        }
    )

    zones.write_csv(path)


def localize_pickup_datetimes(
    values: list[datetime],
) -> pl.Series:
    """Create a timezone-aware Polars datetime series."""
    frame = (
        pl.DataFrame(
            {
                "pickup_datetime": values,
            }
        )
        .with_columns(
            pl.col("pickup_datetime")
            .dt.replace_time_zone(
                TIMEZONE,
                ambiguous="raise",
                non_existent="raise",
            )
        )
    )

    return frame["pickup_datetime"]


def write_validated_input(path: Path) -> None:
    """Write synthetic validated trips for demand aggregation."""
    pickup_datetimes = localize_pickup_datetimes(
        [
            datetime(2024, 1, 1, 0, 5),
            datetime(2024, 1, 1, 17, 45),
            datetime(2024, 1, 2, 8, 0),
            datetime(2024, 1, 2, 9, 0),
            datetime(2024, 1, 31, 23, 59),
            datetime(2024, 1, 15, 12, 0),
        ]
    )

    trips = pl.DataFrame(
        {
            "pickup_datetime": pickup_datetimes,
            "PULocationID": [
                2,
                2,
                3,
                3,
                2,
                1,
            ],
            "eligible_for_demand_forecasting": [
                True,
                True,
                True,
                False,
                True,
                False,
            ],
        }
    )

    trips.write_parquet(path)


def write_input_missing_eligibility(
    path: Path,
) -> None:
    """Write an input file missing a required aggregation column."""
    pickup_datetimes = localize_pickup_datetimes(
        [
            datetime(2024, 1, 1, 0, 5),
        ]
    )

    trips = pl.DataFrame(
        {
            "pickup_datetime": pickup_datetimes,
            "PULocationID": [2],
        }
    )

    trips.write_parquet(path)


def write_aggregation_config(
    tmp_path: Path,
    *,
    expected_year: int = 2024,
    expected_month: int = 1,
    timezone: str = TIMEZONE,
    interval: str = "1d",
    input_filename: str = "validated.parquet",
) -> Path:
    """Write a temporary demand aggregation configuration."""
    config: dict[str, Any] = {
        "dataset": {
            "name": "test_zone_demand",
            "expected_year": expected_year,
            "expected_month": expected_month,
            "timezone": timezone,
        },
        "paths": {
            "input": str(
                tmp_path / input_filename
            ),
            "zone_lookup": str(
                tmp_path / "zones.csv"
            ),
            "output": str(
                tmp_path / "zone_demand.parquet"
            ),
            "report": str(
                tmp_path / "aggregation-report.json"
            ),
        },
        "aggregation": {
            "interval": interval,
            "core_boroughs": [
                "Bronx",
                "Brooklyn",
                "Manhattan",
                "Queens",
                "Staten Island",
            ],
        },
        "output": {
            "compression": "zstd",
        },
    }

    config_path = (
        tmp_path / "aggregation.yaml"
    )

    config_path.write_text(
        yaml.safe_dump(
            config,
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    return config_path


def prepare_aggregation_case(
    tmp_path: Path,
) -> Path:
    """Create input, reference, and configuration files."""
    write_zone_lookup(
        tmp_path / "zones.csv"
    )

    write_validated_input(
        tmp_path / "validated.parquet"
    )

    return write_aggregation_config(
        tmp_path
    )


def test_build_zone_demand_creates_dense_panel(
    tmp_path: Path,
) -> None:
    """Build a dense daily panel and preserve eligible pickups."""
    config_path = prepare_aggregation_case(
        tmp_path
    )

    report = build_zone_demand(
        config_path
    )

    output_path = (
        tmp_path / "zone_demand.parquet"
    )

    report_path = (
        tmp_path / "aggregation-report.json"
    )

    assert output_path.exists()
    assert report_path.exists()

    output = pl.read_parquet(
        output_path
    )

    assert report["output"]["expected_row_count"] == 62
    assert report["output"]["row_count"] == 62

    assert report["panel"] == {
        "interval_count": 31,
        "zone_count": 2,
        "total_pickups": 4,
        "zero_demand_rows": 59,
        "zero_demand_rate": round(
            59 / 62,
            8,
        ),
    }

    assert report["pickup_count_distribution"][
        "minimum"
    ] == 0

    assert report["pickup_count_distribution"][
        "maximum"
    ] == 2

    assert report["invariants"] == {
        "row_count_matches_dense_grid": True,
        "pickup_count_is_conserved": True,
        "duplicate_key_count": 0,
        "null_value_count": 0,
        "minimum_pickup_count": 0,
    }

    assert output.height == 62
    assert output["zone_id"].n_unique() == 2

    assert set(
        output["zone_id"].unique().to_list()
    ) == {
        2,
        3,
    }

    assert output["interval_start"].n_unique() == 31
    assert output["pickup_count"].sum() == 4

    duplicate_count = (
        output.group_by(
            "interval_start",
            "zone_id",
        )
        .len()
        .filter(
            pl.col("len") > 1
        )
        .height
    )

    assert duplicate_count == 0

    null_count = sum(
        output.null_count().row(0)
    )

    assert null_count == 0

    assert TIMEZONE in str(
        output.schema["interval_start"]
    )

    report_from_disk = json.loads(
        report_path.read_text(
            encoding="utf-8"
        )
    )

    assert report_from_disk["panel"] == (
        report["panel"]
    )


def test_observed_pickups_are_assigned_to_expected_cells(
    tmp_path: Path,
) -> None:
    """Verify bucket assignment and zero-demand rows."""
    config_path = prepare_aggregation_case(
        tmp_path
    )

    build_zone_demand(
        config_path
    )

    output = pl.read_parquet(
        tmp_path / "zone_demand.parquet"
    )

    january_first_zone_two = (
        output.filter(
            (
                pl.col("service_date")
                == date(2024, 1, 1)
            )
            & (
                pl.col("zone_id") == 2
            )
        )
        .select(
            "pickup_count",
            "time_step",
            "day_of_week",
            "is_weekend",
        )
        .row(
            0,
            named=True,
        )
    )

    assert january_first_zone_two == {
        "pickup_count": 2,
        "time_step": 0,
        "day_of_week": 1,
        "is_weekend": False,
    }

    january_second_zone_three = (
        output.filter(
            (
                pl.col("service_date")
                == date(2024, 1, 2)
            )
            & (
                pl.col("zone_id") == 3
            )
        )
        .select(
            "pickup_count",
            "time_step",
        )
        .row(
            0,
            named=True,
        )
    )

    assert january_second_zone_three == {
        "pickup_count": 1,
        "time_step": 1,
    }

    january_thirty_first_zone_two = (
        output.filter(
            (
                pl.col("service_date")
                == date(2024, 1, 31)
            )
            & (
                pl.col("zone_id") == 2
            )
        )
        .select(
            "pickup_count",
            "time_step",
        )
        .row(
            0,
            named=True,
        )
    )

    assert january_thirty_first_zone_two == {
        "pickup_count": 1,
        "time_step": 30,
    }

    january_first_zone_three = (
        output.filter(
            (
                pl.col("service_date")
                == date(2024, 1, 1)
            )
            & (
                pl.col("zone_id") == 3
            )
        )
        .select(
            "pickup_count"
        )
        .item()
    )

    assert january_first_zone_three == 0


def test_time_index_respects_daylight_saving_transition(
    tmp_path: Path,
) -> None:
    """Generate the correct hourly grid during DST transition."""
    settings = DemandAggregationSettings(
        expected_year=2024,
        expected_month=3,
        timezone=TIMEZONE,
        input_path=(
            tmp_path / "unused.parquet"
        ),
        zone_lookup_path=(
            tmp_path / "unused.csv"
        ),
        output_path=(
            tmp_path / "unused-output.parquet"
        ),
        report_path=(
            tmp_path / "unused-report.json"
        ),
        interval="1h",
        core_boroughs=(
            "Queens",
            "Brooklyn",
        ),
        compression="zstd",
    )

    settings.validate()

    time_index = build_time_index(
        settings
    )

    assert time_index.height == 743

    march_tenth = (
        time_index.filter(
            pl.col("interval_start").dt.date()
            == date(2024, 3, 10)
        )
    )

    assert march_tenth.height == 23

    march_tenth_hours = (
        march_tenth[
            "interval_start"
        ]
        .dt.hour()
        .to_list()
    )

    assert 2 not in march_tenth_hours

    assert TIMEZONE in str(
        time_index.schema["interval_start"]
    )


@pytest.mark.parametrize(
    (
        "timezone",
        "interval",
        "expected_message",
    ),
    [
        (
            "Mars/Olympus_Mons",
            "1h",
            "Unknown timezone",
        ),
        (
            TIMEZONE,
            "0m",
            "Aggregation interval",
        ),
        (
            TIMEZONE,
            "fifteen-minutes",
            "Aggregation interval",
        ),
    ],
)
def test_invalid_aggregation_configuration_is_rejected(
    tmp_path: Path,
    timezone: str,
    interval: str,
    expected_message: str,
) -> None:
    """Reject invalid timezones and interval definitions."""
    config_path = write_aggregation_config(
        tmp_path,
        timezone=timezone,
        interval=interval,
    )

    with pytest.raises(
        DemandAggregationError,
        match=expected_message,
    ):
        DemandAggregationSettings.from_yaml(
            config_path
        )


def test_missing_input_column_is_rejected(
    tmp_path: Path,
) -> None:
    """Reject validated inputs missing aggregation columns."""
    write_zone_lookup(
        tmp_path / "zones.csv"
    )

    write_input_missing_eligibility(
        tmp_path / "missing-column.parquet"
    )

    config_path = write_aggregation_config(
        tmp_path,
        input_filename=(
            "missing-column.parquet"
        ),
    )

    with pytest.raises(
        DemandAggregationError,
        match="Validated input is missing columns",
    ):
        build_zone_demand(
            config_path
        )