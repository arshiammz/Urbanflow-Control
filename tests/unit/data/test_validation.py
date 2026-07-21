from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl
import pytest
import yaml

from urbanflow.data.validation import (
    ValidationError,
    ValidationSettings,
    validate_trip_data,
)


def write_zone_lookup(path: Path) -> None:
    """Write a minimal TLC zone lookup for validation tests."""
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


def write_trip_data(path: Path) -> None:
    """Write synthetic trips covering validation outcomes."""
    trips = pl.DataFrame(
        {
            "test_record_id": [
                "accepted",
                "request_after_pickup",
                "non_positive_trip",
                "non_core_pickup",
                "non_core_dropoff",
                "invalid_dropoff",
                "long_trip",
                "outside_month",
            ],
            "hvfhs_license_num": [
                "HV0003",
                "HV0003",
                "HV0003",
                "HV0003",
                "HV0003",
                "HV0003",
                "HV0003",
                "HV0003",
            ],
            "request_datetime": [
                datetime(2024, 1, 10, 9, 50),
                datetime(2024, 1, 10, 10, 5),
                datetime(2024, 1, 10, 10, 50),
                datetime(2024, 1, 10, 11, 50),
                datetime(2024, 1, 10, 12, 50),
                datetime(2024, 1, 10, 13, 50),
                datetime(2024, 1, 10, 14, 50),
                datetime(2024, 1, 31, 23, 50),
            ],
            "pickup_datetime": [
                datetime(2024, 1, 10, 10, 0),
                datetime(2024, 1, 10, 10, 0),
                datetime(2024, 1, 10, 11, 0),
                datetime(2024, 1, 10, 12, 0),
                datetime(2024, 1, 10, 13, 0),
                datetime(2024, 1, 10, 14, 0),
                datetime(2024, 1, 10, 15, 0),
                datetime(2024, 2, 1, 0, 0),
            ],
            "dropoff_datetime": [
                datetime(2024, 1, 10, 10, 30),
                datetime(2024, 1, 10, 10, 20),
                datetime(2024, 1, 10, 11, 0),
                datetime(2024, 1, 10, 12, 30),
                datetime(2024, 1, 10, 13, 30),
                datetime(2024, 1, 10, 14, 30),
                datetime(2024, 1, 10, 20, 0),
                datetime(2024, 2, 1, 0, 10),
            ],
            "PULocationID": [
                2,
                2,
                2,
                265,
                2,
                2,
                2,
                2,
            ],
            "DOLocationID": [
                3,
                3,
                3,
                3,
                265,
                999,
                3,
                3,
            ],
            "trip_miles": [
                8.0,
                5.0,
                0.0,
                15.0,
                20.0,
                4.0,
                100.0,
                3.0,
            ],
            "trip_time": [
                1800,
                1200,
                0,
                1800,
                1800,
                1800,
                18000,
                600,
            ],
        }
    )

    trips.write_parquet(path)


def write_validation_config(
    tmp_path: Path,
    *,
    include_rejected_records: bool = True,
    timezone: str = "America/New_York",
) -> Path:
    """Write a validation configuration using temporary paths."""
    config: dict[str, Any] = {
        "dataset": {
            "name": "hvfhv",
            "expected_year": 2024,
            "expected_month": 1,
            "timezone": timezone,
        },
        "paths": {
            "input": str(
                tmp_path / "trips.parquet"
            ),
            "zone_lookup": str(
                tmp_path / "zones.csv"
            ),
            "output": str(
                tmp_path / "validated.parquet"
            ),
            "report": str(
                tmp_path / "validation-report.json"
            ),
        },
        "validation": {
            "required_columns": [
                "hvfhs_license_num",
                "request_datetime",
                "pickup_datetime",
                "dropoff_datetime",
                "PULocationID",
                "DOLocationID",
                "trip_miles",
                "trip_time",
            ],
            "core_boroughs": [
                "Bronx",
                "Brooklyn",
                "Manhattan",
                "Queens",
                "Staten Island",
            ],
            "long_trip_duration_seconds": 14400,
        },
        "output": {
            "compression": "zstd",
            "include_rejected_records": (
                include_rejected_records
            ),
        },
    }

    config_path = (
        tmp_path / "validation.yaml"
    )

    config_path.write_text(
        yaml.safe_dump(
            config,
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    return config_path


def prepare_validation_case(
    tmp_path: Path,
    *,
    include_rejected_records: bool = True,
) -> Path:
    """Create all files required by a validation test."""
    write_trip_data(
        tmp_path / "trips.parquet"
    )

    write_zone_lookup(
        tmp_path / "zones.csv"
    )

    return write_validation_config(
        tmp_path,
        include_rejected_records=(
            include_rejected_records
        ),
    )


def test_validation_assigns_expected_statuses(
    tmp_path: Path,
) -> None:
    """Validate row status, flags, and task eligibility."""
    config_path = prepare_validation_case(
        tmp_path
    )

    report = validate_trip_data(
        config_path
    )

    assert report["status_counts"] == {
        "total": 8,
        "accepted": 1,
        "retained_with_flag": 4,
        "rejected": 3,
    }

    assert report["flag_counts"] == {
        "flag_missing_required_value": 0,
        "flag_invalid_pickup_zone": 0,
        "flag_invalid_dropoff_zone": 1,
        "flag_pickup_outside_expected_month": 1,
        "flag_non_positive_trip_duration": 1,
        "flag_non_positive_trip_time": 1,
        "flag_request_after_pickup": 1,
        "flag_long_trip_duration": 1,
        "flag_pickup_non_core_zone": 1,
        "flag_dropoff_non_core_zone": 1,
    }

    assert report["eligibility_counts"] == {
        "eligible_for_demand_forecasting": 4,
        "eligible_for_od_modeling": 2,
        "eligible_for_wait_time_analysis": 4,
    }

    output = pl.read_parquet(
        tmp_path / "validated.parquet"
    )

    rows = {
        row["test_record_id"]: row
        for row in output.to_dicts()
    }

    assert rows["accepted"]["record_status"] == (
        "accepted"
    )

    assert rows[
        "request_after_pickup"
    ]["record_status"] == "retained_with_flag"

    assert rows[
        "request_after_pickup"
    ]["eligible_for_wait_time_analysis"] is False

    assert rows[
        "non_positive_trip"
    ]["record_status"] == "rejected"

    assert rows[
        "non_core_pickup"
    ]["flag_pickup_non_core_zone"] is True

    assert rows[
        "non_core_pickup"
    ]["eligible_for_demand_forecasting"] is False

    assert rows[
        "non_core_dropoff"
    ]["eligible_for_demand_forecasting"] is True

    assert rows[
        "non_core_dropoff"
    ]["eligible_for_od_modeling"] is False

    assert rows[
        "invalid_dropoff"
    ]["flag_invalid_dropoff_zone"] is True

    assert rows[
        "invalid_dropoff"
    ]["record_status"] == "rejected"

    assert rows[
        "long_trip"
    ]["flag_long_trip_duration"] is True

    assert rows[
        "long_trip"
    ]["eligible_for_od_modeling"] is False

    assert rows[
        "outside_month"
    ]["flag_pickup_outside_expected_month"] is True

    assert rows[
        "outside_month"
    ]["record_status"] == "rejected"

    pickup_dtype = output.schema[
        "pickup_datetime"
    ]

    assert "America/New_York" in str(
        pickup_dtype
    )


def test_validation_can_exclude_rejected_rows(
    tmp_path: Path,
) -> None:
    """Exclude rejected rows while preserving full summary counts."""
    config_path = prepare_validation_case(
        tmp_path,
        include_rejected_records=False,
    )

    report = validate_trip_data(
        config_path
    )

    output = pl.read_parquet(
        tmp_path / "validated.parquet"
    )

    assert report["status_counts"] == {
        "total": 8,
        "accepted": 1,
        "retained_with_flag": 4,
        "rejected": 3,
    }

    assert report["output"]["record_count"] == 5
    assert output.height == 5
    assert not output["row_rejected"].any()
    assert "rejected" not in (
        output["record_status"].unique().to_list()
    )


def test_invalid_timezone_is_rejected(
    tmp_path: Path,
) -> None:
    """Reject unknown IANA timezone identifiers."""
    config_path = write_validation_config(
        tmp_path,
        timezone="Mars/Olympus_Mons",
    )

    with pytest.raises(
        ValidationError,
        match="Unknown timezone",
    ):
        ValidationSettings.from_yaml(
            config_path
        )