from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import polars as pl
import pytest
import yaml

from urbanflow.forecasting.features import (
    ForecastFeatureError,
    ForecastFeatureSettings,
    build_forecast_features,
)


TIMEZONE = "America/New_York"


def write_dense_demand_panel(
    path: Path,
) -> None:
    """Write a complete hourly zone-demand panel for January."""
    intervals = pl.datetime_range(
        start=datetime(2024, 1, 1),
        end=datetime(2024, 2, 1),
        interval="1h",
        closed="left",
        time_unit="ns",
        time_zone=TIMEZONE,
        eager=True,
    )

    time_index = (
        pl.DataFrame(
            {
                "interval_start": intervals,
            }
        )
        .with_row_index(
            name="time_step",
            offset=0,
        )
        .with_columns(
            pl.col("time_step").cast(
                pl.Int32
            )
        )
    )

    zones = pl.DataFrame(
        {
            "zone_id": [
                2,
                3,
            ],
            "borough": [
                "Queens",
                "Brooklyn",
            ],
            "zone_name": [
                "Test Queens Zone",
                "Test Brooklyn Zone",
            ],
            "service_zone": [
                "Boro Zone",
                "Boro Zone",
            ],
        },
        schema={
            "zone_id": pl.Int32,
            "borough": pl.String,
            "zone_name": pl.String,
            "service_zone": pl.String,
        },
    )

    panel = (
        time_index.join(
            zones,
            how="cross",
        )
        .with_columns(
            (
                pl.col("time_step")
                + pl.when(
                    pl.col("zone_id") == 2
                )
                .then(0)
                .otherwise(1000)
            )
            .cast(pl.Int32)
            .alias("pickup_count")
        )
        .with_columns(
            pl.col("interval_start")
            .dt.date()
            .alias("service_date"),

            pl.col("interval_start")
            .dt.year()
            .cast(pl.Int16)
            .alias("year"),

            pl.col("interval_start")
            .dt.month()
            .cast(pl.Int8)
            .alias("month"),

            pl.col("interval_start")
            .dt.day()
            .cast(pl.Int8)
            .alias("day"),

            pl.col("interval_start")
            .dt.weekday()
            .cast(pl.Int8)
            .alias("day_of_week"),

            pl.col("interval_start")
            .dt.hour()
            .cast(pl.Int8)
            .alias("hour"),

            pl.col("interval_start")
            .dt.minute()
            .cast(pl.Int8)
            .alias("minute"),
        )
        .with_columns(
            pl.col("day_of_week")
            .is_in([6, 7])
            .alias("is_weekend")
        )
        .select(
            "time_step",
            "interval_start",
            "service_date",
            "year",
            "month",
            "day",
            "day_of_week",
            "hour",
            "minute",
            "is_weekend",
            "zone_id",
            "borough",
            "zone_name",
            "service_zone",
            "pickup_count",
        )
        .sort(
            "interval_start",
            "zone_id",
        )
    )

    panel.write_parquet(path)


def write_feature_config(
    tmp_path: Path,
    *,
    timezone: str = TIMEZONE,
    lags: list[int] | None = None,
    rolling_windows: list[int] | None = None,
) -> Path:
    """Write a temporary forecasting feature configuration."""
    config: dict[str, Any] = {
        "dataset": {
            "name": "test_forecast_features",
            "expected_year": 2024,
            "expected_month": 1,
            "timezone": timezone,
        },
        "paths": {
            "input": str(
                tmp_path / "zone-demand.parquet"
            ),
            "output": str(
                tmp_path / "features.parquet"
            ),
            "report": str(
                tmp_path / "feature-report.json"
            ),
        },
        "features": {
            "target_column": "pickup_count",
            "lags": (
                lags
                if lags is not None
                else [1, 2, 24]
            ),
            "rolling_windows": (
                rolling_windows
                if rolling_windows is not None
                else [2, 24]
            ),
        },
        "splits": {
            "train_end": "2024-01-20",
            "validation_end": "2024-01-26",
            "test_end": "2024-02-01",
        },
        "output": {
            "compression": "zstd",
        },
    }

    config_path = (
        tmp_path / "features.yaml"
    )

    config_path.write_text(
        yaml.safe_dump(
            config,
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    return config_path


def prepare_feature_case(
    tmp_path: Path,
) -> Path:
    """Create the synthetic input panel and feature config."""
    write_dense_demand_panel(
        tmp_path / "zone-demand.parquet"
    )

    return write_feature_config(
        tmp_path
    )


def test_build_forecast_features_creates_expected_dataset(
    tmp_path: Path,
) -> None:
    """Build features and enforce output dimensions."""
    config_path = prepare_feature_case(
        tmp_path
    )

    report = build_forecast_features(
        config_path
    )

    output_path = (
        tmp_path / "features.parquet"
    )

    output = pl.read_parquet(
        output_path
    )

    assert output_path.exists()
    assert (
        tmp_path / "feature-report.json"
    ).exists()

    assert report["output"][
        "expected_row_count"
    ] == 1440

    assert report["output"][
        "row_count"
    ] == 1440

    assert report["output"][
        "interval_count"
    ] == 720

    assert report["output"][
        "zone_count"
    ] == 2

    assert report["splits"] == {
        "train_rows": 864,
        "validation_rows": 288,
        "test_rows": 288,
    }

    assert report["feature_schema"][
        "feature_count"
    ] == 19

    assert report["invariants"][
        "history_uses_past_only"
    ] is True

    assert report["invariants"][
        "duplicate_key_count"
    ] == 0

    assert report["invariants"][
        "null_value_count"
    ] == 0

    assert output.height == 1440
    assert output["zone_id"].n_unique() == 2
    assert output["interval_start"].n_unique() == 720
    assert output.null_count().row(0) == (
        0,
    ) * output.width

    assert TIMEZONE in str(
        output.schema["interval_start"]
    )


def test_history_features_use_only_previous_values(
    tmp_path: Path,
) -> None:
    """Verify lag and rolling features exclude the current target."""
    config_path = prepare_feature_case(
        tmp_path
    )

    build_forecast_features(
        config_path
    )

    output = pl.read_parquet(
        tmp_path / "features.parquet"
    )

    zone_two_row = (
        output.filter(
            (pl.col("zone_id") == 2)
            & (
                pl.col("service_date")
                == date(2024, 1, 2)
            )
            & (pl.col("hour") == 0)
        )
        .row(
            0,
            named=True,
        )
    )

    assert zone_two_row[
        "target_pickup_count"
    ] == 24

    assert zone_two_row[
        "pickup_count_lag_1"
    ] == 23

    assert zone_two_row[
        "pickup_count_lag_2"
    ] == 22

    assert zone_two_row[
        "pickup_count_lag_24"
    ] == 0

    assert zone_two_row[
        "pickup_count_rolling_mean_2"
    ] == pytest.approx(
        22.5,
    )

    assert zone_two_row[
        "pickup_count_rolling_std_2"
    ] == pytest.approx(
        0.70710678,
        rel=1e-5,
    )

    assert zone_two_row[
        "pickup_count_rolling_mean_24"
    ] == pytest.approx(
        11.5,
    )

    assert zone_two_row[
        "pickup_count_rolling_std_24"
    ] == pytest.approx(
        7.07106781,
        rel=1e-5,
    )


def test_history_is_calculated_independently_per_zone(
    tmp_path: Path,
) -> None:
    """Prevent history values from crossing zone boundaries."""
    config_path = prepare_feature_case(
        tmp_path
    )

    build_forecast_features(
        config_path
    )

    output = pl.read_parquet(
        tmp_path / "features.parquet"
    )

    zone_three_row = (
        output.filter(
            (pl.col("zone_id") == 3)
            & (
                pl.col("service_date")
                == date(2024, 1, 2)
            )
            & (pl.col("hour") == 0)
        )
        .row(
            0,
            named=True,
        )
    )

    assert zone_three_row[
        "target_pickup_count"
    ] == 1024

    assert zone_three_row[
        "pickup_count_lag_1"
    ] == 1023

    assert zone_three_row[
        "pickup_count_lag_24"
    ] == 1000

    assert zone_three_row[
        "pickup_count_rolling_mean_24"
    ] == pytest.approx(
        1011.5,
    )


def test_temporal_split_boundaries_are_respected(
    tmp_path: Path,
) -> None:
    """Assign rows to non-overlapping chronological splits."""
    config_path = prepare_feature_case(
        tmp_path
    )

    build_forecast_features(
        config_path
    )

    output = pl.read_parquet(
        tmp_path / "features.parquet"
    )

    train_row = (
        output.filter(
            (
                pl.col("service_date")
                == date(2024, 1, 19)
            )
            & (pl.col("hour") == 23)
            & (pl.col("zone_id") == 2)
        )
        .select("split")
        .item()
    )

    validation_row = (
        output.filter(
            (
                pl.col("service_date")
                == date(2024, 1, 20)
            )
            & (pl.col("hour") == 0)
            & (pl.col("zone_id") == 2)
        )
        .select("split")
        .item()
    )

    test_row = (
        output.filter(
            (
                pl.col("service_date")
                == date(2024, 1, 26)
            )
            & (pl.col("hour") == 0)
            & (pl.col("zone_id") == 2)
        )
        .select("split")
        .item()
    )

    assert train_row == "train"
    assert validation_row == "validation"
    assert test_row == "test"


@pytest.mark.parametrize(
    (
        "timezone",
        "lags",
        "rolling_windows",
        "expected_message",
    ),
    [
        (
            "Mars/Olympus_Mons",
            [1, 2],
            [2],
            "Unknown timezone",
        ),
        (
            TIMEZONE,
            [1, 1],
            [2],
            "Lag values must be unique",
        ),
        (
            TIMEZONE,
            [0, 1],
            [2],
            "All lag values must be positive",
        ),
        (
            TIMEZONE,
            [1, 2],
            [1],
            "Rolling windows must be greater than one",
        ),
    ],
)
def test_invalid_feature_configuration_is_rejected(
    tmp_path: Path,
    timezone: str,
    lags: list[int],
    rolling_windows: list[int],
    expected_message: str,
) -> None:
    """Reject unsafe or malformed feature configuration."""
    config_path = write_feature_config(
        tmp_path,
        timezone=timezone,
        lags=lags,
        rolling_windows=rolling_windows,
    )

    with pytest.raises(
        ForecastFeatureError,
        match=expected_message,
    ):
        ForecastFeatureSettings.from_yaml(
            config_path
        )


def test_feature_dataset_preserves_timezone(
    tmp_path: Path,
) -> None:
    """Keep the configured timezone in the feature output."""
    config_path = prepare_feature_case(
        tmp_path
    )

    build_forecast_features(
        config_path
    )

    schema = pl.scan_parquet(
        tmp_path / "features.parquet"
    ).collect_schema()

    assert schema[
        "interval_start"
    ] == pl.Datetime(
        time_unit="ns",
        time_zone=TIMEZONE,
    )

    first_interval = (
        pl.scan_parquet(
            tmp_path / "features.parquet"
        )
        .select(
            pl.col("interval_start").min()
        )
        .collect()
        .item()
    )

    assert first_interval == datetime(
        2024,
        1,
        2,
        0,
        0,
        tzinfo=ZoneInfo(TIMEZONE),
    )