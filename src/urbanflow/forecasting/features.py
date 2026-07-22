from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from math import pi
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import polars as pl
import yaml


INPUT_REQUIRED_COLUMNS = {
    "time_step",
    "interval_start",
    "service_date",
    "day_of_week",
    "hour",
    "minute",
    "is_weekend",
    "zone_id",
    "borough",
    "zone_name",
    "service_zone",
    "pickup_count",
}

BASE_OUTPUT_COLUMNS = (
    "interval_start",
    "time_step",
    "service_date",
    "zone_id",
    "borough",
    "zone_name",
    "service_zone",
    "split",
    "target_pickup_count",
    "day_of_week",
    "hour",
    "minute",
    "minute_of_day",
    "is_weekend",
    "time_of_day_sin",
    "time_of_day_cos",
    "day_of_week_sin",
    "day_of_week_cos",
)


class ForecastFeatureError(RuntimeError):
    """Raised when forecast feature generation cannot be completed."""


@dataclass(frozen=True)
class ForecastFeatureSettings:
    """Configuration for leakage-safe forecasting features."""

    expected_year: int
    expected_month: int
    timezone: str

    input_path: Path
    output_path: Path
    report_path: Path

    target_column: str
    lags: tuple[int, ...]
    rolling_windows: tuple[int, ...]

    train_end: date
    validation_end: date
    test_end: date

    compression: str

    @classmethod
    def from_yaml(
        cls,
        config_path: Path,
    ) -> ForecastFeatureSettings:
        """Load feature-generation settings from YAML."""
        if not config_path.exists():
            raise ForecastFeatureError(
                f"Feature config does not exist: {config_path}"
            )

        try:
            raw_config = yaml.safe_load(
                config_path.read_text(
                    encoding="utf-8",
                )
            )
        except (OSError, yaml.YAMLError) as error:
            raise ForecastFeatureError(
                f"Could not read feature config: {error}"
            ) from error

        if not isinstance(raw_config, dict):
            raise ForecastFeatureError(
                "Feature config must contain a YAML mapping."
            )

        try:
            dataset_config = raw_config["dataset"]
            paths_config = raw_config["paths"]
            feature_config = raw_config["features"]
            split_config = raw_config["splits"]
            output_config = raw_config["output"]

            settings = cls(
                expected_year=int(
                    dataset_config["expected_year"]
                ),
                expected_month=int(
                    dataset_config["expected_month"]
                ),
                timezone=str(
                    dataset_config["timezone"]
                ),
                input_path=Path(
                    paths_config["input"]
                ),
                output_path=Path(
                    paths_config["output"]
                ),
                report_path=Path(
                    paths_config["report"]
                ),
                target_column=str(
                    feature_config["target_column"]
                ),
                lags=tuple(
                    int(value)
                    for value in feature_config["lags"]
                ),
                rolling_windows=tuple(
                    int(value)
                    for value in feature_config[
                        "rolling_windows"
                    ]
                ),
                train_end=date.fromisoformat(
                    str(split_config["train_end"])
                ),
                validation_end=date.fromisoformat(
                    str(
                        split_config[
                            "validation_end"
                        ]
                    )
                ),
                test_end=date.fromisoformat(
                    str(split_config["test_end"])
                ),
                compression=str(
                    output_config["compression"]
                ),
            )

        except (
            KeyError,
            TypeError,
            ValueError,
        ) as error:
            raise ForecastFeatureError(
                f"Invalid feature configuration: {error}"
            ) from error

        settings.validate()

        return settings

    @property
    def maximum_history_steps(self) -> int:
        """Return the longest required historical window."""
        return max(
            *self.lags,
            *self.rolling_windows,
        )

    @property
    def lag_columns(self) -> tuple[str, ...]:
        """Return configured lag-feature names."""
        return tuple(
            f"pickup_count_lag_{lag}"
            for lag in self.lags
        )

    @property
    def rolling_columns(self) -> tuple[str, ...]:
        """Return configured rolling-feature names."""
        columns: list[str] = []

        for window in self.rolling_windows:
            columns.extend(
                [
                    (
                        "pickup_count_rolling_"
                        f"mean_{window}"
                    ),
                    (
                        "pickup_count_rolling_"
                        f"std_{window}"
                    ),
                ]
            )

        return tuple(columns)

    @property
    def history_feature_columns(
        self,
    ) -> tuple[str, ...]:
        """Return every lagged and rolling feature column."""
        return (
            *self.lag_columns,
            *self.rolling_columns,
        )

    @property
    def model_feature_columns(
        self,
    ) -> tuple[str, ...]:
        """Return columns intended for model input."""
        return (
            "zone_id",
            "borough",
            "service_zone",
            "day_of_week",
            "hour",
            "minute",
            "minute_of_day",
            "is_weekend",
            "time_of_day_sin",
            "time_of_day_cos",
            "day_of_week_sin",
            "day_of_week_cos",
            *self.history_feature_columns,
        )

    def validate(self) -> None:
        """Validate configuration values."""
        if self.expected_year < 2009:
            raise ForecastFeatureError(
                f"Invalid expected year: {self.expected_year}"
            )

        if not 1 <= self.expected_month <= 12:
            raise ForecastFeatureError(
                f"Invalid expected month: {self.expected_month}"
            )

        try:
            ZoneInfo(self.timezone)
        except ZoneInfoNotFoundError as error:
            raise ForecastFeatureError(
                f"Unknown timezone: {self.timezone}"
            ) from error

        if self.target_column != "pickup_count":
            raise ForecastFeatureError(
                "The current feature pipeline requires "
                "target_column='pickup_count'."
            )

        if not self.lags:
            raise ForecastFeatureError(
                "At least one lag must be configured."
            )

        if not self.rolling_windows:
            raise ForecastFeatureError(
                "At least one rolling window must be configured."
            )

        if any(
            value <= 0
            for value in self.lags
        ):
            raise ForecastFeatureError(
                "All lag values must be positive."
            )

        if any(
            value <= 1
            for value in self.rolling_windows
        ):
            raise ForecastFeatureError(
                "Rolling windows must be greater than one."
            )

        if len(set(self.lags)) != len(
            self.lags
        ):
            raise ForecastFeatureError(
                "Lag values must be unique."
            )

        if len(
            set(self.rolling_windows)
        ) != len(self.rolling_windows):
            raise ForecastFeatureError(
                "Rolling-window values must be unique."
            )

        dataset_start, _ = (
            get_month_date_bounds(
                year=self.expected_year,
                month=self.expected_month,
            )
        )

        if not (
            dataset_start
            < self.train_end
            < self.validation_end
            < self.test_end
        ):
            raise ForecastFeatureError(
                "Split boundaries must satisfy: "
                "dataset_start < train_end < "
                "validation_end < test_end."
            )

        if self.test_end.day != 1:
            raise ForecastFeatureError(
                "test_end must be the first day after "
                "the final complete dataset month."
            )

        if not self.compression:
            raise ForecastFeatureError(
                "Output compression must not be empty."
            )

    def split_datetimes(
        self,
    ) -> tuple[datetime, datetime, datetime]:
        """Return timezone-aware split boundaries."""
        timezone_info = ZoneInfo(
            self.timezone
        )

        train_end_datetime = datetime.combine(
            self.train_end,
            time.min,
            tzinfo=timezone_info,
        )

        validation_end_datetime = datetime.combine(
            self.validation_end,
            time.min,
            tzinfo=timezone_info,
        )

        test_end_datetime = datetime.combine(
            self.test_end,
            time.min,
            tzinfo=timezone_info,
        )

        return (
            train_end_datetime,
            validation_end_datetime,
            test_end_datetime,
        )

    def as_report_dict(self) -> dict[str, Any]:
        """Return effective configuration as JSON data."""
        return {
            "dataset": {
                "expected_year": self.expected_year,
                "expected_month": self.expected_month,
                "dataset_start": (
                    date(
                        self.expected_year,
                        self.expected_month,
                        1,
                    ).isoformat()
                ),
                "dataset_end": self.test_end.isoformat(),
                "timezone": self.timezone,
            },
            "paths": {
                "input": str(self.input_path),
                "output": str(self.output_path),
                "report": str(self.report_path),
            },
            "features": {
                "target_column": self.target_column,
                "lags": list(self.lags),
                "rolling_windows": list(
                    self.rolling_windows
                ),
                "maximum_history_steps": (
                    self.maximum_history_steps
                ),
            },
            "splits": {
                "train_end": (
                    self.train_end.isoformat()
                ),
                "validation_end": (
                    self.validation_end.isoformat()
                ),
                "test_end": (
                    self.test_end.isoformat()
                ),
            },
            "output": {
                "compression": self.compression,
            },
        }


def get_month_date_bounds(
    year: int,
    month: int,
) -> tuple[date, date]:
    """Return the first day of this and the next month."""
    month_start = date(
        year=year,
        month=month,
        day=1,
    )

    if month == 12:
        next_month_start = date(
            year=year + 1,
            month=1,
            day=1,
        )
    else:
        next_month_start = date(
            year=year,
            month=month + 1,
            day=1,
        )

    return month_start, next_month_start


def validate_input_panel(
    settings: ForecastFeatureSettings,
) -> dict[str, int]:
    """Validate the dense zone-demand input panel."""
    if not settings.input_path.exists():
        raise ForecastFeatureError(
            f"Demand panel does not exist: {settings.input_path}"
        )

    if settings.input_path.stat().st_size == 0:
        raise ForecastFeatureError(
            f"Demand panel is empty: {settings.input_path}"
        )

    try:
        source = pl.scan_parquet(
            settings.input_path
        )

        schema = source.collect_schema()

    except Exception as error:
        raise ForecastFeatureError(
            f"Could not read demand panel: {error}"
        ) from error

    missing_columns = sorted(
        INPUT_REQUIRED_COLUMNS.difference(
            schema.names()
        )
    )

    if missing_columns:
        missing_text = ", ".join(
            missing_columns
        )

        raise ForecastFeatureError(
            "Demand panel is missing columns: "
            f"{missing_text}"
        )

    interval_dtype = schema[
        "interval_start"
    ]

    if settings.timezone not in str(
        interval_dtype
    ):
        raise ForecastFeatureError(
            "interval_start does not use the configured "
            f"timezone {settings.timezone}: "
            f"{interval_dtype}"
        )

    summary = (
        source.select(
            pl.len().alias(
                "row_count"
            ),
            pl.col("interval_start")
            .n_unique()
            .alias(
                "interval_count"
            ),
            pl.col("zone_id")
            .n_unique()
            .alias(
                "zone_count"
            ),
            pl.col("pickup_count")
            .min()
            .alias(
                "minimum_pickup_count"
            ),
            pl.col("time_step")
            .n_unique()
            .alias(
                "time_step_count"
            ),
            pl.col("time_step")
            .min()
            .alias(
                "minimum_time_step"
            ),
            pl.col("time_step")
            .max()
            .alias(
                "maximum_time_step"
            ),
            pl.col("interval_start")
            .min()
            .alias(
                "first_interval"
            ),
            pl.col("interval_start")
            .max()
            .alias(
                "last_interval"
            ),
        )
        .collect()
        .row(
            0,
            named=True,
        )
    )

    duplicate_key_count = int(
        source.group_by(
            "interval_start",
            "zone_id",
        )
        .agg(
            pl.len().alias(
                "key_count"
            )
        )
        .filter(
            pl.col("key_count") > 1
        )
        .select(
            pl.len().alias(
                "duplicate_key_count"
            )
        )
        .collect()
        .item()
    )

    null_row = (
        source.select(
            [
                pl.col(column)
                .null_count()
                .alias(column)
                for column in sorted(
                    INPUT_REQUIRED_COLUMNS
                )
            ]
        )
        .collect()
        .row(0)
    )

    required_null_count = sum(
        int(value)
        for value in null_row
    )

    row_count = int(
        summary["row_count"]
    )

    interval_count = int(
        summary["interval_count"]
    )

    zone_count = int(
        summary["zone_count"]
    )

    minimum_pickup_count = int(
        summary["minimum_pickup_count"]
    )

    time_step_count = int(
        summary["time_step_count"]
    )

    minimum_time_step = int(
        summary["minimum_time_step"]
    )

    maximum_time_step = int(
        summary["maximum_time_step"]
    )

    if duplicate_key_count:
        raise ForecastFeatureError(
            "Demand panel contains duplicate "
            "interval_start and zone_id keys."
        )

    if required_null_count:
        raise ForecastFeatureError(
            "Demand panel contains null values "
            "in required columns."
        )

    if row_count != interval_count * zone_count:
        raise ForecastFeatureError(
            "Demand panel is not a complete dense grid."
        )

    if minimum_pickup_count < 0:
        raise ForecastFeatureError(
            "Demand panel contains negative pickup counts."
        )

    if (
        interval_count
        <= settings.maximum_history_steps
    ):
        raise ForecastFeatureError(
            "Demand panel does not contain enough "
            "intervals for the configured history."
        )

    if time_step_count != interval_count:
        raise ForecastFeatureError(
            "Demand panel time_step values do not match "
            "the interval count."
        )

    if minimum_time_step != 0:
        raise ForecastFeatureError(
            "Demand panel time_step must start at zero."
        )

    if maximum_time_step != interval_count - 1:
        raise ForecastFeatureError(
            "Demand panel time_step values are not contiguous."
        )

    first_interval = summary[
        "first_interval"
    ]

    last_interval = summary[
        "last_interval"
    ]

    timezone_info = ZoneInfo(
        settings.timezone
    )

    dataset_start = datetime(
        year=settings.expected_year,
        month=settings.expected_month,
        day=1,
        tzinfo=timezone_info,
    )

    dataset_end = datetime.combine(
        settings.test_end,
        time.min,
        tzinfo=timezone_info,
    )

    if first_interval != dataset_start:
        raise ForecastFeatureError(
            "Demand panel does not begin at the configured "
            f"dataset start: expected {dataset_start!s}, "
            f"received {first_interval!s}."
        )

    first_two_intervals = (
        source.select(
            pl.col("interval_start")
        )
        .unique()
        .sort("interval_start")
        .limit(2)
        .collect()
        .get_column("interval_start")
        .to_list()
    )

    if len(first_two_intervals) != 2:
        raise ForecastFeatureError(
            "Demand panel does not contain enough intervals "
            "to derive its frequency."
        )

    interval_seconds = int(
        (
            first_two_intervals[1].astimezone(UTC)
            - first_two_intervals[0].astimezone(UTC)
        ).total_seconds()
    )

    if interval_seconds <= 0:
        raise ForecastFeatureError(
            "Demand panel interval duration must be positive."
        )

    expected_interval_count = int(
        (
            dataset_end.astimezone(UTC)
            - dataset_start.astimezone(UTC)
        ).total_seconds()
        // interval_seconds
    )

    expected_last_interval = (
        dataset_end.astimezone(UTC)
        - timedelta(
            seconds=interval_seconds
        )
    ).astimezone(
        timezone_info
    )

    if interval_count != expected_interval_count:
        raise ForecastFeatureError(
            "Demand panel interval count does not cover the "
            "configured dataset range: "
            f"expected {expected_interval_count:,}, "
            f"received {interval_count:,}."
        )

    if last_interval != expected_last_interval:
        raise ForecastFeatureError(
            "Demand panel does not end at the configured "
            f"dataset boundary: expected "
            f"{expected_last_interval!s}, received "
            f"{last_interval!s}."
        )

    return {
        "row_count": row_count,
        "interval_count": interval_count,
        "zone_count": zone_count,
        "duplicate_key_count": (
            duplicate_key_count
        ),
        "required_null_count": (
            required_null_count
        ),
        "interval_seconds": interval_seconds,
        "minimum_time_step": minimum_time_step,
        "maximum_time_step": maximum_time_step,
    }


def build_feature_frame(
    settings: ForecastFeatureSettings,
) -> pl.LazyFrame:
    """Build leakage-safe historical and calendar features."""
    train_end, validation_end, test_end = (
        settings.split_datetimes()
    )

    lag_expressions = [
        (
            pl.col(settings.target_column)
            .shift(lag)
            .over(
                "zone_id",
                order_by="interval_start",
            )
            .cast(pl.Int32)
            .alias(
                f"pickup_count_lag_{lag}"
            )
        )
        for lag in settings.lags
    ]

    rolling_expressions: list[
        pl.Expr
    ] = []

    for window in settings.rolling_windows:
        rolling_expressions.extend(
            [
                (
                    pl.col(
                        settings.target_column
                    )
                    .shift(1)
                    .rolling_mean(
                        window_size=window,
                        min_samples=window,
                    )
                    .over(
                        "zone_id",
                        order_by="interval_start",
                    )
                    .cast(pl.Float32)
                    .alias(
                        "pickup_count_rolling_"
                        f"mean_{window}"
                    )
                ),
                (
                    pl.col(
                        settings.target_column
                    )
                    .shift(1)
                    .rolling_std(
                        window_size=window,
                        min_samples=window,
                        ddof=1,
                    )
                    .over(
                        "zone_id",
                        order_by="interval_start",
                    )
                    .cast(pl.Float32)
                    .alias(
                        "pickup_count_rolling_"
                        f"std_{window}"
                    )
                ),
            ]
        )

    frame = (
        pl.scan_parquet(
            settings.input_path
        )
        .with_columns(
            *lag_expressions,
            *rolling_expressions,
        )
        .with_columns(
            (
                pl.col("hour").cast(
                    pl.Int16
                )
                * 60
                + pl.col("minute").cast(
                    pl.Int16
                )
            )
            .cast(pl.Int16)
            .alias(
                "minute_of_day"
            )
        )
        .with_columns(
            (
                pl.col("minute_of_day")
                .cast(pl.Float64)
                * (
                    2.0
                    * pi
                    / 1440.0
                )
            )
            .sin()
            .cast(pl.Float32)
            .alias(
                "time_of_day_sin"
            ),

            (
                pl.col("minute_of_day")
                .cast(pl.Float64)
                * (
                    2.0
                    * pi
                    / 1440.0
                )
            )
            .cos()
            .cast(pl.Float32)
            .alias(
                "time_of_day_cos"
            ),

            (
                (
                    pl.col("day_of_week")
                    .cast(pl.Float64)
                    - 1.0
                )
                * (
                    2.0
                    * pi
                    / 7.0
                )
            )
            .sin()
            .cast(pl.Float32)
            .alias(
                "day_of_week_sin"
            ),

            (
                (
                    pl.col("day_of_week")
                    .cast(pl.Float64)
                    - 1.0
                )
                * (
                    2.0
                    * pi
                    / 7.0
                )
            )
            .cos()
            .cast(pl.Float32)
            .alias(
                "day_of_week_cos"
            ),

            pl.col(
                settings.target_column
            )
            .cast(pl.Int32)
            .alias(
                "target_pickup_count"
            ),

            pl.when(
                pl.col("interval_start")
                < train_end
            )
            .then(
                pl.lit("train")
            )
            .when(
                pl.col("interval_start")
                < validation_end
            )
            .then(
                pl.lit("validation")
            )
            .when(
                pl.col("interval_start")
                < test_end
            )
            .then(
                pl.lit("test")
            )
            .otherwise(
                pl.lit(
                    None,
                    dtype=pl.String,
                )
            )
            .alias(
                "split"
            ),
        )
    )

    history_complete = pl.all_horizontal(
        [
            pl.col(column).is_not_null()
            for column in (
                settings.history_feature_columns
            )
        ]
    )

    return (
        frame.filter(
            history_complete
            & pl.col("split").is_not_null()
        )
        .select(
            *BASE_OUTPUT_COLUMNS,
            *settings.history_feature_columns,
        )
        .sort(
            "interval_start",
            "zone_id",
        )
    )


def collect_output_summary(
    output_path: Path,
) -> dict[str, Any]:
    """Collect output quality and target statistics."""
    output = pl.scan_parquet(
        output_path
    )

    schema = output.collect_schema()

    row = (
        output.select(
            pl.len().alias(
                "row_count"
            ),
            pl.col("interval_start")
            .n_unique()
            .alias(
                "interval_count"
            ),
            pl.col("zone_id")
            .n_unique()
            .alias(
                "zone_count"
            ),
            (
                pl.col("split")
                == "train"
            )
            .sum()
            .alias(
                "train_rows"
            ),
            (
                pl.col("split")
                == "validation"
            )
            .sum()
            .alias(
                "validation_rows"
            ),
            (
                pl.col("split")
                == "test"
            )
            .sum()
            .alias(
                "test_rows"
            ),
            pl.col("target_pickup_count")
            .sum()
            .alias(
                "target_sum"
            ),
            pl.col("target_pickup_count")
            .min()
            .alias(
                "target_minimum"
            ),
            pl.col("target_pickup_count")
            .max()
            .alias(
                "target_maximum"
            ),
            pl.col("target_pickup_count")
            .mean()
            .alias(
                "target_mean"
            ),
            pl.col("target_pickup_count")
            .quantile(0.50)
            .alias(
                "target_p50"
            ),
            pl.col("target_pickup_count")
            .quantile(0.95)
            .alias(
                "target_p95"
            ),
            pl.col("target_pickup_count")
            .quantile(0.99)
            .alias(
                "target_p99"
            ),
            pl.col("interval_start")
            .min()
            .alias(
                "first_interval"
            ),
            pl.col("interval_start")
            .max()
            .alias(
                "last_interval"
            ),
        )
        .collect()
        .row(
            0,
            named=True,
        )
    )

    row_count = int(
        row["row_count"]
    )

    if row_count == 0:
        raise ForecastFeatureError(
            "Forecast feature dataset is empty."
        )

    duplicate_key_count = int(
        output.group_by(
            "interval_start",
            "zone_id",
        )
        .agg(
            pl.len().alias(
                "key_count"
            )
        )
        .filter(
            pl.col("key_count") > 1
        )
        .select(
            pl.len().alias(
                "duplicate_key_count"
            )
        )
        .collect()
        .item()
    )

    null_row = (
        output.select(
            [
                pl.col(column)
                .null_count()
                .alias(column)
                for column in schema.names()
            ]
        )
        .collect()
        .row(0)
    )

    null_value_count = sum(
        int(value)
        for value in null_row
    )

    first_interval = row[
        "first_interval"
    ]

    last_interval = row[
        "last_interval"
    ]

    return {
        "row_count": row_count,
        "interval_count": int(
            row["interval_count"]
        ),
        "zone_count": int(
            row["zone_count"]
        ),
        "train_rows": int(
            row["train_rows"]
        ),
        "validation_rows": int(
            row["validation_rows"]
        ),
        "test_rows": int(
            row["test_rows"]
        ),
        "target_sum": int(
            row["target_sum"]
        ),
        "target_minimum": int(
            row["target_minimum"]
        ),
        "target_maximum": int(
            row["target_maximum"]
        ),
        "target_mean": float(
            row["target_mean"]
        ),
        "target_p50": float(
            row["target_p50"]
        ),
        "target_p95": float(
            row["target_p95"]
        ),
        "target_p99": float(
            row["target_p99"]
        ),
        "first_interval": (
            first_interval.isoformat()
        ),
        "last_interval": (
            last_interval.isoformat()
        ),
        "duplicate_key_count": (
            duplicate_key_count
        ),
        "null_value_count": (
            null_value_count
        ),
    }


def enforce_output_invariants(
    *,
    settings: ForecastFeatureSettings,
    input_summary: dict[str, int],
    output_summary: dict[str, Any],
) -> int:
    """Fail if the feature dataset violates its contract."""
    expected_interval_count = (
        input_summary["interval_count"]
        - settings.maximum_history_steps
    )

    expected_row_count = (
        expected_interval_count
        * input_summary["zone_count"]
    )

    if (
        output_summary["row_count"]
        != expected_row_count
    ):
        raise ForecastFeatureError(
            "Unexpected feature-dataset row count: "
            f"expected {expected_row_count:,}, "
            f"received "
            f"{output_summary['row_count']:,}"
        )

    if (
        output_summary["interval_count"]
        != expected_interval_count
    ):
        raise ForecastFeatureError(
            "Unexpected feature interval count: "
            f"expected {expected_interval_count:,}, "
            f"received "
            f"{output_summary['interval_count']:,}"
        )

    if (
        output_summary["zone_count"]
        != input_summary["zone_count"]
    ):
        raise ForecastFeatureError(
            "Feature dataset lost one or more zones."
        )

    if output_summary[
        "duplicate_key_count"
    ]:
        raise ForecastFeatureError(
            "Feature dataset contains duplicate keys."
        )

    if output_summary[
        "null_value_count"
    ]:
        raise ForecastFeatureError(
            "Feature dataset contains null values."
        )

    split_total = (
        output_summary["train_rows"]
        + output_summary[
            "validation_rows"
        ]
        + output_summary["test_rows"]
    )

    if (
        split_total
        != output_summary["row_count"]
    ):
        raise ForecastFeatureError(
            "Split counts do not cover every output row."
        )

    for split_column in (
        "train_rows",
        "validation_rows",
        "test_rows",
    ):
        if output_summary[split_column] <= 0:
            raise ForecastFeatureError(
                "Train, validation, and test splits "
                "must all contain records."
            )

    if (
        output_summary[
            "target_minimum"
        ]
        < 0
    ):
        raise ForecastFeatureError(
            "Feature dataset contains a negative target."
        )

    return expected_row_count


def write_feature_report(
    *,
    settings: ForecastFeatureSettings,
    input_summary: dict[str, int],
    output_summary: dict[str, Any],
    expected_row_count: int,
) -> dict[str, Any]:
    """Write the feature-generation report."""
    report: dict[str, Any] = {
        "feature_pipeline_version": "0.1.0",
        "generated_at": datetime.now(
            UTC
        ).isoformat(),
        "effective_config": (
            settings.as_report_dict()
        ),
        "input": {
            "path": str(
                settings.input_path
            ),
            "size_bytes": (
                settings.input_path
                .stat()
                .st_size
            ),
            **input_summary,
        },
        "output": {
            "path": str(
                settings.output_path
            ),
            "size_bytes": (
                settings.output_path
                .stat()
                .st_size
            ),
            "expected_row_count": (
                expected_row_count
            ),
            "row_count": (
                output_summary[
                    "row_count"
                ]
            ),
            "interval_count": (
                output_summary[
                    "interval_count"
                ]
            ),
            "zone_count": (
                output_summary[
                    "zone_count"
                ]
            ),
            "first_interval": (
                output_summary[
                    "first_interval"
                ]
            ),
            "last_interval": (
                output_summary[
                    "last_interval"
                ]
            ),
        },
        "splits": {
            "train_rows": (
                output_summary[
                    "train_rows"
                ]
            ),
            "validation_rows": (
                output_summary[
                    "validation_rows"
                ]
            ),
            "test_rows": (
                output_summary[
                    "test_rows"
                ]
            ),
        },
        "target_distribution": {
            "sum": (
                output_summary[
                    "target_sum"
                ]
            ),
            "minimum": (
                output_summary[
                    "target_minimum"
                ]
            ),
            "maximum": (
                output_summary[
                    "target_maximum"
                ]
            ),
            "mean": round(
                output_summary[
                    "target_mean"
                ],
                6,
            ),
            "p50": (
                output_summary[
                    "target_p50"
                ]
            ),
            "p95": (
                output_summary[
                    "target_p95"
                ]
            ),
            "p99": (
                output_summary[
                    "target_p99"
                ]
            ),
        },
        "feature_schema": {
            "target_column": (
                "target_pickup_count"
            ),
            "feature_count": len(
                settings.model_feature_columns
            ),
            "feature_columns": list(
                settings.model_feature_columns
            ),
            "history_feature_columns": list(
                settings.history_feature_columns
            ),
        },
        "invariants": {
            "history_uses_past_only": True,
            "row_count_matches_expected": True,
            "duplicate_key_count": (
                output_summary[
                    "duplicate_key_count"
                ]
            ),
            "null_value_count": (
                output_summary[
                    "null_value_count"
                ]
            ),
            "minimum_target": (
                output_summary[
                    "target_minimum"
                ]
            ),
        },
    }

    settings.report_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    settings.report_path.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    return report


def build_forecast_features(
    config_path: Path,
) -> dict[str, Any]:
    """Build the configured forecasting feature dataset."""
    settings = ForecastFeatureSettings.from_yaml(
        config_path
    )

    input_summary = validate_input_panel(
        settings
    )

    feature_frame = build_feature_frame(
        settings
    )

    settings.output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_output = (
        settings.output_path.with_name(
            f"{settings.output_path.name}.part"
        )
    )

    temporary_output.unlink(
        missing_ok=True
    )

    try:
        feature_frame.sink_parquet(
            temporary_output,
            compression=settings.compression,
            statistics=True,
        )

        temporary_output.replace(
            settings.output_path
        )

    except Exception as error:
        temporary_output.unlink(
            missing_ok=True
        )

        raise ForecastFeatureError(
            "Could not write forecast features: "
            f"{error}"
        ) from error

    output_summary = collect_output_summary(
        settings.output_path
    )

    expected_row_count = (
        enforce_output_invariants(
            settings=settings,
            input_summary=input_summary,
            output_summary=output_summary,
        )
    )

    return write_feature_report(
        settings=settings,
        input_summary=input_summary,
        output_summary=output_summary,
        expected_row_count=expected_row_count,
    )