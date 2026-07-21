from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import polars as pl
import yaml


ZONE_LOOKUP_REQUIRED_COLUMNS = {
    "LocationID",
    "Borough",
    "Zone",
    "service_zone",
}

INPUT_REQUIRED_COLUMNS = {
    "pickup_datetime",
    "PULocationID",
    "eligible_for_demand_forecasting",
}

OUTPUT_REQUIRED_COLUMNS = (
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

INTERVAL_PATTERN = re.compile(
    r"^[1-9][0-9]*(?:ns|us|ms|s|m|h|d)$"
)


class DemandAggregationError(RuntimeError):
    """Raised when zone-demand aggregation cannot be completed."""


@dataclass(frozen=True)
class DemandAggregationSettings:
    """Configuration for zone-demand aggregation."""

    expected_year: int
    expected_month: int
    timezone: str

    input_path: Path
    zone_lookup_path: Path
    output_path: Path
    report_path: Path

    interval: str
    core_boroughs: tuple[str, ...]

    compression: str

    @classmethod
    def from_yaml(
        cls,
        config_path: Path,
    ) -> DemandAggregationSettings:
        """Load aggregation settings from YAML."""
        if not config_path.exists():
            raise DemandAggregationError(
                f"Aggregation config does not exist: {config_path}"
            )

        try:
            raw_config = yaml.safe_load(
                config_path.read_text(
                    encoding="utf-8"
                )
            )
        except (OSError, yaml.YAMLError) as error:
            raise DemandAggregationError(
                f"Could not read aggregation config: {error}"
            ) from error

        if not isinstance(raw_config, dict):
            raise DemandAggregationError(
                "Aggregation config must contain a YAML mapping."
            )

        try:
            dataset_config = raw_config["dataset"]
            paths_config = raw_config["paths"]
            aggregation_config = raw_config[
                "aggregation"
            ]
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
                zone_lookup_path=Path(
                    paths_config["zone_lookup"]
                ),
                output_path=Path(
                    paths_config["output"]
                ),
                report_path=Path(
                    paths_config["report"]
                ),
                interval=str(
                    aggregation_config["interval"]
                ),
                core_boroughs=tuple(
                    str(borough)
                    for borough in aggregation_config[
                        "core_boroughs"
                    ]
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
            raise DemandAggregationError(
                f"Invalid aggregation configuration: {error}"
            ) from error

        settings.validate()

        return settings

    def validate(self) -> None:
        """Validate configuration values."""
        if self.expected_year < 2009:
            raise DemandAggregationError(
                f"Invalid expected year: {self.expected_year}"
            )

        if not 1 <= self.expected_month <= 12:
            raise DemandAggregationError(
                f"Invalid expected month: {self.expected_month}"
            )

        try:
            ZoneInfo(self.timezone)
        except ZoneInfoNotFoundError as error:
            raise DemandAggregationError(
                f"Unknown timezone: {self.timezone}"
            ) from error

        if not INTERVAL_PATTERN.fullmatch(
            self.interval
        ):
            raise DemandAggregationError(
                "Aggregation interval must be a positive "
                "Polars duration such as '15m' or '1h'."
            )

        if not self.core_boroughs:
            raise DemandAggregationError(
                "At least one core borough must be configured."
            )

        if not self.compression:
            raise DemandAggregationError(
                "Output compression must not be empty."
            )

    def as_report_dict(self) -> dict[str, Any]:
        """Return the effective JSON-compatible configuration."""
        return {
            "dataset": {
                "expected_year": self.expected_year,
                "expected_month": self.expected_month,
                "timezone": self.timezone,
            },
            "paths": {
                "input": str(self.input_path),
                "zone_lookup": str(
                    self.zone_lookup_path
                ),
                "output": str(self.output_path),
                "report": str(self.report_path),
            },
            "aggregation": {
                "interval": self.interval,
                "core_boroughs": list(
                    self.core_boroughs
                ),
            },
            "output": {
                "compression": self.compression,
            },
        }


def get_month_bounds(
    year: int,
    month: int,
) -> tuple[datetime, datetime]:
    """Return naive monthly boundaries for range generation."""
    month_start = datetime(
        year=year,
        month=month,
        day=1,
    )

    if month == 12:
        next_month_start = datetime(
            year=year + 1,
            month=1,
            day=1,
        )
    else:
        next_month_start = datetime(
            year=year,
            month=month + 1,
            day=1,
        )

    return month_start, next_month_start


def load_core_zones(
    settings: DemandAggregationSettings,
) -> pl.DataFrame:
    """Load and validate core TLC operating zones."""
    if not settings.zone_lookup_path.exists():
        raise DemandAggregationError(
            "Taxi-zone lookup does not exist: "
            f"{settings.zone_lookup_path}"
        )

    if settings.zone_lookup_path.stat().st_size == 0:
        raise DemandAggregationError(
            "Taxi-zone lookup is empty: "
            f"{settings.zone_lookup_path}"
        )

    try:
        zones = pl.read_csv(
            settings.zone_lookup_path
        )
    except Exception as error:
        raise DemandAggregationError(
            f"Could not read taxi-zone lookup: {error}"
        ) from error

    missing_columns = sorted(
        ZONE_LOOKUP_REQUIRED_COLUMNS.difference(
            zones.columns
        )
    )

    if missing_columns:
        missing_text = ", ".join(
            missing_columns
        )

        raise DemandAggregationError(
            "Taxi-zone lookup is missing columns: "
            f"{missing_text}"
        )

    duplicate_location_ids = int(
        zones.select(
            pl.col("LocationID")
            .is_duplicated()
            .sum()
            .alias("duplicate_count")
        ).item()
    )

    if duplicate_location_ids:
        raise DemandAggregationError(
            "Taxi-zone lookup contains duplicate LocationIDs."
        )

    core_zones = (
        zones.filter(
            pl.col("Borough").is_in(
                list(settings.core_boroughs)
            )
        )
        .select(
            pl.col("LocationID")
            .cast(pl.Int32)
            .alias("zone_id"),
            pl.col("Borough").alias(
                "borough"
            ),
            pl.col("Zone").alias(
                "zone_name"
            ),
            pl.col("service_zone"),
        )
        .sort("zone_id")
    )

    if core_zones.height == 0:
        raise DemandAggregationError(
            "No core operating zones were found."
        )

    null_count = int(
        core_zones.null_count()
        .select(
            pl.sum_horizontal(
                pl.all()
            ).alias("null_count")
        )
        .item()
    )

    if null_count:
        raise DemandAggregationError(
            "Core zone lookup contains null values."
        )

    return core_zones


def build_time_index(
    settings: DemandAggregationSettings,
) -> pl.DataFrame:
    """Build every interval in the configured month."""
    month_start, next_month_start = get_month_bounds(
        year=settings.expected_year,
        month=settings.expected_month,
    )

    try:
        interval_start = pl.datetime_range(
            start=month_start,
            end=next_month_start,
            interval=settings.interval,
            closed="left",
            time_unit="ns",
            time_zone=settings.timezone,
            eager=True,
        )
    except Exception as error:
        raise DemandAggregationError(
            f"Could not generate time index: {error}"
        ) from error

    if interval_start.len() == 0:
        raise DemandAggregationError(
            "Generated time index is empty."
        )

    return (
        pl.DataFrame(
            {
                "interval_start": (
                    interval_start
                ),
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


def validate_input_schema(
    settings: DemandAggregationSettings,
) -> pl.Schema:
    """Validate the input Parquet schema."""
    if not settings.input_path.exists():
        raise DemandAggregationError(
            f"Validated input does not exist: {settings.input_path}"
        )

    if settings.input_path.stat().st_size == 0:
        raise DemandAggregationError(
            f"Validated input is empty: {settings.input_path}"
        )

    try:
        schema = pl.scan_parquet(
            settings.input_path
        ).collect_schema()
    except Exception as error:
        raise DemandAggregationError(
            f"Could not read validated input schema: {error}"
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

        raise DemandAggregationError(
            "Validated input is missing columns: "
            f"{missing_text}"
        )

    pickup_dtype = schema[
        "pickup_datetime"
    ]

    if settings.timezone not in str(
        pickup_dtype
    ):
        raise DemandAggregationError(
            "pickup_datetime does not use the configured "
            f"timezone {settings.timezone}: {pickup_dtype}"
        )

    return schema


def build_observed_demand(
    settings: DemandAggregationSettings,
) -> pl.LazyFrame:
    """Aggregate eligible pickups into zone-time buckets."""
    validate_input_schema(
        settings
    )

    return (
        pl.scan_parquet(
            settings.input_path
        )
        .filter(
            pl.col(
                "eligible_for_demand_forecasting"
            ).fill_null(False)
        )
        .select(
            pl.col("pickup_datetime")
            .dt.truncate(
                settings.interval
            )
            .alias("interval_start"),
            pl.col("PULocationID")
            .cast(pl.Int32)
            .alias("zone_id"),
        )
        .group_by(
            "interval_start",
            "zone_id",
        )
        .agg(
            pl.len()
            .cast(pl.Int32)
            .alias("pickup_count")
        )
    )


def build_dense_demand_panel(
    settings: DemandAggregationSettings,
    core_zones: pl.DataFrame,
    time_index: pl.DataFrame,
) -> pl.LazyFrame:
    """Build a complete zone-time grid and attach demand."""
    observed_demand = build_observed_demand(
        settings
    )

    dense_grid = (
        time_index.lazy()
        .join(
            core_zones.lazy(),
            how="cross",
        )
    )

    return (
        dense_grid.join(
            observed_demand,
            on=[
                "interval_start",
                "zone_id",
            ],
            how="left",
        )
        .with_columns(
            pl.col("pickup_count")
            .fill_null(0)
            .cast(pl.Int32)
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
            *OUTPUT_REQUIRED_COLUMNS
        )
        .sort(
            "interval_start",
            "zone_id",
        )
    )


def count_eligible_input_records(
    settings: DemandAggregationSettings,
) -> int:
    """Count validated records eligible for demand forecasting."""
    return int(
        pl.scan_parquet(
            settings.input_path
        )
        .filter(
            pl.col(
                "eligible_for_demand_forecasting"
            ).fill_null(False)
        )
        .select(
            pl.len().alias("record_count")
        )
        .collect()
        .item()
    )


def collect_output_summary(
    output_path: Path,
) -> dict[str, int | float]:
    """Collect summary metrics from the output panel."""
    output = pl.scan_parquet(
        output_path
    )

    summary_row = (
        output.select(
            pl.len().alias(
                "row_count"
            ),
            pl.col("interval_start")
            .n_unique()
            .alias("interval_count"),
            pl.col("zone_id")
            .n_unique()
            .alias("zone_count"),
            pl.col("pickup_count")
            .sum()
            .alias("total_pickups"),
            (
                pl.col("pickup_count") == 0
            )
            .sum()
            .alias("zero_demand_rows"),
            pl.col("pickup_count")
            .min()
            .alias("minimum_pickup_count"),
            pl.col("pickup_count")
            .max()
            .alias("maximum_pickup_count"),
            pl.col("pickup_count")
            .mean()
            .alias("mean_pickup_count"),
            pl.col("pickup_count")
            .quantile(0.50)
            .alias("p50_pickup_count"),
            pl.col("pickup_count")
            .quantile(0.95)
            .alias("p95_pickup_count"),
            pl.col("pickup_count")
            .quantile(0.99)
            .alias("p99_pickup_count"),
        )
        .collect()
        .row(
            0,
            named=True,
        )
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

    null_count_row = (
        output.select(
            [
                pl.col(column)
                .null_count()
                .alias(column)
                for column in OUTPUT_REQUIRED_COLUMNS
            ]
        )
        .collect()
        .row(0)
    )

    null_value_count = sum(
        int(value)
        for value in null_count_row
    )

    return {
        "row_count": int(
            summary_row["row_count"]
        ),
        "interval_count": int(
            summary_row["interval_count"]
        ),
        "zone_count": int(
            summary_row["zone_count"]
        ),
        "total_pickups": int(
            summary_row["total_pickups"]
        ),
        "zero_demand_rows": int(
            summary_row["zero_demand_rows"]
        ),
        "minimum_pickup_count": int(
            summary_row["minimum_pickup_count"]
        ),
        "maximum_pickup_count": int(
            summary_row["maximum_pickup_count"]
        ),
        "mean_pickup_count": float(
            summary_row["mean_pickup_count"]
        ),
        "p50_pickup_count": float(
            summary_row["p50_pickup_count"]
        ),
        "p95_pickup_count": float(
            summary_row["p95_pickup_count"]
        ),
        "p99_pickup_count": float(
            summary_row["p99_pickup_count"]
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
    summary: dict[str, int | float],
    expected_row_count: int,
    expected_interval_count: int,
    expected_zone_count: int,
    eligible_input_count: int,
) -> None:
    """Fail when the demand panel violates invariants."""
    if summary["row_count"] != expected_row_count:
        raise DemandAggregationError(
            "Unexpected demand-panel row count: "
            f"expected {expected_row_count:,}, "
            f"received {summary['row_count']:,}"
        )

    if (
        summary["interval_count"]
        != expected_interval_count
    ):
        raise DemandAggregationError(
            "Unexpected interval count: "
            f"expected {expected_interval_count:,}, "
            f"received {summary['interval_count']:,}"
        )

    if summary["zone_count"] != expected_zone_count:
        raise DemandAggregationError(
            "Unexpected zone count: "
            f"expected {expected_zone_count:,}, "
            f"received {summary['zone_count']:,}"
        )

    if summary["total_pickups"] != eligible_input_count:
        raise DemandAggregationError(
            "Demand aggregation did not preserve pickup count: "
            f"expected {eligible_input_count:,}, "
            f"received {summary['total_pickups']:,}"
        )

    if summary["duplicate_key_count"] != 0:
        raise DemandAggregationError(
            "Demand panel contains duplicate "
            "interval_start and zone_id keys."
        )

    if summary["null_value_count"] != 0:
        raise DemandAggregationError(
            "Demand panel contains null values."
        )

    if summary["minimum_pickup_count"] < 0:
        raise DemandAggregationError(
            "Demand panel contains negative pickup counts."
        )


def write_aggregation_report(
    *,
    settings: DemandAggregationSettings,
    summary: dict[str, int | float],
    eligible_input_count: int,
    expected_row_count: int,
) -> dict[str, Any]:
    """Write the demand aggregation report."""
    row_count = int(
        summary["row_count"]
    )

    zero_demand_rate = (
        int(summary["zero_demand_rows"])
        / row_count
        if row_count
        else 0.0
    )

    report: dict[str, Any] = {
        "aggregation_version": "0.1.0",
        "generated_at": datetime.now(
            UTC
        ).isoformat(),
        "effective_config": (
            settings.as_report_dict()
        ),
        "input": {
            "path": str(settings.input_path),
            "size_bytes": (
                settings.input_path.stat().st_size
            ),
            "eligible_record_count": (
                eligible_input_count
            ),
        },
        "output": {
            "path": str(settings.output_path),
            "size_bytes": (
                settings.output_path.stat().st_size
            ),
            "expected_row_count": (
                expected_row_count
            ),
            "row_count": row_count,
        },
        "panel": {
            "interval_count": int(
                summary["interval_count"]
            ),
            "zone_count": int(
                summary["zone_count"]
            ),
            "total_pickups": int(
                summary["total_pickups"]
            ),
            "zero_demand_rows": int(
                summary["zero_demand_rows"]
            ),
            "zero_demand_rate": round(
                zero_demand_rate,
                8,
            ),
        },
        "pickup_count_distribution": {
            "minimum": int(
                summary[
                    "minimum_pickup_count"
                ]
            ),
            "maximum": int(
                summary[
                    "maximum_pickup_count"
                ]
            ),
            "mean": round(
                float(
                    summary[
                        "mean_pickup_count"
                    ]
                ),
                6,
            ),
            "p50": float(
                summary[
                    "p50_pickup_count"
                ]
            ),
            "p95": float(
                summary[
                    "p95_pickup_count"
                ]
            ),
            "p99": float(
                summary[
                    "p99_pickup_count"
                ]
            ),
        },
        "invariants": {
            "row_count_matches_dense_grid": True,
            "pickup_count_is_conserved": True,
            "duplicate_key_count": int(
                summary[
                    "duplicate_key_count"
                ]
            ),
            "null_value_count": int(
                summary[
                    "null_value_count"
                ]
            ),
            "minimum_pickup_count": int(
                summary[
                    "minimum_pickup_count"
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


def build_zone_demand(
    config_path: Path,
) -> dict[str, Any]:
    """Build the configured dense zone-demand panel."""
    settings = DemandAggregationSettings.from_yaml(
        config_path
    )

    core_zones = load_core_zones(
        settings
    )

    time_index = build_time_index(
        settings
    )

    expected_zone_count = (
        core_zones.height
    )
    expected_interval_count = (
        time_index.height
    )
    expected_row_count = (
        expected_zone_count
        * expected_interval_count
    )

    demand_panel = build_dense_demand_panel(
        settings=settings,
        core_zones=core_zones,
        time_index=time_index,
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
        demand_panel.sink_parquet(
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

        raise DemandAggregationError(
            f"Could not write demand panel: {error}"
        ) from error

    eligible_input_count = (
        count_eligible_input_records(
            settings
        )
    )

    summary = collect_output_summary(
        settings.output_path
    )

    enforce_output_invariants(
        summary=summary,
        expected_row_count=expected_row_count,
        expected_interval_count=(
            expected_interval_count
        ),
        expected_zone_count=(
            expected_zone_count
        ),
        eligible_input_count=(
            eligible_input_count
        ),
    )

    return write_aggregation_report(
        settings=settings,
        summary=summary,
        eligible_input_count=(
            eligible_input_count
        ),
        expected_row_count=(
            expected_row_count
        ),
    )