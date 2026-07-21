from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl
import pyarrow.parquet as pq


REQUIRED_COLUMNS = {
    "hvfhs_license_num",
    "request_datetime",
    "pickup_datetime",
    "dropoff_datetime",
    "PULocationID",
    "DOLocationID",
    "trip_miles",
    "trip_time",
}


class DataAuditError(RuntimeError):
    """Raised when a trip-data audit cannot be completed safely."""


def get_month_bounds(
    year: int,
    month: int,
) -> tuple[datetime, datetime]:
    """Return inclusive month start and exclusive next-month start."""
    if not 1 <= month <= 12:
        raise ValueError(f"Invalid month: {month}")

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


def serialize_value(value: Any) -> Any:
    """Convert audit values to JSON-compatible Python values."""
    if isinstance(value, datetime):
        return value.isoformat()

    if isinstance(value, Path):
        return str(value)

    return value


def build_null_report(
    summary: dict[str, Any],
    columns: list[str],
    row_count: int,
) -> dict[str, dict[str, float | int]]:
    """Build null counts and percentages for every source column."""
    report: dict[str, dict[str, float | int]] = {}

    for column in columns:
        null_count = int(summary[f"null__{column}"])

        null_percentage = (
            null_count / row_count * 100
            if row_count > 0
            else 0.0
        )

        report[column] = {
            "count": null_count,
            "percentage": round(
                null_percentage,
                6,
            ),
        }

    return report


def audit_trip_data(
    input_path: Path,
    output_path: Path,
    *,
    expected_year: int,
    expected_month: int,
) -> dict[str, Any]:
    """Audit one monthly NYC TLC HVFHV Parquet file."""
    if not input_path.exists():
        raise DataAuditError(
            f"Input file does not exist: {input_path}"
        )

    if not input_path.is_file():
        raise DataAuditError(
            f"Input path is not a file: {input_path}"
        )

    if input_path.stat().st_size == 0:
        raise DataAuditError(
            f"Input file is empty: {input_path}"
        )

    parquet_file = pq.ParquetFile(input_path)
    frame = pl.scan_parquet(input_path)

    schema = frame.collect_schema()
    columns = schema.names()

    missing_columns = sorted(
        REQUIRED_COLUMNS.difference(columns)
    )

    if missing_columns:
        missing_text = ", ".join(missing_columns)
        raise DataAuditError(
            f"Required columns are missing: {missing_text}"
        )

    month_start, next_month_start = get_month_bounds(
        year=expected_year,
        month=expected_month,
    )

    request_to_pickup_seconds = (
        pl.col("pickup_datetime")
        - pl.col("request_datetime")
    ).dt.total_seconds()

    derived_trip_duration_seconds = (
        pl.col("dropoff_datetime")
        - pl.col("pickup_datetime")
    ).dt.total_seconds()

    null_expressions = [
        pl.col(column)
        .null_count()
        .alias(f"null__{column}")
        for column in columns
    ]

    summary_expressions: list[pl.Expr] = [
        pl.len().alias("row_count"),
        pl.col("request_datetime")
        .min()
        .alias("min_request_datetime"),
        pl.col("request_datetime")
        .max()
        .alias("max_request_datetime"),
        pl.col("pickup_datetime")
        .min()
        .alias("min_pickup_datetime"),
        pl.col("pickup_datetime")
        .max()
        .alias("max_pickup_datetime"),
        pl.col("dropoff_datetime")
        .min()
        .alias("min_dropoff_datetime"),
        pl.col("dropoff_datetime")
        .max()
        .alias("max_dropoff_datetime"),
        pl.col("PULocationID")
        .min()
        .alias("min_pickup_zone_id"),
        pl.col("PULocationID")
        .max()
        .alias("max_pickup_zone_id"),
        pl.col("PULocationID")
        .n_unique()
        .alias("unique_pickup_zone_count"),
        pl.col("DOLocationID")
        .min()
        .alias("min_dropoff_zone_id"),
        pl.col("DOLocationID")
        .max()
        .alias("max_dropoff_zone_id"),
        pl.col("DOLocationID")
        .n_unique()
        .alias("unique_dropoff_zone_count"),
        (
            pl.col("request_datetime") < month_start
        )
        .sum()
        .alias("request_before_month_count"),
        (
            pl.col("request_datetime") >= next_month_start
        )
        .sum()
        .alias("request_after_month_count"),
        (
            pl.col("pickup_datetime") < month_start
        )
        .sum()
        .alias("pickup_before_month_count"),
        (
            pl.col("pickup_datetime") >= next_month_start
        )
        .sum()
        .alias("pickup_after_month_count"),
        (
            pl.col("dropoff_datetime") < month_start
        )
        .sum()
        .alias("dropoff_before_month_count"),
        (
            pl.col("dropoff_datetime") >= next_month_start
        )
        .sum()
        .alias("dropoff_after_month_count"),
        (
            pl.col("request_datetime")
            > pl.col("pickup_datetime")
        )
        .sum()
        .alias("request_after_pickup_count"),
        (
            pl.col("pickup_datetime")
            >= pl.col("dropoff_datetime")
        )
        .sum()
        .alias("non_positive_derived_trip_duration_count"),
        (
            pl.col("trip_time") <= 0
        )
        .sum()
        .alias("non_positive_trip_time_count"),
        (
            pl.col("trip_miles") < 0
        )
        .sum()
        .alias("negative_trip_miles_count"),
        request_to_pickup_seconds
        .min()
        .alias("request_to_pickup_min_seconds"),
        request_to_pickup_seconds
        .quantile(0.01)
        .alias("request_to_pickup_p01_seconds"),
        request_to_pickup_seconds
        .quantile(0.50)
        .alias("request_to_pickup_p50_seconds"),
        request_to_pickup_seconds
        .quantile(0.95)
        .alias("request_to_pickup_p95_seconds"),
        request_to_pickup_seconds
        .quantile(0.99)
        .alias("request_to_pickup_p99_seconds"),
        request_to_pickup_seconds
        .max()
        .alias("request_to_pickup_max_seconds"),
        derived_trip_duration_seconds
        .min()
        .alias("trip_duration_min_seconds"),
        derived_trip_duration_seconds
        .quantile(0.01)
        .alias("trip_duration_p01_seconds"),
        derived_trip_duration_seconds
        .quantile(0.50)
        .alias("trip_duration_p50_seconds"),
        derived_trip_duration_seconds
        .quantile(0.95)
        .alias("trip_duration_p95_seconds"),
        derived_trip_duration_seconds
        .quantile(0.99)
        .alias("trip_duration_p99_seconds"),
        derived_trip_duration_seconds
        .max()
        .alias("trip_duration_max_seconds"),
        (
            pl.col("trip_time")
            - derived_trip_duration_seconds
        )
        .abs()
        .quantile(0.50)
        .alias("trip_time_difference_p50_seconds"),
        (
            pl.col("trip_time")
            - derived_trip_duration_seconds
        )
        .abs()
        .quantile(0.99)
        .alias("trip_time_difference_p99_seconds"),
        (
            (
                pl.col("trip_time")
                - derived_trip_duration_seconds
            )
            .abs()
            > 60
        )
        .sum()
        .alias("trip_time_difference_over_60_seconds_count"),
        pl.col("trip_miles")
        .min()
        .alias("trip_miles_min"),
        pl.col("trip_miles")
        .quantile(0.50)
        .alias("trip_miles_p50"),
        pl.col("trip_miles")
        .quantile(0.95)
        .alias("trip_miles_p95"),
        pl.col("trip_miles")
        .quantile(0.99)
        .alias("trip_miles_p99"),
        pl.col("trip_miles")
        .max()
        .alias("trip_miles_max"),
    ]

    summary_frame = frame.select(
        summary_expressions + null_expressions
    ).collect()

    summary = summary_frame.row(
        0,
        named=True,
    )

    row_count = int(summary["row_count"])

    provider_counts = (
        frame.group_by("hvfhs_license_num")
        .agg(
            pl.len().alias("row_count")
        )
        .sort(
            "row_count",
            descending=True,
        )
        .collect()
        .to_dicts()
    )

    schema_report = {
        column: str(schema[column])
        for column in columns
    }

    report: dict[str, Any] = {
        "audit_version": "0.1.0",
        "dataset": {
            "name": "NYC TLC HVFHV Trip Records",
            "expected_year": expected_year,
            "expected_month": expected_month,
            "expected_month_start": month_start.isoformat(),
            "expected_next_month_start": (
                next_month_start.isoformat()
            ),
        },
        "file": {
            "path": str(input_path),
            "size_bytes": input_path.stat().st_size,
            "parquet_rows": (
                parquet_file.metadata.num_rows
            ),
            "parquet_columns": (
                parquet_file.metadata.num_columns
            ),
            "parquet_row_groups": (
                parquet_file.num_row_groups
            ),
        },
        "schema": schema_report,
        "required_columns": sorted(REQUIRED_COLUMNS),
        "missing_required_columns": missing_columns,
        "row_count": row_count,
        "nulls": build_null_report(
            summary=summary,
            columns=columns,
            row_count=row_count,
        ),
        "temporal_bounds": {
            "request_datetime": {
                "min": serialize_value(
                    summary["min_request_datetime"]
                ),
                "max": serialize_value(
                    summary["max_request_datetime"]
                ),
            },
            "pickup_datetime": {
                "min": serialize_value(
                    summary["min_pickup_datetime"]
                ),
                "max": serialize_value(
                    summary["max_pickup_datetime"]
                ),
            },
            "dropoff_datetime": {
                "min": serialize_value(
                    summary["min_dropoff_datetime"]
                ),
                "max": serialize_value(
                    summary["max_dropoff_datetime"]
                ),
            },
        },
        "month_boundary_counts": {
            "request_before_month": int(
                summary["request_before_month_count"]
            ),
            "request_after_month": int(
                summary["request_after_month_count"]
            ),
            "pickup_before_month": int(
                summary["pickup_before_month_count"]
            ),
            "pickup_after_month": int(
                summary["pickup_after_month_count"]
            ),
            "dropoff_before_month": int(
                summary["dropoff_before_month_count"]
            ),
            "dropoff_after_month": int(
                summary["dropoff_after_month_count"]
            ),
        },
        "zone_statistics": {
            "pickup": {
                "minimum_id": int(
                    summary["min_pickup_zone_id"]
                ),
                "maximum_id": int(
                    summary["max_pickup_zone_id"]
                ),
                "unique_count": int(
                    summary["unique_pickup_zone_count"]
                ),
            },
            "dropoff": {
                "minimum_id": int(
                    summary["min_dropoff_zone_id"]
                ),
                "maximum_id": int(
                    summary["max_dropoff_zone_id"]
                ),
                "unique_count": int(
                    summary["unique_dropoff_zone_count"]
                ),
            },
        },
        "quality_checks": {
            "request_after_pickup": int(
                summary["request_after_pickup_count"]
            ),
            "non_positive_derived_trip_duration": int(
                summary[
                    "non_positive_derived_trip_duration_count"
                ]
            ),
            "non_positive_trip_time": int(
                summary["non_positive_trip_time_count"]
            ),
            "negative_trip_miles": int(
                summary["negative_trip_miles_count"]
            ),
            "trip_time_difference_over_60_seconds": int(
                summary[
                    "trip_time_difference_over_60_seconds_count"
                ]
            ),
        },
        "request_to_pickup_seconds": {
            "minimum": serialize_value(
                summary["request_to_pickup_min_seconds"]
            ),
            "p01": serialize_value(
                summary["request_to_pickup_p01_seconds"]
            ),
            "p50": serialize_value(
                summary["request_to_pickup_p50_seconds"]
            ),
            "p95": serialize_value(
                summary["request_to_pickup_p95_seconds"]
            ),
            "p99": serialize_value(
                summary["request_to_pickup_p99_seconds"]
            ),
            "maximum": serialize_value(
                summary["request_to_pickup_max_seconds"]
            ),
        },
        "trip_duration_seconds": {
            "minimum": serialize_value(
                summary["trip_duration_min_seconds"]
            ),
            "p01": serialize_value(
                summary["trip_duration_p01_seconds"]
            ),
            "p50": serialize_value(
                summary["trip_duration_p50_seconds"]
            ),
            "p95": serialize_value(
                summary["trip_duration_p95_seconds"]
            ),
            "p99": serialize_value(
                summary["trip_duration_p99_seconds"]
            ),
            "maximum": serialize_value(
                summary["trip_duration_max_seconds"]
            ),
        },
        "trip_time_consistency": {
            "absolute_difference_p50_seconds": serialize_value(
                summary["trip_time_difference_p50_seconds"]
            ),
            "absolute_difference_p99_seconds": serialize_value(
                summary["trip_time_difference_p99_seconds"]
            ),
        },
        "trip_miles": {
            "minimum": serialize_value(
                summary["trip_miles_min"]
            ),
            "p50": serialize_value(
                summary["trip_miles_p50"]
            ),
            "p95": serialize_value(
                summary["trip_miles_p95"]
            ),
            "p99": serialize_value(
                summary["trip_miles_p99"]
            ),
            "maximum": serialize_value(
                summary["trip_miles_max"]
            ),
        },
        "provider_counts": provider_counts,
    }

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    return report