from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import reduce
from operator import or_
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

REJECTION_FLAG_COLUMNS = (
    "flag_missing_required_value",
    "flag_invalid_pickup_zone",
    "flag_invalid_dropoff_zone",
    "flag_pickup_outside_expected_month",
    "flag_non_positive_trip_duration",
    "flag_non_positive_trip_time",
)

RETAINED_FLAG_COLUMNS = (
    "flag_request_after_pickup",
    "flag_long_trip_duration",
    "flag_pickup_non_core_zone",
    "flag_dropoff_non_core_zone",
)

ALL_FLAG_COLUMNS = (
    *REJECTION_FLAG_COLUMNS,
    *RETAINED_FLAG_COLUMNS,
)

ELIGIBILITY_COLUMNS = (
    "eligible_for_demand_forecasting",
    "eligible_for_od_modeling",
    "eligible_for_wait_time_analysis",
)


class ValidationError(RuntimeError):
    """Raised when TLC trip-data validation cannot be completed."""


@dataclass(frozen=True)
class ValidationSettings:
    """Configuration required by the validation pipeline."""

    expected_year: int
    expected_month: int
    timezone: str

    input_path: Path
    zone_lookup_path: Path
    output_path: Path
    report_path: Path

    required_columns: tuple[str, ...]
    core_boroughs: tuple[str, ...]
    long_trip_duration_seconds: int

    compression: str
    include_rejected_records: bool

    @classmethod
    def from_yaml(
        cls,
        config_path: Path,
    ) -> ValidationSettings:
        """Load validation settings from a YAML file."""
        if not config_path.exists():
            raise ValidationError(
                f"Validation config does not exist: {config_path}"
            )

        try:
            raw_config = yaml.safe_load(
                config_path.read_text(encoding="utf-8")
            )
        except (OSError, yaml.YAMLError) as error:
            raise ValidationError(
                f"Could not read validation config: {error}"
            ) from error

        if not isinstance(raw_config, dict):
            raise ValidationError(
                "Validation config must contain a YAML mapping."
            )

        try:
            dataset_config = raw_config["dataset"]
            paths_config = raw_config["paths"]
            validation_config = raw_config["validation"]
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
                required_columns=tuple(
                    str(column)
                    for column in validation_config[
                        "required_columns"
                    ]
                ),
                core_boroughs=tuple(
                    str(borough)
                    for borough in validation_config[
                        "core_boroughs"
                    ]
                ),
                long_trip_duration_seconds=int(
                    validation_config[
                        "long_trip_duration_seconds"
                    ]
                ),
                compression=str(
                    output_config["compression"]
                ),
                include_rejected_records=bool(
                    output_config[
                        "include_rejected_records"
                    ]
                ),
            )

        except (
            KeyError,
            TypeError,
            ValueError,
        ) as error:
            raise ValidationError(
                f"Invalid validation configuration: {error}"
            ) from error

        settings.validate()

        return settings

    def validate(self) -> None:
        """Validate configuration values."""
        if not 1 <= self.expected_month <= 12:
            raise ValidationError(
                f"Invalid expected month: {self.expected_month}"
            )

        if self.expected_year < 2009:
            raise ValidationError(
                f"Invalid expected year: {self.expected_year}"
            )

        if not self.required_columns:
            raise ValidationError(
                "At least one required column must be configured."
            )

        if not self.core_boroughs:
            raise ValidationError(
                "At least one core borough must be configured."
            )

        if self.long_trip_duration_seconds <= 0:
            raise ValidationError(
                "Long-trip threshold must be positive."
            )

        try:
            ZoneInfo(self.timezone)
        except ZoneInfoNotFoundError as error:
            raise ValidationError(
                f"Unknown timezone: {self.timezone}"
            ) from error

    def as_report_dict(self) -> dict[str, Any]:
        """Return JSON-compatible effective configuration."""
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
            "validation": {
                "required_columns": list(
                    self.required_columns
                ),
                "core_boroughs": list(
                    self.core_boroughs
                ),
                "long_trip_duration_seconds": (
                    self.long_trip_duration_seconds
                ),
            },
            "output": {
                "compression": self.compression,
                "include_rejected_records": (
                    self.include_rejected_records
                ),
            },
        }


def get_month_bounds(
    year: int,
    month: int,
    timezone: str,
) -> tuple[datetime, datetime]:
    """Return timezone-aware monthly boundaries."""
    try:
        timezone_info = ZoneInfo(timezone)
    except ZoneInfoNotFoundError as error:
        raise ValidationError(
            f"Unknown timezone: {timezone}"
        ) from error

    month_start = datetime(
        year=year,
        month=month,
        day=1,
        tzinfo=timezone_info,
    )

    if month == 12:
        next_month_start = datetime(
            year=year + 1,
            month=1,
            day=1,
            tzinfo=timezone_info,
        )
    else:
        next_month_start = datetime(
            year=year,
            month=month + 1,
            day=1,
            tzinfo=timezone_info,
        )

    return month_start, next_month_start


def combine_with_or(
    expressions: list[pl.Expr],
) -> pl.Expr:
    """Combine boolean expressions with logical OR."""
    if not expressions:
        raise ValidationError(
            "Cannot combine an empty expression list."
        )

    return reduce(or_, expressions)


def load_zone_identifiers(
    settings: ValidationSettings,
) -> tuple[list[int], list[int]]:
    """Load all valid and core operating-zone identifiers."""
    if not settings.zone_lookup_path.exists():
        raise ValidationError(
            "Taxi-zone lookup does not exist: "
            f"{settings.zone_lookup_path}"
        )

    try:
        zones = pl.read_csv(
            settings.zone_lookup_path
        )
    except Exception as error:
        raise ValidationError(
            f"Could not read taxi-zone lookup: {error}"
        ) from error

    missing_columns = sorted(
        ZONE_LOOKUP_REQUIRED_COLUMNS.difference(
            zones.columns
        )
    )

    if missing_columns:
        missing_text = ", ".join(missing_columns)

        raise ValidationError(
            "Taxi-zone lookup is missing columns: "
            f"{missing_text}"
        )

    duplicate_count = zones.select(
        pl.col("LocationID")
        .is_duplicated()
        .sum()
        .alias("duplicate_count")
    ).item()

    if duplicate_count:
        raise ValidationError(
            "Taxi-zone lookup contains duplicate LocationIDs."
        )

    all_zone_ids = [
        int(value)
        for value in zones[
            "LocationID"
        ].to_list()
    ]

    core_zone_ids = [
        int(value)
        for value in (
            zones.filter(
                pl.col("Borough").is_in(
                    list(settings.core_boroughs)
                )
            )["LocationID"].to_list()
        )
    ]

    if not core_zone_ids:
        raise ValidationError(
            "No core operating zones were found."
        )

    return all_zone_ids, core_zone_ids


def build_validated_lazy_frame(
    settings: ValidationSettings,
    all_zone_ids: list[int],
    core_zone_ids: list[int],
) -> pl.LazyFrame:
    """Build the lazy row-validation transformation."""
    if not settings.input_path.exists():
        raise ValidationError(
            f"Input dataset does not exist: {settings.input_path}"
        )

    if settings.input_path.stat().st_size == 0:
        raise ValidationError(
            f"Input dataset is empty: {settings.input_path}"
        )

    source = pl.scan_parquet(
        settings.input_path
    )

    schema = source.collect_schema()

    missing_columns = sorted(
        set(settings.required_columns).difference(
            schema.names()
        )
    )

    if missing_columns:
        missing_text = ", ".join(missing_columns)

        raise ValidationError(
            f"Input dataset is missing columns: {missing_text}"
        )

    timestamp_columns = [
        "request_datetime",
        "pickup_datetime",
        "dropoff_datetime",
    ]

    if "on_scene_datetime" in schema.names():
        timestamp_columns.append(
            "on_scene_datetime"
        )

    source = source.with_columns(
        [
            pl.col(column)
            .dt.replace_time_zone(
                settings.timezone,
                ambiguous="raise",
                non_existent="raise",
            )
            .alias(column)
            for column in timestamp_columns
        ]
    )

    month_start, next_month_start = get_month_bounds(
        year=settings.expected_year,
        month=settings.expected_month,
        timezone=settings.timezone,
    )

    missing_required_expression = combine_with_or(
        [
            pl.col(column).is_null()
            for column in settings.required_columns
        ]
    )

    pickup_zone_known = (
        pl.col("PULocationID")
        .is_in(all_zone_ids)
        .fill_null(False)
    )

    dropoff_zone_known = (
        pl.col("DOLocationID")
        .is_in(all_zone_ids)
        .fill_null(False)
    )

    pickup_zone_core = (
        pl.col("PULocationID")
        .is_in(core_zone_ids)
        .fill_null(False)
    )

    dropoff_zone_core = (
        pl.col("DOLocationID")
        .is_in(core_zone_ids)
        .fill_null(False)
    )

    validated = source.with_columns(
        (
            pl.col("dropoff_datetime")
            - pl.col("pickup_datetime")
        )
        .dt.total_seconds()
        .alias("derived_trip_duration_seconds"),

        (
            pl.col("pickup_datetime")
            - pl.col("request_datetime")
        )
        .dt.total_seconds()
        .alias("request_to_pickup_seconds"),

        pl.col("pickup_datetime")
        .dt.strftime("%Y-%m")
        .alias("pickup_month"),
    )

    validated = validated.with_columns(
        missing_required_expression.alias(
            "flag_missing_required_value"
        ),

        (~pickup_zone_known).alias(
            "flag_invalid_pickup_zone"
        ),

        (~dropoff_zone_known).alias(
            "flag_invalid_dropoff_zone"
        ),

        (
            (
                pl.col("pickup_datetime")
                < month_start
            )
            |
            (
                pl.col("pickup_datetime")
                >= next_month_start
            )
        )
        .fill_null(False)
        .alias(
            "flag_pickup_outside_expected_month"
        ),

        (
            pl.col(
                "derived_trip_duration_seconds"
            )
            <= 0
        )
        .fill_null(False)
        .alias(
            "flag_non_positive_trip_duration"
        ),

        (
            pl.col("trip_time") <= 0
        )
        .fill_null(False)
        .alias(
            "flag_non_positive_trip_time"
        ),

        (
            pl.col("request_datetime")
            > pl.col("pickup_datetime")
        )
        .fill_null(False)
        .alias(
            "flag_request_after_pickup"
        ),

        (
            pl.col(
                "derived_trip_duration_seconds"
            )
            > settings.long_trip_duration_seconds
        )
        .fill_null(False)
        .alias(
            "flag_long_trip_duration"
        ),

        (
            pickup_zone_known
            & ~pickup_zone_core
        )
        .fill_null(False)
        .alias(
            "flag_pickup_non_core_zone"
        ),

        (
            dropoff_zone_known
            & ~dropoff_zone_core
        )
        .fill_null(False)
        .alias(
            "flag_dropoff_non_core_zone"
        ),
    )

    rejected_expression = combine_with_or(
        [
            pl.col(column)
            for column in REJECTION_FLAG_COLUMNS
        ]
    )

    retained_expression = combine_with_or(
        [
            pl.col(column)
            for column in RETAINED_FLAG_COLUMNS
        ]
    )

    validated = validated.with_columns(
        rejected_expression.alias(
            "row_rejected"
        )
    )

    validated = validated.with_columns(
        (
            ~pl.col("row_rejected")
            & retained_expression
        ).alias(
            "row_retained_with_flag"
        )
    )

    validated = validated.with_columns(
        pl.when(
            pl.col("row_rejected")
        )
        .then(
            pl.lit("rejected")
        )
        .when(
            pl.col("row_retained_with_flag")
        )
        .then(
            pl.lit("retained_with_flag")
        )
        .otherwise(
            pl.lit("accepted")
        )
        .alias("record_status"),

        (
            ~pl.col("row_rejected")
            & pickup_zone_core
        ).alias(
            "eligible_for_demand_forecasting"
        ),

        (
            ~pl.col("row_rejected")
            & pickup_zone_core
            & dropoff_zone_core
            & ~pl.col("flag_long_trip_duration")
        ).alias(
            "eligible_for_od_modeling"
        ),

        (
            ~pl.col("row_rejected")
            & ~pl.col("flag_request_after_pickup")
        ).alias(
            "eligible_for_wait_time_analysis"
        ),
    )

    return validated


def collect_validation_summary(
    frame: pl.LazyFrame,
) -> dict[str, int]:
    """Collect validation, flag, and eligibility counts."""
    expressions: list[pl.Expr] = [
        pl.len().alias("total_records"),

        (
            pl.col("record_status")
            == "accepted"
        )
        .sum()
        .alias("accepted_records"),

        (
            pl.col("record_status")
            == "retained_with_flag"
        )
        .sum()
        .alias("retained_with_flag_records"),

        (
            pl.col("record_status")
            == "rejected"
        )
        .sum()
        .alias("rejected_records"),
    ]

    expressions.extend(
        pl.col(column)
        .sum()
        .alias(column)
        for column in ALL_FLAG_COLUMNS
    )

    expressions.extend(
        pl.col(column)
        .sum()
        .alias(column)
        for column in ELIGIBILITY_COLUMNS
    )

    row = (
        frame.select(expressions)
        .collect()
        .row(
            0,
            named=True,
        )
    )

    return {
        key: int(value)
        for key, value in row.items()
    }


def write_validation_report(
    settings: ValidationSettings,
    summary: dict[str, int],
    output_record_count: int,
) -> dict[str, Any]:
    """Write the machine-readable validation summary."""
    total_records = summary["total_records"]

    rejection_rate = (
        summary["rejected_records"]
        / total_records
        if total_records
        else 0.0
    )

    retained_rate = (
        summary["retained_with_flag_records"]
        / total_records
        if total_records
        else 0.0
    )

    report: dict[str, Any] = {
        "validation_version": "0.1.0",
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
        },
        "output": {
            "path": str(settings.output_path),
            "size_bytes": (
                settings.output_path.stat().st_size
            ),
            "record_count": output_record_count,
            "includes_rejected_records": (
                settings.include_rejected_records
            ),
        },
        "status_counts": {
            "total": total_records,
            "accepted": summary[
                "accepted_records"
            ],
            "retained_with_flag": summary[
                "retained_with_flag_records"
            ],
            "rejected": summary[
                "rejected_records"
            ],
        },
        "rates": {
            "rejection_rate": round(
                rejection_rate,
                8,
            ),
            "retained_with_flag_rate": round(
                retained_rate,
                8,
            ),
        },
        "flag_counts": {
            column: summary[column]
            for column in ALL_FLAG_COLUMNS
        },
        "eligibility_counts": {
            column: summary[column]
            for column in ELIGIBILITY_COLUMNS
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


def validate_trip_data(
    config_path: Path,
) -> dict[str, Any]:
    """Run the configured TLC validation pipeline."""
    settings = ValidationSettings.from_yaml(
        config_path
    )

    all_zone_ids, core_zone_ids = (
        load_zone_identifiers(settings)
    )

    validated = build_validated_lazy_frame(
        settings=settings,
        all_zone_ids=all_zone_ids,
        core_zone_ids=core_zone_ids,
    )

    settings.output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_output = settings.output_path.with_name(
        f"{settings.output_path.name}.part"
    )

    temporary_output.unlink(
        missing_ok=True
    )

    if settings.include_rejected_records:
        output_frame = validated
    else:
        output_frame = validated.filter(
            ~pl.col("row_rejected")
        )

    try:
        output_frame.sink_parquet(
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

        raise ValidationError(
            f"Could not write validated dataset: {error}"
        ) from error

    if settings.include_rejected_records:
        summary_source = pl.scan_parquet(
            settings.output_path
        )
    else:
        summary_source = validated

    summary = collect_validation_summary(
        summary_source
    )

    output_record_count = int(
        pl.scan_parquet(
            settings.output_path
        )
        .select(
            pl.len().alias("row_count")
        )
        .collect()
        .item()
    )

    return write_validation_report(
        settings=settings,
        summary=summary,
        output_record_count=output_record_count,
    )