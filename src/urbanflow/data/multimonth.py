from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import polars as pl
import yaml

from urbanflow.data.aggregation import (
    DemandAggregationError,
    build_zone_demand,
)
from urbanflow.data.download import (
    DownloadError,
    download_trip_data,
)
from urbanflow.data.validation import (
    ValidationError,
    validate_trip_data,
)


TLC_FILE_PREFIXES = {
    "hvfhv": "fhvhv",
}


class MultiMonthPipelineError(RuntimeError):
    """Raised when multi-month data preparation cannot complete."""


@dataclass(frozen=True)
class MonthPaths:
    """Paths generated for one monthly pipeline execution."""

    year: int
    month: int
    period: str

    raw_path: Path
    validated_path: Path
    demand_path: Path

    validation_config_path: Path
    aggregation_config_path: Path

    validation_report_path: Path
    aggregation_report_path: Path


@dataclass(frozen=True)
class MultiMonthSettings:
    """Configuration for resumable multi-month data preparation."""

    dataset: str
    year: int
    start_month: int
    end_month: int
    timezone: str

    validation_template_path: Path
    aggregation_template_path: Path
    aggregation_interval: str

    raw_directory: Path
    zone_lookup_path: Path
    interim_directory: Path
    monthly_demand_directory: Path
    generated_config_directory: Path

    combined_output_path: Path
    report_path: Path

    download_missing: bool
    validate_missing: bool
    aggregate_missing: bool

    force_download: bool
    force_validation: bool
    force_aggregation: bool
    force_combine: bool

    cleanup_interim_after_aggregation: bool
    compression: str

    @classmethod
    def from_yaml(
        cls,
        config_path: Path,
    ) -> MultiMonthSettings:
        """Load multi-month settings from YAML."""
        raw_config = load_yaml_mapping(
            config_path,
            description="multi-month config",
        )

        try:
            dataset_config = raw_config["dataset"]
            template_config = raw_config["templates"]
            path_config = raw_config["paths"]
            execution_config = raw_config["execution"]
            output_config = raw_config["output"]

            validation_template_path = Path(
                template_config["validation"]
            )

            aggregation_template_path = Path(
                template_config[
                    "demand_aggregation"
                ]
            )

            aggregation_template = load_yaml_mapping(
                aggregation_template_path,
                description=(
                    "demand aggregation template"
                ),
            )

            aggregation_section = (
                aggregation_template["aggregation"]
            )

            aggregation_interval = str(
                aggregation_section["interval"]
            )

            settings = cls(
                dataset=str(
                    dataset_config["dataset"]
                ),
                year=int(
                    dataset_config["year"]
                ),
                start_month=int(
                    dataset_config["start_month"]
                ),
                end_month=int(
                    dataset_config["end_month"]
                ),
                timezone=str(
                    dataset_config["timezone"]
                ),
                validation_template_path=(
                    validation_template_path
                ),
                aggregation_template_path=(
                    aggregation_template_path
                ),
                aggregation_interval=(
                    aggregation_interval
                ),
                raw_directory=Path(
                    path_config["raw_directory"]
                ),
                zone_lookup_path=Path(
                    path_config["zone_lookup"]
                ),
                interim_directory=Path(
                    path_config[
                        "interim_directory"
                    ]
                ),
                monthly_demand_directory=Path(
                    path_config[
                        "monthly_demand_directory"
                    ]
                ),
                generated_config_directory=Path(
                    path_config[
                        "generated_config_directory"
                    ]
                ),
                combined_output_path=Path(
                    path_config["combined_output"]
                ),
                report_path=Path(
                    path_config["report"]
                ),
                download_missing=bool(
                    execution_config[
                        "download_missing"
                    ]
                ),
                validate_missing=bool(
                    execution_config[
                        "validate_missing"
                    ]
                ),
                aggregate_missing=bool(
                    execution_config[
                        "aggregate_missing"
                    ]
                ),
                force_download=bool(
                    execution_config[
                        "force_download"
                    ]
                ),
                force_validation=bool(
                    execution_config[
                        "force_validation"
                    ]
                ),
                force_aggregation=bool(
                    execution_config[
                        "force_aggregation"
                    ]
                ),
                force_combine=bool(
                    execution_config[
                        "force_combine"
                    ]
                ),
                cleanup_interim_after_aggregation=bool(
                    execution_config[
                        "cleanup_interim_after_aggregation"
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
            raise MultiMonthPipelineError(
                "Invalid multi-month configuration: "
                f"{error}"
            ) from error

        settings.validate()

        return settings

    @property
    def months(self) -> tuple[int, ...]:
        """Return configured month numbers."""
        return tuple(
            range(
                self.start_month,
                self.end_month + 1,
            )
        )

    @property
    def interval_slug(self) -> str:
        """Return a filesystem-safe interval label."""
        return (
            self.aggregation_interval
            .strip()
            .replace(" ", "")
            .replace("/", "-")
        )

    @property
    def month_count(self) -> int:
        """Return the number of configured months."""
        return len(self.months)

    def validate(self) -> None:
        """Validate multi-month configuration."""
        if self.dataset not in TLC_FILE_PREFIXES:
            supported = ", ".join(
                sorted(TLC_FILE_PREFIXES)
            )

            raise MultiMonthPipelineError(
                f"Unsupported dataset: {self.dataset}. "
                f"Supported datasets: {supported}"
            )

        if self.year < 2009:
            raise MultiMonthPipelineError(
                f"Invalid dataset year: {self.year}"
            )

        if not 1 <= self.start_month <= 12:
            raise MultiMonthPipelineError(
                "start_month must be between 1 and 12."
            )

        if not 1 <= self.end_month <= 12:
            raise MultiMonthPipelineError(
                "end_month must be between 1 and 12."
            )

        if self.start_month > self.end_month:
            raise MultiMonthPipelineError(
                "start_month must not be after end_month."
            )

        try:
            ZoneInfo(self.timezone)
        except ZoneInfoNotFoundError as error:
            raise MultiMonthPipelineError(
                f"Unknown timezone: {self.timezone}"
            ) from error

        if not self.validation_template_path.exists():
            raise MultiMonthPipelineError(
                "Validation template does not exist: "
                f"{self.validation_template_path}"
            )

        if not self.aggregation_template_path.exists():
            raise MultiMonthPipelineError(
                "Demand aggregation template does not "
                "exist: "
                f"{self.aggregation_template_path}"
            )

        if not self.zone_lookup_path.exists():
            raise MultiMonthPipelineError(
                "Taxi-zone lookup does not exist: "
                f"{self.zone_lookup_path}"
            )

        if not self.aggregation_interval:
            raise MultiMonthPipelineError(
                "Aggregation interval must not be empty."
            )

        if not self.compression:
            raise MultiMonthPipelineError(
                "Output compression must not be empty."
            )

        resolved_monthly_directory = (
            self.monthly_demand_directory.resolve()
        )

        resolved_combined_output = (
            self.combined_output_path.resolve()
        )

        if (
            resolved_combined_output.parent
            == resolved_monthly_directory
            and resolved_combined_output.name.startswith(
                "zone_demand_"
            )
        ):
            return

    def as_report_dict(self) -> dict[str, Any]:
        """Return effective settings as JSON-compatible data."""
        return {
            "dataset": {
                "dataset": self.dataset,
                "year": self.year,
                "start_month": self.start_month,
                "end_month": self.end_month,
                "timezone": self.timezone,
            },
            "templates": {
                "validation": str(
                    self.validation_template_path
                ),
                "demand_aggregation": str(
                    self.aggregation_template_path
                ),
                "aggregation_interval": (
                    self.aggregation_interval
                ),
            },
            "paths": {
                "raw_directory": str(
                    self.raw_directory
                ),
                "zone_lookup": str(
                    self.zone_lookup_path
                ),
                "interim_directory": str(
                    self.interim_directory
                ),
                "monthly_demand_directory": str(
                    self.monthly_demand_directory
                ),
                "generated_config_directory": str(
                    self.generated_config_directory
                ),
                "combined_output": str(
                    self.combined_output_path
                ),
                "report": str(
                    self.report_path
                ),
            },
            "execution": {
                "download_missing": (
                    self.download_missing
                ),
                "validate_missing": (
                    self.validate_missing
                ),
                "aggregate_missing": (
                    self.aggregate_missing
                ),
                "force_download": (
                    self.force_download
                ),
                "force_validation": (
                    self.force_validation
                ),
                "force_aggregation": (
                    self.force_aggregation
                ),
                "force_combine": (
                    self.force_combine
                ),
                "cleanup_interim_after_aggregation": (
                    self.cleanup_interim_after_aggregation
                ),
            },
            "output": {
                "compression": self.compression,
            },
        }


def load_yaml_mapping(
    path: Path,
    *,
    description: str,
) -> dict[str, Any]:
    """Read a YAML file and require a top-level mapping."""
    if not path.exists():
        raise MultiMonthPipelineError(
            f"{description.capitalize()} does not exist: "
            f"{path}"
        )

    try:
        content = yaml.safe_load(
            path.read_text(
                encoding="utf-8"
            )
        )
    except (OSError, yaml.YAMLError) as error:
        raise MultiMonthPipelineError(
            f"Could not read {description}: {error}"
        ) from error

    if not isinstance(content, dict):
        raise MultiMonthPipelineError(
            f"{description.capitalize()} must contain "
            "a YAML mapping."
        )

    return content


def require_mapping_section(
    config: dict[str, Any],
    section_name: str,
    *,
    description: str,
) -> dict[str, Any]:
    """Return one required mapping section."""
    section = config.get(
        section_name
    )

    if not isinstance(section, dict):
        raise MultiMonthPipelineError(
            f"{description.capitalize()} must contain "
            f"a '{section_name}' mapping."
        )

    return section


def next_month_start(
    year: int,
    month: int,
    *,
    timezone: ZoneInfo,
) -> datetime:
    """Return the first instant of the following month."""
    if month == 12:
        return datetime(
            year=year + 1,
            month=1,
            day=1,
            tzinfo=timezone,
        )

    return datetime(
        year=year,
        month=month + 1,
        day=1,
        tzinfo=timezone,
    )


def month_start(
    year: int,
    month: int,
    *,
    timezone: ZoneInfo,
) -> datetime:
    """Return the first instant of one local month."""
    return datetime(
        year=year,
        month=month,
        day=1,
        tzinfo=timezone,
    )


def build_month_paths(
    settings: MultiMonthSettings,
    month: int,
) -> MonthPaths:
    """Build all paths needed for one month."""
    period = (
        f"{settings.year}-"
        f"{month:02d}"
    )

    file_prefix = TLC_FILE_PREFIXES[
        settings.dataset
    ]

    raw_filename = (
        f"{file_prefix}_tripdata_"
        f"{period}.parquet"
    )

    validated_filename = (
        f"{settings.dataset}_"
        f"{period}_validated.parquet"
    )

    demand_filename = (
        "zone_demand_"
        f"{settings.interval_slug}_"
        f"{period}.parquet"
    )

    return MonthPaths(
        year=settings.year,
        month=month,
        period=period,
        raw_path=(
            settings.raw_directory
            / raw_filename
        ),
        validated_path=(
            settings.interim_directory
            / validated_filename
        ),
        demand_path=(
            settings.monthly_demand_directory
            / demand_filename
        ),
        validation_config_path=(
            settings.generated_config_directory
            / f"{period}-validation.yaml"
        ),
        aggregation_config_path=(
            settings.generated_config_directory
            / f"{period}-demand-aggregation.yaml"
        ),
        validation_report_path=Path(
            "reports/data_quality"
        )
        / (
            f"{period}-full-"
            "validation-summary.json"
        ),
        aggregation_report_path=Path(
            "reports/data_quality"
        )
        / (
            f"{period}-full-"
            "demand-aggregation.json"
        ),
    )


def write_yaml_file(
    path: Path,
    config: dict[str, Any],
) -> None:
    """Write one generated YAML configuration."""
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        yaml.safe_dump(
            config,
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def generate_validation_config(
    settings: MultiMonthSettings,
    paths: MonthPaths,
) -> None:
    """Generate one monthly validation configuration."""
    template = load_yaml_mapping(
        settings.validation_template_path,
        description="validation template",
    )

    config = copy.deepcopy(
        template
    )

    dataset_section = require_mapping_section(
        config,
        "dataset",
        description="validation template",
    )

    path_section = require_mapping_section(
        config,
        "paths",
        description="validation template",
    )

    dataset_section[
        "expected_year"
    ] = paths.year

    dataset_section[
        "expected_month"
    ] = paths.month

    dataset_section[
        "timezone"
    ] = settings.timezone

    path_section[
        "input"
    ] = str(paths.raw_path)

    path_section[
        "zone_lookup"
    ] = str(settings.zone_lookup_path)

    path_section[
        "output"
    ] = str(paths.validated_path)

    path_section[
        "report"
    ] = str(
        paths.validation_report_path
    )

    write_yaml_file(
        paths.validation_config_path,
        config,
    )


def generate_aggregation_config(
    settings: MultiMonthSettings,
    paths: MonthPaths,
) -> None:
    """Generate one monthly aggregation configuration."""
    template = load_yaml_mapping(
        settings.aggregation_template_path,
        description=(
            "demand aggregation template"
        ),
    )

    config = copy.deepcopy(
        template
    )

    dataset_section = require_mapping_section(
        config,
        "dataset",
        description=(
            "demand aggregation template"
        ),
    )

    path_section = require_mapping_section(
        config,
        "paths",
        description=(
            "demand aggregation template"
        ),
    )

    dataset_section[
        "expected_year"
    ] = paths.year

    dataset_section[
        "expected_month"
    ] = paths.month

    dataset_section[
        "timezone"
    ] = settings.timezone

    path_section[
        "input"
    ] = str(paths.validated_path)

    path_section[
        "zone_lookup"
    ] = str(settings.zone_lookup_path)

    path_section[
        "output"
    ] = str(paths.demand_path)

    path_section[
        "report"
    ] = str(
        paths.aggregation_report_path
    )

    write_yaml_file(
        paths.aggregation_config_path,
        config,
    )


def generate_month_configs(
    settings: MultiMonthSettings,
    paths: MonthPaths,
) -> None:
    """Generate validation and aggregation configs."""
    generate_validation_config(
        settings,
        paths,
    )

    generate_aggregation_config(
        settings,
        paths,
    )


def file_is_newer_than(
    candidate: Path,
    dependency: Path,
) -> bool:
    """Return whether candidate is at least as new as dependency."""
    if not candidate.exists():
        return False

    if not dependency.exists():
        return True

    return (
        candidate.stat().st_mtime_ns
        >= dependency.stat().st_mtime_ns
    )


def download_month(
    settings: MultiMonthSettings,
    paths: MonthPaths,
) -> str:
    """Download one missing or forced monthly raw file."""
    if (
        paths.raw_path.exists()
        and not settings.force_download
    ):
        return "skipped"

    if (
        not paths.raw_path.exists()
        and not settings.download_missing
        and not settings.force_download
    ):
        raise MultiMonthPipelineError(
            "Raw monthly file is missing and "
            "download_missing is disabled: "
            f"{paths.raw_path}"
        )

    try:
        destination = download_trip_data(
            dataset=settings.dataset,
            year=paths.year,
            month=paths.month,
            output_directory=(
                settings.raw_directory
            ),
            force=settings.force_download,
        )
    except (
        DownloadError,
        ValueError,
    ) as error:
        raise MultiMonthPipelineError(
            f"Download failed for {paths.period}: "
            f"{error}"
        ) from error

    if destination.resolve() != (
        paths.raw_path.resolve()
    ):
        raise MultiMonthPipelineError(
            "Downloader returned an unexpected path "
            f"for {paths.period}: {destination}"
        )

    if not destination.exists():
        raise MultiMonthPipelineError(
            "Downloader completed without creating "
            f"the expected file: {destination}"
        )

    if settings.force_download:
        return "refreshed"

    return "created"


def monthly_demand_can_be_reused(
    settings: MultiMonthSettings,
    paths: MonthPaths,
) -> bool:
    """Return whether a completed monthly demand file is reusable."""
    if not paths.demand_path.exists():
        return False

    if (
        settings.force_validation
        or settings.force_aggregation
    ):
        return False

    if not paths.raw_path.exists():
        return True

    return file_is_newer_than(
        paths.demand_path,
        paths.raw_path,
    )


def validate_month(
    settings: MultiMonthSettings,
    paths: MonthPaths,
) -> str:
    """Validate one monthly raw trip dataset."""
    output_exists = (
        paths.validated_path.exists()
    )

    output_is_current = (
        output_exists
        and file_is_newer_than(
            paths.validated_path,
            paths.raw_path,
        )
    )

    must_run = (
        settings.force_validation
        or not output_is_current
    )

    if not must_run:
        return "skipped"

    if (
        not settings.validate_missing
        and not settings.force_validation
    ):
        raise MultiMonthPipelineError(
            "Validated monthly file is missing or stale "
            "and validate_missing is disabled: "
            f"{paths.validated_path}"
        )

    paths.validated_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    try:
        validate_trip_data(
            config_path=(
                paths.validation_config_path
            )
        )
    except ValidationError as error:
        raise MultiMonthPipelineError(
            f"Validation failed for {paths.period}: "
            f"{error}"
        ) from error

    if not paths.validated_path.exists():
        raise MultiMonthPipelineError(
            "Validation completed without creating "
            f"the expected output: {paths.validated_path}"
        )

    if output_exists:
        return "refreshed"

    return "created"


def aggregate_month(
    settings: MultiMonthSettings,
    paths: MonthPaths,
) -> str:
    """Aggregate one validated month into zone demand."""
    output_exists = (
        paths.demand_path.exists()
    )

    output_is_current = (
        output_exists
        and file_is_newer_than(
            paths.demand_path,
            paths.validated_path,
        )
    )

    must_run = (
        settings.force_aggregation
        or not output_is_current
    )

    if not must_run:
        return "skipped"

    if (
        not settings.aggregate_missing
        and not settings.force_aggregation
    ):
        raise MultiMonthPipelineError(
            "Monthly demand file is missing or stale "
            "and aggregate_missing is disabled: "
            f"{paths.demand_path}"
        )

    if not paths.validated_path.exists():
        raise MultiMonthPipelineError(
            "Cannot aggregate without a validated "
            f"monthly file: {paths.validated_path}"
        )

    paths.demand_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    try:
        build_zone_demand(
            config_path=(
                paths.aggregation_config_path
            )
        )
    except DemandAggregationError as error:
        raise MultiMonthPipelineError(
            f"Demand aggregation failed for "
            f"{paths.period}: {error}"
        ) from error

    if not paths.demand_path.exists():
        raise MultiMonthPipelineError(
            "Demand aggregation completed without "
            "creating the expected output: "
            f"{paths.demand_path}"
        )

    if output_exists:
        return "refreshed"

    return "created"


def maybe_cleanup_interim(
    settings: MultiMonthSettings,
    paths: MonthPaths,
) -> bool:
    """Optionally remove one validated monthly artifact."""
    if (
        not settings.cleanup_interim_after_aggregation
    ):
        return False

    if not paths.demand_path.exists():
        return False

    if not paths.validated_path.exists():
        return False

    paths.validated_path.unlink()

    return True


def collect_null_count(
    frame: pl.LazyFrame,
    columns: list[str],
) -> int:
    """Count null values across selected columns."""
    null_row = (
        frame.select(
            [
                pl.col(column)
                .null_count()
                .alias(column)
                for column in columns
            ]
        )
        .collect()
        .row(0)
    )

    return sum(
        int(value)
        for value in null_row
    )


def collect_duplicate_key_count(
    frame: pl.LazyFrame,
) -> int:
    """Count duplicate zone-time keys."""
    return int(
        frame.group_by(
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


def collect_zone_ids(
    path: Path,
) -> tuple[int, ...]:
    """Collect sorted unique zone identifiers."""
    values = (
        pl.scan_parquet(
            path
        )
        .select(
            pl.col("zone_id")
        )
        .unique()
        .sort("zone_id")
        .collect()
        .get_column("zone_id")
        .to_list()
    )

    return tuple(
        int(value)
        for value in values
    )


def collect_interval_seconds(
    path: Path,
) -> int:
    """Derive interval duration from one monthly panel."""
    intervals = (
        pl.scan_parquet(
            path
        )
        .select(
            pl.col("interval_start")
        )
        .unique()
        .sort("interval_start")
        .limit(2)
        .collect()
        .get_column("interval_start")
        .to_list()
    )

    if len(intervals) < 2:
        raise MultiMonthPipelineError(
            "Monthly demand panel does not contain "
            "enough intervals to derive frequency: "
            f"{path}"
        )

    first_interval = intervals[0]
    second_interval = intervals[1]

    delta_seconds = int(
        (
            second_interval.astimezone(UTC)
            - first_interval.astimezone(UTC)
        ).total_seconds()
    )

    if delta_seconds <= 0:
        raise MultiMonthPipelineError(
            "Derived aggregation interval is not "
            f"positive for {path}."
        )

    return delta_seconds


def collect_monthly_summary(
    path: Path,
) -> dict[str, Any]:
    """Collect invariants from one monthly demand panel."""
    if not path.exists():
        raise MultiMonthPipelineError(
            f"Monthly demand file does not exist: {path}"
        )

    if path.stat().st_size == 0:
        raise MultiMonthPipelineError(
            f"Monthly demand file is empty: {path}"
        )

    try:
        frame = pl.scan_parquet(
            path
        )

        schema = frame.collect_schema()
    except Exception as error:
        raise MultiMonthPipelineError(
            f"Could not read monthly demand file "
            f"{path}: {error}"
        ) from error

    required_columns = {
        "time_step",
        "interval_start",
        "zone_id",
        "pickup_count",
    }

    missing_columns = sorted(
        required_columns.difference(
            schema.names()
        )
    )

    if missing_columns:
        raise MultiMonthPipelineError(
            "Monthly demand file is missing columns: "
            + ", ".join(missing_columns)
        )

    row = (
        frame.select(
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
            pl.col("pickup_count")
            .sum()
            .alias(
                "total_pickups"
            ),
            (
                pl.col("pickup_count") == 0
            )
            .sum()
            .alias(
                "zero_demand_rows"
            ),
            pl.col("pickup_count")
            .min()
            .alias(
                "minimum_pickup_count"
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

    duplicate_key_count = (
        collect_duplicate_key_count(
            frame
        )
    )

    null_value_count = (
        collect_null_count(
            frame,
            schema.names(),
        )
    )

    row_count = int(
        row["row_count"]
    )

    interval_count = int(
        row["interval_count"]
    )

    zone_count = int(
        row["zone_count"]
    )

    time_step_count = int(
        row["time_step_count"]
    )

    minimum_time_step = int(
        row["minimum_time_step"]
    )

    maximum_time_step = int(
        row["maximum_time_step"]
    )

    if row_count != (
        interval_count
        * zone_count
    ):
        raise MultiMonthPipelineError(
            "Monthly demand panel is not a complete "
            f"dense grid: {path}"
        )

    if duplicate_key_count:
        raise MultiMonthPipelineError(
            "Monthly demand panel contains duplicate "
            f"zone-time keys: {path}"
        )

    if null_value_count:
        raise MultiMonthPipelineError(
            "Monthly demand panel contains null "
            f"values: {path}"
        )

    if int(
        row["minimum_pickup_count"]
    ) < 0:
        raise MultiMonthPipelineError(
            "Monthly demand panel contains negative "
            f"pickup counts: {path}"
        )

    if time_step_count != interval_count:
        raise MultiMonthPipelineError(
            "Monthly time_step count does not match "
            f"interval count: {path}"
        )

    if minimum_time_step != 0:
        raise MultiMonthPipelineError(
            "Monthly time_step must start at zero: "
            f"{path}"
        )

    if maximum_time_step != (
        interval_count - 1
    ):
        raise MultiMonthPipelineError(
            "Monthly time_step is not contiguous: "
            f"{path}"
        )

    return {
        "path": str(path),
        "size_bytes": (
            path.stat().st_size
        ),
        "schema": [
            {
                "name": name,
                "dtype": str(
                    schema[name]
                ),
            }
            for name in schema.names()
        ],
        "row_count": row_count,
        "interval_count": interval_count,
        "zone_count": zone_count,
        "time_step_count": (
            time_step_count
        ),
        "minimum_time_step": (
            minimum_time_step
        ),
        "maximum_time_step": (
            maximum_time_step
        ),
        "total_pickups": int(
            row["total_pickups"]
        ),
        "zero_demand_rows": int(
            row["zero_demand_rows"]
        ),
        "minimum_pickup_count": int(
            row["minimum_pickup_count"]
        ),
        "first_interval": row[
            "first_interval"
        ],
        "last_interval": row[
            "last_interval"
        ],
        "duplicate_key_count": (
            duplicate_key_count
        ),
        "null_value_count": (
            null_value_count
        ),
        "zone_ids": collect_zone_ids(
            path
        ),
    }


def enforce_month_boundaries(
    *,
    settings: MultiMonthSettings,
    paths: MonthPaths,
    summary: dict[str, Any],
    interval_seconds: int,
) -> None:
    """Verify one monthly panel covers its exact month."""
    timezone = ZoneInfo(
        settings.timezone
    )

    expected_start = month_start(
        paths.year,
        paths.month,
        timezone=timezone,
    )

    expected_end = next_month_start(
        paths.year,
        paths.month,
        timezone=timezone,
    )

    expected_interval_count = int(
        (
            expected_end.astimezone(UTC)
            - expected_start.astimezone(UTC)
        ).total_seconds()
        // interval_seconds
    )

    expected_last = (
        expected_end.astimezone(UTC)
        - timedelta(
            seconds=interval_seconds
        )
    ).astimezone(
        timezone
    )

    if summary[
        "first_interval"
    ] != expected_start:
        raise MultiMonthPipelineError(
            f"{paths.period} demand panel starts at "
            f"{summary['first_interval']!s}; expected "
            f"{expected_start!s}."
        )

    if summary[
        "last_interval"
    ] != expected_last:
        raise MultiMonthPipelineError(
            f"{paths.period} demand panel ends at "
            f"{summary['last_interval']!s}; expected "
            f"{expected_last!s}."
        )

    if summary[
        "interval_count"
    ] != expected_interval_count:
        raise MultiMonthPipelineError(
            f"{paths.period} interval count is "
            f"{summary['interval_count']:,}; expected "
            f"{expected_interval_count:,}."
        )


def validate_monthly_outputs(
    settings: MultiMonthSettings,
    month_paths: list[MonthPaths],
) -> tuple[
    list[dict[str, Any]],
    int,
]:
    """Validate all monthly demand artifacts."""
    if not month_paths:
        raise MultiMonthPipelineError(
            "No monthly demand paths were generated."
        )

    interval_seconds = (
        collect_interval_seconds(
            month_paths[0].demand_path
        )
    )

    summaries: list[
        dict[str, Any]
    ] = []

    reference_schema: list[
        dict[str, str]
    ] | None = None

    reference_zone_ids: tuple[
        int,
        ...
    ] | None = None

    previous_last_interval: (
        datetime | None
    ) = None

    for paths in month_paths:
        summary = collect_monthly_summary(
            paths.demand_path
        )

        enforce_month_boundaries(
            settings=settings,
            paths=paths,
            summary=summary,
            interval_seconds=interval_seconds,
        )

        current_schema = summary[
            "schema"
        ]

        current_zone_ids = summary[
            "zone_ids"
        ]

        if reference_schema is None:
            reference_schema = (
                current_schema
            )
        elif current_schema != reference_schema:
            raise MultiMonthPipelineError(
                "Monthly demand schemas differ. "
                f"Mismatch detected in {paths.period}."
            )

        if reference_zone_ids is None:
            reference_zone_ids = (
                current_zone_ids
            )
        elif (
            current_zone_ids
            != reference_zone_ids
        ):
            raise MultiMonthPipelineError(
                "Monthly demand zone sets differ. "
                f"Mismatch detected in {paths.period}."
            )

        if previous_last_interval is not None:
            expected_next = (
                previous_last_interval
                .astimezone(UTC)
                + timedelta(
                    seconds=interval_seconds
                )
            ).astimezone(
                ZoneInfo(
                    settings.timezone
                )
            )

            if summary[
                "first_interval"
            ] != expected_next:
                raise MultiMonthPipelineError(
                    "Gap or overlap detected between "
                    "monthly demand panels before "
                    f"{paths.period}."
                )

        previous_last_interval = (
            summary["last_interval"]
        )

        summaries.append(
            summary
        )

    return (
        summaries,
        interval_seconds,
    )


def combined_output_is_current(
    settings: MultiMonthSettings,
    month_paths: list[MonthPaths],
) -> bool:
    """Return whether the combined output is current."""
    if not settings.combined_output_path.exists():
        return False

    if settings.force_combine:
        return False

    combined_mtime = (
        settings.combined_output_path
        .stat()
        .st_mtime_ns
    )

    return all(
        combined_mtime
        >= paths.demand_path
        .stat()
        .st_mtime_ns
        for paths in month_paths
    )


def build_combined_frame(
    month_paths: list[MonthPaths],
    monthly_summaries: list[
        dict[str, Any]
    ],
) -> pl.LazyFrame:
    """Build one globally indexed multi-month panel."""
    frames: list[
        pl.LazyFrame
    ] = []

    interval_offset = 0

    for paths, summary in zip(
        month_paths,
        monthly_summaries,
        strict=True,
    ):
        frame = (
            pl.scan_parquet(
                paths.demand_path
            )
            .with_columns(
                (
                    pl.col("time_step")
                    .cast(pl.Int64)
                    + interval_offset
                )
                .cast(pl.Int32)
                .alias("time_step")
            )
        )

        frames.append(
            frame
        )

        interval_offset += int(
            summary["interval_count"]
        )

    return (
        pl.concat(
            frames,
            how="vertical",
        )
        .sort(
            "interval_start",
            "zone_id",
        )
    )


def write_combined_output(
    *,
    settings: MultiMonthSettings,
    month_paths: list[MonthPaths],
    monthly_summaries: list[
        dict[str, Any]
    ],
) -> str:
    """Write or reuse the combined multi-month panel."""
    if combined_output_is_current(
        settings,
        month_paths,
    ):
        return "skipped"

    output_existed = (
        settings.combined_output_path
        .exists()
    )

    settings.combined_output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_output = (
        settings.combined_output_path
        .with_name(
            f"{settings.combined_output_path.name}"
            ".part"
        )
    )

    temporary_output.unlink(
        missing_ok=True
    )

    combined = build_combined_frame(
        month_paths,
        monthly_summaries,
    )

    try:
        combined.sink_parquet(
            temporary_output,
            compression=settings.compression,
            statistics=True,
        )

        temporary_output.replace(
            settings.combined_output_path
        )

    except Exception as error:
        temporary_output.unlink(
            missing_ok=True
        )

        raise MultiMonthPipelineError(
            "Could not write combined multi-month "
            f"demand panel: {error}"
        ) from error

    if output_existed:
        return "refreshed"

    return "created"


def collect_combined_summary(
    path: Path,
) -> dict[str, Any]:
    """Collect invariants from the combined demand panel."""
    summary = collect_monthly_summary(
        path
    )

    return summary


def enforce_combined_invariants(
    *,
    monthly_summaries: list[
        dict[str, Any]
    ],
    combined_summary: dict[str, Any],
) -> None:
    """Ensure combination conserves all monthly data."""
    expected_row_count = sum(
        int(summary["row_count"])
        for summary in monthly_summaries
    )

    expected_interval_count = sum(
        int(summary["interval_count"])
        for summary in monthly_summaries
    )

    expected_total_pickups = sum(
        int(summary["total_pickups"])
        for summary in monthly_summaries
    )

    expected_zero_demand_rows = sum(
        int(summary["zero_demand_rows"])
        for summary in monthly_summaries
    )

    expected_zone_count = int(
        monthly_summaries[0][
            "zone_count"
        ]
    )

    if (
        combined_summary["row_count"]
        != expected_row_count
    ):
        raise MultiMonthPipelineError(
            "Combined row count does not equal "
            "the sum of monthly row counts."
        )

    if (
        combined_summary[
            "interval_count"
        ]
        != expected_interval_count
    ):
        raise MultiMonthPipelineError(
            "Combined interval count does not equal "
            "the sum of monthly interval counts."
        )

    if (
        combined_summary["zone_count"]
        != expected_zone_count
    ):
        raise MultiMonthPipelineError(
            "Combined output has an unexpected "
            "zone count."
        )

    if (
        combined_summary[
            "total_pickups"
        ]
        != expected_total_pickups
    ):
        raise MultiMonthPipelineError(
            "Combined pickup count does not equal "
            "the sum of monthly pickup counts."
        )

    if (
        combined_summary[
            "zero_demand_rows"
        ]
        != expected_zero_demand_rows
    ):
        raise MultiMonthPipelineError(
            "Combined zero-demand row count does not "
            "equal the monthly total."
        )

    if combined_summary[
        "minimum_time_step"
    ] != 0:
        raise MultiMonthPipelineError(
            "Combined time_step must start at zero."
        )

    if (
        combined_summary[
            "maximum_time_step"
        ]
        != expected_interval_count - 1
    ):
        raise MultiMonthPipelineError(
            "Combined time_step does not end at the "
            "expected value."
        )

    if (
        combined_summary[
            "time_step_count"
        ]
        != expected_interval_count
    ):
        raise MultiMonthPipelineError(
            "Combined time_step values are not "
            "globally contiguous."
        )


def sha256_file(
    path: Path,
    *,
    chunk_size: int = 1024 * 1024,
) -> str:
    """Calculate SHA-256 for one local file."""
    digest = hashlib.sha256()

    try:
        with path.open("rb") as file_handle:
            while chunk := file_handle.read(
                chunk_size
            ):
                digest.update(
                    chunk
                )
    except OSError as error:
        raise MultiMonthPipelineError(
            f"Could not hash file {path}: {error}"
        ) from error

    return digest.hexdigest()


def serialize_month_summary(
    *,
    paths: MonthPaths,
    summary: dict[str, Any],
    actions: dict[str, Any],
) -> dict[str, Any]:
    """Convert one monthly summary to report data."""
    return {
        "period": paths.period,
        "actions": actions,
        "paths": {
            "raw": str(
                paths.raw_path
            ),
            "validated": str(
                paths.validated_path
            ),
            "demand": str(
                paths.demand_path
            ),
            "validation_config": str(
                paths.validation_config_path
            ),
            "aggregation_config": str(
                paths.aggregation_config_path
            ),
            "validation_report": str(
                paths.validation_report_path
            ),
            "aggregation_report": str(
                paths.aggregation_report_path
            ),
        },
        "artifacts": {
            "raw_exists": (
                paths.raw_path.exists()
            ),
            "raw_size_bytes": (
                paths.raw_path.stat().st_size
                if paths.raw_path.exists()
                else None
            ),
            "validated_exists": (
                paths.validated_path.exists()
            ),
            "validated_size_bytes": (
                paths.validated_path
                .stat()
                .st_size
                if paths.validated_path.exists()
                else None
            ),
            "validation_report_exists": (
                paths.validation_report_path
                .exists()
            ),
            "aggregation_report_exists": (
                paths.aggregation_report_path
                .exists()
            ),
        },
        "demand": {
            "path": summary["path"],
            "size_bytes": (
                summary["size_bytes"]
            ),
            "row_count": (
                summary["row_count"]
            ),
            "interval_count": (
                summary["interval_count"]
            ),
            "zone_count": (
                summary["zone_count"]
            ),
            "total_pickups": (
                summary["total_pickups"]
            ),
            "zero_demand_rows": (
                summary["zero_demand_rows"]
            ),
            "first_interval": (
                summary["first_interval"]
                .isoformat()
            ),
            "last_interval": (
                summary["last_interval"]
                .isoformat()
            ),
            "sha256": sha256_file(
                paths.demand_path
            ),
        },
    }


def write_multimonth_report(
    *,
    settings: MultiMonthSettings,
    month_paths: list[MonthPaths],
    monthly_summaries: list[
        dict[str, Any]
    ],
    month_actions: list[
        dict[str, Any]
    ],
    combined_action: str,
    combined_summary: dict[str, Any],
    interval_seconds: int,
) -> dict[str, Any]:
    """Write the final multi-month pipeline report."""
    monthly_reports = [
        serialize_month_summary(
            paths=paths,
            summary=summary,
            actions=actions,
        )
        for paths, summary, actions in zip(
            month_paths,
            monthly_summaries,
            month_actions,
            strict=True,
        )
    ]

    report: dict[str, Any] = {
        "multimonth_pipeline_version": "0.1.0",
        "generated_at": datetime.now(
            UTC
        ).isoformat(),
        "effective_config": (
            settings.as_report_dict()
        ),
        "execution": {
            "month_count": (
                settings.month_count
            ),
            "combined_action": (
                combined_action
            ),
        },
        "frequency": {
            "configured_interval": (
                settings.aggregation_interval
            ),
            "derived_interval_seconds": (
                interval_seconds
            ),
        },
        "months": monthly_reports,
        "combined_output": {
            "path": str(
                settings.combined_output_path
            ),
            "size_bytes": (
                settings.combined_output_path
                .stat()
                .st_size
            ),
            "sha256": sha256_file(
                settings.combined_output_path
            ),
            "row_count": (
                combined_summary["row_count"]
            ),
            "interval_count": (
                combined_summary[
                    "interval_count"
                ]
            ),
            "zone_count": (
                combined_summary["zone_count"]
            ),
            "total_pickups": (
                combined_summary[
                    "total_pickups"
                ]
            ),
            "zero_demand_rows": (
                combined_summary[
                    "zero_demand_rows"
                ]
            ),
            "first_interval": (
                combined_summary[
                    "first_interval"
                ].isoformat()
            ),
            "last_interval": (
                combined_summary[
                    "last_interval"
                ].isoformat()
            ),
            "minimum_time_step": (
                combined_summary[
                    "minimum_time_step"
                ]
            ),
            "maximum_time_step": (
                combined_summary[
                    "maximum_time_step"
                ]
            ),
        },
        "invariants": {
            "monthly_schemas_match": True,
            "monthly_zone_sets_match": True,
            "monthly_boundaries_match": True,
            "months_are_contiguous": True,
            "combined_row_count_conserved": True,
            "combined_pickup_count_conserved": True,
            "combined_zero_rows_conserved": True,
            "combined_time_step_contiguous": True,
            "duplicate_key_count": (
                combined_summary[
                    "duplicate_key_count"
                ]
            ),
            "null_value_count": (
                combined_summary[
                    "null_value_count"
                ]
            ),
            "minimum_pickup_count": (
                combined_summary[
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


def run_month_pipeline(
    settings: MultiMonthSettings,
    paths: MonthPaths,
) -> dict[str, Any]:
    """Run or resume one monthly data pipeline."""
    generate_month_configs(
        settings,
        paths,
    )

    download_action = download_month(
        settings,
        paths,
    )

    if monthly_demand_can_be_reused(
        settings,
        paths,
    ):
        return {
            "download": download_action,
            "validation": (
                "skipped_existing_demand"
            ),
            "aggregation": "skipped",
            "interim_cleaned": False,
        }

    validation_action = validate_month(
        settings,
        paths,
    )

    aggregation_action = aggregate_month(
        settings,
        paths,
    )

    interim_cleaned = maybe_cleanup_interim(
        settings,
        paths,
    )

    return {
        "download": download_action,
        "validation": validation_action,
        "aggregation": aggregation_action,
        "interim_cleaned": interim_cleaned,
    }


def build_multimonth_demand(
    config_path: Path,
) -> dict[str, Any]:
    """Run the configured resumable multi-month pipeline."""
    settings = MultiMonthSettings.from_yaml(
        config_path
    )

    settings.raw_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    settings.interim_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    settings.monthly_demand_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    settings.generated_config_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    month_paths = [
        build_month_paths(
            settings,
            month,
        )
        for month in settings.months
    ]

    month_actions = [
        run_month_pipeline(
            settings,
            paths,
        )
        for paths in month_paths
    ]

    (
        monthly_summaries,
        interval_seconds,
    ) = validate_monthly_outputs(
        settings,
        month_paths,
    )

    combined_action = write_combined_output(
        settings=settings,
        month_paths=month_paths,
        monthly_summaries=monthly_summaries,
    )

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

    return write_multimonth_report(
        settings=settings,
        month_paths=month_paths,
        monthly_summaries=monthly_summaries,
        month_actions=month_actions,
        combined_action=combined_action,
        combined_summary=combined_summary,
        interval_seconds=interval_seconds,
    )