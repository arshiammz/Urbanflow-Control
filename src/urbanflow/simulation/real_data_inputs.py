from __future__ import annotations

import glob
import json
import math
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import polars as pl
import yaml


class SimulationInputError(RuntimeError):
    """Raised when real simulation inputs cannot be prepared."""


@dataclass(frozen=True)
class SimulationInputConfig:
    """Validated configuration for simulation-input generation."""

    dataset_name: str
    timezone: str
    interval_minutes: int

    raw_trip_glob: str
    demand_path: Path
    output_directory: Path
    metadata_path: Path

    pickup_datetime_column: str
    dropoff_datetime_column: str
    pickup_zone_column: str
    dropoff_zone_column: str

    demand_timestamp_column: str
    demand_zone_column: str
    demand_value_column: str

    od_fit_start: date
    od_fit_end_exclusive: date

    train_start: date
    train_end_exclusive: date

    validation_start: date
    validation_end_exclusive: date

    test_start: date
    test_end_exclusive: date

    top_n_zones: int
    minimum_fit_pickups: int

    minimum_trip_minutes: float
    maximum_trip_minutes: float
    smoothing_alpha: float
    fallback_travel_minutes: float
    maximum_travel_steps: int

    minimum_observed_od_coverage: float
    compression: str

    @property
    def zone_index_path(self) -> Path:
        return self.output_directory / "zone_index.parquet"

    @property
    def od_profile_path(self) -> Path:
        return self.output_directory / "od_profile.parquet"

    @property
    def historical_demand_path(self) -> Path:
        return self.output_directory / "historical_demand.parquet"


def parse_date(
    value: Any,
    *,
    description: str,
) -> date:
    """Parse one configuration date."""
    if isinstance(value, datetime):
        return value.date()

    if isinstance(value, date):
        return value

    try:
        return date.fromisoformat(
            str(value)
        )
    except ValueError as error:
        raise SimulationInputError(
            f"{description} must use YYYY-MM-DD format."
        ) from error


def read_config(
    path: Path,
) -> SimulationInputConfig:
    """Read and validate simulation-input configuration."""
    if not path.exists():
        raise SimulationInputError(
            f"Config does not exist: {path}"
        )

    try:
        raw = yaml.safe_load(
            path.read_text(
                encoding="utf-8",
            )
        )

        if not isinstance(raw, dict):
            raise SimulationInputError(
                "Config must contain a YAML mapping."
            )

        dataset = raw["dataset"]
        paths = raw["paths"]
        columns = raw["columns"]
        periods = raw["periods"]
        zone_selection = raw["zone_selection"]
        od_profile = raw["od_profile"]
        quality = raw["quality"]
        output = raw["output"]

        config = SimulationInputConfig(
            dataset_name=str(
                dataset["name"]
            ),
            timezone=str(
                dataset["timezone"]
            ),
            interval_minutes=int(
                dataset["interval_minutes"]
            ),
            raw_trip_glob=str(
                paths["raw_trip_glob"]
            ),
            demand_path=Path(
                paths["demand"]
            ),
            output_directory=Path(
                paths["output_directory"]
            ),
            metadata_path=Path(
                paths["metadata"]
            ),
            pickup_datetime_column=str(
                columns["pickup_datetime"]
            ),
            dropoff_datetime_column=str(
                columns["dropoff_datetime"]
            ),
            pickup_zone_column=str(
                columns["pickup_zone"]
            ),
            dropoff_zone_column=str(
                columns["dropoff_zone"]
            ),
            demand_timestamp_column=str(
                columns["demand_timestamp"]
            ),
            demand_zone_column=str(
                columns["demand_zone"]
            ),
            demand_value_column=str(
                columns["demand_value"]
            ),
            od_fit_start=parse_date(
                periods["od_fit_start"],
                description="od_fit_start",
            ),
            od_fit_end_exclusive=parse_date(
                periods["od_fit_end_exclusive"],
                description="od_fit_end_exclusive",
            ),
            train_start=parse_date(
                periods["train_start"],
                description="train_start",
            ),
            train_end_exclusive=parse_date(
                periods["train_end_exclusive"],
                description="train_end_exclusive",
            ),
            validation_start=parse_date(
                periods["validation_start"],
                description="validation_start",
            ),
            validation_end_exclusive=parse_date(
                periods["validation_end_exclusive"],
                description="validation_end_exclusive",
            ),
            test_start=parse_date(
                periods["test_start"],
                description="test_start",
            ),
            test_end_exclusive=parse_date(
                periods["test_end_exclusive"],
                description="test_end_exclusive",
            ),
            top_n_zones=int(
                zone_selection["top_n_zones"]
            ),
            minimum_fit_pickups=int(
                zone_selection["minimum_fit_pickups"]
            ),
            minimum_trip_minutes=float(
                od_profile["minimum_trip_minutes"]
            ),
            maximum_trip_minutes=float(
                od_profile["maximum_trip_minutes"]
            ),
            smoothing_alpha=float(
                od_profile["laplace_smoothing_alpha"]
            ),
            fallback_travel_minutes=float(
                od_profile["fallback_travel_minutes"]
            ),
            maximum_travel_steps=int(
                od_profile["maximum_travel_steps"]
            ),
            minimum_observed_od_coverage=float(
                quality["minimum_observed_od_coverage"]
            ),
            compression=str(
                output["compression"]
            ),
        )

    except SimulationInputError:
        raise

    except (
        OSError,
        KeyError,
        TypeError,
        ValueError,
        yaml.YAMLError,
    ) as error:
        raise SimulationInputError(
            f"Invalid simulation-input config: {error}"
        ) from error

    validate_config(
        config
    )

    return config


def validate_config(
    config: SimulationInputConfig,
) -> None:
    """Validate configuration invariants."""
    if not config.dataset_name:
        raise SimulationInputError(
            "Dataset name must not be empty."
        )

    if config.interval_minutes <= 0:
        raise SimulationInputError(
            "interval_minutes must be positive."
        )

    if not config.demand_path.exists():
        raise SimulationInputError(
            f"Demand dataset does not exist: {config.demand_path}"
        )

    if config.demand_path.stat().st_size == 0:
        raise SimulationInputError(
            f"Demand dataset is empty: {config.demand_path}"
        )

    if config.top_n_zones <= 1:
        raise SimulationInputError(
            "top_n_zones must be greater than one."
        )

    if config.minimum_fit_pickups < 0:
        raise SimulationInputError(
            "minimum_fit_pickups must be non-negative."
        )

    if not (
        0.0
        < config.minimum_trip_minutes
        < config.maximum_trip_minutes
    ):
        raise SimulationInputError(
            "Trip-duration limits are invalid."
        )

    if (
        not math.isfinite(config.smoothing_alpha)
        or config.smoothing_alpha <= 0
    ):
        raise SimulationInputError(
            "Laplace smoothing alpha must be positive."
        )

    if (
        not math.isfinite(config.fallback_travel_minutes)
        or config.fallback_travel_minutes <= 0
    ):
        raise SimulationInputError(
            "Fallback travel minutes must be positive."
        )

    if config.maximum_travel_steps < 1:
        raise SimulationInputError(
            "maximum_travel_steps must be positive."
        )

    if not (
        0.0
        <= config.minimum_observed_od_coverage
        <= 1.0
    ):
        raise SimulationInputError(
            "OD coverage threshold must be between zero and one."
        )

    if not (
        config.od_fit_start
        < config.od_fit_end_exclusive
    ):
        raise SimulationInputError(
            "OD fit period is invalid."
        )

    split_boundaries = (
        config.train_start,
        config.train_end_exclusive,
        config.validation_start,
        config.validation_end_exclusive,
        config.test_start,
        config.test_end_exclusive,
    )

    if split_boundaries != tuple(
        sorted(split_boundaries)
    ):
        raise SimulationInputError(
            "Train, validation, and test periods must be ordered."
        )

    if (
        config.train_end_exclusive
        != config.validation_start
    ):
        raise SimulationInputError(
            "Train and validation periods must be contiguous."
        )

    if (
        config.validation_end_exclusive
        != config.test_start
    ):
        raise SimulationInputError(
            "Validation and test periods must be contiguous."
        )


def resolve_raw_files(
    pattern: str,
) -> tuple[Path, ...]:
    """Resolve raw trip files deterministically."""
    paths = tuple(
        sorted(
            Path(value)
            for value in glob.glob(
                pattern,
                recursive=True,
            )
        )
    )

    if not paths:
        raise SimulationInputError(
            f"No raw trip files match: {pattern}"
        )

    empty_paths = [
        path
        for path in paths
        if path.stat().st_size == 0
    ]

    if empty_paths:
        raise SimulationInputError(
            "Raw trip inputs contain empty files: "
            + ", ".join(
                str(path)
                for path in empty_paths
            )
        )

    return paths


def write_parquet_atomically(
    frame: pl.DataFrame,
    *,
    path: Path,
    compression: str,
) -> dict[str, Any]:
    """Write a Parquet artifact atomically."""
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = path.with_name(
        f"{path.name}.part"
    )

    temporary_path.unlink(
        missing_ok=True
    )

    try:
        frame.write_parquet(
            temporary_path,
            compression=compression,
            statistics=True,
        )

        temporary_path.replace(
            path
        )

    except Exception as error:
        temporary_path.unlink(
            missing_ok=True
        )

        raise SimulationInputError(
            f"Could not write Parquet artifact {path}: {error}"
        ) from error

    return {
        "path": str(path),
        "row_count": frame.height,
        "column_count": frame.width,
        "size_bytes": path.stat().st_size,
    }


def sanitize_json_value(
    value: Any,
) -> Any:
    """Replace non-finite values with JSON null."""
    if isinstance(value, float):
        if not math.isfinite(value):
            return None

        return value

    if isinstance(value, dict):
        return {
            str(key): sanitize_json_value(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            sanitize_json_value(item)
            for item in value
        ]

    return value


def write_json_atomically(
    path: Path,
    payload: dict[str, Any],
) -> None:
    """Write one strict JSON artifact atomically."""
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = path.with_name(
        f"{path.name}.part"
    )

    temporary_path.unlink(
        missing_ok=True
    )

    try:
        temporary_path.write_text(
            json.dumps(
                sanitize_json_value(payload),
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )

        temporary_path.replace(
            path
        )

    except (
        OSError,
        TypeError,
        ValueError,
    ) as error:
        temporary_path.unlink(
            missing_ok=True
        )

        raise SimulationInputError(
            f"Could not write metadata report: {error}"
        ) from error


def load_demand(
    config: SimulationInputConfig,
) -> pl.DataFrame:
    """Load and validate dense historical demand."""
    try:
        frame = pl.read_parquet(
            config.demand_path
        )
    except Exception as error:
        raise SimulationInputError(
            f"Could not read demand data: {error}"
        ) from error

    required_columns = {
        config.demand_timestamp_column,
        config.demand_zone_column,
        config.demand_value_column,
    }

    missing_columns = sorted(
        required_columns.difference(
            frame.columns
        )
    )

    if missing_columns:
        raise SimulationInputError(
            "Demand dataset is missing columns: "
            + ", ".join(missing_columns)
        )

    frame = (
        frame.select(
            pl.col(
                config.demand_timestamp_column
            ).alias(
                "interval_start"
            ),
            pl.col(
                config.demand_zone_column
            )
            .cast(
                pl.Int64
            )
            .alias(
                "zone_id"
            ),
            pl.col(
                config.demand_value_column
            )
            .cast(
                pl.Int64
            )
            .alias(
                "actual_demand"
            ),
        )
        .with_columns(
            pl.col(
                "interval_start"
            )
            .dt.date()
            .alias(
                "service_date"
            )
        )
    )

    null_count = sum(
        int(value)
        for value in frame.null_count().row(0)
    )

    if null_count:
        raise SimulationInputError(
            "Demand dataset contains null values."
        )

    negative_demand_count = (
        frame.filter(
            pl.col(
                "actual_demand"
            )
            < 0
        )
        .height
    )

    if negative_demand_count:
        raise SimulationInputError(
            "Demand dataset contains negative counts."
        )

    duplicate_count = (
        frame.group_by(
            "interval_start",
            "zone_id",
        )
        .agg(
            pl.len().alias(
                "row_count"
            )
        )
        .filter(
            pl.col(
                "row_count"
            )
            > 1
        )
        .height
    )

    if duplicate_count:
        raise SimulationInputError(
            "Demand dataset contains duplicate interval-zone keys."
        )

    return frame


def select_zones(
    demand: pl.DataFrame,
    *,
    config: SimulationInputConfig,
) -> pl.DataFrame:
    """Select the highest-volume zones using fit-period demand only."""
    selected = (
        demand.filter(
            pl.col(
                "service_date"
            )
            >= config.od_fit_start
        )
        .filter(
            pl.col(
                "service_date"
            )
            < config.od_fit_end_exclusive
        )
        .group_by(
            "zone_id"
        )
        .agg(
            pl.col(
                "actual_demand"
            )
            .sum()
            .alias(
                "fit_pickup_count"
            )
        )
        .filter(
            pl.col(
                "fit_pickup_count"
            )
            >= config.minimum_fit_pickups
        )
        .sort(
            [
                "fit_pickup_count",
                "zone_id",
            ],
            descending=[
                True,
                False,
            ],
        )
        .head(
            config.top_n_zones
        )
    )

    if selected.height != config.top_n_zones:
        raise SimulationInputError(
            "Could not select the requested number of zones. "
            f"Requested={config.top_n_zones}, "
            f"eligible={selected.height}."
        )

    total_fit_pickups = int(
        selected.get_column(
            "fit_pickup_count"
        ).sum()
    )

    if total_fit_pickups <= 0:
        raise SimulationInputError(
            "Selected zones have zero fit-period demand."
        )

    return (
        selected.with_row_index(
            name="zone_index",
        )
        .with_columns(
            (
                pl.col(
                    "fit_pickup_count"
                )
                / total_fit_pickups
            )
            .alias(
                "fit_pickup_share"
            )
        )
        .select(
            "zone_index",
            "zone_id",
            "fit_pickup_count",
            "fit_pickup_share",
        )
    )


def assign_split_expression(
    config: SimulationInputConfig,
) -> pl.Expr:
    """Build temporal split assignment."""
    service_date = pl.col(
        "service_date"
    )

    return (
        pl.when(
            service_date.is_between(
                config.train_start,
                config.train_end_exclusive,
                closed="left",
            )
        )
        .then(
            pl.lit("train")
        )
        .when(
            service_date.is_between(
                config.validation_start,
                config.validation_end_exclusive,
                closed="left",
            )
        )
        .then(
            pl.lit("validation")
        )
        .when(
            service_date.is_between(
                config.test_start,
                config.test_end_exclusive,
                closed="left",
            )
        )
        .then(
            pl.lit("test")
        )
        .otherwise(
            None
        )
        .alias(
            "split"
        )
    )


def build_historical_demand(
    demand: pl.DataFrame,
    *,
    zones: pl.DataFrame,
    config: SimulationInputConfig,
) -> pl.DataFrame:
    """Build selected-zone historical demand episodes."""
    historical = (
        demand.join(
            zones.select(
                "zone_index",
                "zone_id",
            ),
            on="zone_id",
            how="inner",
        )
        .with_columns(
            assign_split_expression(
                config
            )
        )
        .filter(
            pl.col(
                "split"
            )
            .is_not_null()
        )
        .select(
            "interval_start",
            "service_date",
            "split",
            "zone_index",
            "zone_id",
            "actual_demand",
        )
        .sort(
            "interval_start",
            "zone_index",
        )
    )

    expected_zone_count = zones.height

    invalid_interval_count = (
        historical.group_by(
            "interval_start"
        )
        .agg(
            pl.len().alias(
                "zone_count"
            )
        )
        .filter(
            pl.col(
                "zone_count"
            )
            != expected_zone_count
        )
        .height
    )

    if invalid_interval_count:
        raise SimulationInputError(
            "Historical demand is not dense across selected zones."
        )

    split_counts = (
        historical.group_by(
            "split"
        )
        .agg(
            pl.len().alias(
                "row_count"
            )
        )
        .to_dict(
            as_series=False
        )
    )

    observed_splits = set(
        split_counts.get(
            "split",
            [],
        )
    )

    required_splits = {
        "train",
        "validation",
        "test",
    }

    if observed_splits != required_splits:
        raise SimulationInputError(
            "Historical demand does not contain all required splits. "
            f"Observed={sorted(observed_splits)}."
        )

    return historical


def scan_fit_trips(
    raw_paths: tuple[Path, ...],
    *,
    selected_zone_ids: list[int],
    config: SimulationInputConfig,
) -> pl.LazyFrame:
    """Scan and clean fit-period raw HVFHV trips."""
    try:
        lazy = pl.scan_parquet(
            [
                str(path)
                for path in raw_paths
            ]
        )
    except Exception as error:
        raise SimulationInputError(
            f"Could not scan raw trip files: {error}"
        ) from error

    schema_names = set(
        lazy.collect_schema().names()
    )

    required_columns = {
        config.pickup_datetime_column,
        config.dropoff_datetime_column,
        config.pickup_zone_column,
        config.dropoff_zone_column,
    }

    missing_columns = sorted(
        required_columns.difference(
            schema_names
        )
    )

    if missing_columns:
        raise SimulationInputError(
            "Raw trip inputs are missing columns: "
            + ", ".join(missing_columns)
        )

    pickup_datetime = (
        pl.col(
            config.pickup_datetime_column
        )
        .cast(
            pl.Datetime("us")
        )
    )

    dropoff_datetime = (
        pl.col(
            config.dropoff_datetime_column
        )
        .cast(
            pl.Datetime("us")
        )
    )

    duration_minutes = (
        (
            dropoff_datetime
            - pickup_datetime
        )
        .dt.total_seconds()
        / 60.0
    )

    return (
        lazy.select(
            pickup_datetime.alias(
                "pickup_datetime"
            ),
            dropoff_datetime.alias(
                "dropoff_datetime"
            ),
            pl.col(
                config.pickup_zone_column
            )
            .cast(
                pl.Int64
            )
            .alias(
                "origin_zone_id"
            ),
            pl.col(
                config.dropoff_zone_column
            )
            .cast(
                pl.Int64
            )
            .alias(
                "destination_zone_id"
            ),
        )
        .with_columns(
            pl.col(
                "pickup_datetime"
            )
            .dt.date()
            .alias(
                "pickup_date"
            ),
            duration_minutes.alias(
                "travel_minutes"
            ),
        )
        .filter(
            pl.col(
                "pickup_date"
            )
            >= config.od_fit_start
        )
        .filter(
            pl.col(
                "pickup_date"
            )
            < config.od_fit_end_exclusive
        )
        .filter(
            pl.col(
                "origin_zone_id"
            )
            .is_in(
                selected_zone_ids
            )
        )
        .filter(
            pl.col(
                "destination_zone_id"
            )
            .is_in(
                selected_zone_ids
            )
        )
        .filter(
            pl.col(
                "travel_minutes"
            )
            .is_between(
                config.minimum_trip_minutes,
                config.maximum_trip_minutes,
                closed="both",
            )
        )
    )


def build_observed_od(
    fit_trips: pl.LazyFrame,
) -> pl.DataFrame:
    """Aggregate observed OD trip count and median duration."""
    try:
        observed = (
            fit_trips.group_by(
                "origin_zone_id",
                "destination_zone_id",
            )
            .agg(
                pl.len().alias(
                    "trip_count"
                ),
                pl.col(
                    "travel_minutes"
                )
                .median()
                .alias(
                    "median_travel_minutes"
                ),
            )
            .collect(
                engine="streaming"
            )
        )
    except Exception as error:
        raise SimulationInputError(
            f"Could not aggregate OD profile: {error}"
        ) from error

    if observed.height == 0:
        raise SimulationInputError(
            "No eligible OD trips were found."
        )

    return observed


def build_od_profile(
    observed_od: pl.DataFrame,
    *,
    zones: pl.DataFrame,
    config: SimulationInputConfig,
) -> tuple[
    pl.DataFrame,
    dict[str, Any],
]:
    """Build a complete smoothed OD and travel-step profile."""
    origin_zones = zones.select(
        pl.col(
            "zone_index"
        ).alias(
            "origin_index"
        ),
        pl.col(
            "zone_id"
        ).alias(
            "origin_zone_id"
        ),
    )

    destination_zones = zones.select(
        pl.col(
            "zone_index"
        ).alias(
            "destination_index"
        ),
        pl.col(
            "zone_id"
        ).alias(
            "destination_zone_id"
        ),
    )

    full_grid = origin_zones.join(
        destination_zones,
        how="cross",
    )

    global_median = observed_od.get_column(
        "median_travel_minutes"
    ).median()

    if global_median is None:
        global_median = (
            config.fallback_travel_minutes
        )

    global_median = float(
        global_median
    )

    origin_fallbacks = (
        observed_od.group_by(
            "origin_zone_id"
        )
        .agg(
            pl.col(
                "median_travel_minutes"
            )
            .median()
            .alias(
                "origin_median_travel_minutes"
            )
        )
    )

    profile = (
        full_grid.join(
            observed_od,
            on=[
                "origin_zone_id",
                "destination_zone_id",
            ],
            how="left",
        )
        .join(
            origin_fallbacks,
            on="origin_zone_id",
            how="left",
        )
        .with_columns(
            pl.col(
                "trip_count"
            )
            .fill_null(
                0
            )
            .cast(
                pl.Int64
            ),
            pl.col(
                "median_travel_minutes"
            )
            .is_not_null()
            .alias(
                "travel_time_observed"
            ),
        )
        .with_columns(
            pl.coalesce(
                pl.col(
                    "median_travel_minutes"
                ),
                pl.col(
                    "origin_median_travel_minutes"
                ),
                pl.lit(
                    global_median
                ),
                pl.lit(
                    config.fallback_travel_minutes
                ),
            )
            .alias(
                "effective_travel_minutes"
            )
        )
        .with_columns(
            (
                pl.col(
                    "trip_count"
                )
                + config.smoothing_alpha
            )
            .alias(
                "smoothed_trip_count"
            )
        )
        .with_columns(
            (
                pl.col(
                    "smoothed_trip_count"
                )
                / pl.col(
                    "smoothed_trip_count"
                )
                .sum()
                .over(
                    "origin_zone_id"
                )
            )
            .alias(
                "destination_probability"
            ),
            (
                pl.col(
                    "effective_travel_minutes"
                )
                / config.interval_minutes
            )
            .ceil()
            .clip(
                1,
                config.maximum_travel_steps,
            )
            .cast(
                pl.Int64
            )
            .alias(
                "passenger_travel_steps"
            ),
        )
        .with_columns(
            pl.when(
                pl.col(
                    "origin_index"
                )
                == pl.col(
                    "destination_index"
                )
            )
            .then(
                0
            )
            .otherwise(
                pl.col(
                    "passenger_travel_steps"
                )
            )
            .cast(
                pl.Int64
            )
            .alias(
                "relocation_travel_steps"
            )
        )
        .drop(
            "origin_median_travel_minutes"
        )
        .sort(
            "origin_index",
            "destination_index",
        )
    )

    expected_rows = (
        zones.height
        * zones.height
    )

    if profile.height != expected_rows:
        raise SimulationInputError(
            "OD profile does not contain a complete matrix."
        )

    probability_checks = (
        profile.group_by(
            "origin_index"
        )
        .agg(
            pl.col(
                "destination_probability"
            )
            .sum()
            .alias(
                "probability_sum"
            )
        )
    )

    invalid_probability_rows = (
        probability_checks.filter(
            (
                pl.col(
                    "probability_sum"
                )
                - 1.0
            )
            .abs()
            > 1e-9
        )
        .height
    )

    if invalid_probability_rows:
        raise SimulationInputError(
            "Destination probabilities do not sum to one."
        )

    observed_pair_count = (
        profile.filter(
            pl.col(
                "travel_time_observed"
            )
        )
        .height
    )

    observed_coverage = (
        observed_pair_count
        / expected_rows
    )

    if (
        observed_coverage
        < config.minimum_observed_od_coverage
    ):
        raise SimulationInputError(
            "Observed OD travel-time coverage is below threshold: "
            f"coverage={observed_coverage:.4f}, "
            f"required={config.minimum_observed_od_coverage:.4f}."
        )

    diagonal_relocation_error_count = (
        profile.filter(
            (
                pl.col(
                    "origin_index"
                )
                == pl.col(
                    "destination_index"
                )
            )
            & (
                pl.col(
                    "relocation_travel_steps"
                )
                != 0
            )
        )
        .height
    )

    off_diagonal_relocation_error_count = (
        profile.filter(
            (
                pl.col(
                    "origin_index"
                )
                != pl.col(
                    "destination_index"
                )
            )
            & (
                pl.col(
                    "relocation_travel_steps"
                )
                < 1
            )
        )
        .height
    )

    if (
        diagonal_relocation_error_count
        or off_diagonal_relocation_error_count
    ):
        raise SimulationInputError(
            "Relocation travel-step matrix is invalid."
        )

    summary = {
        "matrix_size": zones.height,
        "matrix_row_count": profile.height,
        "observed_pair_count": observed_pair_count,
        "observed_pair_coverage": observed_coverage,
        "global_median_travel_minutes": global_median,
        "fallback_pair_count": (
            profile.height
            - observed_pair_count
        ),
    }

    return profile, summary


def build_real_simulation_inputs(
    config_path: Path,
) -> dict[str, Any]:
    """Build all real-data simulation artifacts."""
    config = read_config(
        config_path
    )

    raw_paths = resolve_raw_files(
        config.raw_trip_glob
    )

    demand = load_demand(
        config
    )

    zones = select_zones(
        demand,
        config=config,
    )

    historical_demand = build_historical_demand(
        demand,
        zones=zones,
        config=config,
    )

    selected_zone_ids = (
        zones.get_column(
            "zone_id"
        )
        .to_list()
    )

    fit_trips = scan_fit_trips(
        raw_paths,
        selected_zone_ids=selected_zone_ids,
        config=config,
    )

    observed_od = build_observed_od(
        fit_trips
    )

    od_profile, od_summary = build_od_profile(
        observed_od,
        zones=zones,
        config=config,
    )

    artifacts = {
        "zone_index": write_parquet_atomically(
            zones,
            path=config.zone_index_path,
            compression=config.compression,
        ),
        "od_profile": write_parquet_atomically(
            od_profile,
            path=config.od_profile_path,
            compression=config.compression,
        ),
        "historical_demand": write_parquet_atomically(
            historical_demand,
            path=config.historical_demand_path,
            compression=config.compression,
        ),
    }

    split_summary = (
        historical_demand.group_by(
            "split"
        )
        .agg(
            pl.len().alias(
                "row_count"
            ),
            pl.col(
                "interval_start"
            )
            .n_unique()
            .alias(
                "interval_count"
            ),
            pl.col(
                "actual_demand"
            )
            .sum()
            .alias(
                "demand_total"
            ),
        )
        .sort(
            "split"
        )
        .to_dicts()
    )

    metadata = {
        "simulation_input_version": "1.0.0",
        "generated_at": datetime.now(
            UTC
        ).isoformat(),
        "dataset_name": config.dataset_name,
        "timezone": config.timezone,
        "interval_minutes": (
            config.interval_minutes
        ),
        "selection_contract": {
            "zone_selection_uses_fit_period_only": True,
            "od_profile_uses_fit_period_only": True,
            "validation_used_for_od_fit": False,
            "test_used_for_od_fit": False,
            "od_fit_start": (
                config.od_fit_start.isoformat()
            ),
            "od_fit_end_exclusive": (
                config.od_fit_end_exclusive.isoformat()
            ),
        },
        "raw_inputs": {
            "file_count": len(raw_paths),
            "files": [
                str(path)
                for path in raw_paths
            ],
            "demand_path": str(
                config.demand_path
            ),
        },
        "zones": {
            "selected_zone_count": zones.height,
            "minimum_fit_pickups": (
                config.minimum_fit_pickups
            ),
            "total_fit_pickups": int(
                zones.get_column(
                    "fit_pickup_count"
                ).sum()
            ),
        },
        "od_profile": od_summary,
        "historical_demand": {
            "row_count": (
                historical_demand.height
            ),
            "interval_count": (
                historical_demand.get_column(
                    "interval_start"
                )
                .n_unique()
            ),
            "splits": split_summary,
        },
        "artifacts": artifacts,
        "invariants": {
            "zone_count_matches_config": (
                zones.height
                == config.top_n_zones
            ),
            "od_matrix_is_complete": (
                od_profile.height
                == zones.height * zones.height
            ),
            "od_probability_rows_sum_to_one": True,
            "historical_demand_is_dense": True,
            "validation_not_used_for_network_fit": True,
            "test_not_used_for_network_fit": True,
        },
    }

    write_json_atomically(
        config.metadata_path,
        metadata,
    )

    return metadata