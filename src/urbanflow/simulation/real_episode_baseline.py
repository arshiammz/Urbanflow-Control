from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import yaml
from numpy.typing import NDArray

from urbanflow.simulation.fleet_simulator import (
    FleetSimulator,
    SimulationCosts,
    allocate_integer_flow,
)


FloatVector = NDArray[np.float64]
FloatMatrix = NDArray[np.float64]
IntegerVector = NDArray[np.int64]
IntegerMatrix = NDArray[np.int64]


class RealEpisodeBaselineError(RuntimeError):
    """Raised when the real-data baseline episode fails."""


@dataclass(frozen=True)
class RealEpisodeBaselineConfig:
    """Validated real-data episode configuration."""

    dataset_name: str
    timezone: str
    interval_minutes: int

    zone_index_path: Path
    od_profile_path: Path
    historical_demand_path: Path

    step_metrics_path: Path
    interval_zone_metrics_path: Path
    zone_summary_path: Path
    report_path: Path

    evaluation_split: str
    service_date: date
    warmup_days: int

    sizing_split: str
    reserve_multiplier: float
    minimum_vehicles_per_zone: int
    initial_distribution_column: str

    unmet_demand_penalty: float
    relocation_cost_per_vehicle_step: float
    idle_vehicle_cost: float

    policy_name: str
    compression: str


@dataclass(frozen=True)
class NetworkInputs:
    """Dense zone-network inputs for the simulator."""

    zone_ids: tuple[int, ...]
    fit_pickup_shares: FloatVector

    destination_probabilities: FloatMatrix
    passenger_travel_steps: IntegerMatrix
    relocation_travel_steps: IntegerMatrix


@dataclass(frozen=True)
class FleetSizingResult:
    """Result of training-only fleet sizing."""

    training_mean_demand: FloatVector
    expected_travel_steps: FloatVector

    estimated_busy_vehicles: float
    reserve_multiplier: float

    total_fleet: int
    initial_vehicles: IntegerVector

    def as_dict(
        self,
        *,
        zone_ids: tuple[int, ...],
    ) -> dict[str, Any]:
        """Return JSON-compatible fleet-sizing metadata."""
        return {
            "method": (
                "little_law_expected_concurrent_vehicles"
            ),
            "estimated_busy_vehicles": (
                self.estimated_busy_vehicles
            ),
            "reserve_multiplier": (
                self.reserve_multiplier
            ),
            "total_fleet": self.total_fleet,
            "zone_allocations": [
                {
                    "zone_id": zone_id,
                    "training_mean_demand": float(
                        self.training_mean_demand[
                            zone_index
                        ]
                    ),
                    "expected_travel_steps": float(
                        self.expected_travel_steps[
                            zone_index
                        ]
                    ),
                    "initial_vehicles": int(
                        self.initial_vehicles[
                            zone_index
                        ]
                    ),
                }
                for zone_index, zone_id in enumerate(
                    zone_ids
                )
            ],
        }


def parse_date(
    value: Any,
    *,
    description: str,
) -> date:
    """Parse a configuration date."""
    if isinstance(
        value,
        datetime,
    ):
        return value.date()

    if isinstance(
        value,
        date,
    ):
        return value

    try:
        return date.fromisoformat(
            str(value)
        )

    except ValueError as error:
        raise RealEpisodeBaselineError(
            f"{description} must use YYYY-MM-DD format."
        ) from error


def read_config(
    path: Path,
) -> RealEpisodeBaselineConfig:
    """Read and validate baseline episode configuration."""
    if not path.exists():
        raise RealEpisodeBaselineError(
            f"Episode config does not exist: {path}"
        )

    try:
        raw = yaml.safe_load(
            path.read_text(
                encoding="utf-8",
            )
        )

        if not isinstance(
            raw,
            dict,
        ):
            raise RealEpisodeBaselineError(
                "Episode config must contain a YAML mapping."
            )

        dataset = raw["dataset"]
        paths = raw["paths"]
        episode = raw["episode"]
        fleet = raw["fleet"]
        costs = raw["costs"]
        policy = raw["policy"]
        output = raw["output"]

        config = RealEpisodeBaselineConfig(
            dataset_name=str(
                dataset["name"]
            ),
            timezone=str(
                dataset["timezone"]
            ),
            interval_minutes=int(
                dataset["interval_minutes"]
            ),
            zone_index_path=Path(
                paths["zone_index"]
            ),
            od_profile_path=Path(
                paths["od_profile"]
            ),
            historical_demand_path=Path(
                paths["historical_demand"]
            ),
            step_metrics_path=Path(
                paths["step_metrics"]
            ),
            interval_zone_metrics_path=Path(
                paths["interval_zone_metrics"]
            ),
            zone_summary_path=Path(
                paths["zone_summary"]
            ),
            report_path=Path(
                paths["report"]
            ),
            evaluation_split=str(
                episode["evaluation_split"]
            ),
            service_date=parse_date(
                episode["service_date"],
                description="service_date",
            ),
            warmup_days=int(
                episode["warmup_days"]
            ),
            sizing_split=str(
                fleet["sizing_split"]
            ),
            reserve_multiplier=float(
                fleet["reserve_multiplier"]
            ),
            minimum_vehicles_per_zone=int(
                fleet["minimum_vehicles_per_zone"]
            ),
            initial_distribution_column=str(
                fleet[
                    "initial_distribution_column"
                ]
            ),
            unmet_demand_penalty=float(
                costs["unmet_demand_penalty"]
            ),
            relocation_cost_per_vehicle_step=float(
                costs[
                    "relocation_cost_per_vehicle_step"
                ]
            ),
            idle_vehicle_cost=float(
                costs["idle_vehicle_cost"]
            ),
            policy_name=str(
                policy["name"]
            ),
            compression=str(
                output["compression"]
            ),
        )

    except RealEpisodeBaselineError:
        raise

    except (
        OSError,
        KeyError,
        TypeError,
        ValueError,
        yaml.YAMLError,
    ) as error:
        raise RealEpisodeBaselineError(
            f"Invalid episode config: {error}"
        ) from error

    validate_config(
        config
    )

    return config


def validate_config(
    config: RealEpisodeBaselineConfig,
) -> None:
    """Validate configuration invariants."""
    if not config.dataset_name:
        raise RealEpisodeBaselineError(
            "Dataset name must not be empty."
        )

    if config.interval_minutes <= 0:
        raise RealEpisodeBaselineError(
            "interval_minutes must be positive."
        )

    if config.warmup_days < 0:
        raise RealEpisodeBaselineError(
            "warmup_days must be non-negative."
        )

    if (
        not math.isfinite(
            config.reserve_multiplier
        )
        or config.reserve_multiplier < 1.0
    ):
        raise RealEpisodeBaselineError(
            "reserve_multiplier must be finite "
            "and at least one."
        )

    if config.minimum_vehicles_per_zone < 0:
        raise RealEpisodeBaselineError(
            "minimum_vehicles_per_zone "
            "must be non-negative."
        )

    if not config.evaluation_split:
        raise RealEpisodeBaselineError(
            "evaluation_split must not be empty."
        )

    if not config.sizing_split:
        raise RealEpisodeBaselineError(
            "sizing_split must not be empty."
        )

    if not config.initial_distribution_column:
        raise RealEpisodeBaselineError(
            "initial_distribution_column "
            "must not be empty."
        )

    if config.policy_name != "no_rebalancing":
        raise RealEpisodeBaselineError(
            "This runner only supports the "
            "no_rebalancing policy."
        )

    if not config.compression:
        raise RealEpisodeBaselineError(
            "Output compression must not be empty."
        )

    for name, value in (
        (
            "unmet_demand_penalty",
            config.unmet_demand_penalty,
        ),
        (
            "relocation_cost_per_vehicle_step",
            config.relocation_cost_per_vehicle_step,
        ),
        (
            "idle_vehicle_cost",
            config.idle_vehicle_cost,
        ),
    ):
        if (
            not math.isfinite(
                value
            )
            or value < 0
        ):
            raise RealEpisodeBaselineError(
                f"{name} must be finite and non-negative."
            )

    for input_path in (
        config.zone_index_path,
        config.od_profile_path,
        config.historical_demand_path,
    ):
        if not input_path.exists():
            raise RealEpisodeBaselineError(
                f"Required input does not exist: {input_path}"
            )

        if input_path.stat().st_size == 0:
            raise RealEpisodeBaselineError(
                f"Required input is empty: {input_path}"
            )


def validate_no_nulls(
    frame: pl.DataFrame,
    *,
    description: str,
) -> None:
    """Reject a frame containing null values."""
    null_count = sum(
        int(value)
        for value in frame.null_count().row(0)
    )

    if null_count:
        raise RealEpisodeBaselineError(
            f"{description} contains {null_count} null values."
        )


def load_zone_index(
    config: RealEpisodeBaselineConfig,
) -> tuple[
    pl.DataFrame,
    tuple[int, ...],
    FloatVector,
]:
    """Load and validate deterministic zone ordering."""
    try:
        frame = pl.read_parquet(
            config.zone_index_path
        )

    except Exception as error:
        raise RealEpisodeBaselineError(
            f"Could not read zone index: {error}"
        ) from error

    required_columns = {
        "zone_index",
        "zone_id",
        config.initial_distribution_column,
    }

    missing_columns = sorted(
        required_columns.difference(
            frame.columns
        )
    )

    if missing_columns:
        raise RealEpisodeBaselineError(
            "Zone index is missing columns: "
            + ", ".join(
                missing_columns
            )
        )

    frame = (
        frame.select(
            pl.col(
                "zone_index"
            )
            .cast(
                pl.Int64
            ),
            pl.col(
                "zone_id"
            )
            .cast(
                pl.Int64
            ),
            pl.col(
                config.initial_distribution_column
            )
            .cast(
                pl.Float64
            )
            .alias(
                "initial_distribution_weight"
            ),
        )
        .sort(
            "zone_index"
        )
    )

    validate_no_nulls(
        frame,
        description="Zone index",
    )

    if frame.height <= 1:
        raise RealEpisodeBaselineError(
            "At least two simulation zones are required."
        )

    expected_indices = list(
        range(
            frame.height
        )
    )

    observed_indices = frame.get_column(
        "zone_index"
    ).to_list()

    if observed_indices != expected_indices:
        raise RealEpisodeBaselineError(
            "zone_index must be contiguous and zero-based."
        )

    if (
        frame.get_column(
            "zone_id"
        )
        .n_unique()
        != frame.height
    ):
        raise RealEpisodeBaselineError(
            "zone_id values must be unique."
        )

    weights = (
        frame.get_column(
            "initial_distribution_weight"
        )
        .to_numpy()
        .astype(
            np.float64,
            copy=False,
        )
    )

    if not np.isfinite(
        weights
    ).all():
        raise RealEpisodeBaselineError(
            "Initial distribution weights "
            "contain non-finite values."
        )

    if (
        weights
        < 0
    ).any():
        raise RealEpisodeBaselineError(
            "Initial distribution weights "
            "contain negative values."
        )

    weight_sum = float(
        weights.sum()
    )

    if weight_sum <= 0:
        raise RealEpisodeBaselineError(
            "Initial distribution weights sum to zero."
        )

    weights = (
        weights
        / weight_sum
    )

    zone_ids = tuple(
        int(value)
        for value in frame.get_column(
            "zone_id"
        ).to_list()
    )

    return (
        frame,
        zone_ids,
        weights,
    )


def load_od_matrices(
    config: RealEpisodeBaselineConfig,
    *,
    zone_count: int,
) -> tuple[
    FloatMatrix,
    IntegerMatrix,
    IntegerMatrix,
]:
    """Load and validate dense OD matrices."""
    try:
        frame = pl.read_parquet(
            config.od_profile_path
        )

    except Exception as error:
        raise RealEpisodeBaselineError(
            f"Could not read OD profile: {error}"
        ) from error

    required_columns = {
        "origin_index",
        "destination_index",
        "destination_probability",
        "passenger_travel_steps",
        "relocation_travel_steps",
    }

    missing_columns = sorted(
        required_columns.difference(
            frame.columns
        )
    )

    if missing_columns:
        raise RealEpisodeBaselineError(
            "OD profile is missing columns: "
            + ", ".join(
                missing_columns
            )
        )

    frame = (
        frame.select(
            pl.col(
                "origin_index"
            )
            .cast(
                pl.Int64
            ),
            pl.col(
                "destination_index"
            )
            .cast(
                pl.Int64
            ),
            pl.col(
                "destination_probability"
            )
            .cast(
                pl.Float64
            ),
            pl.col(
                "passenger_travel_steps"
            )
            .cast(
                pl.Int64
            ),
            pl.col(
                "relocation_travel_steps"
            )
            .cast(
                pl.Int64
            ),
        )
        .sort(
            "origin_index",
            "destination_index",
        )
    )

    validate_no_nulls(
        frame,
        description="OD profile",
    )

    expected_rows = (
        zone_count
        * zone_count
    )

    if frame.height != expected_rows:
        raise RealEpisodeBaselineError(
            "OD profile is not a complete matrix: "
            f"expected={expected_rows}, "
            f"observed={frame.height}."
        )

    expected_origin_indices = np.repeat(
        np.arange(
            zone_count,
            dtype=np.int64,
        ),
        zone_count,
    )

    expected_destination_indices = np.tile(
        np.arange(
            zone_count,
            dtype=np.int64,
        ),
        zone_count,
    )

    observed_origin_indices = (
        frame.get_column(
            "origin_index"
        )
        .to_numpy()
        .astype(
            np.int64,
            copy=False,
        )
    )

    observed_destination_indices = (
        frame.get_column(
            "destination_index"
        )
        .to_numpy()
        .astype(
            np.int64,
            copy=False,
        )
    )

    if not np.array_equal(
        observed_origin_indices,
        expected_origin_indices,
    ):
        raise RealEpisodeBaselineError(
            "OD origin ordering is invalid."
        )

    if not np.array_equal(
        observed_destination_indices,
        expected_destination_indices,
    ):
        raise RealEpisodeBaselineError(
            "OD destination ordering is invalid."
        )

    probabilities = (
        frame.get_column(
            "destination_probability"
        )
        .to_numpy()
        .reshape(
            zone_count,
            zone_count,
        )
        .astype(
            np.float64,
            copy=False,
        )
    )

    passenger_steps = (
        frame.get_column(
            "passenger_travel_steps"
        )
        .to_numpy()
        .reshape(
            zone_count,
            zone_count,
        )
        .astype(
            np.int64,
            copy=False,
        )
    )

    relocation_steps = (
        frame.get_column(
            "relocation_travel_steps"
        )
        .to_numpy()
        .reshape(
            zone_count,
            zone_count,
        )
        .astype(
            np.int64,
            copy=False,
        )
    )

    if not np.isfinite(
        probabilities
    ).all():
        raise RealEpisodeBaselineError(
            "OD probabilities contain non-finite values."
        )

    if (
        probabilities
        < 0
    ).any():
        raise RealEpisodeBaselineError(
            "OD probabilities contain negative values."
        )

    if not np.allclose(
        probabilities.sum(
            axis=1
        ),
        1.0,
        rtol=0.0,
        atol=1e-9,
    ):
        raise RealEpisodeBaselineError(
            "OD probability rows do not sum to one."
        )

    if (
        passenger_steps
        < 1
    ).any():
        raise RealEpisodeBaselineError(
            "Passenger travel steps must be at least one."
        )

    if (
        np.diag(
            relocation_steps
        )
        != 0
    ).any():
        raise RealEpisodeBaselineError(
            "Relocation-step diagonal must be zero."
        )

    off_diagonal_mask = ~np.eye(
        zone_count,
        dtype=bool,
    )

    if (
        relocation_steps[
            off_diagonal_mask
        ]
        < 1
    ).any():
        raise RealEpisodeBaselineError(
            "Off-diagonal relocation steps "
            "must be at least one."
        )

    return (
        probabilities,
        passenger_steps,
        relocation_steps,
    )


def load_historical_demand(
    config: RealEpisodeBaselineConfig,
    *,
    zone_count: int,
) -> pl.DataFrame:
    """Load and validate historical simulation demand."""
    try:
        frame = pl.read_parquet(
            config.historical_demand_path
        )

    except Exception as error:
        raise RealEpisodeBaselineError(
            f"Could not read historical demand: {error}"
        ) from error

    required_columns = {
        "interval_start",
        "service_date",
        "split",
        "zone_index",
        "zone_id",
        "actual_demand",
    }

    missing_columns = sorted(
        required_columns.difference(
            frame.columns
        )
    )

    if missing_columns:
        raise RealEpisodeBaselineError(
            "Historical demand is missing columns: "
            + ", ".join(
                missing_columns
            )
        )

    frame = (
        frame.select(
            "interval_start",
            pl.col(
                "service_date"
            )
            .cast(
                pl.Date
            ),
            pl.col(
                "split"
            )
            .cast(
                pl.String
            ),
            pl.col(
                "zone_index"
            )
            .cast(
                pl.Int64
            ),
            pl.col(
                "zone_id"
            )
            .cast(
                pl.Int64
            ),
            pl.col(
                "actual_demand"
            )
            .cast(
                pl.Int64
            ),
        )
        .sort(
            "interval_start",
            "zone_index",
        )
    )

    validate_no_nulls(
        frame,
        description="Historical demand",
    )

    if (
        frame.get_column(
            "actual_demand"
        )
        < 0
    ).any():
        raise RealEpisodeBaselineError(
            "Historical demand contains negative values."
        )

    duplicate_count = (
        frame.group_by(
            "interval_start",
            "zone_index",
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
        raise RealEpisodeBaselineError(
            "Historical demand contains duplicate "
            "interval-zone keys."
        )

    invalid_zone_index_count = (
        frame.filter(
            (
                pl.col(
                    "zone_index"
                )
                < 0
            )
            | (
                pl.col(
                    "zone_index"
                )
                >= zone_count
            )
        )
        .height
    )

    if invalid_zone_index_count:
        raise RealEpisodeBaselineError(
            "Historical demand contains invalid zone indices."
        )

    return frame


def estimate_initial_fleet(
    historical_demand: pl.DataFrame,
    *,
    network: NetworkInputs,
    config: RealEpisodeBaselineConfig,
) -> FleetSizingResult:
    """Estimate fleet size from training-only concurrent demand."""
    zone_count = len(
        network.zone_ids
    )

    training = historical_demand.filter(
        pl.col(
            "split"
        )
        == config.sizing_split
    )

    if training.height == 0:
        raise RealEpisodeBaselineError(
            f"Fleet-sizing split is empty: {config.sizing_split}"
        )

    mean_demand_frame = (
        training.group_by(
            "zone_index"
        )
        .agg(
            pl.col(
                "actual_demand"
            )
            .mean()
            .alias(
                "mean_demand"
            )
        )
        .sort(
            "zone_index"
        )
    )

    if mean_demand_frame.height != zone_count:
        raise RealEpisodeBaselineError(
            "Fleet-sizing demand does not cover every zone."
        )

    expected_indices = list(
        range(
            zone_count
        )
    )

    if (
        mean_demand_frame.get_column(
            "zone_index"
        ).to_list()
        != expected_indices
    ):
        raise RealEpisodeBaselineError(
            "Fleet-sizing zone ordering is invalid."
        )

    training_mean_demand = (
        mean_demand_frame.get_column(
            "mean_demand"
        )
        .to_numpy()
        .astype(
            np.float64,
            copy=False,
        )
    )

    expected_travel_steps = np.sum(
        network.destination_probabilities
        * network.passenger_travel_steps,
        axis=1,
        dtype=np.float64,
    )

    estimated_busy_vehicles = float(
        np.dot(
            training_mean_demand,
            expected_travel_steps,
        )
    )

    if (
        not math.isfinite(
            estimated_busy_vehicles
        )
        or estimated_busy_vehicles <= 0
    ):
        raise RealEpisodeBaselineError(
            "Estimated busy fleet is invalid."
        )

    minimum_total = (
        zone_count
        * config.minimum_vehicles_per_zone
    )

    total_fleet = max(
        int(
            math.ceil(
                estimated_busy_vehicles
                * config.reserve_multiplier
            )
        ),
        minimum_total,
    )

    initial_vehicles = np.full(
        zone_count,
        config.minimum_vehicles_per_zone,
        dtype=np.int64,
    )

    remaining_vehicles = (
        total_fleet
        - int(
            initial_vehicles.sum()
        )
    )

    if remaining_vehicles < 0:
        raise RealEpisodeBaselineError(
            "Fleet size is smaller than the configured "
            "per-zone minimum."
        )

    if remaining_vehicles:
        initial_vehicles += allocate_integer_flow(
            remaining_vehicles,
            network.fit_pickup_shares,
        )

    if int(
        initial_vehicles.sum()
    ) != total_fleet:
        raise RealEpisodeBaselineError(
            "Initial fleet allocation failed conservation."
        )

    return FleetSizingResult(
        training_mean_demand=(
            training_mean_demand
        ),
        expected_travel_steps=(
            expected_travel_steps
        ),
        estimated_busy_vehicles=(
            estimated_busy_vehicles
        ),
        reserve_multiplier=(
            config.reserve_multiplier
        ),
        total_fleet=total_fleet,
        initial_vehicles=initial_vehicles,
    )


def extract_episode(
    historical_demand: pl.DataFrame,
    *,
    zone_count: int,
    config: RealEpisodeBaselineConfig,
) -> tuple[
    list[datetime],
    IntegerMatrix,
]:
    """Extract a dense warmup-plus-evaluation episode."""
    start_date = (
        config.service_date
        - timedelta(
            days=config.warmup_days
        )
    )

    frame = (
        historical_demand.filter(
            pl.col(
                "service_date"
            )
            .is_between(
                start_date,
                config.service_date,
                closed="both",
            )
        )
        .sort(
            "interval_start",
            "zone_index",
        )
    )

    if frame.height == 0:
        raise RealEpisodeBaselineError(
            "Selected episode contains no demand rows."
        )

    evaluation_rows = frame.filter(
        pl.col(
            "service_date"
        )
        == config.service_date
    )

    if evaluation_rows.height == 0:
        raise RealEpisodeBaselineError(
            "Evaluation date contains no demand rows."
        )

    invalid_evaluation_split_count = (
        evaluation_rows.filter(
            pl.col(
                "split"
            )
            != config.evaluation_split
        )
        .height
    )

    if invalid_evaluation_split_count:
        observed_splits = (
            evaluation_rows.get_column(
                "split"
            )
            .unique()
            .sort()
            .to_list()
        )

        raise RealEpisodeBaselineError(
            "Evaluation date does not match configured split: "
            f"expected={config.evaluation_split}, "
            f"observed={observed_splits}."
        )

    interval_counts = (
        frame.group_by(
            "interval_start"
        )
        .agg(
            pl.len().alias(
                "zone_count"
            ),
            pl.col(
                "zone_index"
            )
            .n_unique()
            .alias(
                "unique_zone_count"
            ),
        )
    )

    invalid_interval_count = (
        interval_counts.filter(
            (
                pl.col(
                    "zone_count"
                )
                != zone_count
            )
            | (
                pl.col(
                    "unique_zone_count"
                )
                != zone_count
            )
        )
        .height
    )

    if invalid_interval_count:
        raise RealEpisodeBaselineError(
            "Episode demand is not dense across zones."
        )

    timestamps = (
        frame.select(
            "interval_start"
        )
        .unique()
        .sort(
            "interval_start"
        )
        .get_column(
            "interval_start"
        )
        .to_list()
    )

    expected_delta = timedelta(
        minutes=config.interval_minutes
    )

    for previous, current in pairwise(
        timestamps
    ):
        if (
            current.astimezone(UTC)
            - previous.astimezone(UTC)
        ) != expected_delta:
            raise RealEpisodeBaselineError(
                "Episode intervals are not contiguous: "
                f"{previous!s} → {current!s}."
            )

    expected_zone_indices = np.tile(
        np.arange(
            zone_count,
            dtype=np.int64,
        ),
        len(
            timestamps
        ),
    )

    observed_zone_indices = (
        frame.get_column(
            "zone_index"
        )
        .to_numpy()
        .astype(
            np.int64,
            copy=False,
        )
    )

    if not np.array_equal(
        observed_zone_indices,
        expected_zone_indices,
    ):
        raise RealEpisodeBaselineError(
            "Episode zone ordering is invalid."
        )

    demand_matrix = (
        frame.get_column(
            "actual_demand"
        )
        .to_numpy()
        .reshape(
            len(
                timestamps
            ),
            zone_count,
        )
        .astype(
            np.int64,
            copy=False,
        )
    )

    return (
        timestamps,
        demand_matrix,
    )


def extract_episode_range(
    historical_demand: pl.DataFrame,
    *,
    zone_count: int,
    config: RealEpisodeBaselineConfig,
    evaluation_start_date: date,
    evaluation_end_date: date,
) -> tuple[list[datetime], IntegerMatrix]:
    """Extract one continuous warmup + multi-day evaluation window.

    The warmup is shared and occurs exactly once before ``evaluation_start_date``.
    Every interval in the inclusive evaluation date range must belong to the
    configured holdout split.  This is used by the final untouched holdout so
    fleet state carries continuously across the evaluation month.
    """
    if evaluation_end_date < evaluation_start_date:
        raise RealEpisodeBaselineError(
            "evaluation_end_date must not precede evaluation_start_date."
        )
    warmup_start = evaluation_start_date - timedelta(days=config.warmup_days)
    frame = (
        historical_demand.filter(
            pl.col("service_date").is_between(
                warmup_start,
                evaluation_end_date,
                closed="both",
            )
        )
        .sort("interval_start", "zone_index")
    )
    if frame.height == 0:
        raise RealEpisodeBaselineError("Selected multi-day episode contains no rows.")

    evaluation = frame.filter(
        pl.col("service_date").is_between(
            evaluation_start_date,
            evaluation_end_date,
            closed="both",
        )
    )
    if evaluation.height == 0:
        raise RealEpisodeBaselineError("Evaluation date range contains no demand rows.")
    observed_splits = sorted(
        str(value) for value in evaluation.get_column("split").unique().to_list()
    )
    if observed_splits != [config.evaluation_split]:
        raise RealEpisodeBaselineError(
            "Evaluation date range does not match configured split: "
            f"expected={config.evaluation_split}, observed={observed_splits}."
        )

    interval_counts = frame.group_by("interval_start").agg(
        pl.len().alias("row_count"),
        pl.col("zone_index").n_unique().alias("unique_zone_count"),
        pl.col("zone_index").min().alias("minimum_zone_index"),
        pl.col("zone_index").max().alias("maximum_zone_index"),
    )
    invalid = interval_counts.filter(
        (pl.col("row_count") != zone_count)
        | (pl.col("unique_zone_count") != zone_count)
        | (pl.col("minimum_zone_index") != 0)
        | (pl.col("maximum_zone_index") != zone_count - 1)
    )
    if invalid.height:
        raise RealEpisodeBaselineError(
            "Multi-day episode is not a dense zone-time panel: "
            f"invalid_intervals={invalid.height}."
        )

    timestamps = [
        value
        for value in frame.get_column("interval_start").unique().sort().to_list()
    ]
    if any(value.tzinfo is None for value in timestamps):
        raise RealEpisodeBaselineError("Episode timestamps must be timezone-aware.")
    expected_seconds = config.interval_minutes * 60
    for previous, current in zip(timestamps, timestamps[1:], strict=False):
        elapsed = (current.astimezone(UTC) - previous.astimezone(UTC)).total_seconds()
        if elapsed != expected_seconds:
            raise RealEpisodeBaselineError(
                "Multi-day episode timestamps are not contiguous in UTC: "
                f"previous={previous}, current={current}."
            )

    expected_zone_indices = np.tile(
        np.arange(zone_count, dtype=np.int64),
        len(timestamps),
    )
    observed_zone_indices = frame.get_column("zone_index").to_numpy().astype(
        np.int64, copy=False
    )
    if not np.array_equal(observed_zone_indices, expected_zone_indices):
        raise RealEpisodeBaselineError("Multi-day zone ordering is invalid.")
    demand_matrix = frame.get_column("actual_demand").to_numpy().reshape(
        len(timestamps), zone_count
    ).astype(np.int64, copy=False)
    if demand_matrix.shape != (len(timestamps), zone_count) or (demand_matrix < 0).any():
        raise RealEpisodeBaselineError("Multi-day demand matrix has an invalid shape or values.")
    warmup_steps = sum(value.date() < evaluation_start_date for value in timestamps)
    evaluation_steps = len(timestamps) - warmup_steps
    if warmup_steps <= 0 or evaluation_steps <= 0:
        raise RealEpisodeBaselineError("Multi-day episode needs warmup and evaluation rows.")
    return timestamps, demand_matrix


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

        raise RealEpisodeBaselineError(
            f"Could not write Parquet artifact {path}: {error}"
        ) from error

    return {
        "path": str(
            path
        ),
        "row_count": frame.height,
        "column_count": frame.width,
        "size_bytes": path.stat().st_size,
    }


def sanitize_json_value(
    value: Any,
) -> Any:
    """Convert values into strict JSON-compatible forms."""
    if isinstance(
        value,
        np.integer,
    ):
        return int(
            value
        )

    if isinstance(
        value,
        np.floating,
    ):
        value = float(
            value
        )

    if isinstance(
        value,
        float,
    ):
        if not math.isfinite(
            value
        ):
            return None

        return value

    if isinstance(
        value,
        Path,
    ):
        return str(
            value
        )

    if isinstance(
        value,
        (
            datetime,
            date,
        ),
    ):
        return value.isoformat()

    if isinstance(
        value,
        dict,
    ):
        return {
            str(key): sanitize_json_value(
                item
            )
            for key, item in value.items()
        }

    if isinstance(
        value,
        (
            list,
            tuple,
        ),
    ):
        return [
            sanitize_json_value(
                item
            )
            for item in value
        ]

    return value


def write_json_atomically(
    path: Path,
    payload: dict[str, Any],
) -> None:
    """Write a strict JSON report atomically."""
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
                sanitize_json_value(
                    payload
                ),
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

        raise RealEpisodeBaselineError(
            f"Could not write episode report: {error}"
        ) from error


def run_real_episode_baseline(
    config_path: Path,
) -> dict[str, Any]:
    """Run one real historical no-rebalancing episode."""
    config = read_config(
        config_path
    )

    (
        zone_index,
        zone_ids,
        fit_pickup_shares,
    ) = load_zone_index(
        config
    )

    zone_count = len(
        zone_ids
    )

    (
        destination_probabilities,
        passenger_travel_steps,
        relocation_travel_steps,
    ) = load_od_matrices(
        config,
        zone_count=zone_count,
    )

    network = NetworkInputs(
        zone_ids=zone_ids,
        fit_pickup_shares=fit_pickup_shares,
        destination_probabilities=(
            destination_probabilities
        ),
        passenger_travel_steps=(
            passenger_travel_steps
        ),
        relocation_travel_steps=(
            relocation_travel_steps
        ),
    )

    historical_demand = load_historical_demand(
        config,
        zone_count=zone_count,
    )

    fleet_sizing = estimate_initial_fleet(
        historical_demand,
        network=network,
        config=config,
    )

    timestamps, demand_matrix = extract_episode(
        historical_demand,
        zone_count=zone_count,
        config=config,
    )

    simulator = FleetSimulator(
        zone_ids=zone_ids,
        initial_vehicles=(
            fleet_sizing.initial_vehicles
        ),
        passenger_destination_probabilities=(
            destination_probabilities
        ),
        relocation_travel_steps=(
            relocation_travel_steps
        ),
        passenger_travel_steps=(
            passenger_travel_steps
        ),
        costs=SimulationCosts(
            unmet_demand_penalty=(
                config.unmet_demand_penalty
            ),
            relocation_cost_per_vehicle_step=(
                config.relocation_cost_per_vehicle_step
            ),
            idle_vehicle_cost=(
                config.idle_vehicle_cost
            ),
        ),
    )

    zero_relocation_action = np.zeros(
        (
            zone_count,
            zone_count,
        ),
        dtype=np.int64,
    )

    step_records: list[
        dict[str, Any]
    ] = []

    interval_zone_records: list[
        dict[str, Any]
    ] = []

    evaluation_results = []

    for timestamp, demand in zip(
        timestamps,
        demand_matrix,
        strict=True,
    ):
        result = simulator.step(
            demand=demand,
            relocation_action=(
                zero_relocation_action
            ),
        )

        is_evaluation = (
            timestamp.date()
            == config.service_date
        )

        phase = (
            "evaluation"
            if is_evaluation
            else "warmup"
        )

        step_records.append(
            {
                "simulator_step": result.step,
                "interval_start": timestamp,
                "service_date": timestamp.date(),
                "phase": phase,
                "demand_total": (
                    result.metrics.demand_total
                ),
                "served_demand_total": (
                    result.metrics.served_demand_total
                ),
                "unmet_demand_total": (
                    result.metrics.unmet_demand_total
                ),
                "idle_vehicle_total": (
                    result.metrics.idle_vehicle_total
                ),
                "arrivals_released_total": (
                    result.metrics.arrivals_released_total
                ),
                "relocated_vehicle_total": (
                    result.metrics.relocated_vehicle_total
                ),
                "service_rate": (
                    result.metrics.service_rate
                ),
                "unmet_demand_cost": (
                    result.metrics.unmet_demand_cost
                ),
                "relocation_cost": (
                    result.metrics.relocation_cost
                ),
                "idle_vehicle_cost": (
                    result.metrics.idle_vehicle_cost
                ),
                "total_operating_cost": (
                    result.metrics.total_operating_cost
                ),
                "available_vehicle_total_end": int(
                    result.available_vehicles_end.sum()
                ),
                "in_transit_vehicle_total_end": int(
                    result.in_transit_vehicles_end.sum()
                ),
                "fleet_conservation_passed": (
                    result.fleet_conservation_passed
                ),
            }
        )

        if not is_evaluation:
            continue

        evaluation_results.append(
            result
        )

        for zone_index_value, zone_id in enumerate(
            zone_ids
        ):
            interval_zone_records.append(
                {
                    "interval_start": timestamp,
                    "service_date": timestamp.date(),
                    "zone_index": zone_index_value,
                    "zone_id": zone_id,
                    "demand": int(
                        result.demand[
                            zone_index_value
                        ]
                    ),
                    "served_demand": int(
                        result.served_demand[
                            zone_index_value
                        ]
                    ),
                    "unmet_demand": int(
                        result.unmet_demand[
                            zone_index_value
                        ]
                    ),
                    "idle_vehicles_end": int(
                        result.idle_vehicles[
                            zone_index_value
                        ]
                    ),
                    "arrivals_released": int(
                        result.arrivals_released[
                            zone_index_value
                        ]
                    ),
                    "available_vehicles_end": int(
                        result.available_vehicles_end[
                            zone_index_value
                        ]
                    ),
                    "in_transit_vehicles_end": int(
                        result.in_transit_vehicles_end[
                            zone_index_value
                        ]
                    ),
                }
            )

    if not evaluation_results:
        raise RealEpisodeBaselineError(
            "No evaluation transitions were produced."
        )

    step_metrics = (
        pl.DataFrame(
            step_records
        )
        .sort(
            "interval_start"
        )
    )

    interval_zone_metrics = (
        pl.DataFrame(
            interval_zone_records
        )
        .sort(
            "interval_start",
            "zone_index",
        )
    )

    zone_summary = (
        interval_zone_metrics.group_by(
            "zone_index",
            "zone_id",
        )
        .agg(
            pl.len().alias(
                "interval_count"
            ),
            pl.col(
                "demand"
            )
            .sum()
            .alias(
                "demand_total"
            ),
            pl.col(
                "served_demand"
            )
            .sum()
            .alias(
                "served_demand_total"
            ),
            pl.col(
                "unmet_demand"
            )
            .sum()
            .alias(
                "unmet_demand_total"
            ),
            pl.col(
                "unmet_demand"
            )
            .max()
            .alias(
                "maximum_interval_unmet_demand"
            ),
            pl.col(
                "idle_vehicles_end"
            )
            .mean()
            .alias(
                "mean_idle_vehicles"
            ),
            pl.col(
                "available_vehicles_end"
            )
            .mean()
            .alias(
                "mean_available_vehicles_end"
            ),
            pl.col(
                "in_transit_vehicles_end"
            )
            .mean()
            .alias(
                "mean_in_transit_vehicles_end"
            ),
        )
        .with_columns(
            pl.when(
                pl.col(
                    "demand_total"
                )
                > 0
            )
            .then(
                pl.col(
                    "served_demand_total"
                )
                / pl.col(
                    "demand_total"
                )
            )
            .otherwise(
                1.0
            )
            .alias(
                "service_rate"
            )
        )
        .sort(
            "zone_index"
        )
    )

    evaluation_step_metrics = (
        step_metrics.filter(
            pl.col(
                "phase"
            )
            == "evaluation"
        )
    )

    evaluation_step_count = (
        evaluation_step_metrics.height
    )

    warmup_step_count = (
        step_metrics.height
        - evaluation_step_count
    )

    demand_total = sum(
        result.metrics.demand_total
        for result in evaluation_results
    )

    served_total = sum(
        result.metrics.served_demand_total
        for result in evaluation_results
    )

    unmet_total = sum(
        result.metrics.unmet_demand_total
        for result in evaluation_results
    )

    idle_vehicle_interval_sum = sum(
        result.metrics.idle_vehicle_total
        for result in evaluation_results
    )

    relocation_total = sum(
        result.metrics.relocated_vehicle_total
        for result in evaluation_results
    )

    unmet_cost = sum(
        result.metrics.unmet_demand_cost
        for result in evaluation_results
    )

    relocation_cost = sum(
        result.metrics.relocation_cost
        for result in evaluation_results
    )

    idle_cost = sum(
        result.metrics.idle_vehicle_cost
        for result in evaluation_results
    )

    total_operating_cost = sum(
        result.metrics.total_operating_cost
        for result in evaluation_results
    )

    service_rate = (
        served_total
        / demand_total
        if demand_total > 0
        else 1.0
    )

    artifacts = {
        "step_metrics": write_parquet_atomically(
            step_metrics,
            path=config.step_metrics_path,
            compression=config.compression,
        ),
        "interval_zone_metrics": (
            write_parquet_atomically(
                interval_zone_metrics,
                path=(
                    config.interval_zone_metrics_path
                ),
                compression=config.compression,
            )
        ),
        "zone_summary": write_parquet_atomically(
            zone_summary,
            path=config.zone_summary_path,
            compression=config.compression,
        ),
    }

    top_unmet_zones = (
        zone_summary.sort(
            [
                "unmet_demand_total",
                "demand_total",
            ],
            descending=[
                True,
                True,
            ],
        )
        .head(
            10
        )
        .to_dicts()
    )

    all_steps_preserve_fleet = all(
        bool(
            record[
                "fleet_conservation_passed"
            ]
        )
        for record in step_records
    )

    final_snapshot = simulator.snapshot()

    final_observed_fleet = (
        int(
            final_snapshot.available_vehicles.sum()
        )
        + int(
            final_snapshot.in_transit_vehicles.sum()
        )
    )

    report = {
        "simulation_report_version": "1.0.0",
        "generated_at": datetime.now(
            UTC
        ).isoformat(),
        "dataset_name": config.dataset_name,
        "policy": {
            "name": config.policy_name,
            "uses_forecast": False,
            "uses_relocation": False,
        },
        "episode": {
            "evaluation_split": (
                config.evaluation_split
            ),
            "service_date": (
                config.service_date
            ),
            "warmup_days": (
                config.warmup_days
            ),
            "warmup_step_count": (
                warmup_step_count
            ),
            "evaluation_step_count": (
                evaluation_step_count
            ),
            "first_interval": (
                timestamps[0]
            ),
            "last_interval": (
                timestamps[-1]
            ),
        },
        "network": {
            "zone_count": zone_count,
            "zone_ids": list(
                zone_ids
            ),
            "od_pair_count": (
                zone_count
                * zone_count
            ),
        },
        "fleet_sizing": fleet_sizing.as_dict(
            zone_ids=zone_ids
        ),
        "cost_configuration": {
            "unmet_demand_penalty": (
                config.unmet_demand_penalty
            ),
            "relocation_cost_per_vehicle_step": (
                config.relocation_cost_per_vehicle_step
            ),
            "idle_vehicle_cost": (
                config.idle_vehicle_cost
            ),
            "currency": None,
            "costs_are_normalized_objective_units": True,
        },
        "evaluation_metrics": {
            "demand_total": demand_total,
            "served_demand_total": served_total,
            "unmet_demand_total": unmet_total,
            "service_rate": service_rate,
            "average_idle_vehicles": (
                idle_vehicle_interval_sum
                / evaluation_step_count
            ),
            "relocated_vehicle_total": (
                relocation_total
            ),
            "unmet_demand_cost": float(
                unmet_cost
            ),
            "relocation_cost": float(
                relocation_cost
            ),
            "idle_vehicle_cost": float(
                idle_cost
            ),
            "total_operating_cost": float(
                total_operating_cost
            ),
        },
        "top_unmet_zones": (
            top_unmet_zones
        ),
        "artifacts": artifacts,
        "invariants": {
            "all_steps_preserve_fleet": (
                all_steps_preserve_fleet
            ),
            "final_fleet_is_conserved": (
                final_observed_fleet
                == simulator.total_fleet
            ),
            "no_relocation_was_applied": (
                relocation_total
                == 0
            ),
            "served_plus_unmet_equals_demand": (
                served_total
                + unmet_total
                == demand_total
            ),
            "evaluation_demand_matches_zone_records": (
                demand_total
                == int(
                    interval_zone_metrics.get_column(
                        "demand"
                    ).sum()
                )
            ),
            "evaluation_zone_grid_is_dense": (
                interval_zone_metrics.height
                == evaluation_step_count
                * zone_count
            ),
            "fleet_sizing_uses_training_split_only": True,
        },
    }

    write_json_atomically(
        config.report_path,
        report,
    )

    return report