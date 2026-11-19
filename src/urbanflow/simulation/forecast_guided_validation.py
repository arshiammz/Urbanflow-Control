from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import yaml
from numpy.typing import NDArray

from urbanflow.simulation.decision_time_forecasting import fit_train_only_projector
from urbanflow.simulation.forecast_guided_heuristic import (
    ForecastGuidedConfig,
    build_simulator,
    greedy_relocation_action,
    load_forecast_matrix,
    read_config as read_heuristic_config,
    write_json_atomically,
    write_parquet_atomically,
)
from urbanflow.simulation.real_episode_baseline import (
    NetworkInputs,
    RealEpisodeBaselineConfig,
    estimate_initial_fleet,
    load_historical_demand,
    load_od_matrices,
    load_zone_index,
    read_config as read_baseline_config,
)


FloatMatrix = NDArray[np.float64]
IntegerMatrix = NDArray[np.int64]


class ForecastGuidedValidationError(RuntimeError):
    """Raised when continuous multi-day validation fails."""


@dataclass(frozen=True)
class MultiDayValidationConfig:
    """Configuration for continuous multi-day validation."""

    dataset_name: str

    baseline_config_path: Path
    heuristic_config_path: Path

    step_metrics_path: Path
    daily_metrics_path: Path
    daily_zone_metrics_path: Path
    zone_summary_path: Path
    report_path: Path

    validation_split: str
    start_date: date
    end_date_inclusive: date

    compression: str


def parse_date(
    value: Any,
    *,
    description: str,
) -> date:
    """Parse an ISO calendar date."""
    if isinstance(value, datetime):
        return value.date()

    if isinstance(value, date):
        return value

    try:
        return date.fromisoformat(str(value))

    except ValueError as error:
        raise ForecastGuidedValidationError(
            f"{description} must use YYYY-MM-DD format."
        ) from error


def read_config(
    path: Path,
) -> MultiDayValidationConfig:
    """Read and validate multi-day validation configuration."""
    if not path.exists():
        raise ForecastGuidedValidationError(
            f"Validation config does not exist: {path}"
        )

    try:
        raw = yaml.safe_load(
            path.read_text(
                encoding="utf-8",
            )
        )

        if not isinstance(raw, dict):
            raise ForecastGuidedValidationError(
                "Validation config must contain a YAML mapping."
            )

        dataset = raw["dataset"]
        paths = raw["paths"]
        validation = raw["validation"]
        output = raw["output"]

        config = MultiDayValidationConfig(
            dataset_name=str(
                dataset["name"]
            ),
            baseline_config_path=Path(
                paths["baseline_config"]
            ),
            heuristic_config_path=Path(
                paths["heuristic_config"]
            ),
            step_metrics_path=Path(
                paths["step_metrics"]
            ),
            daily_metrics_path=Path(
                paths["daily_metrics"]
            ),
            daily_zone_metrics_path=Path(
                paths["daily_zone_metrics"]
            ),
            zone_summary_path=Path(
                paths["zone_summary"]
            ),
            report_path=Path(
                paths["report"]
            ),
            validation_split=str(
                validation["split"]
            ),
            start_date=parse_date(
                validation["start_date"],
                description="start_date",
            ),
            end_date_inclusive=parse_date(
                validation["end_date_inclusive"],
                description="end_date_inclusive",
            ),
            compression=str(
                output["compression"]
            ),
        )

    except ForecastGuidedValidationError:
        raise

    except (
        OSError,
        KeyError,
        TypeError,
        ValueError,
        yaml.YAMLError,
    ) as error:
        raise ForecastGuidedValidationError(
            f"Invalid validation config: {error}"
        ) from error

    validate_config(
        config
    )

    return config


def validate_config(
    config: MultiDayValidationConfig,
) -> None:
    """Validate multi-day configuration invariants."""
    if not config.dataset_name:
        raise ForecastGuidedValidationError(
            "Dataset name must not be empty."
        )

    if not config.validation_split:
        raise ForecastGuidedValidationError(
            "Validation split must not be empty."
        )

    if config.end_date_inclusive < config.start_date:
        raise ForecastGuidedValidationError(
            "end_date_inclusive must not be before start_date."
        )

    if not config.compression:
        raise ForecastGuidedValidationError(
            "Output compression must not be empty."
        )

    for input_path in (
        config.baseline_config_path,
        config.heuristic_config_path,
    ):
        if not input_path.exists():
            raise ForecastGuidedValidationError(
                f"Required input does not exist: {input_path}"
            )

        if input_path.stat().st_size == 0:
            raise ForecastGuidedValidationError(
                f"Required input is empty: {input_path}"
            )


def validate_component_configs(
    *,
    config: MultiDayValidationConfig,
    baseline_config: RealEpisodeBaselineConfig,
    heuristic_config: ForecastGuidedConfig,
) -> None:
    """Validate compatibility between component configs."""
    configured_baseline = (
        config.baseline_config_path.resolve()
    )

    heuristic_baseline = (
        heuristic_config.baseline_config_path.resolve()
    )

    if configured_baseline != heuristic_baseline:
        raise ForecastGuidedValidationError(
            "Validation and heuristic configs reference "
            "different baseline configs."
        )

    if heuristic_config.apply_during_warmup:
        raise ForecastGuidedValidationError(
            "Multi-day validation requires "
            "apply_during_warmup=false."
        )

    if baseline_config.warmup_days < 1:
        raise ForecastGuidedValidationError(
            "At least one warmup day is required for "
            "continuous validation."
        )


def extract_continuous_window(
    historical_demand: pl.DataFrame,
    *,
    config: MultiDayValidationConfig,
    baseline_config: RealEpisodeBaselineConfig,
    zone_count: int,
) -> tuple[
    list[datetime],
    IntegerMatrix,
]:
    """Extract a dense warmup-plus-validation demand window."""
    warmup_start_date = (
        config.start_date
        - timedelta(
            days=baseline_config.warmup_days
        )
    )

    frame = (
        historical_demand.filter(
            pl.col(
                "service_date"
            ).is_between(
                warmup_start_date,
                config.end_date_inclusive,
                closed="both",
            )
        )
        .sort(
            "interval_start",
            "zone_index",
        )
    )

    if frame.height == 0:
        raise ForecastGuidedValidationError(
            "Continuous validation window contains no demand."
        )

    evaluation_rows = frame.filter(
        pl.col(
            "service_date"
        ).is_between(
            config.start_date,
            config.end_date_inclusive,
            closed="both",
        )
    )

    if evaluation_rows.height == 0:
        raise ForecastGuidedValidationError(
            "Evaluation window contains no demand rows."
        )

    invalid_split_count = (
        evaluation_rows.filter(
            pl.col(
                "split"
            )
            != config.validation_split
        )
        .height
    )

    if invalid_split_count:
        observed_splits = (
            evaluation_rows.get_column(
                "split"
            )
            .unique()
            .sort()
            .to_list()
        )

        raise ForecastGuidedValidationError(
            "Evaluation demand does not match the configured "
            "validation split: "
            f"expected={config.validation_split}, "
            f"observed={observed_splits}."
        )

    interval_counts = (
        frame.group_by(
            "interval_start"
        )
        .agg(
            pl.len().alias(
                "row_count"
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
                    "row_count"
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
        raise ForecastGuidedValidationError(
            "Historical demand is not dense across zones."
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
        minutes=baseline_config.interval_minutes
    )

    for previous, current in pairwise(
        timestamps
    ):
        if (
            current
            - previous
        ) != expected_delta:
            raise ForecastGuidedValidationError(
                "Historical intervals are not contiguous: "
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
        expected_zone_indices,
        observed_zone_indices,
    ):
        raise ForecastGuidedValidationError(
            "Historical demand zone ordering is invalid."
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


def validate_forecast_split_window(
    *,
    config: MultiDayValidationConfig,
    heuristic_config: ForecastGuidedConfig,
    zone_ids: tuple[int, ...],
    first_timestamp: datetime,
    last_required_timestamp: datetime,
) -> dict[str, Any]:
    """Ensure every required forecast belongs to validation."""
    try:
        frame = (
            pl.scan_parquet(
                heuristic_config.forecast_predictions_path
            )
            .select(
                pl.col(
                    heuristic_config.forecast_timestamp_column
                ).alias(
                    "interval_start"
                ),
                pl.col(
                    heuristic_config.forecast_zone_column
                )
                .cast(
                    pl.Int64
                )
                .alias(
                    "zone_id"
                ),
                pl.col(
                    heuristic_config.forecast_split_column
                )
                .cast(
                    pl.String
                )
                .alias(
                    "split"
                ),
            )
            .filter(
                pl.col(
                    "zone_id"
                ).is_in(
                    list(
                        zone_ids
                    )
                )
            )
            .filter(
                pl.col(
                    "interval_start"
                )
                >= first_timestamp
            )
            .filter(
                pl.col(
                    "interval_start"
                )
                <= last_required_timestamp
            )
            .collect()
        )

    except Exception as error:
        raise ForecastGuidedValidationError(
            "Could not validate forecast split window: "
            f"{error}"
        ) from error

    if frame.height == 0:
        raise ForecastGuidedValidationError(
            "Forecast split validation returned no rows."
        )

    observed_splits = (
        frame.get_column(
            "split"
        )
        .unique()
        .sort()
        .to_list()
    )

    invalid_split_count = (
        frame.filter(
            pl.col(
                "split"
            )
            != config.validation_split
        )
        .height
    )

    if invalid_split_count:
        raise ForecastGuidedValidationError(
            "Forecast window crosses outside validation: "
            f"observed_splits={observed_splits}, "
            f"invalid_rows={invalid_split_count}."
        )

    return {
        "first_timestamp": first_timestamp,
        "last_required_timestamp": last_required_timestamp,
        "observed_splits": observed_splits,
        "row_count": frame.height,
        "invalid_split_count": invalid_split_count,
    }


def build_daily_metrics(
    step_metrics: pl.DataFrame,
) -> pl.DataFrame:
    """Aggregate interval-level metrics by evaluation day."""
    daily = (
        step_metrics.filter(
            pl.col(
                "phase"
            )
            == "evaluation"
        )
        .group_by(
            "service_date"
        )
        .agg(
            pl.len().alias(
                "interval_count"
            ),
            pl.col(
                "actual_demand_total"
            )
            .sum()
            .alias(
                "demand_total"
            ),
            pl.col(
                "baseline_served_total"
            )
            .sum()
            .alias(
                "baseline_served_total"
            ),
            pl.col(
                "policy_served_total"
            )
            .sum()
            .alias(
                "policy_served_total"
            ),
            pl.col(
                "baseline_unmet_total"
            )
            .sum()
            .alias(
                "baseline_unmet_total"
            ),
            pl.col(
                "policy_unmet_total"
            )
            .sum()
            .alias(
                "policy_unmet_total"
            ),
            pl.col(
                "baseline_idle_total"
            )
            .mean()
            .alias(
                "baseline_average_idle"
            ),
            pl.col(
                "policy_idle_total"
            )
            .mean()
            .alias(
                "policy_average_idle"
            ),
            pl.col(
                "policy_relocated_total"
            )
            .sum()
            .alias(
                "policy_relocated_total"
            ),
            pl.col(
                "policy_relocation_vehicle_steps"
            )
            .sum()
            .alias(
                "policy_relocation_vehicle_steps"
            ),
            pl.col(
                "baseline_total_cost"
            )
            .sum()
            .alias(
                "baseline_total_cost"
            ),
            pl.col(
                "policy_total_cost"
            )
            .sum()
            .alias(
                "policy_total_cost"
            ),
            pl.col(
                "baseline_fleet_conserved"
            )
            .all()
            .alias(
                "baseline_fleet_conserved"
            ),
            pl.col(
                "policy_fleet_conserved"
            )
            .all()
            .alias(
                "policy_fleet_conserved"
            ),
        )
        .with_columns(
            (
                pl.col(
                    "baseline_served_total"
                )
                / pl.col(
                    "demand_total"
                )
            )
            .alias(
                "baseline_service_rate"
            ),
            (
                pl.col(
                    "policy_served_total"
                )
                / pl.col(
                    "demand_total"
                )
            )
            .alias(
                "policy_service_rate"
            ),
            (
                pl.col(
                    "baseline_unmet_total"
                )
                - pl.col(
                    "policy_unmet_total"
                )
            )
            .alias(
                "unmet_demand_reduction"
            ),
            (
                pl.col(
                    "baseline_total_cost"
                )
                - pl.col(
                    "policy_total_cost"
                )
            )
            .alias(
                "operating_cost_reduction"
            ),
            (
                pl.col(
                    "baseline_average_idle"
                )
                - pl.col(
                    "policy_average_idle"
                )
            )
            .alias(
                "average_idle_reduction"
            ),
        )
        .with_columns(
            (
                pl.col(
                    "policy_service_rate"
                )
                - pl.col(
                    "baseline_service_rate"
                )
            )
            .alias(
                "service_rate_improvement"
            ),
            (
                pl.col(
                    "unmet_demand_reduction"
                )
                / pl.col(
                    "baseline_unmet_total"
                )
            )
            .alias(
                "unmet_demand_relative_reduction"
            ),
            (
                pl.col(
                    "operating_cost_reduction"
                )
                / pl.col(
                    "baseline_total_cost"
                )
            )
            .alias(
                "operating_cost_relative_reduction"
            ),
            (
                pl.col(
                    "policy_unmet_total"
                )
                < pl.col(
                    "baseline_unmet_total"
                )
            )
            .alias(
                "policy_reduces_unmet_demand"
            ),
            (
                pl.col(
                    "policy_total_cost"
                )
                < pl.col(
                    "baseline_total_cost"
                )
            )
            .alias(
                "policy_beats_baseline_cost"
            ),
        )
        .sort(
            "service_date"
        )
    )

    return daily


def build_daily_zone_metrics(
    interval_zone_metrics: pl.DataFrame,
) -> pl.DataFrame:
    """Aggregate interval-zone metrics by day and zone."""
    daily_zone = (
        interval_zone_metrics.group_by(
            "service_date",
            "zone_index",
            "zone_id",
        )
        .agg(
            pl.len().alias(
                "interval_count"
            ),
            pl.col(
                "actual_demand"
            )
            .sum()
            .alias(
                "demand_total"
            ),
            pl.col(
                "baseline_served"
            )
            .sum()
            .alias(
                "baseline_served_total"
            ),
            pl.col(
                "policy_served"
            )
            .sum()
            .alias(
                "policy_served_total"
            ),
            pl.col(
                "baseline_unmet"
            )
            .sum()
            .alias(
                "baseline_unmet_total"
            ),
            pl.col(
                "policy_unmet"
            )
            .sum()
            .alias(
                "policy_unmet_total"
            ),
            pl.col(
                "baseline_idle_end"
            )
            .mean()
            .alias(
                "baseline_mean_idle"
            ),
            pl.col(
                "policy_idle_end"
            )
            .mean()
            .alias(
                "policy_mean_idle"
            ),
            pl.col(
                "policy_outbound_relocation"
            )
            .sum()
            .alias(
                "policy_outbound_relocation_total"
            ),
            pl.col(
                "policy_inbound_relocation"
            )
            .sum()
            .alias(
                "policy_inbound_relocation_total"
            ),
        )
        .with_columns(
            (
                pl.col(
                    "baseline_unmet_total"
                )
                - pl.col(
                    "policy_unmet_total"
                )
            )
            .alias(
                "unmet_demand_reduction"
            ),
            pl.when(
                pl.col(
                    "demand_total"
                )
                > 0
            )
            .then(
                pl.col(
                    "baseline_served_total"
                )
                / pl.col(
                    "demand_total"
                )
            )
            .otherwise(
                1.0
            )
            .alias(
                "baseline_service_rate"
            ),
            pl.when(
                pl.col(
                    "demand_total"
                )
                > 0
            )
            .then(
                pl.col(
                    "policy_served_total"
                )
                / pl.col(
                    "demand_total"
                )
            )
            .otherwise(
                1.0
            )
            .alias(
                "policy_service_rate"
            ),
        )
        .with_columns(
            (
                pl.col(
                    "policy_service_rate"
                )
                - pl.col(
                    "baseline_service_rate"
                )
            )
            .alias(
                "service_rate_improvement"
            )
        )
        .sort(
            "service_date",
            "zone_index",
        )
    )

    return daily_zone


def build_zone_summary(
    daily_zone_metrics: pl.DataFrame,
) -> pl.DataFrame:
    """Build aggregate multi-day zone diagnostics."""
    summary = (
        daily_zone_metrics.group_by(
            "zone_index",
            "zone_id",
        )
        .agg(
            pl.len().alias(
                "day_count"
            ),
            pl.col(
                "demand_total"
            )
            .sum()
            .alias(
                "demand_total"
            ),
            pl.col(
                "baseline_served_total"
            )
            .sum()
            .alias(
                "baseline_served_total"
            ),
            pl.col(
                "policy_served_total"
            )
            .sum()
            .alias(
                "policy_served_total"
            ),
            pl.col(
                "baseline_unmet_total"
            )
            .sum()
            .alias(
                "baseline_unmet_total"
            ),
            pl.col(
                "policy_unmet_total"
            )
            .sum()
            .alias(
                "policy_unmet_total"
            ),
            pl.col(
                "unmet_demand_reduction"
            )
            .sum()
            .alias(
                "unmet_demand_reduction"
            ),
            pl.col(
                "unmet_demand_reduction"
            )
            .mean()
            .alias(
                "mean_daily_unmet_reduction"
            ),
            (
                pl.col(
                    "unmet_demand_reduction"
                )
                > 0
            )
            .sum()
            .alias(
                "days_improved"
            ),
            (
                pl.col(
                    "unmet_demand_reduction"
                )
                < 0
            )
            .sum()
            .alias(
                "days_harmed"
            ),
            pl.col(
                "baseline_mean_idle"
            )
            .mean()
            .alias(
                "baseline_mean_idle"
            ),
            pl.col(
                "policy_mean_idle"
            )
            .mean()
            .alias(
                "policy_mean_idle"
            ),
            pl.col(
                "policy_outbound_relocation_total"
            )
            .sum()
            .alias(
                "policy_outbound_relocation_total"
            ),
            pl.col(
                "policy_inbound_relocation_total"
            )
            .sum()
            .alias(
                "policy_inbound_relocation_total"
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
                    "baseline_served_total"
                )
                / pl.col(
                    "demand_total"
                )
            )
            .otherwise(
                1.0
            )
            .alias(
                "baseline_service_rate"
            ),
            pl.when(
                pl.col(
                    "demand_total"
                )
                > 0
            )
            .then(
                pl.col(
                    "policy_served_total"
                )
                / pl.col(
                    "demand_total"
                )
            )
            .otherwise(
                1.0
            )
            .alias(
                "policy_service_rate"
            ),
        )
        .with_columns(
            (
                pl.col(
                    "policy_service_rate"
                )
                - pl.col(
                    "baseline_service_rate"
                )
            )
            .alias(
                "service_rate_improvement"
            )
        )
        .sort(
            "zone_index"
        )
    )

    return summary


def run_forecast_guided_validation(
    config_path: Path,
) -> dict[str, Any]:
    """Run continuous forecast-guided validation."""
    config = read_config(
        config_path
    )

    baseline_config = read_baseline_config(
        config.baseline_config_path
    )

    heuristic_config = read_heuristic_config(
        config.heuristic_config_path
    )

    validate_component_configs(
        config=config,
        baseline_config=baseline_config,
        heuristic_config=heuristic_config,
    )

    (
        _,
        zone_ids,
        fit_pickup_shares,
    ) = load_zone_index(
        baseline_config
    )

    zone_count = len(
        zone_ids
    )

    (
        destination_probabilities,
        passenger_travel_steps,
        relocation_travel_steps,
    ) = load_od_matrices(
        baseline_config,
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
        baseline_config,
        zone_count=zone_count,
    )

    fleet_sizing = estimate_initial_fleet(
        historical_demand,
        network=network,
        config=baseline_config,
    )

    episode_timestamps, demand_matrix = (
        extract_continuous_window(
            historical_demand,
            config=config,
            baseline_config=baseline_config,
            zone_count=zone_count,
        )
    )

    (
        forecast_timestamps,
        forecast_matrix,
    ) = load_forecast_matrix(
        config=heuristic_config,
        baseline_config=baseline_config,
        zone_ids=zone_ids,
        episode_timestamps=episode_timestamps,
        expected_split=config.validation_split,
    )

    projector = fit_train_only_projector(
        historical_demand,
        zone_ids=zone_ids,
        interval_minutes=baseline_config.interval_minutes,
        training_split=baseline_config.sizing_split,
    )

    if (
        forecast_timestamps[
            :len(
                episode_timestamps
            )
        ]
        != episode_timestamps
    ):
        raise ForecastGuidedValidationError(
            "Forecast and demand timestamps do not align."
        )

    last_required_timestamp = episode_timestamps[-1]
    forecast_split_validation = validate_forecast_split_window(
        config=config,
        heuristic_config=heuristic_config,
        zone_ids=zone_ids,
        first_timestamp=episode_timestamps[0],
        last_required_timestamp=last_required_timestamp,
    )

    baseline_simulator = build_simulator(
        baseline_config=baseline_config,
        network=network,
        initial_vehicles=(
            fleet_sizing.initial_vehicles
        ),
    )

    policy_simulator = build_simulator(
        baseline_config=baseline_config,
        network=network,
        initial_vehicles=(
            fleet_sizing.initial_vehicles
        ),
    )

    zero_action = np.zeros(
        (
            zone_count,
            zone_count,
        ),
        dtype=np.int64,
    )

    step_records: list[
        dict[str, Any]
    ] = []

    zone_records: list[
        dict[str, Any]
    ] = []

    baseline_fleet_conserved = True
    policy_fleet_conserved = True

    for step_index, (
        timestamp,
        actual_demand,
    ) in enumerate(
        zip(
            episode_timestamps,
            demand_matrix,
            strict=True,
        )
    ):
        is_evaluation = (
            config.start_date
            <= timestamp.date()
            <= config.end_date_inclusive
        )

        if is_evaluation:
            future_forecasts = projector.project(
                decision_time=timestamp,
                current_prediction=forecast_matrix[step_index],
                horizon_steps=heuristic_config.forecast_horizon_steps,
            )

            decision = greedy_relocation_action(
                snapshot=policy_simulator.snapshot(),
                current_forecast=forecast_matrix[
                    step_index
                ],
                future_forecasts=future_forecasts,
                relocation_travel_steps=(
                    relocation_travel_steps
                ),
                config=heuristic_config,
            )

            policy_action = (
                decision.relocation_action
            )

        else:
            policy_action = zero_action

        baseline_result = baseline_simulator.step(
            demand=actual_demand,
            relocation_action=zero_action,
        )

        policy_result = policy_simulator.step(
            demand=actual_demand,
            relocation_action=policy_action,
        )

        baseline_fleet_conserved = (
            baseline_fleet_conserved
            and baseline_result.fleet_conservation_passed
        )

        policy_fleet_conserved = (
            policy_fleet_conserved
            and policy_result.fleet_conservation_passed
        )

        phase = (
            "evaluation"
            if is_evaluation
            else "warmup"
        )

        step_records.append(
            {
                "simulator_step": step_index,
                "interval_start": timestamp,
                "service_date": timestamp.date(),
                "phase": phase,
                "actual_demand_total": int(
                    actual_demand.sum()
                ),
                "baseline_served_total": (
                    baseline_result
                    .metrics
                    .served_demand_total
                ),
                "policy_served_total": (
                    policy_result
                    .metrics
                    .served_demand_total
                ),
                "baseline_unmet_total": (
                    baseline_result
                    .metrics
                    .unmet_demand_total
                ),
                "policy_unmet_total": (
                    policy_result
                    .metrics
                    .unmet_demand_total
                ),
                "baseline_idle_total": (
                    baseline_result
                    .metrics
                    .idle_vehicle_total
                ),
                "policy_idle_total": (
                    policy_result
                    .metrics
                    .idle_vehicle_total
                ),
                "policy_relocated_total": (
                    policy_result
                    .metrics
                    .relocated_vehicle_total
                ),
                "policy_relocation_vehicle_steps": (
                    policy_result
                    .metrics
                    .relocation_vehicle_steps
                ),
                "baseline_total_cost": (
                    baseline_result
                    .metrics
                    .total_operating_cost
                ),
                "policy_total_cost": (
                    policy_result
                    .metrics
                    .total_operating_cost
                ),
                "baseline_fleet_conserved": (
                    baseline_result
                    .fleet_conservation_passed
                ),
                "policy_fleet_conserved": (
                    policy_result
                    .fleet_conservation_passed
                ),
            }
        )

        if not is_evaluation:
            continue

        for zone_index, zone_id in enumerate(
            zone_ids
        ):
            zone_records.append(
                {
                    "interval_start": timestamp,
                    "service_date": timestamp.date(),
                    "zone_index": zone_index,
                    "zone_id": zone_id,
                    "actual_demand": int(
                        actual_demand[
                            zone_index
                        ]
                    ),
                    "baseline_served": int(
                        baseline_result.served_demand[
                            zone_index
                        ]
                    ),
                    "policy_served": int(
                        policy_result.served_demand[
                            zone_index
                        ]
                    ),
                    "baseline_unmet": int(
                        baseline_result.unmet_demand[
                            zone_index
                        ]
                    ),
                    "policy_unmet": int(
                        policy_result.unmet_demand[
                            zone_index
                        ]
                    ),
                    "baseline_idle_end": int(
                        baseline_result.idle_vehicles[
                            zone_index
                        ]
                    ),
                    "policy_idle_end": int(
                        policy_result.idle_vehicles[
                            zone_index
                        ]
                    ),
                    "policy_outbound_relocation": int(
                        policy_action[
                            zone_index
                        ].sum()
                    ),
                    "policy_inbound_relocation": int(
                        policy_action[
                            :,
                            zone_index
                        ].sum()
                    ),
                }
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
            zone_records
        )
        .sort(
            "interval_start",
            "zone_index",
        )
    )

    daily_metrics = build_daily_metrics(
        step_metrics
    )

    daily_zone_metrics = build_daily_zone_metrics(
        interval_zone_metrics
    )

    zone_summary = build_zone_summary(
        daily_zone_metrics
    )

    expected_day_count = (
        config.end_date_inclusive
        - config.start_date
    ).days + 1

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

    demand_total = int(
        daily_metrics.get_column(
            "demand_total"
        ).sum()
    )

    baseline_served_total = int(
        daily_metrics.get_column(
            "baseline_served_total"
        ).sum()
    )

    policy_served_total = int(
        daily_metrics.get_column(
            "policy_served_total"
        ).sum()
    )

    baseline_unmet_total = int(
        daily_metrics.get_column(
            "baseline_unmet_total"
        ).sum()
    )

    policy_unmet_total = int(
        daily_metrics.get_column(
            "policy_unmet_total"
        ).sum()
    )

    baseline_total_cost = float(
        daily_metrics.get_column(
            "baseline_total_cost"
        ).sum()
    )

    policy_total_cost = float(
        daily_metrics.get_column(
            "policy_total_cost"
        ).sum()
    )

    baseline_service_rate = (
        baseline_served_total
        / demand_total
    )

    policy_service_rate = (
        policy_served_total
        / demand_total
    )

    service_improvements = (
        daily_metrics.get_column(
            "service_rate_improvement"
        )
        .to_numpy()
        .astype(
            np.float64,
            copy=False,
        )
    )

    days_reducing_unmet = int(
        daily_metrics.get_column(
            "policy_reduces_unmet_demand"
        ).sum()
    )

    days_beating_cost = int(
        daily_metrics.get_column(
            "policy_beats_baseline_cost"
        ).sum()
    )

    best_day = (
        daily_metrics.sort(
            "service_rate_improvement",
            descending=True,
        )
        .head(
            1
        )
        .to_dicts()[0]
    )

    worst_day = (
        daily_metrics.sort(
            "service_rate_improvement"
        )
        .head(
            1
        )
        .to_dicts()[0]
    )

    invariants = {
        "validation_day_count_matches_configuration": (
            daily_metrics.height
            == expected_day_count
        ),
        "evaluation_zone_grid_is_dense": (
            interval_zone_metrics.height
            == evaluation_step_count
            * zone_count
        ),
        "baseline_fleet_conserved": (
            baseline_fleet_conserved
        ),
        "policy_fleet_conserved": (
            policy_fleet_conserved
        ),
        "baseline_served_plus_unmet_equals_demand": (
            baseline_served_total
            + baseline_unmet_total
            == demand_total
        ),
        "policy_served_plus_unmet_equals_demand": (
            policy_served_total
            + policy_unmet_total
            == demand_total
        ),
        "forecast_window_uses_validation_split_only": (
            forecast_split_validation[
                "invalid_split_count"
            ]
            == 0
        ),
        "no_test_forecasts_used": (
            forecast_split_validation[
                "observed_splits"
            ]
            == [
                config.validation_split
            ]
        ),
        "warmup_policy_is_identical": (
            not heuristic_config.apply_during_warmup
        ),
        "fleet_sizing_uses_training_split_only": True,
        "future_predictions_not_read": True,
        "future_demand_not_used_for_actions": True,
    }

    if not all(
        invariants.values()
    ):
        failed = [
            name
            for name, passed in invariants.items()
            if not passed
        ]

        raise ForecastGuidedValidationError(
            "Validation invariants failed: "
            + ", ".join(
                failed
            )
        )

    artifacts = {
        "step_metrics": write_parquet_atomically(
            step_metrics,
            path=config.step_metrics_path,
            compression=config.compression,
        ),
        "daily_metrics": write_parquet_atomically(
            daily_metrics,
            path=config.daily_metrics_path,
            compression=config.compression,
        ),
        "daily_zone_metrics": write_parquet_atomically(
            daily_zone_metrics,
            path=config.daily_zone_metrics_path,
            compression=config.compression,
        ),
        "zone_summary": write_parquet_atomically(
            zone_summary,
            path=config.zone_summary_path,
            compression=config.compression,
        ),
    }

    report = {
        "simulation_report_version": "1.0.0",
        "generated_at": datetime.now(
            UTC
        ).isoformat(),
        "dataset_name": config.dataset_name,
        "methodology": {
            "evaluation_mode": (
                "continuous_multi_day_simulation"
            ),
            "actual_demand_drives_transitions": True,
            "fleet_resets_between_days": False,
            "training_only_fleet_sizing": True,
            "warmup_policy": "no_rebalancing",
            "validation_predictions_only": True,
            "test_predictions_used": False,
            "asof_method": projector.method,
            "future_offline_predictions_used": False,
            "future_realized_demand_used_in_decisions": False,
            "seasonal_profile_fitted_on_split": baseline_config.sizing_split,
        },
        "validation_window": {
            "split": config.validation_split,
            "start_date": config.start_date,
            "end_date_inclusive": (
                config.end_date_inclusive
            ),
            "day_count": daily_metrics.height,
            "warmup_step_count": warmup_step_count,
            "evaluation_step_count": (
                evaluation_step_count
            ),
            "first_timestamp": (
                episode_timestamps[0]
            ),
            "last_evaluation_timestamp": (
                episode_timestamps[-1]
            ),
            "last_required_forecast_timestamp": (
                last_required_timestamp
            ),
        },
        "network": {
            "zone_count": zone_count,
            "od_pair_count": (
                zone_count
                * zone_count
            ),
        },
        "fleet": {
            "total_fleet": (
                fleet_sizing.total_fleet
            ),
            "estimated_busy_vehicles": (
                fleet_sizing.estimated_busy_vehicles
            ),
            "reserve_multiplier": (
                fleet_sizing.reserve_multiplier
            ),
        },
        "policy": {
            "name": heuristic_config.policy_name,
            "forecast_horizon_steps": (
                heuristic_config.forecast_horizon_steps
            ),
            "demand_buffer_multiplier": (
                heuristic_config.demand_buffer_multiplier
            ),
            "minimum_zone_reserve": (
                heuristic_config.minimum_zone_reserve
            ),
            "maximum_relocation_steps": (
                heuristic_config.maximum_relocation_steps
            ),
            "maximum_relocation_fraction": (
                heuristic_config.maximum_relocation_fraction
            ),
            "in_transit_supply_credit": (
                heuristic_config.in_transit_supply_credit
            ),
        },
        "aggregate_metrics": {
            "demand_total": demand_total,
            "baseline": {
                "served_demand_total": (
                    baseline_served_total
                ),
                "unmet_demand_total": (
                    baseline_unmet_total
                ),
                "service_rate": (
                    baseline_service_rate
                ),
                "average_idle_vehicles": float(
                    evaluation_step_metrics
                    .get_column(
                        "baseline_idle_total"
                    )
                    .mean()
                ),
                "total_operating_cost": (
                    baseline_total_cost
                ),
            },
            "policy": {
                "served_demand_total": (
                    policy_served_total
                ),
                "unmet_demand_total": (
                    policy_unmet_total
                ),
                "service_rate": (
                    policy_service_rate
                ),
                "average_idle_vehicles": float(
                    evaluation_step_metrics
                    .get_column(
                        "policy_idle_total"
                    )
                    .mean()
                ),
                "relocated_vehicle_total": int(
                    daily_metrics.get_column(
                        "policy_relocated_total"
                    ).sum()
                ),
                "relocation_vehicle_steps": int(
                    daily_metrics.get_column(
                        "policy_relocation_vehicle_steps"
                    ).sum()
                ),
                "total_operating_cost": (
                    policy_total_cost
                ),
            },
        },
        "comparison": {
            "unmet_demand_absolute_reduction": (
                baseline_unmet_total
                - policy_unmet_total
            ),
            "unmet_demand_relative_reduction": (
                (
                    baseline_unmet_total
                    - policy_unmet_total
                )
                / baseline_unmet_total
            ),
            "service_rate_absolute_improvement": (
                policy_service_rate
                - baseline_service_rate
            ),
            "operating_cost_absolute_reduction": (
                baseline_total_cost
                - policy_total_cost
            ),
            "operating_cost_relative_reduction": (
                (
                    baseline_total_cost
                    - policy_total_cost
                )
                / baseline_total_cost
            ),
            "days_reducing_unmet_demand": (
                days_reducing_unmet
            ),
            "days_beating_baseline_cost": (
                days_beating_cost
            ),
            "mean_daily_service_rate_improvement": float(
                np.mean(
                    service_improvements
                )
            ),
            "median_daily_service_rate_improvement": float(
                np.median(
                    service_improvements
                )
            ),
            "daily_service_rate_improvement_std": float(
                np.std(
                    service_improvements
                )
            ),
            "best_day": best_day,
            "worst_day": worst_day,
        },
        "top_zone_improvements": (
            zone_summary.sort(
                "unmet_demand_reduction",
                descending=True,
            )
            .head(
                10
            )
            .to_dicts()
        ),
        "most_harmed_zones": (
            zone_summary.sort(
                "unmet_demand_reduction"
            )
            .head(
                10
            )
            .to_dicts()
        ),
        "daily_metrics": (
            daily_metrics.to_dicts()
        ),
        "forecast_split_validation": (
            forecast_split_validation
        ),
        "artifacts": artifacts,
        "invariants": invariants,
    }

    write_json_atomically(
        config.report_path,
        report,
    )

    return report