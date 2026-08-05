from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import yaml
from numpy.typing import NDArray

from urbanflow.simulation.fleet_simulator import (
    FleetSimulator,
    FleetSnapshot,
    SimulationCosts,
)
from urbanflow.simulation.real_episode_baseline import (
    NetworkInputs,
    RealEpisodeBaselineConfig,
    estimate_initial_fleet,
    extract_episode,
    load_historical_demand,
    load_od_matrices,
    load_zone_index,
    read_config as read_baseline_config,
)


FloatVector = NDArray[np.float64]
FloatMatrix = NDArray[np.float64]
IntegerVector = NDArray[np.int64]
IntegerMatrix = NDArray[np.int64]


class ForecastGuidedPolicyError(RuntimeError):
    """Raised when forecast-guided simulation fails."""


@dataclass(frozen=True)
class ForecastGuidedConfig:
    """Validated forecast-guided policy configuration."""

    dataset_name: str

    baseline_config_path: Path
    forecast_predictions_path: Path

    step_metrics_path: Path
    interval_zone_metrics_path: Path
    zone_summary_path: Path
    report_path: Path

    forecast_timestamp_column: str
    forecast_zone_column: str
    forecast_prediction_column: str
    forecast_split_column: str

    policy_name: str

    forecast_horizon_steps: int
    demand_buffer_multiplier: float
    minimum_zone_reserve: int

    maximum_relocation_steps: int
    maximum_relocation_fraction: float

    in_transit_supply_credit: float
    apply_during_warmup: bool

    compression: str


@dataclass(frozen=True)
class PolicyDecision:
    """One forecast-guided relocation decision."""

    relocation_action: IntegerMatrix

    forecast_target: IntegerVector
    projected_supply: IntegerVector

    surplus: IntegerVector
    deficit: IntegerVector

    requested_relocations: int
    relocation_vehicle_steps: int

    def as_dict(self) -> dict[str, Any]:
        """Return JSON-compatible policy diagnostics."""
        return {
            "forecast_target": (
                self.forecast_target.tolist()
            ),
            "projected_supply": (
                self.projected_supply.tolist()
            ),
            "surplus": self.surplus.tolist(),
            "deficit": self.deficit.tolist(),
            "requested_relocations": (
                self.requested_relocations
            ),
            "relocation_vehicle_steps": (
                self.relocation_vehicle_steps
            ),
        }


def read_config(
    path: Path,
) -> ForecastGuidedConfig:
    """Read forecast-guided experiment configuration."""
    if not path.exists():
        raise ForecastGuidedPolicyError(
            f"Policy config does not exist: {path}"
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
            raise ForecastGuidedPolicyError(
                "Policy config must contain a YAML mapping."
            )

        dataset = raw["dataset"]
        paths = raw["paths"]
        forecast = raw["forecast"]
        policy = raw["policy"]
        output = raw["output"]

        config = ForecastGuidedConfig(
            dataset_name=str(
                dataset["name"]
            ),
            baseline_config_path=Path(
                paths["baseline_config"]
            ),
            forecast_predictions_path=Path(
                paths["forecast_predictions"]
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
            forecast_timestamp_column=str(
                forecast["timestamp_column"]
            ),
            forecast_zone_column=str(
                forecast["zone_column"]
            ),
            forecast_prediction_column=str(
                forecast["prediction_column"]
            ),
            forecast_split_column=str(
                forecast["split_column"]
            ),
            policy_name=str(
                policy["name"]
            ),
            forecast_horizon_steps=int(
                policy["forecast_horizon_steps"]
            ),
            demand_buffer_multiplier=float(
                policy["demand_buffer_multiplier"]
            ),
            minimum_zone_reserve=int(
                policy["minimum_zone_reserve"]
            ),
            maximum_relocation_steps=int(
                policy["maximum_relocation_steps"]
            ),
            maximum_relocation_fraction=float(
                policy["maximum_relocation_fraction"]
            ),
            in_transit_supply_credit=float(
                policy["in_transit_supply_credit"]
            ),
            apply_during_warmup=bool(
                policy["apply_during_warmup"]
            ),
            compression=str(
                output["compression"]
            ),
        )

    except ForecastGuidedPolicyError:
        raise

    except (
        OSError,
        KeyError,
        TypeError,
        ValueError,
        yaml.YAMLError,
    ) as error:
        raise ForecastGuidedPolicyError(
            f"Invalid policy config: {error}"
        ) from error

    validate_config(
        config
    )

    return config


def validate_config(
    config: ForecastGuidedConfig,
) -> None:
    """Validate policy configuration invariants."""
    if not config.dataset_name:
        raise ForecastGuidedPolicyError(
            "Dataset name must not be empty."
        )

    for path in (
        config.baseline_config_path,
        config.forecast_predictions_path,
    ):
        if not path.exists():
            raise ForecastGuidedPolicyError(
                f"Required input does not exist: {path}"
            )

        if path.stat().st_size == 0:
            raise ForecastGuidedPolicyError(
                f"Required input is empty: {path}"
            )

    if config.policy_name != "forecast_guided_greedy":
        raise ForecastGuidedPolicyError(
            "Unsupported policy name: "
            f"{config.policy_name}"
        )

    if config.forecast_horizon_steps < 1:
        raise ForecastGuidedPolicyError(
            "forecast_horizon_steps must be positive."
        )

    if (
        not math.isfinite(
            config.demand_buffer_multiplier
        )
        or config.demand_buffer_multiplier <= 0
    ):
        raise ForecastGuidedPolicyError(
            "demand_buffer_multiplier must be positive."
        )

    if config.minimum_zone_reserve < 0:
        raise ForecastGuidedPolicyError(
            "minimum_zone_reserve must be non-negative."
        )

    if config.maximum_relocation_steps < 1:
        raise ForecastGuidedPolicyError(
            "maximum_relocation_steps must be positive."
        )

    if (
        not math.isfinite(
            config.maximum_relocation_fraction
        )
        or not (
            0.0
            <= config.maximum_relocation_fraction
            <= 1.0
        )
    ):
        raise ForecastGuidedPolicyError(
            "maximum_relocation_fraction must be "
            "between zero and one."
        )

    if (
        not math.isfinite(
            config.in_transit_supply_credit
        )
        or not (
            0.0
            <= config.in_transit_supply_credit
            <= 1.0
        )
    ):
        raise ForecastGuidedPolicyError(
            "in_transit_supply_credit must be "
            "between zero and one."
        )

    if not config.compression:
        raise ForecastGuidedPolicyError(
            "Output compression must not be empty."
        )


def load_forecast_matrix(
    *,
    config: ForecastGuidedConfig,
    baseline_config: RealEpisodeBaselineConfig,
    zone_ids: tuple[int, ...],
    episode_timestamps: list[datetime],
) -> tuple[
    list[datetime],
    FloatMatrix,
]:
    """Load a dense forecast matrix for episode and horizon."""
    if not episode_timestamps:
        raise ForecastGuidedPolicyError(
            "Episode timestamps must not be empty."
        )

    interval_delta = timedelta(
        minutes=baseline_config.interval_minutes
    )

    future_timestamps = [
        episode_timestamps[-1]
        + interval_delta * offset
        for offset in range(
            1,
            config.forecast_horizon_steps + 1,
        )
    ]

    expected_timestamps = [
        *episode_timestamps,
        *future_timestamps,
    ]

    expected_timestamp_count = len(
        expected_timestamps
    )

    first_timestamp = expected_timestamps[0]
    last_required_timestamp = (
        expected_timestamps[-1]
    )

    try:
        frame = pl.read_parquet(
            config.forecast_predictions_path
        )

    except Exception as error:
        raise ForecastGuidedPolicyError(
            f"Could not read forecast predictions: {error}"
        ) from error

    required_columns = {
        config.forecast_timestamp_column,
        config.forecast_zone_column,
        config.forecast_prediction_column,
        config.forecast_split_column,
    }

    missing_columns = sorted(
        required_columns.difference(
            frame.columns
        )
    )

    if missing_columns:
        raise ForecastGuidedPolicyError(
            "Forecast predictions are missing columns: "
            + ", ".join(
                missing_columns
            )
        )

    frame = (
        frame.select(
            pl.col(
                config.forecast_timestamp_column
            ).alias(
                "interval_start"
            ),
            pl.col(
                config.forecast_zone_column
            )
            .cast(
                pl.Int64
            )
            .alias(
                "zone_id"
            ),
            pl.col(
                config.forecast_prediction_column
            )
            .cast(
                pl.Float64
            )
            .alias(
                "forecast_demand"
            ),
            pl.col(
                config.forecast_split_column
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
    )

    if frame.height == 0:
        raise ForecastGuidedPolicyError(
            "No forecasts were found for the episode range."
        )

    null_count = sum(
        int(value)
        for value in frame.null_count().row(0)
    )

    if null_count:
        raise ForecastGuidedPolicyError(
            "Forecast dataset contains null values."
        )

    invalid_prediction_count = (
        frame.filter(
            ~pl.col(
                "forecast_demand"
            ).is_finite()
            | (
                pl.col(
                    "forecast_demand"
                )
                < 0
            )
        )
        .height
    )

    if invalid_prediction_count:
        raise ForecastGuidedPolicyError(
            "Forecasts contain invalid predictions."
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
        raise ForecastGuidedPolicyError(
            "Forecast dataset contains duplicate keys."
        )

    zone_order = pl.DataFrame(
        {
            "zone_id": list(
                zone_ids
            ),
            "zone_index": list(
                range(
                    len(
                        zone_ids
                    )
                )
            ),
        },
        schema={
            "zone_id": pl.Int64,
            "zone_index": pl.Int64,
        },
    )

    timestamp_dtype = frame.schema[
        "interval_start"
    ]

    timestamp_index = pl.DataFrame(
        {
            "interval_start": pl.Series(
                name="interval_start",
                values=expected_timestamps,
                dtype=timestamp_dtype,
            ),
            "forecast_step": pl.Series(
                name="forecast_step",
                values=list(
                    range(
                        expected_timestamp_count
                    )
                ),
                dtype=pl.Int64,
            ),
        }
    )

    expected_grid = (
        timestamp_index.join(
            zone_order,
            how="cross",
        )
        .sort(
            "forecast_step",
            "zone_index",
        )
    )

    selected = (
        expected_grid.join(
            frame,
            on=[
                "interval_start",
                "zone_id",
            ],
            how="left",
        )
        .sort(
            "forecast_step",
            "zone_index",
        )
    )

    expected_rows = (
        expected_timestamp_count
        * len(
            zone_ids
        )
    )

    if selected.height != expected_rows:
        raise ForecastGuidedPolicyError(
            "Expected forecast grid construction failed: "
            f"expected={expected_rows}, "
            f"observed={selected.height}."
        )

    missing_forecast_count = (
        selected.get_column(
            "forecast_demand"
        )
        .null_count()
    )

    if missing_forecast_count:
        missing_examples = (
            selected.filter(
                pl.col(
                    "forecast_demand"
                ).is_null()
            )
            .select(
                "interval_start",
                "zone_id",
                "zone_index",
            )
            .head(
                10
            )
            .to_dicts()
        )

        raise ForecastGuidedPolicyError(
            "Forecast matrix contains missing interval-zone rows: "
            f"missing={missing_forecast_count}, "
            f"examples={missing_examples}."
        )

    observed_timestamps = (
        selected.select(
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

    if observed_timestamps != expected_timestamps:
        raise ForecastGuidedPolicyError(
            "Forecast timestamps do not match the "
            "expected episode and horizon."
        )

    expected_zone_indices = np.tile(
        np.arange(
            len(
                zone_ids
            ),
            dtype=np.int64,
        ),
        expected_timestamp_count,
    )

    observed_zone_indices = (
        selected.get_column(
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
        raise ForecastGuidedPolicyError(
            "Forecast zone ordering is invalid."
        )

    matrix = (
        selected.get_column(
            "forecast_demand"
        )
        .to_numpy()
        .reshape(
            expected_timestamp_count,
            len(
                zone_ids
            ),
        )
        .astype(
            np.float64,
            copy=False,
        )
    )

    if not np.isfinite(
        matrix
    ).all():
        raise ForecastGuidedPolicyError(
            "Final forecast matrix contains "
            "non-finite values."
        )

    if (
        matrix
        < 0
    ).any():
        raise ForecastGuidedPolicyError(
            "Final forecast matrix contains "
            "negative values."
        )

    return (
        expected_timestamps,
        matrix,
    )


def build_forecast_target(
    future_forecasts: FloatMatrix,
    *,
    demand_buffer_multiplier: float,
    minimum_zone_reserve: int,
) -> IntegerVector:
    """Build conservative zone inventory targets."""
    if future_forecasts.ndim != 2:
        raise ForecastGuidedPolicyError(
            "Future forecasts must be a matrix."
        )

    if future_forecasts.shape[0] == 0:
        raise ForecastGuidedPolicyError(
            "At least one future forecast is required."
        )

    peak_forecast = np.max(
        future_forecasts,
        axis=0,
    )

    target = np.ceil(
        peak_forecast
        * demand_buffer_multiplier
    ).astype(
        np.int64
    )

    target += minimum_zone_reserve

    return target


def greedy_relocation_action(
    *,
    snapshot: FleetSnapshot,
    current_forecast: FloatVector,
    future_forecasts: FloatMatrix,
    relocation_travel_steps: IntegerMatrix,
    config: ForecastGuidedConfig,
) -> PolicyDecision:
    """Create one feasible greedy relocation matrix."""
    available = (
        snapshot.available_vehicles
        .astype(
            np.int64,
            copy=True,
        )
    )

    in_transit = (
        snapshot.in_transit_vehicles
        .astype(
            np.int64,
            copy=True,
        )
    )

    zone_count = available.shape[0]

    expected_shape = (
        zone_count,
        zone_count,
    )

    if (
        relocation_travel_steps.shape
        != expected_shape
    ):
        raise ForecastGuidedPolicyError(
            "Relocation matrix shape does not match fleet state."
        )

    current_reserve = np.ceil(
        current_forecast
        * config.demand_buffer_multiplier
    ).astype(
        np.int64
    )

    current_reserve += (
        config.minimum_zone_reserve
    )

    forecast_target = build_forecast_target(
        future_forecasts,
        demand_buffer_multiplier=(
            config.demand_buffer_multiplier
        ),
        minimum_zone_reserve=(
            config.minimum_zone_reserve
        ),
    )

    credited_in_transit = np.floor(
        in_transit
        * config.in_transit_supply_credit
    ).astype(
        np.int64
    )

    projected_supply = (
        available
        + credited_in_transit
    )

    surplus = np.maximum(
        available
        - current_reserve,
        0,
    ).astype(
        np.int64
    )

    deficit = np.maximum(
        forecast_target
        - projected_supply,
        0,
    ).astype(
        np.int64
    )

    relocation_action = np.zeros(
        expected_shape,
        dtype=np.int64,
    )

    relocation_cap = int(
        math.floor(
            int(
                available.sum()
            )
            * config.maximum_relocation_fraction
        )
    )

    if (
        relocation_cap <= 0
        or int(
            surplus.sum()
        )
        == 0
        or int(
            deficit.sum()
        )
        == 0
    ):
        return PolicyDecision(
            relocation_action=relocation_action,
            forecast_target=forecast_target,
            projected_supply=projected_supply,
            surplus=surplus,
            deficit=deficit,
            requested_relocations=0,
            relocation_vehicle_steps=0,
        )

    origin_order = np.argsort(
        -surplus,
        kind="stable",
    )

    relocated_total = 0

    for origin in origin_order:
        origin_surplus = int(
            surplus[
                origin
            ]
        )

        if origin_surplus <= 0:
            continue

        if relocated_total >= relocation_cap:
            break

        eligible_destinations = [
            destination
            for destination in range(
                zone_count
            )
            if (
                destination != origin
                and deficit[
                    destination
                ]
                > 0
                and relocation_travel_steps[
                    origin,
                    destination,
                ]
                <= config.maximum_relocation_steps
            )
        ]

        eligible_destinations.sort(
            key=lambda destination: (
                int(
                    relocation_travel_steps[
                        origin,
                        destination,
                    ]
                ),
                -int(
                    deficit[
                        destination
                    ]
                ),
                destination,
            )
        )

        for destination in eligible_destinations:
            if origin_surplus <= 0:
                break

            remaining_cap = (
                relocation_cap
                - relocated_total
            )

            if remaining_cap <= 0:
                break

            flow = min(
                origin_surplus,
                int(
                    deficit[
                        destination
                    ]
                ),
                remaining_cap,
            )

            if flow <= 0:
                continue

            relocation_action[
                origin,
                destination,
            ] += flow

            origin_surplus -= flow

            deficit[
                destination
            ] -= flow

            relocated_total += flow

    outbound = relocation_action.sum(
        axis=1
    )

    if (
        outbound
        > available
    ).any():
        raise ForecastGuidedPolicyError(
            "Generated relocation action exceeds "
            "available vehicles."
        )

    vehicle_steps = int(
        np.sum(
            relocation_action
            * relocation_travel_steps
        )
    )

    return PolicyDecision(
        relocation_action=relocation_action,
        forecast_target=forecast_target,
        projected_supply=projected_supply,
        surplus=surplus,
        deficit=deficit,
        requested_relocations=relocated_total,
        relocation_vehicle_steps=vehicle_steps,
    )


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

        raise ForecastGuidedPolicyError(
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
    """Convert values to strict JSON-compatible objects."""
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
        np.ndarray,
    ):
        return [
            sanitize_json_value(
                item
            )
            for item in value.tolist()
        ]

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

        raise ForecastGuidedPolicyError(
            f"Could not write policy report: {error}"
        ) from error


def build_simulator(
    *,
    baseline_config: RealEpisodeBaselineConfig,
    network: NetworkInputs,
    initial_vehicles: IntegerVector,
) -> FleetSimulator:
    """Build one identically configured simulator."""
    return FleetSimulator(
        zone_ids=network.zone_ids,
        initial_vehicles=initial_vehicles,
        passenger_destination_probabilities=(
            network.destination_probabilities
        ),
        relocation_travel_steps=(
            network.relocation_travel_steps
        ),
        passenger_travel_steps=(
            network.passenger_travel_steps
        ),
        costs=SimulationCosts(
            unmet_demand_penalty=(
                baseline_config.unmet_demand_penalty
            ),
            relocation_cost_per_vehicle_step=(
                baseline_config
                .relocation_cost_per_vehicle_step
            ),
            idle_vehicle_cost=(
                baseline_config.idle_vehicle_cost
            ),
        ),
    )


def run_forecast_guided_comparison(
    config_path: Path,
) -> dict[str, Any]:
    """Run no-rebalancing and forecast-guided policies."""
    config = read_config(
        config_path
    )

    baseline_config = read_baseline_config(
        config.baseline_config_path
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
        extract_episode(
            historical_demand,
            zone_count=zone_count,
            config=baseline_config,
        )
    )

    (
        forecast_timestamps,
        forecast_matrix,
    ) = load_forecast_matrix(
        config=config,
        baseline_config=baseline_config,
        zone_ids=zone_ids,
        episode_timestamps=episode_timestamps,
    )

    if (
        forecast_timestamps[
            :len(
                episode_timestamps
            )
        ]
        != episode_timestamps
    ):
        raise ForecastGuidedPolicyError(
            "Forecast and demand episode timestamps do not align."
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

    evaluation_baseline_results = []
    evaluation_policy_results = []

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
            timestamp.date()
            == baseline_config.service_date
        )

        apply_policy = (
            is_evaluation
            or config.apply_during_warmup
        )

        current_forecast = forecast_matrix[
            step_index
        ]

        future_start = (
            step_index
            + 1
        )

        future_end = (
            future_start
            + config.forecast_horizon_steps
        )

        future_forecasts = forecast_matrix[
            future_start:future_end
        ]

        if (
            apply_policy
            and future_forecasts.shape[0]
            != config.forecast_horizon_steps
        ):
            raise ForecastGuidedPolicyError(
                "Forecast horizon is incomplete "
                f"at step {step_index}."
            )

        if apply_policy:
            decision = greedy_relocation_action(
                snapshot=policy_simulator.snapshot(),
                current_forecast=current_forecast,
                future_forecasts=future_forecasts,
                relocation_travel_steps=(
                    relocation_travel_steps
                ),
                config=config,
            )

            policy_action = (
                decision.relocation_action
            )

        else:
            decision = PolicyDecision(
                relocation_action=zero_action.copy(),
                forecast_target=np.zeros(
                    zone_count,
                    dtype=np.int64,
                ),
                projected_supply=np.zeros(
                    zone_count,
                    dtype=np.int64,
                ),
                surplus=np.zeros(
                    zone_count,
                    dtype=np.int64,
                ),
                deficit=np.zeros(
                    zone_count,
                    dtype=np.int64,
                ),
                requested_relocations=0,
                relocation_vehicle_steps=0,
            )

            policy_action = zero_action

        baseline_result = baseline_simulator.step(
            demand=actual_demand,
            relocation_action=zero_action,
        )

        policy_result = policy_simulator.step(
            demand=actual_demand,
            relocation_action=policy_action,
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
                "policy_applied": apply_policy,
                "forecast_demand_total": float(
                    current_forecast.sum()
                ),
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
                "cost_improvement": (
                    baseline_result
                    .metrics
                    .total_operating_cost
                    - policy_result
                    .metrics
                    .total_operating_cost
                ),
                "baseline_service_rate": (
                    baseline_result
                    .metrics
                    .service_rate
                ),
                "policy_service_rate": (
                    policy_result
                    .metrics
                    .service_rate
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

        evaluation_baseline_results.append(
            baseline_result
        )

        evaluation_policy_results.append(
            policy_result
        )

        for zone_index, zone_id in enumerate(
            zone_ids
        ):
            zone_records.append(
                {
                    "interval_start": timestamp,
                    "zone_index": zone_index,
                    "zone_id": zone_id,
                    "forecast_demand": float(
                        current_forecast[
                            zone_index
                        ]
                    ),
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
                    "policy_forecast_target": int(
                        decision.forecast_target[
                            zone_index
                        ]
                    ),
                    "policy_projected_supply": int(
                        decision.projected_supply[
                            zone_index
                        ]
                    ),
                }
            )

    if not evaluation_policy_results:
        raise ForecastGuidedPolicyError(
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
            zone_records
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
        .sort(
            "zone_index"
        )
    )

    def aggregate_results(
        results: list[Any],
    ) -> dict[str, Any]:
        demand_total = sum(
            result.metrics.demand_total
            for result in results
        )

        served_total = sum(
            result.metrics.served_demand_total
            for result in results
        )

        unmet_total = sum(
            result.metrics.unmet_demand_total
            for result in results
        )

        idle_total = sum(
            result.metrics.idle_vehicle_total
            for result in results
        )

        relocated_total = sum(
            result.metrics.relocated_vehicle_total
            for result in results
        )

        relocation_vehicle_steps = sum(
            result.metrics.relocation_vehicle_steps
            for result in results
        )

        unmet_cost = sum(
            result.metrics.unmet_demand_cost
            for result in results
        )

        relocation_cost = sum(
            result.metrics.relocation_cost
            for result in results
        )

        idle_cost = sum(
            result.metrics.idle_vehicle_cost
            for result in results
        )

        total_cost = sum(
            result.metrics.total_operating_cost
            for result in results
        )

        return {
            "step_count": len(
                results
            ),
            "demand_total": demand_total,
            "served_demand_total": served_total,
            "unmet_demand_total": unmet_total,
            "service_rate": (
                served_total
                / demand_total
                if demand_total > 0
                else 1.0
            ),
            "average_idle_vehicles": (
                idle_total
                / len(
                    results
                )
            ),
            "relocated_vehicle_total": (
                relocated_total
            ),
            "relocation_vehicle_steps": (
                relocation_vehicle_steps
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
                total_cost
            ),
        }

    baseline_metrics = aggregate_results(
        evaluation_baseline_results
    )

    policy_metrics = aggregate_results(
        evaluation_policy_results
    )

    baseline_cost = float(
        baseline_metrics[
            "total_operating_cost"
        ]
    )

    policy_cost = float(
        policy_metrics[
            "total_operating_cost"
        ]
    )

    baseline_unmet = int(
        baseline_metrics[
            "unmet_demand_total"
        ]
    )

    policy_unmet = int(
        policy_metrics[
            "unmet_demand_total"
        ]
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

    report = {
        "simulation_report_version": "1.0.0",
        "generated_at": datetime.now(
            UTC
        ).isoformat(),
        "dataset_name": config.dataset_name,
        "episode": {
            "service_date": (
                baseline_config.service_date
            ),
            "evaluation_split": (
                baseline_config.evaluation_split
            ),
            "warmup_days": (
                baseline_config.warmup_days
            ),
            "evaluation_step_count": len(
                evaluation_policy_results
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
            "name": config.policy_name,
            "forecast_horizon_steps": (
                config.forecast_horizon_steps
            ),
            "demand_buffer_multiplier": (
                config.demand_buffer_multiplier
            ),
            "minimum_zone_reserve": (
                config.minimum_zone_reserve
            ),
            "maximum_relocation_steps": (
                config.maximum_relocation_steps
            ),
            "maximum_relocation_fraction": (
                config.maximum_relocation_fraction
            ),
            "in_transit_supply_credit": (
                config.in_transit_supply_credit
            ),
            "apply_during_warmup": (
                config.apply_during_warmup
            ),
        },
        "baseline_metrics": baseline_metrics,
        "policy_metrics": policy_metrics,
        "comparison": {
            "unmet_demand_absolute_reduction": (
                baseline_unmet
                - policy_unmet
            ),
            "unmet_demand_relative_reduction": (
                (
                    baseline_unmet
                    - policy_unmet
                )
                / baseline_unmet
                if baseline_unmet > 0
                else 0.0
            ),
            "service_rate_absolute_improvement": (
                float(
                    policy_metrics[
                        "service_rate"
                    ]
                )
                - float(
                    baseline_metrics[
                        "service_rate"
                    ]
                )
            ),
            "operating_cost_absolute_reduction": (
                baseline_cost
                - policy_cost
            ),
            "operating_cost_relative_reduction": (
                (
                    baseline_cost
                    - policy_cost
                )
                / baseline_cost
                if baseline_cost > 0
                else 0.0
            ),
            "policy_beats_baseline_cost": (
                policy_cost
                < baseline_cost
            ),
            "policy_reduces_unmet_demand": (
                policy_unmet
                < baseline_unmet
            ),
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
        "worst_zone_changes": (
            zone_summary.sort(
                "unmet_demand_reduction"
            )
            .head(
                10
            )
            .to_dicts()
        ),
        "artifacts": artifacts,
        "invariants": {
            "policies_use_identical_actual_demand": True,
            "policies_use_identical_initial_fleet": True,
            "policies_use_identical_warmup_policy": (
                not config.apply_during_warmup
            ),
            "baseline_fleet_conserved": all(
                result.fleet_conservation_passed
                for result in evaluation_baseline_results
            ),
            "policy_fleet_conserved": all(
                result.fleet_conservation_passed
                for result in evaluation_policy_results
            ),
            "baseline_served_plus_unmet_equals_demand": (
                int(
                    baseline_metrics[
                        "served_demand_total"
                    ]
                )
                + int(
                    baseline_metrics[
                        "unmet_demand_total"
                    ]
                )
                == int(
                    baseline_metrics[
                        "demand_total"
                    ]
                )
            ),
            "policy_served_plus_unmet_equals_demand": (
                int(
                    policy_metrics[
                        "served_demand_total"
                    ]
                )
                + int(
                    policy_metrics[
                        "unmet_demand_total"
                    ]
                )
                == int(
                    policy_metrics[
                        "demand_total"
                    ]
                )
            ),
        },
    }

    write_json_atomically(
        config.report_path,
        report,
    )

    return report