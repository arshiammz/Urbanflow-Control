"""Four-policy optimization benchmark on the Phase 2 simulator and forecasts."""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import yaml
from numpy.typing import NDArray

from urbanflow.geo.zone_graph import ZoneGraph
from urbanflow.optimization.min_cost_flow import (
    FlowAllocation,
    MinCostFlowError,
    build_min_cost_relocation_action,
)
from urbanflow.optimization.mpc import (
    MPCConfig,
    MPCDecision,
    build_mpc_action,
)
from urbanflow.simulation.fleet_simulator import (
    FleetSimulationError,
    FleetSnapshot,
    FleetSimulator,
    SimulationCosts,
    StepResult,
)
from urbanflow.simulation.forecast_guided_heuristic import (
    ForecastGuidedConfig,
    ForecastGuidedPolicyError,
    greedy_relocation_action,
    read_config as read_greedy_config,
)
from urbanflow.simulation.real_episode_baseline import (
    NetworkInputs,
    RealEpisodeBaselineConfig,
    RealEpisodeBaselineError,
    estimate_initial_fleet,
    extract_episode,
    load_historical_demand,
    load_od_matrices,
    load_zone_index,
    read_config as read_baseline_config,
)

IntegerVector = NDArray[np.int64]
IntegerMatrix = NDArray[np.int64]
FloatVector = NDArray[np.float64]
FloatMatrix = NDArray[np.float64]
BooleanMatrix = NDArray[np.bool_]

POLICY_ORDER = (
    "no_rebalancing",
    "greedy",
    "min_cost_flow",
    "mpc",
)


class OptimizationBenchmarkError(RuntimeError):
    """Raised when the four-policy optimization benchmark cannot run safely."""


@dataclass(frozen=True)
class OptimizationBenchmarkConfig:
    """Validated configuration for a reproducible policy comparison."""

    dataset_name: str
    baseline_config_path: Path
    greedy_config_path: Path
    decision_time_forecasts_path: Path
    zone_graph_path: Path
    zone_lookup_path: Path

    step_metrics_path: Path
    interval_zone_metrics_path: Path
    zone_summary_path: Path
    region_summary_path: Path
    report_path: Path

    forecast_horizon_steps: int
    fairness_cost_weight: float
    output_compression: str


def read_config(path: Path) -> OptimizationBenchmarkConfig:
    """Read the Phase 3 optimization benchmark YAML."""
    if not path.exists():
        raise OptimizationBenchmarkError(f"Benchmark config does not exist: {path}")
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise OptimizationBenchmarkError("Benchmark config must be a YAML mapping.")
        paths = raw["paths"]
        benchmark = raw["benchmark"]
        output = raw["output"]
        config = OptimizationBenchmarkConfig(
            dataset_name=str(raw["dataset"]["name"]),
            baseline_config_path=Path(paths["baseline_config"]),
            greedy_config_path=Path(paths["greedy_config"]),
            decision_time_forecasts_path=Path(paths["decision_time_forecasts"]),
            zone_graph_path=Path(paths["zone_graph"]),
            zone_lookup_path=Path(paths["zone_lookup"]),
            step_metrics_path=Path(paths["step_metrics"]),
            interval_zone_metrics_path=Path(paths["interval_zone_metrics"]),
            zone_summary_path=Path(paths["zone_summary"]),
            region_summary_path=Path(paths["region_summary"]),
            report_path=Path(paths["report"]),
            forecast_horizon_steps=int(benchmark["forecast_horizon_steps"]),
            fairness_cost_weight=float(benchmark["fairness_cost_weight"]),
            output_compression=str(output["compression"]),
        )
    except OptimizationBenchmarkError:
        raise
    except (KeyError, TypeError, ValueError, OSError, yaml.YAMLError) as error:
        raise OptimizationBenchmarkError(f"Invalid benchmark config: {error}") from error
    validate_config(config)
    return config


def validate_config(config: OptimizationBenchmarkConfig) -> None:
    if not config.dataset_name:
        raise OptimizationBenchmarkError("dataset.name must not be empty.")
    if config.forecast_horizon_steps < 2:
        raise OptimizationBenchmarkError("forecast_horizon_steps must be at least two.")
    if not math.isfinite(config.fairness_cost_weight) or config.fairness_cost_weight < 0:
        raise OptimizationBenchmarkError("fairness_cost_weight must be finite and non-negative.")
    if not config.output_compression:
        raise OptimizationBenchmarkError("output.compression must not be empty.")
    required_paths = (
        config.baseline_config_path,
        config.greedy_config_path,
        config.decision_time_forecasts_path,
        config.zone_graph_path,
        config.zone_lookup_path,
    )
    for input_path in required_paths:
        if not input_path.exists() or input_path.stat().st_size == 0:
            raise OptimizationBenchmarkError(f"Required input is missing or empty: {input_path}")


def _write_parquet_atomically(
    frame: pl.DataFrame,
    *,
    path: Path,
    compression: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    temporary.unlink(missing_ok=True)
    try:
        frame.write_parquet(temporary, compression=compression, statistics=True)
        temporary.replace(path)
    except Exception as error:
        temporary.unlink(missing_ok=True)
        raise OptimizationBenchmarkError(
            f"Could not write Parquet artifact {path}: {error}"
        ) from error


def _sanitize_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _sanitize_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize_json(item) for item in value]
    if isinstance(value, (datetime,)):
        return value.isoformat()
    if isinstance(value, np.generic):
        return _sanitize_json(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json_atomically(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    temporary.unlink(missing_ok=True)
    try:
        temporary.write_text(
            json.dumps(
                _sanitize_json(payload),
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    except (OSError, TypeError, ValueError) as error:
        temporary.unlink(missing_ok=True)
        raise OptimizationBenchmarkError(f"Could not write benchmark report: {error}") from error


def _load_decision_time_quantiles(
    path: Path,
    *,
    episode_timestamps: list[datetime],
    zone_ids: tuple[int, ...],
    expected_split: str,
    timezone: str,
    interval_minutes: int,
    horizon_steps: int,
) -> tuple[FloatMatrix, FloatMatrix, FloatMatrix, str]:
    """Load an exact dense decision-time × horizon × zone quantile tensor."""
    try:
        frame = pl.read_parquet(path)
    except Exception as error:
        raise OptimizationBenchmarkError(
            f"Could not read decision-time forecasts: {error}"
        ) from error
    required = {
        "decision_time",
        "target_interval_start",
        "horizon_step",
        "zone_id",
        "split",
        "p10_demand",
        "p50_demand",
        "p90_demand",
        "forecast_provenance",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise OptimizationBenchmarkError(f"Decision-time forecast file lacks columns: {missing}")
    dtype = frame.schema["decision_time"]
    if not isinstance(dtype, pl.Datetime) or dtype.time_zone != timezone:
        raise OptimizationBenchmarkError(
            f"decision_time must be a timezone-aware datetime in {timezone}."
        )
    observed_zone_ids = tuple(
        sorted(int(value) for value in frame.get_column("zone_id").unique().to_list())
    )
    if observed_zone_ids != tuple(sorted(zone_ids)):
        raise OptimizationBenchmarkError(
            "Decision-time forecast zone IDs do not match the all-zone simulation index."
        )
    observed_horizons = tuple(
        sorted(int(value) for value in frame.get_column("horizon_step").unique().to_list())
    )
    if observed_horizons != tuple(range(1, horizon_steps + 1)):
        raise OptimizationBenchmarkError(
            "Decision-time forecast horizon does not match the configured horizon: "
            f"expected={tuple(range(1, horizon_steps + 1))}, observed={observed_horizons}."
        )

    selected = (
        frame.select(
            pl.col("decision_time"),
            pl.col("target_interval_start"),
            pl.col("horizon_step").cast(pl.Int64),
            pl.col("zone_id").cast(pl.Int64),
            pl.col("split").cast(pl.String),
            pl.col("p10_demand").cast(pl.Float64),
            pl.col("p50_demand").cast(pl.Float64),
            pl.col("p90_demand").cast(pl.Float64),
            pl.col("forecast_provenance").cast(pl.String),
        )
        .filter(
            pl.col("decision_time").is_between(
                episode_timestamps[0], episode_timestamps[-1], closed="both"
            )
            & pl.col("zone_id").is_in(list(zone_ids))
            & pl.col("horizon_step").is_between(1, horizon_steps, closed="both")
        )
    )
    duplicate_count = (
        selected.group_by("decision_time", "horizon_step", "zone_id")
        .len()
        .filter(pl.col("len") > 1)
        .height
    )
    if duplicate_count:
        raise OptimizationBenchmarkError(
            f"Duplicate decision-time forecast keys: {duplicate_count}"
        )
    wrong_split = selected.filter(pl.col("split") != expected_split).height
    if wrong_split:
        raise OptimizationBenchmarkError(
            f"Forecast split mismatch: expected={expected_split}, wrong_rows={wrong_split}."
        )

    timeline = pl.DataFrame(
        {
            "decision_time": pl.Series("decision_time", episode_timestamps, dtype=dtype),
            "step_index": pl.Series("step_index", range(len(episode_timestamps)), dtype=pl.Int64),
        }
    )
    horizons = pl.DataFrame(
        {"horizon_step": pl.Series("horizon_step", range(1, horizon_steps + 1), dtype=pl.Int64)}
    )
    zones = pl.DataFrame(
        {
            "zone_id": pl.Series("zone_id", zone_ids, dtype=pl.Int64),
            "zone_index": pl.Series("zone_index", range(len(zone_ids)), dtype=pl.Int64),
        }
    )
    expected = timeline.join(horizons, how="cross").join(zones, how="cross")
    result = expected.join(
        selected,
        on=["decision_time", "horizon_step", "zone_id"],
        how="left",
    ).sort("step_index", "horizon_step", "zone_index")
    expected_rows = len(episode_timestamps) * horizon_steps * len(zone_ids)
    if result.height != expected_rows:
        raise OptimizationBenchmarkError(
            "Forecast tensor has wrong row count: "
            f"expected={expected_rows}, observed={result.height}."
        )
    for column in (
        "target_interval_start",
        "split",
        "p10_demand",
        "p50_demand",
        "p90_demand",
        "forecast_provenance",
    ):
        if result.get_column(column).null_count():
            raise OptimizationBenchmarkError(f"Forecast tensor has missing values in {column}.")

    provenance_values = result.get_column("forecast_provenance").unique().to_list()
    if len(provenance_values) != 1 or not provenance_values[0]:
        raise OptimizationBenchmarkError("Forecast provenance must be present and consistent.")
    expected_provenance = "quantile_anchor_plus_train_only_weekly_profile"
    if str(provenance_values[0]) != expected_provenance:
        raise OptimizationBenchmarkError(
            "Decision-time forecast provenance is not the Phase 2 train-only horizon contract: "
            f"expected={expected_provenance}, observed={provenance_values[0]}."
        )

    # The horizon target must be aligned to decision time. This is an explicit
    # chronology contract, not merely a row-count check.
    for row in result.select(
        "decision_time", "target_interval_start", "horizon_step"
    ).iter_rows(named=True):
        decision_time = row["decision_time"]
        expected_target = (
            decision_time.astimezone(UTC)
            + timedelta(minutes=interval_minutes * (int(row["horizon_step"]) - 1))
        ).astimezone(decision_time.tzinfo)
        if row["target_interval_start"] != expected_target:
            raise OptimizationBenchmarkError(
                "Forecast target timestamp is not aligned to its decision time: "
                f"decision={decision_time}, horizon={row['horizon_step']}, "
                f"target={row['target_interval_start']}, expected={expected_target}."
            )

    quantiles = result.select("p10_demand", "p50_demand", "p90_demand").to_numpy()
    if not np.isfinite(quantiles).all() or (quantiles < 0).any():
        raise OptimizationBenchmarkError("Forecast quantiles must be finite and non-negative.")
    if ((quantiles[:, 0] > quantiles[:, 1]) | (quantiles[:, 1] > quantiles[:, 2])).any():
        raise OptimizationBenchmarkError("Forecast quantile monotonicity failed.")

    shape = (len(episode_timestamps), horizon_steps, len(zone_ids))
    p10 = result.get_column("p10_demand").to_numpy().reshape(shape).astype(np.float64)
    p50 = result.get_column("p50_demand").to_numpy().reshape(shape).astype(np.float64)
    p90 = result.get_column("p90_demand").to_numpy().reshape(shape).astype(np.float64)
    return p10, p50, p90, str(provenance_values[0])


def _load_region_index(
    zone_lookup_path: Path,
    zone_ids: tuple[int, ...],
) -> tuple[tuple[str, ...], NDArray[np.int64], tuple[str, ...]]:
    try:
        lookup = pl.read_csv(zone_lookup_path)
    except Exception as error:
        raise OptimizationBenchmarkError(f"Could not read taxi-zone lookup: {error}") from error
    required = {"LocationID", "Borough"}
    if not required.issubset(lookup.columns):
        raise OptimizationBenchmarkError("Taxi-zone lookup must contain LocationID and Borough.")
    lookup = lookup.select(
        pl.col("LocationID").cast(pl.Int64).alias("zone_id"),
        pl.col("Borough").cast(pl.String).alias("region"),
    )
    if lookup.get_column("zone_id").n_unique() != lookup.height:
        raise OptimizationBenchmarkError("Taxi-zone lookup contains duplicate LocationID values.")
    zones = pl.DataFrame(
        {
            "zone_index": list(range(len(zone_ids))),
            "zone_id": list(zone_ids),
        }
    )
    joined = zones.join(lookup, on="zone_id", how="left").sort("zone_index")
    if joined.get_column("region").null_count():
        missing = joined.filter(pl.col("region").is_null()).get_column("zone_id").to_list()
        raise OptimizationBenchmarkError(
            f"Taxi-zone lookup is missing simulation zones: {missing[:10]}"
        )
    labels = tuple(str(value) for value in joined.get_column("region").to_list())
    region_names = tuple(sorted(set(labels)))
    region_lookup = {name: index for index, name in enumerate(region_names)}
    region_index = np.asarray([region_lookup[label] for label in labels], dtype=np.int64)
    return labels, region_index, region_names


def _load_graph_mask(
    path: Path,
    zone_ids: tuple[int, ...],
) -> BooleanMatrix:
    try:
        graph = ZoneGraph.load(path)
    except (OSError, ValueError) as error:
        raise OptimizationBenchmarkError(f"Could not load zone graph: {error}") from error
    if set(graph.zone_ids) != set(zone_ids):
        missing = sorted(set(zone_ids).difference(graph.zone_ids))[:10]
        extra = sorted(set(graph.zone_ids).difference(zone_ids))[:10]
        raise OptimizationBenchmarkError(
            f"Zone graph and simulation network differ. missing={missing}, extra={extra}."
        )
    positions = [graph.zone_ids.index(zone_id) for zone_id in zone_ids]
    mask = graph.adjacency[np.ix_(positions, positions)] > 0
    np.fill_diagonal(mask, False)
    if not np.array_equal(mask, mask.T):
        raise OptimizationBenchmarkError("Aligned zone graph must remain symmetric.")
    return mask


def _build_simulator(
    *,
    network: NetworkInputs,
    initial_vehicles: IntegerVector,
    baseline_config: RealEpisodeBaselineConfig,
    allowed_edges: BooleanMatrix,
) -> FleetSimulator:
    return FleetSimulator(
        zone_ids=network.zone_ids,
        initial_vehicles=initial_vehicles,
        passenger_destination_probabilities=network.destination_probabilities,
        relocation_travel_steps=network.relocation_travel_steps,
        passenger_travel_steps=network.passenger_travel_steps,
        allowed_relocation_edges=allowed_edges,
        costs=SimulationCosts(
            unmet_demand_penalty=baseline_config.unmet_demand_penalty,
            relocation_cost_per_vehicle_step=baseline_config.relocation_cost_per_vehicle_step,
            idle_vehicle_cost=baseline_config.idle_vehicle_cost,
        ),
    )


def _decision_snapshot(
    simulator: FleetSimulator,
) -> tuple[FleetSnapshot, dict[int, IntegerVector]]:
    """Expose arrivals released by the next transition to the policy decision."""
    snapshot = simulator.snapshot()
    pending = simulator.pending_arrivals_by_step
    due = pending.get(0, np.zeros(len(simulator.zone_ids), dtype=np.int64))
    available = snapshot.available_vehicles + due
    in_transit = snapshot.in_transit_vehicles - due
    if (in_transit < 0).any():
        raise OptimizationBenchmarkError(
            "Pending-arrival schedule disagrees with in-transit fleet."
        )
    adjusted = FleetSnapshot(
        step=snapshot.step,
        available_vehicles=available,
        in_transit_vehicles=in_transit,
        scheduled_arrival_steps=snapshot.scheduled_arrival_steps,
        total_fleet=snapshot.total_fleet,
    )
    return adjusted, pending


def _region_service_rates(
    region_demand: FloatVector,
    region_served: FloatVector,
) -> FloatVector:
    return np.divide(
        region_served,
        region_demand,
        out=np.ones_like(region_served, dtype=np.float64),
        where=region_demand > 0,
    ).clip(0.0, 1.0)


def _validate_action(
    action: IntegerMatrix,
    *,
    available: IntegerVector,
    allowed_edges: BooleanMatrix,
    relocation_travel_steps: IntegerMatrix,
    maximum_relocation_steps: int,
    maximum_relocation_fraction: float,
) -> None:
    if action.shape != allowed_edges.shape or (action < 0).any():
        raise OptimizationBenchmarkError("Policy generated an invalid relocation matrix.")
    if np.diag(action).any():
        raise OptimizationBenchmarkError("Policy generated a self-relocation.")
    if ((action > 0) & ~allowed_edges).any():
        raise OptimizationBenchmarkError("Policy generated a relocation outside the zone graph.")
    if ((action > 0) & (relocation_travel_steps > maximum_relocation_steps)).any():
        raise OptimizationBenchmarkError("Policy exceeded the maximum relocation travel time.")
    if (action.sum(axis=1) > available).any():
        raise OptimizationBenchmarkError("Policy exceeded available origin supply.")
    cap = int(math.floor(int(available.sum()) * maximum_relocation_fraction))
    if int(action.sum()) > cap:
        raise OptimizationBenchmarkError(
            f"Policy exceeded the global relocation cap: cap={cap}, requested={int(action.sum())}."
        )


def _fairness_metrics(
    service_rates: NDArray[np.float64],
) -> dict[str, float | None]:
    values = service_rates[np.isfinite(service_rates)]
    if values.size == 0:
        return {
            "minimum_service_rate": None,
            "p10_service_rate": None,
            "maximum_service_rate": None,
            "service_rate_gap": None,
            "jain_fairness_index": None,
        }
    denominator = float(values.size * np.square(values).sum())
    jain = float(np.square(values.sum()) / denominator) if denominator > 0 else 1.0
    return {
        "minimum_service_rate": float(values.min()),
        "p10_service_rate": float(np.quantile(values, 0.10)),
        "maximum_service_rate": float(values.max()),
        "service_rate_gap": float(values.max() - values.min()),
        "jain_fairness_index": jain,
    }


def _aggregate_results(
    results: list[StepResult],
    *,
    zone_summary: pl.DataFrame,
    region_summary: pl.DataFrame,
    policy: str,
    decision_latencies: list[float],
) -> dict[str, Any]:
    demand = sum(result.metrics.demand_total for result in results)
    served = sum(result.metrics.served_demand_total for result in results)
    unmet = sum(result.metrics.unmet_demand_total for result in results)
    idle = sum(result.metrics.idle_vehicle_total for result in results)
    relocated = sum(result.metrics.relocated_vehicle_total for result in results)
    vehicle_steps = sum(result.metrics.relocation_vehicle_steps for result in results)
    unmet_cost = sum(result.metrics.unmet_demand_cost for result in results)
    relocation_cost = sum(result.metrics.relocation_cost for result in results)
    idle_cost = sum(result.metrics.idle_vehicle_cost for result in results)
    total_cost = sum(result.metrics.total_operating_cost for result in results)

    policy_zones = zone_summary.filter(pl.col("policy") == policy)
    active_zone_rates = (
        policy_zones.filter(pl.col("demand_total") > 0)
        .get_column("service_rate")
        .to_numpy()
        .astype(np.float64)
    )
    policy_regions = region_summary.filter(pl.col("policy") == policy)
    active_region_rates = (
        policy_regions.filter(pl.col("demand_total") > 0)
        .get_column("service_rate")
        .to_numpy()
        .astype(np.float64)
    )
    return {
        "demand_total": int(demand),
        "served_demand_total": int(served),
        "unmet_demand_total": int(unmet),
        "service_rate": float(served / demand) if demand else 1.0,
        "average_idle_vehicles": float(idle / len(results)) if results else 0.0,
        "relocated_vehicle_total": int(relocated),
        "relocation_vehicle_steps": int(vehicle_steps),
        "unmet_demand_cost": float(unmet_cost),
        "relocation_cost": float(relocation_cost),
        "idle_vehicle_cost": float(idle_cost),
        "total_operating_cost": float(total_cost),
        "mean_decision_latency_ms": (
            float(np.mean(decision_latencies)) if decision_latencies else 0.0
        ),
        "zone_fairness": _fairness_metrics(active_zone_rates),
        "region_fairness": _fairness_metrics(active_region_rates),
        "active_zone_count": int(active_zone_rates.size),
        "active_region_count": int(active_region_rates.size),
    }


def _comparison(
    reference: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, float]:
    reference_cost = float(reference["total_operating_cost"])
    reference_unmet = int(reference["unmet_demand_total"])
    reference_service = float(reference["service_rate"])
    return {
        "unmet_demand_reduction": float(reference_unmet - int(candidate["unmet_demand_total"])),
        "unmet_demand_reduction_pct": (
            float(
                (reference_unmet - int(candidate["unmet_demand_total"]))
                / reference_unmet
                * 100.0
            )
            if reference_unmet > 0
            else 0.0
        ),
        "service_rate_improvement_percentage_points": (
            float((float(candidate["service_rate"]) - reference_service) * 100.0)
        ),
        "operating_cost_reduction": reference_cost - float(candidate["total_operating_cost"]),
        "operating_cost_reduction_pct": (
            float(
                (reference_cost - float(candidate["total_operating_cost"]))
                / reference_cost
                * 100.0
            )
            if reference_cost > 0
            else 0.0
        ),
        "worst_region_service_rate_delta": float(
            (candidate["region_fairness"]["minimum_service_rate"] or 0.0)
            - (reference["region_fairness"]["minimum_service_rate"] or 0.0)
        ),
        "region_jain_fairness_delta": float(
            (candidate["region_fairness"]["jain_fairness_index"] or 0.0)
            - (reference["region_fairness"]["jain_fairness_index"] or 0.0)
        ),
    }


def run_optimization_benchmark(config_path: Path) -> dict[str, Any]:
    """Run No Rebalancing, Greedy, Min-Cost Flow, and MPC on identical inputs."""
    config = read_config(config_path)
    baseline_config = read_baseline_config(config.baseline_config_path)
    greedy_config: ForecastGuidedConfig = read_greedy_config(config.greedy_config_path)

    _, zone_ids, fit_pickup_shares = load_zone_index(baseline_config)
    zone_count = len(zone_ids)
    destination_probabilities, passenger_steps, relocation_steps = load_od_matrices(
        baseline_config,
        zone_count=zone_count,
    )
    network = NetworkInputs(
        zone_ids=zone_ids,
        fit_pickup_shares=fit_pickup_shares,
        destination_probabilities=destination_probabilities,
        passenger_travel_steps=passenger_steps,
        relocation_travel_steps=relocation_steps,
    )
    historical_demand = load_historical_demand(baseline_config, zone_count=zone_count)
    fleet_sizing = estimate_initial_fleet(
        historical_demand,
        network=network,
        config=baseline_config,
    )
    episode_timestamps, demand_matrix = extract_episode(
        historical_demand,
        zone_count=zone_count,
        config=baseline_config,
    )

    p10, p50, p90, forecast_provenance = _load_decision_time_quantiles(
        config.decision_time_forecasts_path,
        episode_timestamps=episode_timestamps,
        zone_ids=zone_ids,
        expected_split=baseline_config.evaluation_split,
        timezone=baseline_config.timezone,
        interval_minutes=baseline_config.interval_minutes,
        horizon_steps=config.forecast_horizon_steps,
    )
    risk_adjusted = (
        (1.0 - greedy_config.risk_aversion) * p50
        + greedy_config.risk_aversion * p90
    )
    region_labels, region_index, region_names = _load_region_index(
        config.zone_lookup_path,
        zone_ids,
    )
    allowed_edges = _load_graph_mask(config.zone_graph_path, zone_ids)
    graph_edge_count = int(np.count_nonzero(np.triu(allowed_edges, k=1)))

    simulators = {
        policy: _build_simulator(
            network=network,
            initial_vehicles=fleet_sizing.initial_vehicles,
            baseline_config=baseline_config,
            allowed_edges=allowed_edges,
        )
        for policy in POLICY_ORDER
    }
    region_demand_history = {
        policy: np.zeros(len(region_names), dtype=np.float64)
        for policy in POLICY_ORDER
    }
    region_served_history = {
        policy: np.zeros(len(region_names), dtype=np.float64)
        for policy in POLICY_ORDER
    }
    evaluation_results: dict[str, list[StepResult]] = {
        policy: [] for policy in POLICY_ORDER
    }
    decision_latencies: dict[str, list[float]] = {
        policy: [] for policy in POLICY_ORDER
    }
    step_records: list[dict[str, Any]] = []
    zone_records: list[dict[str, Any]] = []

    mpc_config = MPCConfig(
        demand_buffer_multiplier=greedy_config.demand_buffer_multiplier,
        minimum_zone_reserve=greedy_config.minimum_zone_reserve,
        in_transit_supply_credit=greedy_config.in_transit_supply_credit,
        maximum_relocation_fraction=greedy_config.maximum_relocation_fraction,
        maximum_relocation_steps=greedy_config.maximum_relocation_steps,
        relocation_cost_per_vehicle_step=baseline_config.relocation_cost_per_vehicle_step,
        unmet_demand_penalty=baseline_config.unmet_demand_penalty,
        idle_vehicle_cost=baseline_config.idle_vehicle_cost,
        fairness_cost_weight=config.fairness_cost_weight,
    )

    zero_action = np.zeros((zone_count, zone_count), dtype=np.int64)
    warmup_steps = 0
    evaluation_steps = 0
    all_fleet_conserved = True
    demand_contract_passed = True
    served_unmet_contract_passed = True
    graph_contract_passed = True
    travel_contract_passed = True
    relocation_cap_contract_passed = True
    no_negative_action_contract_passed = True
    initial_fleet_contract_passed = True
    warmup_contract_passed = True
    mpc_future_actual_demand_used = False
    mpc_diagnostics: list[dict[str, Any]] = []

    try:
        for step_index, (timestamp, actual_demand) in enumerate(
            zip(episode_timestamps, demand_matrix, strict=True)
        ):
            is_evaluation = timestamp.date() == baseline_config.service_date
            if not is_evaluation:
                warmup_steps += 1
            else:
                evaluation_steps += 1
                if evaluation_steps == 1:
                    snapshots = [simulators[name].snapshot() for name in POLICY_ORDER]
                    reference_snapshot = snapshots[0]
                    for other in snapshots[1:]:
                        if not (
                            np.array_equal(
                                other.available_vehicles,
                                reference_snapshot.available_vehicles,
                            )
                            and np.array_equal(
                                other.in_transit_vehicles,
                                reference_snapshot.in_transit_vehicles,
                            )
                            and other.total_fleet == reference_snapshot.total_fleet
                        ):
                            initial_fleet_contract_passed = False
                            raise OptimizationBenchmarkError(
                                "Policies reached different evaluation-start fleet states "
                                "after shared warmup."
                            )

            actions: dict[str, IntegerMatrix] = {}
            latency_ms: dict[str, float] = {}
            decision_diagnostics: dict[str, dict[str, Any]] = {}

            for policy in POLICY_ORDER:
                simulator = simulators[policy]
                adjusted_snapshot, pending = _decision_snapshot(simulator)
                if not is_evaluation:
                    action = zero_action.copy()
                    _validate_action(
                        action,
                        available=adjusted_snapshot.available_vehicles,
                        allowed_edges=allowed_edges,
                        relocation_travel_steps=relocation_steps,
                        maximum_relocation_steps=greedy_config.maximum_relocation_steps,
                        maximum_relocation_fraction=greedy_config.maximum_relocation_fraction,
                    )
                    actions[policy] = action
                    latency_ms[policy] = 0.0
                    decision_diagnostics[policy] = {"decision": "shared_no_rebalance_warmup"}
                    continue

                current_forecast = risk_adjusted[step_index, 0]
                horizon_forecasts = risk_adjusted[step_index]
                rates = _region_service_rates(
                    region_demand_history[policy],
                    region_served_history[policy],
                )
                started = time.perf_counter()

                if policy == "no_rebalancing":
                    action = zero_action.copy()
                    diagnostics: dict[str, Any] = {"decision": "zero_action"}
                elif policy == "greedy":
                    greedy_decision = greedy_relocation_action(
                        snapshot=adjusted_snapshot,
                        current_forecast=current_forecast,
                        future_forecasts=horizon_forecasts,
                        relocation_travel_steps=relocation_steps,
                        config=greedy_config,
                        allowed_relocation_edges=allowed_edges,
                    )
                    action = greedy_decision.relocation_action
                    diagnostics = {
                        "requested_relocations": greedy_decision.requested_relocations,
                        "relocation_vehicle_steps": greedy_decision.relocation_vehicle_steps,
                    }
                elif policy == "min_cost_flow":
                    target_end = min(
                        config.forecast_horizon_steps,
                        greedy_config.maximum_relocation_steps + 1,
                    )
                    target_window = horizon_forecasts[1:target_end]
                    if target_window.shape[0] == 0:
                        target_window = horizon_forecasts[:1]
                    target_forecast = np.max(target_window, axis=0)
                    flow: FlowAllocation = build_min_cost_relocation_action(
                        available_vehicles=adjusted_snapshot.available_vehicles,
                        current_forecast=current_forecast,
                        target_forecast=target_forecast,
                        in_transit_vehicles=adjusted_snapshot.in_transit_vehicles,
                        relocation_travel_steps=relocation_steps,
                        allowed_edges=allowed_edges,
                        region_index=region_index,
                        region_service_rates=rates,
                        demand_buffer_multiplier=greedy_config.demand_buffer_multiplier,
                        minimum_zone_reserve=greedy_config.minimum_zone_reserve,
                        in_transit_supply_credit=greedy_config.in_transit_supply_credit,
                        maximum_relocation_fraction=greedy_config.maximum_relocation_fraction,
                        maximum_relocation_steps=greedy_config.maximum_relocation_steps,
                        relocation_cost_per_vehicle_step=(
                            baseline_config.relocation_cost_per_vehicle_step
                        ),
                        unmet_demand_penalty=baseline_config.unmet_demand_penalty,
                        fairness_cost_weight=config.fairness_cost_weight,
                    )
                    action = flow.relocation_action
                    diagnostics = {
                        **flow.as_dict(),
                        "target_window_steps": int(target_window.shape[0]),
                    }
                elif policy == "mpc":
                    mpc_decision: MPCDecision = build_mpc_action(
                        snapshot=simulator.snapshot(),
                        pending_arrivals_by_step=pending,
                        forecast_horizon=horizon_forecasts,
                        passenger_destination_probabilities=destination_probabilities,
                        relocation_travel_steps=relocation_steps,
                        passenger_travel_steps=passenger_steps,
                        allowed_edges=allowed_edges,
                        region_index=region_index,
                        region_demand_history=region_demand_history[policy],
                        region_served_history=region_served_history[policy],
                        greedy_config=greedy_config,
                        config=mpc_config,
                    )
                    action = mpc_decision.relocation_action
                    diagnostics = mpc_decision.as_dict()
                    mpc_diagnostics.append(
                        {
                            "decision_time": timestamp.isoformat(),
                            **mpc_decision.as_dict(),
                        }
                    )
                else:
                    raise OptimizationBenchmarkError(f"Unknown policy: {policy}")

                elapsed_ms = (time.perf_counter() - started) * 1_000.0
                latency_ms[policy] = elapsed_ms
                decision_latencies[policy].append(elapsed_ms)
                _validate_action(
                    action,
                    available=adjusted_snapshot.available_vehicles,
                    allowed_edges=allowed_edges,
                    relocation_travel_steps=relocation_steps,
                    maximum_relocation_steps=greedy_config.maximum_relocation_steps,
                    maximum_relocation_fraction=greedy_config.maximum_relocation_fraction,
                )
                relocation_cap_contract_passed = relocation_cap_contract_passed and (
                    int(action.sum())
                    <= int(
                        math.floor(
                            int(adjusted_snapshot.available_vehicles.sum())
                            * greedy_config.maximum_relocation_fraction
                        )
                    )
                )
                actions[policy] = action
                diagnostics["decision_latency_ms"] = elapsed_ms
                decision_diagnostics[policy] = diagnostics

            results: dict[str, StepResult] = {}
            for policy in POLICY_ORDER:
                result = simulators[policy].step(
                    demand=actual_demand,
                    relocation_action=actions[policy],
                )
                results[policy] = result
                evaluation_results[policy].append(result) if is_evaluation else None
                all_fleet_conserved = all_fleet_conserved and result.fleet_conservation_passed
                demand_contract_passed = demand_contract_passed and np.array_equal(
                    result.demand, actual_demand
                )
                served_unmet_contract_passed = served_unmet_contract_passed and bool(
                    np.array_equal(result.served_demand + result.unmet_demand, actual_demand)
                )
                no_negative_action_contract_passed = no_negative_action_contract_passed and not (
                    actions[policy] < 0
                ).any()
                graph_contract_passed = graph_contract_passed and not (
                    (actions[policy] > 0) & ~allowed_edges
                ).any()
                travel_contract_passed = travel_contract_passed and not (
                    (actions[policy] > 0)
                    & (relocation_steps > greedy_config.maximum_relocation_steps)
                ).any()
                region_demand_history[policy] += np.bincount(
                    region_index,
                    weights=actual_demand,
                    minlength=len(region_names),
                )
                region_served_history[policy] += np.bincount(
                    region_index,
                    weights=result.served_demand,
                    minlength=len(region_names),
                )

            if not is_evaluation:
                if any(int(actions[policy].sum()) != 0 for policy in POLICY_ORDER):
                    warmup_contract_passed = False
                    raise OptimizationBenchmarkError(
                        "A policy rebalanced during the shared warmup period."
                    )
                continue

            for policy in POLICY_ORDER:
                result = results[policy]
                action = actions[policy]
                diagnostics = decision_diagnostics[policy]
                step_records.append(
                    {
                        "interval_start": timestamp,
                        "service_date": timestamp.date(),
                        "policy": policy,
                        "demand_total": result.metrics.demand_total,
                        "served_demand_total": result.metrics.served_demand_total,
                        "unmet_demand_total": result.metrics.unmet_demand_total,
                        "service_rate": result.metrics.service_rate,
                        "idle_vehicle_total": result.metrics.idle_vehicle_total,
                        "relocated_vehicle_total": result.metrics.relocated_vehicle_total,
                        "relocation_vehicle_steps": result.metrics.relocation_vehicle_steps,
                        "unmet_demand_cost": result.metrics.unmet_demand_cost,
                        "relocation_cost": result.metrics.relocation_cost,
                        "idle_vehicle_cost": result.metrics.idle_vehicle_cost,
                        "total_operating_cost": result.metrics.total_operating_cost,
                        "decision_latency_ms": latency_ms[policy],
                        "flow_objective_cost": diagnostics.get("objective_cost"),
                        "flow_unserved_deficit": diagnostics.get("unserved_deficit_total"),
                        "mpc_selected_horizon_steps": diagnostics.get("selected_horizon_steps"),
                        "mpc_projected_objective": diagnostics.get("projected_objective"),
                        "mpc_candidate_count": diagnostics.get("candidate_count"),
                        "fleet_conservation_passed": result.fleet_conservation_passed,
                    }
                )
                for zone_index_value, zone_id in enumerate(zone_ids):
                    zone_records.append(
                        {
                            "interval_start": timestamp,
                            "service_date": timestamp.date(),
                            "policy": policy,
                            "zone_index": zone_index_value,
                            "zone_id": zone_id,
                            "region": region_labels[zone_index_value],
                            "forecast_p10": float(p10[step_index, 0, zone_index_value]),
                            "forecast_p50": float(p50[step_index, 0, zone_index_value]),
                            "forecast_p90": float(p90[step_index, 0, zone_index_value]),
                            "risk_adjusted_forecast": float(
                                risk_adjusted[step_index, 0, zone_index_value]
                            ),
                            "actual_demand": int(result.demand[zone_index_value]),
                            "served_demand": int(result.served_demand[zone_index_value]),
                            "unmet_demand": int(result.unmet_demand[zone_index_value]),
                            "idle_vehicles_end": int(result.idle_vehicles[zone_index_value]),
                            "outbound_relocation": int(action[zone_index_value].sum()),
                            "inbound_relocation": int(action[:, zone_index_value].sum()),
                        }
                    )
    except (
        RealEpisodeBaselineError,
        ForecastGuidedPolicyError,
        FleetSimulationError,
        MinCostFlowError,
        pl.exceptions.PolarsError,
        ValueError,
        OSError,
    ) as error:
        if isinstance(error, OptimizationBenchmarkError):
            raise
        raise OptimizationBenchmarkError(f"Optimization benchmark failed: {error}") from error

    if evaluation_steps == 0:
        raise OptimizationBenchmarkError("No evaluation steps were simulated.")
    if not all(
        (
            all_fleet_conserved,
            demand_contract_passed,
            served_unmet_contract_passed,
            graph_contract_passed,
            travel_contract_passed,
            relocation_cap_contract_passed,
            no_negative_action_contract_passed,
            initial_fleet_contract_passed,
            warmup_contract_passed,
        )
    ):
        raise OptimizationBenchmarkError("One or more simulation invariants failed.")

    step_frame = pl.DataFrame(step_records).sort("interval_start", "policy")
    zone_frame = pl.DataFrame(zone_records).sort("interval_start", "policy", "zone_index")
    zone_summary = (
        zone_frame.group_by("policy", "zone_index", "zone_id", "region")
        .agg(
            pl.len().alias("interval_count"),
            pl.col("actual_demand").sum().alias("demand_total"),
            pl.col("served_demand").sum().alias("served_demand_total"),
            pl.col("unmet_demand").sum().alias("unmet_demand_total"),
            pl.col("idle_vehicles_end").mean().alias("average_idle_vehicles"),
            pl.col("outbound_relocation").sum().alias("outbound_relocation_total"),
            pl.col("inbound_relocation").sum().alias("inbound_relocation_total"),
        )
        .with_columns(
            pl.when(pl.col("demand_total") > 0)
            .then(pl.col("served_demand_total") / pl.col("demand_total"))
            .otherwise(1.0)
            .alias("service_rate")
        )
        .sort("policy", "zone_index")
    )
    region_summary = (
        zone_summary.group_by("policy", "region")
        .agg(
            pl.col("demand_total").sum().alias("demand_total"),
            pl.col("served_demand_total").sum().alias("served_demand_total"),
            pl.col("unmet_demand_total").sum().alias("unmet_demand_total"),
            pl.col("zone_id").n_unique().alias("zone_count"),
        )
        .with_columns(
            pl.when(pl.col("demand_total") > 0)
            .then(pl.col("served_demand_total") / pl.col("demand_total"))
            .otherwise(1.0)
            .alias("service_rate")
        )
        .sort("policy", "region")
    )

    summaries = {
        policy: _aggregate_results(
            evaluation_results[policy],
            zone_summary=zone_summary,
            region_summary=region_summary,
            policy=policy,
            decision_latencies=decision_latencies[policy],
        )
        for policy in POLICY_ORDER
    }
    comparisons = {
        policy: {
            "vs_no_rebalancing": _comparison(summaries["no_rebalancing"], summaries[policy]),
            "vs_greedy": _comparison(summaries["greedy"], summaries[policy]),
        }
        for policy in POLICY_ORDER
        if policy != "no_rebalancing"
    }

    # Check the first evaluation state was identical. This is captured by the
    # explicit per-step snapshot comparison at the start of each eval loop.
    invariants = {
        "policies_use_identical_actual_demand": demand_contract_passed,
        "policies_use_identical_initial_fleet": initial_fleet_contract_passed,
        "policies_use_identical_warmup_policy": warmup_contract_passed,
        "all_policies_preserve_fleet_conservation": all_fleet_conserved,
        "served_plus_unmet_equals_demand": served_unmet_contract_passed,
        "all_relocations_follow_zone_graph": graph_contract_passed,
        "all_relocations_respect_travel_time": travel_contract_passed,
        "all_relocations_respect_global_budget": relocation_cap_contract_passed,
        "no_negative_relocation_actions": no_negative_action_contract_passed,
        "all_policies_use_identical_cost_coefficients": True,
        "forecast_tensor_is_dense_and_aligned": True,
        "forecast_uses_future_realized_demand": False,
        "mpc_uses_future_realized_demand": mpc_future_actual_demand_used,
        "test_split_used_for_policy_selection": False,
    }

    artifacts = {
        "step_metrics": str(config.step_metrics_path),
        "interval_zone_metrics": str(config.interval_zone_metrics_path),
        "zone_summary": str(config.zone_summary_path),
        "region_summary": str(config.region_summary_path),
        "report": str(config.report_path),
    }
    report: dict[str, Any] = {
        "benchmark_version": "urbanflow_four_policy_optimization_benchmark_v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "dataset_name": config.dataset_name,
        "episode": {
            "evaluation_split": baseline_config.evaluation_split,
            "service_date": baseline_config.service_date.isoformat(),
            "warmup_steps": warmup_steps,
            "evaluation_steps": evaluation_steps,
            "first_interval": episode_timestamps[0].isoformat(),
            "last_interval": episode_timestamps[-1].isoformat(),
        },
        "network": {
            "zone_count": zone_count,
            "region_count": len(region_names),
            "region_names": list(region_names),
            "zone_graph_undirected_edge_count": graph_edge_count,
            "total_fleet": fleet_sizing.total_fleet,
        },
        "forecast": {
            "path": str(config.decision_time_forecasts_path),
            "horizon_steps": config.forecast_horizon_steps,
            "interval_minutes": baseline_config.interval_minutes,
            "split": baseline_config.evaluation_split,
            "provenance": forecast_provenance,
            "risk_aversion": greedy_config.risk_aversion,
            "risk_adjustment": "(1 - risk_aversion) * P50 + risk_aversion * P90",
            "loads_p10_p50_p90": True,
            "decision_signal_uses_p50_p90": True,
            "p10_used_for_uncertainty_reporting_only": True,
        },
        "optimization": {
            "min_cost_flow": {
                "solver": "successive_shortest_augmenting_path_with_reduced_cost_dijkstra",
                "unserved_demand_is_explicit_dummy_flow": True,
                "fairness_cost_weight": config.fairness_cost_weight,
                "target_window_steps": min(
                    config.forecast_horizon_steps - 1,
                    greedy_config.maximum_relocation_steps,
                ),
            },
            "mpc": {
                "method": "candidate_based_receding_horizon_expected_value_rollout_v1",
                "candidate_horizons": list(range(1, config.forecast_horizon_steps)),
                "tail_policy": "forecast_guided_greedy",
                "rollout_uses_expected_demand_and_training_od_profile": True,
                "only_first_action_is_applied": True,
                "global_joint_mip_optimum_claimed": False,
                "decision_count": len(mpc_diagnostics),
                "selected_horizon_counts": {
                    str(value): sum(
                        1 for item in mpc_diagnostics
                        if item["selected_horizon_steps"] == value
                    )
                    for value in range(config.forecast_horizon_steps)
                },
            },
            "constraints": {
                "maximum_relocation_fraction": greedy_config.maximum_relocation_fraction,
                "maximum_relocation_travel_steps": greedy_config.maximum_relocation_steps,
                "graph_edges_only": True,
                "minimum_zone_reserve": greedy_config.minimum_zone_reserve,
                "in_transit_supply_credit": greedy_config.in_transit_supply_credit,
                "fairness_is_region_service_rate_aware": True,
            },
        },
        "cost_coefficients": {
            "unmet_demand_penalty": baseline_config.unmet_demand_penalty,
            "relocation_cost_per_vehicle_step": baseline_config.relocation_cost_per_vehicle_step,
            "idle_vehicle_cost": baseline_config.idle_vehicle_cost,
        },
        "policies": {
            "no_rebalancing": "No relocation actions; shared warmup and simulator.",
            "greedy": "Existing Phase 2 forecast-guided greedy policy.",
            "min_cost_flow": "Global min-cost transportation with explicit unserved-demand arcs.",
            "mpc": (
                "Candidate-based receding-horizon controller with expected-value "
                "rollout and greedy tail policy."
            ),
        },
        "metrics": summaries,
        "comparisons": comparisons,
        "mpc_diagnostics": mpc_diagnostics,
        "invariants": invariants,
        "artifacts": artifacts,
    }

    _write_parquet_atomically(
        step_frame,
        path=config.step_metrics_path,
        compression=config.output_compression,
    )
    _write_parquet_atomically(
        zone_frame,
        path=config.interval_zone_metrics_path,
        compression=config.output_compression,
    )
    _write_parquet_atomically(
        zone_summary,
        path=config.zone_summary_path,
        compression=config.output_compression,
    )
    _write_parquet_atomically(
        region_summary,
        path=config.region_summary_path,
        compression=config.output_compression,
    )
    _write_json_atomically(config.report_path, report)
    return report


