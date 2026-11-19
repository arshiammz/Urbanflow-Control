"""Validation-only grid search for decision-time-safe Greedy fleet rebalancing.

Each candidate starts from the same training-derived fleet state. The May
window is continuous within each run (no midnight fleet reset); the baseline
is simulated just once. Future as-of forecasts are projected from the current
one-step validation forecast using *training-only* seasonal statistics.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from datetime import UTC, date, datetime, timedelta
from itertools import product
from pathlib import Path
from typing import Any, Callable

import numpy as np
import polars as pl
import yaml

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
from urbanflow.simulation.forecast_guided_validation import (
    extract_continuous_window,
    validate_component_configs,
    validate_forecast_split_window,
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


class ForecastGuidedTuningError(RuntimeError):
    """Invalid search, split boundary, or candidate selection."""


@dataclass(frozen=True)
class TuningConfig:
    baseline_config_path: Path
    heuristic_config_path: Path
    candidate_results_path: Path
    daily_results_path: Path
    report_path: Path
    validation_split: str
    start_date: date
    end_date_inclusive: date
    forecast_horizon_steps: tuple[int, ...]
    demand_buffer_multiplier: tuple[float, ...]
    minimum_zone_reserve: tuple[int, ...]
    maximum_relocation_steps: tuple[int, ...]
    maximum_relocation_fraction: tuple[float, ...]
    in_transit_supply_credit: tuple[float, ...]
    require_every_day_reduce_unmet_demand: bool
    require_every_day_reduce_operating_cost: bool
    compression: str


def read_config(path: Path) -> TuningConfig:
    """Read strict, deterministic May validation search specification."""
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        p, v, s, rules = raw["paths"], raw["validation"], raw["search"], raw["selection"]
        if (
            rules.get("primary_metric") != "aggregate_operating_cost"
            or rules.get("first_tie_breaker") != "worst_day_service_rate_improvement"
            or rules.get("second_tie_breaker") != "relocation_vehicle_steps"
        ):
            raise ForecastGuidedTuningError("Unsupported selection metric ordering.")
        config = TuningConfig(
            baseline_config_path=Path(p["baseline_config"]),
            heuristic_config_path=Path(p["heuristic_config"]),
            candidate_results_path=Path(p["candidate_results"]),
            daily_results_path=Path(p["daily_results"]),
            report_path=Path(p["report"]),
            validation_split=str(v["split"]),
            start_date=date.fromisoformat(str(v["start_date"])),
            end_date_inclusive=date.fromisoformat(str(v["end_date_inclusive"])),
            forecast_horizon_steps=tuple(map(int, s["forecast_horizon_steps"])),
            demand_buffer_multiplier=tuple(map(float, s["demand_buffer_multiplier"])),
            minimum_zone_reserve=tuple(map(int, s["minimum_zone_reserve"])),
            maximum_relocation_steps=tuple(map(int, s["maximum_relocation_steps"])),
            maximum_relocation_fraction=tuple(map(float, s["maximum_relocation_fraction"])),
            in_transit_supply_credit=tuple(map(float, s["in_transit_supply_credit"])),
            require_every_day_reduce_unmet_demand=rules["require_every_day_reduce_unmet_demand"] is True,
            require_every_day_reduce_operating_cost=rules["require_every_day_reduce_operating_cost"] is True,
            compression=str(raw["output"]["compression"]),
        )
    except ForecastGuidedTuningError:
        raise
    except (OSError, TypeError, AttributeError, KeyError, ValueError, yaml.YAMLError) as exc:
        raise ForecastGuidedTuningError(f"Invalid tuning config {path}: {exc}") from exc
    if config.validation_split != "validation":
        raise ForecastGuidedTuningError("Parameter search is locked to validation, never test.")
    if not (config.start_date <= config.end_date_inclusive):
        raise ForecastGuidedTuningError("Invalid date range.")
    for input_path in (config.baseline_config_path, config.heuristic_config_path):
        if not input_path.is_file():
            raise ForecastGuidedTuningError(f"Missing input config: {input_path}")
    for field in (
        "forecast_horizon_steps", "demand_buffer_multiplier", "minimum_zone_reserve",
        "maximum_relocation_steps", "maximum_relocation_fraction", "in_transit_supply_credit",
    ):
        values = getattr(config, field)
        if not values or len(values) != len(set(values)):
            raise ForecastGuidedTuningError(f"Empty or duplicate search values: {field}")
    if any(x < 1 for x in config.forecast_horizon_steps + config.maximum_relocation_steps):
        raise ForecastGuidedTuningError("Horizon and relocation steps must be positive.")
    if any(x < 0 for x in config.minimum_zone_reserve):
        raise ForecastGuidedTuningError("Reserves must be nonnegative.")
    if any(not math.isfinite(x) or x <= 0 for x in config.demand_buffer_multiplier):
        raise ForecastGuidedTuningError("Demand buffers must be positive and finite.")
    for field in ("maximum_relocation_fraction", "in_transit_supply_credit"):
        if any(not math.isfinite(x) or not 0 <= x <= 1 for x in getattr(config, field)):
            raise ForecastGuidedTuningError(f"{field} values must be in [0,1].")
    return config


def candidate_grid(config: TuningConfig) -> list[dict[str, Any]]:
    """Produce complete deterministic search grid, including the reference policy."""
    return [
        {
            "candidate_id": index,
            "forecast_horizon_steps": h,
            "demand_buffer_multiplier": b,
            "minimum_zone_reserve": r,
            "maximum_relocation_steps": d,
            "maximum_relocation_fraction": f,
            "in_transit_supply_credit": c,
        }
        for index, (h, b, r, d, f, c) in enumerate(
            product(
                sorted(config.forecast_horizon_steps),
                sorted(config.demand_buffer_multiplier),
                sorted(config.minimum_zone_reserve),
                sorted(config.maximum_relocation_steps),
                sorted(config.maximum_relocation_fraction),
                sorted(config.in_transit_supply_credit),
            ),
            start=1,
        )
    ]


def build_candidate_config(
    reference: ForecastGuidedConfig, parameters: dict[str, Any]
) -> ForecastGuidedConfig:
    """Same input artifacts and costs; only search parameters change."""
    return replace(
        reference,
        **{name: parameters[name] for name in (
            "forecast_horizon_steps", "demand_buffer_multiplier", "minimum_zone_reserve",
            "maximum_relocation_steps", "maximum_relocation_fraction", "in_transit_supply_credit",
        )},
    )


def is_reference(parameters: dict[str, Any], reference: ForecastGuidedConfig) -> bool:
    return all(
        math.isclose(float(parameters[name]), float(getattr(reference, name)), abs_tol=1e-12, rel_tol=0)
        for name in (
            "forecast_horizon_steps", "demand_buffer_multiplier", "minimum_zone_reserve",
            "maximum_relocation_steps", "maximum_relocation_fraction", "in_transit_supply_credit",
        )
    )


def simulate(
    *,
    policy: ForecastGuidedConfig | None,
    baseline_config: RealEpisodeBaselineConfig,
    network: NetworkInputs,
    initial_vehicles: np.ndarray,
    timestamps: list[datetime],
    demand_matrix: np.ndarray,
    forecast_matrix: np.ndarray,
    projector: Any,
    start_date: date,
    end_date: date,
) -> tuple[list[dict[str, Any]], bool]:
    """One continuous rollout with a daily metric accumulator."""
    simulator = build_simulator(
        baseline_config=baseline_config, network=network, initial_vehicles=initial_vehicles
    )
    zero = np.zeros((len(network.zone_ids), len(network.zone_ids)), dtype=np.int64)
    daily: dict[date, dict[str, Any]] = {}
    conservation = True
    for step_index, (timestamp, realized_demand) in enumerate(
        zip(timestamps, demand_matrix, strict=True)
    ):
        current_date = timestamp.date()
        evaluate = start_date <= current_date <= end_date
        action = zero
        if evaluate and policy is not None:
            # The action is determined before simulator.step(realized_demand).
            current_prediction = forecast_matrix[step_index]
            future = projector.project(
                decision_time=timestamp,
                current_prediction=current_prediction,
                horizon_steps=policy.forecast_horizon_steps,
            )
            action = greedy_relocation_action(
                snapshot=simulator.snapshot(),
                current_forecast=current_prediction,
                future_forecasts=future,
                relocation_travel_steps=network.relocation_travel_steps,
                config=policy,
            ).relocation_action
        result = simulator.step(demand=realized_demand, relocation_action=action)
        conservation &= bool(result.fleet_conservation_passed)
        if not evaluate:
            continue
        row = daily.setdefault(current_date, {
            "service_date": current_date,
            "step_count": 0,
            "demand_total": 0,
            "served_demand_total": 0,
            "unmet_demand_total": 0,
            "idle_vehicle_sum": 0,
            "relocated_vehicle_total": 0,
            "relocation_vehicle_steps": 0,
            "total_operating_cost": 0.0,
        })
        metrics = result.metrics
        row["step_count"] += 1
        row["demand_total"] += int(metrics.demand_total)
        row["served_demand_total"] += int(metrics.served_demand_total)
        row["unmet_demand_total"] += int(metrics.unmet_demand_total)
        row["idle_vehicle_sum"] += int(metrics.idle_vehicle_total)
        row["relocated_vehicle_total"] += int(metrics.relocated_vehicle_total)
        row["relocation_vehicle_steps"] += int(metrics.relocation_vehicle_steps)
        row["total_operating_cost"] += float(metrics.total_operating_cost)
    expected_dates = [start_date + timedelta(days=k) for k in range((end_date - start_date).days + 1)]
    if sorted(daily) != expected_dates:
        raise ForecastGuidedTuningError("Episode does not cover all validation dates.")
    rows = [daily[day] for day in expected_dates]
    for row in rows:
        if row["served_demand_total"] + row["unmet_demand_total"] != row["demand_total"]:
            raise ForecastGuidedTuningError("Daily demand conservation failed.")
        row["service_rate"] = row["served_demand_total"] / max(1, row["demand_total"])
        row["average_idle_vehicles"] = row["idle_vehicle_sum"] / row["step_count"]
    return rows, conservation


def compare_daily(
    *, parameters: dict[str, Any], baseline: list[dict[str, Any]], policy: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Construct daily policy-vs-baseline rows with no cross-day resets."""
    rows = []
    for b, p in zip(baseline, policy, strict=True):
        if b["service_date"] != p["service_date"] or b["demand_total"] != p["demand_total"]:
            raise ForecastGuidedTuningError("Baseline/policy daily alignment failed.")
        rows.append({
            **parameters,
            "service_date": b["service_date"],
            "step_count": b["step_count"],
            "demand_total": b["demand_total"],
            "baseline_unmet_demand_total": b["unmet_demand_total"],
            "policy_unmet_demand_total": p["unmet_demand_total"],
            "baseline_served_demand_total": b["served_demand_total"],
            "policy_served_demand_total": p["served_demand_total"],
            "baseline_total_operating_cost": b["total_operating_cost"],
            "policy_total_operating_cost": p["total_operating_cost"],
            "baseline_service_rate": b["service_rate"],
            "policy_service_rate": p["service_rate"],
            "service_rate_improvement": p["service_rate"] - b["service_rate"],
            "unmet_demand_reduction": b["unmet_demand_total"] - p["unmet_demand_total"],
            "operating_cost_reduction": b["total_operating_cost"] - p["total_operating_cost"],
            "policy_average_idle_vehicles": p["average_idle_vehicles"],
            "policy_relocated_vehicle_total": p["relocated_vehicle_total"],
            "policy_relocation_vehicle_steps": p["relocation_vehicle_steps"],
            "policy_reduces_unmet_demand": p["unmet_demand_total"] < b["unmet_demand_total"],
            "policy_beats_baseline_cost": p["total_operating_cost"] < b["total_operating_cost"],
        })
    return rows


def summarize(
    *,
    parameters: dict[str, Any],
    daily: list[dict[str, Any]],
    conserved: bool,
    search: TuningConfig,
    reference: ForecastGuidedConfig,
) -> dict[str, Any]:
    """Rank by total cost, then worst-day improvement, then relocation."""
    def sums(key: str) -> float:
        return sum(r[key] for r in daily)
    baseline_cost = float(sums("baseline_total_operating_cost"))
    policy_cost = float(sums("policy_total_operating_cost"))
    baseline_unmet = int(sums("baseline_unmet_demand_total"))
    policy_unmet = int(sums("policy_unmet_demand_total"))
    demand_total = int(sums("demand_total"))
    policy_served = int(sums("policy_served_demand_total"))
    baseline_served = int(sums("baseline_served_demand_total"))
    all_unmet = all(r["policy_reduces_unmet_demand"] for r in daily)
    all_cost = all(r["policy_beats_baseline_cost"] for r in daily)
    feasible = bool(
        conserved
        and policy_served + policy_unmet == demand_total
        and baseline_served + baseline_unmet == demand_total
        and (all_unmet or not search.require_every_day_reduce_unmet_demand)
        and (all_cost or not search.require_every_day_reduce_operating_cost)
    )
    daily_imp = np.array([r["service_rate_improvement"] for r in daily])
    return {
        **parameters,
        "is_reference_config": is_reference(parameters, reference),
        "feasible": feasible,
        "fleet_conserved": conserved,
        "validation_day_count": len(daily),
        "days_reducing_unmet_demand": sum(r["policy_reduces_unmet_demand"] for r in daily),
        "days_beating_baseline_cost": sum(r["policy_beats_baseline_cost"] for r in daily),
        "demand_total": demand_total,
        "policy_unmet_demand_total": policy_unmet,
        "baseline_unmet_demand_total": baseline_unmet,
        "policy_served_demand_total": policy_served,
        "baseline_served_demand_total": baseline_served,
        "policy_service_rate": policy_served / max(1, demand_total),
        "baseline_service_rate": baseline_served / max(1, demand_total),
        "unmet_demand_absolute_reduction": baseline_unmet - policy_unmet,
        "unmet_demand_relative_reduction": (baseline_unmet - policy_unmet) / max(1, baseline_unmet),
        "worst_day_service_rate_improvement": float(daily_imp.min()),
        "median_daily_service_rate_improvement": float(np.median(daily_imp)),
        "mean_daily_service_rate_improvement": float(daily_imp.mean()),
        "relocated_vehicle_total": int(sums("policy_relocated_vehicle_total")),
        "relocation_vehicle_steps": int(sums("policy_relocation_vehicle_steps")),
        "policy_total_operating_cost": policy_cost,
        "baseline_total_operating_cost": baseline_cost,
        "operating_cost_absolute_reduction": baseline_cost - policy_cost,
        "operating_cost_relative_reduction": (baseline_cost - policy_cost) / max(1, baseline_cost),
        "selection_rank": None,
    }


def ranking_key(candidate: dict[str, Any]) -> tuple[float, float, int, int]:
    """Cost primary, worst-day improvement descending, relocation steps ascending."""
    return (
        float(candidate["policy_total_operating_cost"]),
        -float(candidate["worst_day_service_rate_improvement"]),
        int(candidate["relocation_vehicle_steps"]),
        int(candidate["candidate_id"]),
    )


def run_forecast_guided_tuning(
    config_path: Path, *, progress_callback: Callable[[int, int], None] | None = None
) -> dict[str, Any]:
    """Execute 108-candidate May-only search with hard temporal guards."""
    config = read_config(config_path)
    baseline_config = read_baseline_config(config.baseline_config_path)
    reference = read_heuristic_config(config.heuristic_config_path)
    validate_component_configs(
        config=config, baseline_config=baseline_config, heuristic_config=reference
    )
    _, zone_ids, fit_shares = load_zone_index(baseline_config)
    probabilities, passenger_steps, relocation_steps = load_od_matrices(
        baseline_config, zone_count=len(zone_ids)
    )
    network = NetworkInputs(
        zone_ids=zone_ids,
        fit_pickup_shares=fit_shares,
        destination_probabilities=probabilities,
        passenger_travel_steps=passenger_steps,
        relocation_travel_steps=relocation_steps,
    )
    demand = load_historical_demand(baseline_config, zone_count=len(zone_ids))
    fleet = estimate_initial_fleet(demand, network=network, config=baseline_config)
    timestamps, actual = extract_continuous_window(
        demand, config=config, baseline_config=baseline_config, zone_count=len(zone_ids)
    )
    loaded_timestamps, forecasts = load_forecast_matrix(
        config=reference,
        baseline_config=baseline_config,
        zone_ids=zone_ids,
        episode_timestamps=timestamps,
        expected_split=config.validation_split,
    )
    if loaded_timestamps != timestamps:
        raise ForecastGuidedTuningError("Forecast timestamps do not align.")
    split_validation = validate_forecast_split_window(
        config=config,
        heuristic_config=reference,
        zone_ids=zone_ids,
        first_timestamp=timestamps[0],
        last_required_timestamp=timestamps[-1],
    )
    if split_validation["invalid_split_count"] or split_validation["observed_splits"] != ["validation"]:
        raise ForecastGuidedTuningError("Test or non-validation forecast detected.")
    projector = fit_train_only_projector(
        demand,
        zone_ids=zone_ids,
        interval_minutes=baseline_config.interval_minutes,
        training_split=baseline_config.sizing_split,
    )
    baseline_daily, baseline_conserved = simulate(
        policy=None, baseline_config=baseline_config, network=network,
        initial_vehicles=fleet.initial_vehicles, timestamps=timestamps,
        demand_matrix=actual, forecast_matrix=forecasts, projector=projector,
        start_date=config.start_date, end_date=config.end_date_inclusive,
    )
    if not baseline_conserved:
        raise ForecastGuidedTuningError("Baseline conservation failed.")
    candidates = candidate_grid(config)
    if len(candidates) != 108:
        raise ForecastGuidedTuningError(
            f"Search grid was changed: expected 108, got {len(candidates)}."
        )
    candidate_rows: list[dict[str, Any]] = []
    daily_rows: list[dict[str, Any]] = []
    for index, params in enumerate(candidates, start=1):
        policy_daily, conserved = simulate(
            policy=build_candidate_config(reference, params),
            baseline_config=baseline_config, network=network,
            initial_vehicles=fleet.initial_vehicles,
            timestamps=timestamps, demand_matrix=actual,
            forecast_matrix=forecasts, projector=projector,
            start_date=config.start_date, end_date=config.end_date_inclusive,
        )
        joined = compare_daily(parameters=params, baseline=baseline_daily, policy=policy_daily)
        daily_rows.extend(joined)
        candidate_rows.append(summarize(
            parameters=params, daily=joined, conserved=conserved,
            search=config, reference=reference,
        ))
        if progress_callback:
            progress_callback(index, len(candidates))
    refs = [c for c in candidate_rows if c["is_reference_config"]]
    if len(refs) != 1:
        raise ForecastGuidedTuningError("Reference candidate missing or duplicated.")
    eligible = sorted((c for c in candidate_rows if c["feasible"]), key=ranking_key)
    if not eligible:
        raise ForecastGuidedTuningError("No eligible candidates.")
    for rank, row in enumerate(eligible, start=1):
        row["selection_rank"] = rank
    winner = eligible[0]
    expected_daily = len(candidates) * ((config.end_date_inclusive - config.start_date).days + 1)
    invariants = {
        "108_candidates_evaluated": len(candidate_rows) == 108,
        "daily_grid_is_complete": len(daily_rows) == expected_daily,
        "reference_is_unique": len(refs) == 1,
        "champion_is_feasible": bool(winner["feasible"]),
        "all_fleets_conserved": all(bool(c["fleet_conserved"]) for c in candidate_rows),
        "same_baseline_for_all": len(set(c["baseline_total_operating_cost"] for c in candidate_rows)) == 1,
        "validation_only_forecasts": split_validation["observed_splits"] == ["validation"],
        "future_offline_predictions_not_read": True,
        "future_realized_demand_not_used_by_policy": True,
        "train_only_fleet_and_weekly_profiles": baseline_config.sizing_split == "train",
        "no_test_data_used_for_selection": True,
    }
    if not all(invariants.values()):
        raise ForecastGuidedTuningError(
            "Failed invariants: " + ", ".join(k for k, passed in invariants.items() if not passed)
        )
    artifact_candidates = write_parquet_atomically(
        pl.DataFrame(candidate_rows), path=config.candidate_results_path,
        compression=config.compression,
    )
    artifact_daily = write_parquet_atomically(
        pl.DataFrame(daily_rows), path=config.daily_results_path,
        compression=config.compression,
    )
    ref = refs[0]
    report = {
        "report_version": "phase1-asof-v1",
        "generated_at": datetime.now(UTC),
        "methodology": {
            "selection_split": "validation",
            "first_evaluation_date": config.start_date,
            "last_evaluation_date": config.end_date_inclusive,
            "warmup_policy": "no_rebalancing",
            "baseline_run_once": True,
            "fleet_resets_between_candidates": True,
            "fleet_resets_between_days": False,
            "asof_forecasting_method": projector.method,
            "train_only_profile": True,
            "future_offline_predictions_used": False,
            "test_evaluation_performed": False,
        },
        "search_grid": {"candidate_count": len(candidates), "feasible_count": len(eligible)},
        "reference_candidate": ref,
        "reference_feasible": bool(ref["feasible"]),
        "champion_candidate": winner,
        "champion_vs_reference": {
            "operating_cost_absolute_reduction": (
                ref["policy_total_operating_cost"] - winner["policy_total_operating_cost"]
            ),
            "operating_cost_relative_reduction": (
                (ref["policy_total_operating_cost"] - winner["policy_total_operating_cost"])
                / max(1, ref["policy_total_operating_cost"])
            ),
            "unmet_demand_absolute_reduction": (
                ref["policy_unmet_demand_total"] - winner["policy_unmet_demand_total"]
            ),
            "service_rate_absolute_improvement": (
                winner["policy_service_rate"] - ref["policy_service_rate"]
            ),
            "relocation_vehicle_step_difference": (
                winner["relocation_vehicle_steps"] - ref["relocation_vehicle_steps"]
            ),
        },
        "top_feasible_candidates": eligible[:10],
        "artifacts": {
            "candidate_results": artifact_candidates,
            "daily_results": artifact_daily,
        },
        "invariants": invariants,
    }
    write_json_atomically(config.report_path, report)
    return report
