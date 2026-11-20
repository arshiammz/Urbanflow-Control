"""Candidate-based receding-horizon MPC for fleet rebalancing.

The controller evaluates a compact set of first-action candidates produced by
min-cost flow at progressively longer forecast horizons. Each candidate is
rolled through an expected-value fleet model; later decisions use the existing
forecast-guided greedy policy as a common tail policy. Only the first action is
sent to the real simulator. This is a scalable MPC approximation, not a claim
of globally optimal mixed-integer control over the full horizon.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from urbanflow.optimization.min_cost_flow import (
    FlowAllocation,
    build_min_cost_relocation_action,
)
from urbanflow.simulation.fleet_simulator import FleetSnapshot

FloatVector = NDArray[np.float64]
FloatMatrix = NDArray[np.float64]
IntegerVector = NDArray[np.int64]
IntegerMatrix = NDArray[np.int64]
BooleanMatrix = NDArray[np.bool_]


class MPCError(ValueError):
    """Raised when an MPC planning problem is invalid."""


@dataclass(frozen=True)
class MPCConfig:
    """Operational controls shared by MPC candidate generation and rollout."""

    demand_buffer_multiplier: float
    minimum_zone_reserve: int
    in_transit_supply_credit: float
    maximum_relocation_fraction: float
    maximum_relocation_steps: int
    relocation_cost_per_vehicle_step: float
    unmet_demand_penalty: float
    idle_vehicle_cost: float
    fairness_cost_weight: float

    def validate(self) -> None:
        numeric = (
            self.demand_buffer_multiplier,
            self.in_transit_supply_credit,
            self.maximum_relocation_fraction,
            self.relocation_cost_per_vehicle_step,
            self.unmet_demand_penalty,
            self.idle_vehicle_cost,
            self.fairness_cost_weight,
        )
        if not np.isfinite(numeric).all():
            raise MPCError("MPC controls must be finite.")
        if self.demand_buffer_multiplier <= 0:
            raise MPCError("demand_buffer_multiplier must be positive.")
        if not 0 <= self.in_transit_supply_credit <= 1:
            raise MPCError("in_transit_supply_credit must be in [0, 1].")
        if not 0 <= self.maximum_relocation_fraction <= 1:
            raise MPCError("maximum_relocation_fraction must be in [0, 1].")
        if self.minimum_zone_reserve < 0 or self.maximum_relocation_steps < 1:
            raise MPCError("MPC reserve or travel-step limit is invalid.")
        if min(
            self.relocation_cost_per_vehicle_step,
            self.unmet_demand_penalty,
            self.idle_vehicle_cost,
            self.fairness_cost_weight,
        ) < 0:
            raise MPCError("MPC cost coefficients must be non-negative.")


@dataclass(frozen=True)
class MPCDecision:
    """Chosen first action and its candidate-rollout diagnostics."""

    relocation_action: IntegerMatrix
    selected_horizon_steps: int
    projected_operating_cost: float
    projected_fairness_penalty: float
    projected_objective: float
    projected_unmet_demand: float
    planned_relocated_vehicles: int
    candidate_count: int
    candidate_objectives: tuple[dict[str, Any], ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "selected_horizon_steps": self.selected_horizon_steps,
            "projected_operating_cost": self.projected_operating_cost,
            "projected_fairness_penalty": self.projected_fairness_penalty,
            "projected_objective": self.projected_objective,
            "projected_unmet_demand": self.projected_unmet_demand,
            "planned_relocated_vehicles": self.planned_relocated_vehicles,
            "candidate_count": self.candidate_count,
            "candidate_objectives": list(self.candidate_objectives),
        }


def _integer_matrix(values: Any, *, name: str, zone_count: int) -> IntegerMatrix:
    raw = np.asarray(values)
    expected_shape = (zone_count, zone_count)
    if raw.shape != expected_shape or not np.issubdtype(raw.dtype, np.number):
        raise MPCError(f"{name} must be a numeric matrix with shape {expected_shape}.")
    numeric = raw.astype(np.float64, copy=False)
    if not np.isfinite(numeric).all() or (numeric < 0).any():
        raise MPCError(f"{name} must contain finite non-negative values.")
    if not np.equal(numeric, np.floor(numeric)).all():
        raise MPCError(f"{name} must contain integer values.")
    return numeric.astype(np.int64, copy=True)


def _validate_forecasts(values: Any) -> FloatMatrix:
    forecasts = np.asarray(values, dtype=np.float64)
    if forecasts.ndim != 2 or forecasts.shape[0] < 2 or forecasts.shape[1] < 2:
        raise MPCError("forecast_horizon must have shape (horizon>=2, zones>=2).")
    if not np.isfinite(forecasts).all() or (forecasts < 0).any():
        raise MPCError("forecast_horizon must contain finite non-negative values.")
    return forecasts


def _region_rates(
    region_demand: FloatVector,
    region_served: FloatVector,
) -> FloatVector:
    return np.divide(
        region_served,
        region_demand,
        out=np.ones_like(region_served, dtype=np.float64),
        where=region_demand > 0,
    ).clip(0.0, 1.0)


def _add_scheduled(
    schedule: dict[int, FloatVector],
    *,
    offset: int,
    destination: int,
    amount: float,
    zone_count: int,
) -> None:
    if amount <= 0:
        return
    if offset < 1:
        raise MPCError("Newly scheduled vehicle arrivals must be in the future.")
    vector = schedule.setdefault(offset, np.zeros(zone_count, dtype=np.float64))
    vector[destination] += amount




def _greedy_tail_action(
    *,
    available: IntegerVector,
    in_transit: IntegerVector,
    current_forecast: FloatVector,
    future_forecasts: FloatMatrix,
    relocation_travel_steps: IntegerMatrix,
    allowed_edges: BooleanMatrix | None,
    config: Any,
) -> IntegerMatrix:
    """Pure NumPy implementation of the Phase 2 greedy tail policy."""
    zone_count = available.size
    current_reserve = np.ceil(
        current_forecast * config.demand_buffer_multiplier
    ).astype(np.int64) + int(config.minimum_zone_reserve)
    forecast_target = np.ceil(
        np.max(future_forecasts, axis=0) * config.demand_buffer_multiplier
    ).astype(np.int64) + int(config.minimum_zone_reserve)
    projected_supply = available + np.floor(
        in_transit.astype(np.float64) * config.in_transit_supply_credit
    ).astype(np.int64)
    surplus = np.maximum(available - current_reserve, 0).astype(np.int64)
    deficit = np.maximum(forecast_target - projected_supply, 0).astype(np.int64)
    action = np.zeros((zone_count, zone_count), dtype=np.int64)
    relocation_cap = int(
        math.floor(int(available.sum()) * config.maximum_relocation_fraction)
    )
    if relocation_cap <= 0 or int(surplus.sum()) == 0 or int(deficit.sum()) == 0:
        return action

    relocated = 0
    for origin in np.argsort(-surplus, kind="stable"):
        origin_surplus = int(surplus[origin])
        if origin_surplus <= 0 or relocated >= relocation_cap:
            continue
        eligible = (
            (deficit > 0)
            & (relocation_travel_steps[origin] <= config.maximum_relocation_steps)
        )
        eligible[origin] = False
        if allowed_edges is not None:
            eligible &= allowed_edges[origin]
        destinations = np.flatnonzero(eligible).tolist()
        destinations.sort(
            key=lambda destination: (
                int(relocation_travel_steps[origin, destination]),
                -int(deficit[destination]),
                destination,
            )
        )
        for destination in destinations:
            remaining = relocation_cap - relocated
            if origin_surplus <= 0 or remaining <= 0:
                break
            flow = min(origin_surplus, int(deficit[destination]), remaining)
            if flow <= 0:
                continue
            action[origin, destination] += flow
            origin_surplus -= flow
            deficit[destination] -= flow
            relocated += flow
    if (action.sum(axis=1) > available).any():
        raise MPCError("Greedy MPC tail action exceeds available supply.")
    return action


def _rollout_candidate(
    *,
    candidate_action: IntegerMatrix,
    snapshot: FleetSnapshot,
    pending_arrivals_by_step: dict[int, IntegerVector],
    forecasts: FloatMatrix,
    passenger_destination_probabilities: FloatMatrix,
    relocation_travel_steps: IntegerMatrix,
    passenger_travel_steps: IntegerMatrix,
    allowed_edges: BooleanMatrix | None,
    region_index: NDArray[np.int64],
    region_demand_history: FloatVector,
    region_served_history: FloatVector,
    greedy_config: Any,
    mpc_config: MPCConfig,
) -> dict[str, float]:
    """Evaluate one first action under an expected-value greedy tail rollout."""
    horizon_steps, zone_count = forecasts.shape
    available = snapshot.available_vehicles.astype(np.float64, copy=True)
    schedule: dict[int, FloatVector] = {
        int(offset): values.astype(np.float64, copy=True)
        for offset, values in pending_arrivals_by_step.items()
        if int(offset) > 0
    }
    due_now = pending_arrivals_by_step.get(0)
    if due_now is not None:
        available += due_now.astype(np.float64)

    region_demand = region_demand_history.astype(np.float64, copy=True)
    region_served = region_served_history.astype(np.float64, copy=True)
    operating_cost = 0.0
    fairness_penalty = 0.0
    unmet_total = 0.0
    relocated_total = 0

    for horizon_index in range(horizon_steps):
        arrivals = schedule.pop(horizon_index, None)
        if arrivals is not None:
            available += arrivals

        if horizon_index == 0:
            action = candidate_action
        else:
            in_transit = np.zeros(zone_count, dtype=np.float64)
            for offset, values in schedule.items():
                if offset > horizon_index:
                    in_transit += values
            synthetic_snapshot = FleetSnapshot(
                step=snapshot.step + horizon_index,
                available_vehicles=np.floor(available + 1e-9).astype(np.int64),
                in_transit_vehicles=np.floor(in_transit + 1e-9).astype(np.int64),
                scheduled_arrival_steps=tuple(
                    sorted(offset for offset in schedule if offset > horizon_index)
                ),
                total_fleet=snapshot.total_fleet,
            )
            remaining_steps = horizon_steps - horizon_index - 1
            if allowed_edges is None:
                tail_edges = relocation_travel_steps <= remaining_steps
            else:
                tail_edges = allowed_edges & (
                    relocation_travel_steps <= remaining_steps
                )
            action = _greedy_tail_action(
                available=synthetic_snapshot.available_vehicles,
                in_transit=synthetic_snapshot.in_transit_vehicles,
                current_forecast=forecasts[horizon_index],
                future_forecasts=forecasts[horizon_index:],
                relocation_travel_steps=relocation_travel_steps,
                allowed_edges=tail_edges,
                config=greedy_config,
            )

        outbound = action.sum(axis=1).astype(np.float64)
        available_floor = np.floor(available + 1e-9)
        if (outbound > available_floor).any():
            raise MPCError("MPC rollout action exceeds projected available supply.")
        available -= outbound

        for origin, destination in np.argwhere(action > 0):
            count = int(action[origin, destination])
            duration = int(relocation_travel_steps[origin, destination])
            _add_scheduled(
                schedule,
                offset=horizon_index + duration,
                destination=int(destination),
                amount=float(count),
                zone_count=zone_count,
            )
            relocated_total += count

        expected_demand = forecasts[horizon_index]
        served = np.minimum(available, expected_demand)
        unmet = expected_demand - served
        idle = available - served
        vehicle_steps = float(np.sum(action * relocation_travel_steps))
        operating_cost += (
            float(unmet.sum()) * mpc_config.unmet_demand_penalty
            + vehicle_steps * mpc_config.relocation_cost_per_vehicle_step
            + float(idle.sum()) * mpc_config.idle_vehicle_cost
        )
        unmet_total += float(unmet.sum())

        region_demand_step = np.bincount(
            region_index,
            weights=expected_demand,
            minlength=region_demand.size,
        ).astype(np.float64)
        region_served_step = np.bincount(
            region_index,
            weights=served,
            minlength=region_served.size,
        ).astype(np.float64)
        region_demand += region_demand_step
        region_served += region_served_step
        overall_rate = (
            float(region_served.sum() / region_demand.sum())
            if region_demand.sum() > 0
            else 1.0
        )
        rates = _region_rates(region_demand, region_served)
        fairness_gap = np.maximum(overall_rate - rates, 0.0)
        fairness_penalty += mpc_config.fairness_cost_weight * float(
            np.dot(fairness_gap, region_demand_step)
        )

        expected_passenger_flow = served[:, None] * passenger_destination_probabilities
        for duration_value in np.unique(passenger_travel_steps):
            duration = int(duration_value)
            arrivals_by_destination = np.where(
                passenger_travel_steps == duration,
                expected_passenger_flow,
                0.0,
            ).sum(axis=0)
            if float(arrivals_by_destination.sum()) <= 0:
                continue
            target_offset = horizon_index + duration
            scheduled = schedule.setdefault(
                target_offset,
                np.zeros(zone_count, dtype=np.float64),
            )
            scheduled += arrivals_by_destination
        available = idle

    return {
        "operating_cost": operating_cost,
        "fairness_penalty": fairness_penalty,
        "objective": operating_cost + fairness_penalty,
        "unmet_demand": unmet_total,
        "relocated_vehicles": float(relocated_total),
    }


def build_mpc_action(
    *,
    snapshot: FleetSnapshot,
    pending_arrivals_by_step: dict[int, IntegerVector],
    forecast_horizon: Any,
    passenger_destination_probabilities: Any,
    relocation_travel_steps: Any,
    passenger_travel_steps: Any,
    allowed_edges: Any | None,
    region_index: Any,
    region_demand_history: Any,
    region_served_history: Any,
    greedy_config: Any,
    config: MPCConfig,
) -> MPCDecision:
    """Select the first action from a candidate-based receding-horizon plan.

    Candidate first actions are min-cost-flow solutions for horizon lengths
    1..H-1 plus a no-move action. Each candidate is evaluated against the full
    expected-demand horizon using the same deterministic simulator mechanics
    (passenger OD and travel-time schedules) and a greedy tail policy.
    """
    config.validate()
    forecasts = _validate_forecasts(forecast_horizon)
    horizon_steps, zone_count = forecasts.shape
    if snapshot.available_vehicles.shape != (zone_count,):
        raise MPCError("Fleet snapshot zone count does not match forecast horizon.")

    passenger_probabilities = np.asarray(
        passenger_destination_probabilities,
        dtype=np.float64,
    )
    relocation_steps = _integer_matrix(
        relocation_travel_steps,
        name="relocation_travel_steps",
        zone_count=zone_count,
    )
    passenger_steps = _integer_matrix(
        passenger_travel_steps,
        name="passenger_travel_steps",
        zone_count=zone_count,
    )
    expected_matrix_shape = (zone_count, zone_count)
    if passenger_probabilities.shape != expected_matrix_shape:
        raise MPCError("Passenger destination matrix has an invalid shape.")
    if not np.isfinite(passenger_probabilities).all() or (passenger_probabilities < 0).any():
        raise MPCError("Passenger destination probabilities are invalid.")
    if not np.allclose(passenger_probabilities.sum(axis=1), 1.0, atol=1e-9, rtol=0.0):
        raise MPCError("Passenger destination probability rows must sum to one.")
    if (np.diag(relocation_steps) != 0).any():
        raise MPCError("Relocation travel-step diagonal must be zero.")
    off_diagonal = ~np.eye(zone_count, dtype=bool)
    if (relocation_steps[off_diagonal] < 1).any() or (passenger_steps < 1).any():
        raise MPCError("Off-diagonal relocation and all passenger travel steps must be positive.")

    if allowed_edges is None:
        edge_mask = None
    else:
        edge_mask = np.asarray(allowed_edges, dtype=bool)
        if edge_mask.shape != expected_matrix_shape:
            raise MPCError("allowed_edges has an invalid shape.")
        if np.diag(edge_mask).any():
            raise MPCError("allowed_edges cannot contain self-loops.")

    raw_regions = np.asarray(region_index)
    if raw_regions.shape != (zone_count,) or not np.issubdtype(raw_regions.dtype, np.number):
        raise MPCError("region_index has an invalid shape or values.")
    numeric_regions = raw_regions.astype(np.float64, copy=False)
    if (
        not np.isfinite(numeric_regions).all()
        or (numeric_regions < 0).any()
        or not np.equal(numeric_regions, np.floor(numeric_regions)).all()
    ):
        raise MPCError("region_index must contain non-negative integers.")
    regions = numeric_regions.astype(np.int64)
    region_demand = np.asarray(region_demand_history, dtype=np.float64)
    region_served = np.asarray(region_served_history, dtype=np.float64)
    if regions.shape != (zone_count,) or (regions < 0).any():
        raise MPCError("region_index has an invalid shape or values.")
    if region_demand.ndim != 1 or region_served.shape != region_demand.shape:
        raise MPCError("Region history arrays have invalid shapes.")
    if not np.isfinite(region_demand).all() or not np.isfinite(region_served).all():
        raise MPCError("Region history arrays must be finite.")
    invalid_region_counts = (
        (region_demand < 0).any()
        or (region_served < 0).any()
        or (region_served > region_demand + 1e-8).any()
    )
    if invalid_region_counts:
        raise MPCError("Region history arrays contain invalid counts.")
    if regions.max(initial=0) >= region_demand.size:
        raise MPCError("region_index references an unknown region.")

    pending: dict[int, IntegerVector] = {}
    for raw_offset, raw_values in pending_arrivals_by_step.items():
        offset = int(raw_offset)
        values = np.asarray(raw_values)
        if (
            not np.issubdtype(values.dtype, np.number)
            or values.shape != (zone_count,)
        ):
            raise MPCError("pending_arrivals_by_step contains an invalid vector.")
        numeric_values = values.astype(np.float64, copy=False)
        if (
            not np.isfinite(numeric_values).all()
            or (numeric_values < 0).any()
            or not np.equal(numeric_values, np.floor(numeric_values)).all()
        ):
            raise MPCError("pending arrivals must be finite non-negative integers.")
        pending[offset] = numeric_values.astype(np.int64)
    for offset, values in pending.items():
        if offset < 0 or values.shape != (zone_count,) or (values < 0).any():
            raise MPCError("pending_arrivals_by_step contains invalid entries.")

    if snapshot.in_transit_vehicles.shape != (zone_count,):
        raise MPCError("Fleet snapshot in-transit vector has an invalid shape.")
    scheduled_total = np.zeros(zone_count, dtype=np.int64)
    for arrivals in pending.values():
        scheduled_total += arrivals
    if not np.array_equal(scheduled_total, snapshot.in_transit_vehicles):
        raise MPCError("Pending-arrival schedule does not match snapshot in-transit supply.")

    due_now = pending.get(0, np.zeros(zone_count, dtype=np.int64))
    decision_available = snapshot.available_vehicles.astype(np.int64, copy=True) + due_now
    decision_in_transit = snapshot.in_transit_vehicles.astype(np.int64, copy=True) - due_now
    if (decision_in_transit < 0).any():
        raise MPCError("Pending arrivals do not match snapshot in-transit supply.")

    # Build a small, deterministic candidate set. Each candidate uses a
    # progressively longer forecast window and only arcs that can arrive
    # within that candidate's planning window.
    candidates: list[tuple[int, IntegerMatrix]] = [
        (0, np.zeros(expected_matrix_shape, dtype=np.int64))
    ]
    seen = {candidates[0][1].tobytes()}
    for candidate_horizon in range(1, horizon_steps):
        target_forecast = np.max(
            forecasts[1 : candidate_horizon + 1],
            axis=0,
        )
        if edge_mask is None:
            candidate_edges = relocation_steps <= candidate_horizon
        else:
            candidate_edges = edge_mask & (
                relocation_steps <= candidate_horizon
            )
        allocation: FlowAllocation = build_min_cost_relocation_action(
            available_vehicles=decision_available,
            current_forecast=forecasts[0],
            target_forecast=target_forecast,
            in_transit_vehicles=decision_in_transit,
            relocation_travel_steps=relocation_steps,
            allowed_edges=candidate_edges,
            region_index=regions,
            region_service_rates=_region_rates(region_demand, region_served),
            demand_buffer_multiplier=config.demand_buffer_multiplier,
            minimum_zone_reserve=config.minimum_zone_reserve,
            in_transit_supply_credit=config.in_transit_supply_credit,
            maximum_relocation_fraction=config.maximum_relocation_fraction,
            maximum_relocation_steps=config.maximum_relocation_steps,
            relocation_cost_per_vehicle_step=config.relocation_cost_per_vehicle_step,
            unmet_demand_penalty=config.unmet_demand_penalty,
            fairness_cost_weight=config.fairness_cost_weight,
        )
        signature = allocation.relocation_action.tobytes()
        if signature not in seen:
            candidates.append((candidate_horizon, allocation.relocation_action))
            seen.add(signature)

    evaluations: list[tuple[float, int, int, IntegerMatrix, dict[str, float]]] = []
    for candidate_horizon, action in candidates:
        metrics = _rollout_candidate(
            candidate_action=action,
            snapshot=snapshot,
            pending_arrivals_by_step=pending,
            forecasts=forecasts,
            passenger_destination_probabilities=passenger_probabilities,
            relocation_travel_steps=relocation_steps,
            passenger_travel_steps=passenger_steps,
            allowed_edges=edge_mask,
            region_index=regions,
            region_demand_history=region_demand,
            region_served_history=region_served,
            greedy_config=greedy_config,
            mpc_config=config,
        )
        evaluations.append(
            (
                metrics["objective"],
                int(action.sum()),
                candidate_horizon,
                action,
                metrics,
            )
        )

    if not evaluations:
        raise MPCError("MPC produced no candidate actions.")
    # Prefer lower objective; deterministic tie-breaks prefer fewer moves,
    # then the shorter candidate horizon.
    evaluations.sort(key=lambda item: (item[0], item[1], item[2]))
    objective, _, selected_horizon, selected_action, selected_metrics = evaluations[0]
    candidate_diagnostics = tuple(
        {
            "horizon_steps": horizon,
            "relocated_vehicles": int(action.sum()),
            "projected_operating_cost": metrics["operating_cost"],
            "projected_fairness_penalty": metrics["fairness_penalty"],
            "projected_objective": metrics["objective"],
            "projected_unmet_demand": metrics["unmet_demand"],
        }
        for _, _, horizon, action, metrics in evaluations
    )
    return MPCDecision(
        relocation_action=selected_action.copy(),
        selected_horizon_steps=selected_horizon,
        projected_operating_cost=selected_metrics["operating_cost"],
        projected_fairness_penalty=selected_metrics["fairness_penalty"],
        projected_objective=objective,
        projected_unmet_demand=selected_metrics["unmet_demand"],
        planned_relocated_vehicles=int(selected_action.sum()),
        candidate_count=len(evaluations),
        candidate_objectives=candidate_diagnostics,
    )
