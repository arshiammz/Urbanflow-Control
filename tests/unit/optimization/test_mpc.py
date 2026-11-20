from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from urbanflow.optimization.mpc import MPCConfig, build_mpc_action
from urbanflow.simulation.fleet_simulator import FleetSnapshot


def build_inputs() -> tuple[
    FleetSnapshot,
    dict[int, np.ndarray],
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    snapshot = FleetSnapshot(
        step=0,
        available_vehicles=np.asarray([5, 0], dtype=np.int64),
        in_transit_vehicles=np.asarray([0, 0], dtype=np.int64),
        scheduled_arrival_steps=(),
        total_fleet=5,
    )
    forecasts = np.asarray(
        [
            [0.0, 0.0],
            [0.0, 4.0],
            [0.0, 4.0],
        ],
        dtype=np.float64,
    )
    od = np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64)
    relocation_steps = np.asarray([[0, 1], [1, 0]], dtype=np.int64)
    passenger_steps = np.asarray([[1, 1], [1, 1]], dtype=np.int64)
    return snapshot, {}, forecasts, od, np.stack([relocation_steps, passenger_steps])


def test_mpc_selects_first_action_using_horizon_rollout() -> None:
    snapshot, pending, forecasts, od, matrices = build_inputs()
    relocation_steps, passenger_steps = matrices[0], matrices[1]
    allowed = np.asarray([[False, True], [True, False]])
    greedy_config = SimpleNamespace(
        demand_buffer_multiplier=1.0,
        minimum_zone_reserve=0,
        in_transit_supply_credit=0.0,
        maximum_relocation_fraction=0.5,
        maximum_relocation_steps=1,
    )
    config = MPCConfig(
        demand_buffer_multiplier=1.0,
        minimum_zone_reserve=0,
        in_transit_supply_credit=0.0,
        maximum_relocation_fraction=0.5,
        maximum_relocation_steps=1,
        relocation_cost_per_vehicle_step=0.1,
        unmet_demand_penalty=10.0,
        idle_vehicle_cost=0.0,
        fairness_cost_weight=0.5,
    )

    decision = build_mpc_action(
        snapshot=snapshot,
        pending_arrivals_by_step=pending,
        forecast_horizon=forecasts,
        passenger_destination_probabilities=od,
        relocation_travel_steps=relocation_steps,
        passenger_travel_steps=passenger_steps,
        allowed_edges=allowed,
        region_index=[0, 1],
        region_demand_history=[100.0, 100.0],
        region_served_history=[90.0, 50.0],
        greedy_config=greedy_config,
        config=config,
    )

    assert decision.candidate_count >= 2
    assert decision.relocation_action[0, 1] == 2
    assert int(decision.relocation_action.sum()) <= 2
    assert not ((decision.relocation_action > 0) & ~allowed).any()
    assert decision.projected_objective >= 0
    assert decision.selected_horizon_steps in {0, 1, 2}


def test_mpc_returns_zero_action_when_future_demand_is_zero() -> None:
    snapshot, pending, _, od, matrices = build_inputs()
    relocation_steps, passenger_steps = matrices[0], matrices[1]
    forecasts = np.zeros((3, 2), dtype=np.float64)
    allowed = np.asarray([[False, True], [True, False]])
    greedy_config = SimpleNamespace(
        demand_buffer_multiplier=1.0,
        minimum_zone_reserve=0,
        in_transit_supply_credit=0.0,
        maximum_relocation_fraction=0.5,
        maximum_relocation_steps=1,
    )
    config = MPCConfig(
        demand_buffer_multiplier=1.0,
        minimum_zone_reserve=0,
        in_transit_supply_credit=0.0,
        maximum_relocation_fraction=0.5,
        maximum_relocation_steps=1,
        relocation_cost_per_vehicle_step=0.1,
        unmet_demand_penalty=10.0,
        idle_vehicle_cost=0.0,
        fairness_cost_weight=0.5,
    )

    decision = build_mpc_action(
        snapshot=snapshot,
        pending_arrivals_by_step=pending,
        forecast_horizon=forecasts,
        passenger_destination_probabilities=od,
        relocation_travel_steps=relocation_steps,
        passenger_travel_steps=passenger_steps,
        allowed_edges=allowed,
        region_index=[0, 1],
        region_demand_history=[0.0, 0.0],
        region_served_history=[0.0, 0.0],
        greedy_config=greedy_config,
        config=config,
    )

    assert int(decision.relocation_action.sum()) == 0
