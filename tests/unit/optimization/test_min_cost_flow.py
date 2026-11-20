from __future__ import annotations

import numpy as np

from urbanflow.optimization.min_cost_flow import (
    build_min_cost_relocation_action,
    solve_min_cost_transport,
)


def test_min_cost_transport_uses_global_rerouting_not_cheapest_edge_greedily() -> None:
    supply = [5, 5, 0, 0]
    demand = [0, 0, 5, 5]
    costs = np.zeros((4, 4), dtype=np.float64)
    costs[0, 2] = 1.0
    costs[0, 3] = 3.0
    costs[1, 2] = 2.0
    costs[1, 3] = 100.0
    allowed = np.zeros((4, 4), dtype=bool)
    allowed[0, 2] = True
    allowed[0, 3] = True
    allowed[1, 2] = True
    allowed[1, 3] = True

    result = solve_min_cost_transport(
        supply=supply,
        demand=demand,
        unit_costs=costs,
        allowed_edges=allowed,
        relocation_cap=10,
        unserved_unit_costs=[1_000.0, 1_000.0, 1_000.0, 1_000.0],
    )

    assert result.relocation_action[0, 3] == 5
    assert result.relocation_action[1, 2] == 5
    assert result.relocation_action[0, 2] == 0
    assert result.relocation_action[1, 3] == 0
    assert result.relocated_vehicles == 10
    assert int(result.unserved_deficit.sum()) == 0


def test_min_cost_transport_prefers_unserved_arc_when_relocation_is_too_expensive() -> None:
    costs = np.zeros((2, 2), dtype=np.float64)
    costs[0, 1] = 25.0
    allowed = np.asarray([[False, True], [True, False]])

    result = solve_min_cost_transport(
        supply=[5, 0],
        demand=[0, 5],
        unit_costs=costs,
        allowed_edges=allowed,
        relocation_cap=5,
        unserved_unit_costs=[10.0, 10.0],
    )

    assert int(result.relocation_action.sum()) == 0
    assert result.unserved_deficit.tolist() == [0, 5]


def test_min_cost_relocation_respects_fairness_graph_travel_and_global_budget() -> None:
    travel_steps = np.asarray(
        [
            [0, 1, 1, 3],
            [1, 0, 1, 1],
            [1, 1, 0, 1],
            [3, 1, 1, 0],
        ],
        dtype=np.int64,
    )
    allowed = np.asarray(
        [
            [False, True, True, True],
            [True, False, True, True],
            [True, True, False, True],
            [True, True, True, False],
        ],
        dtype=bool,
    )

    result = build_min_cost_relocation_action(
        available_vehicles=[10, 0, 0, 0],
        current_forecast=[0.0, 0.0, 0.0, 0.0],
        target_forecast=[0.0, 8.0, 8.0, 8.0],
        in_transit_vehicles=[0, 0, 0, 0],
        relocation_travel_steps=travel_steps,
        allowed_edges=allowed,
        region_index=[0, 1, 2, 3],
        region_service_rates=[1.0, 0.2, 0.9, 0.5],
        demand_buffer_multiplier=1.0,
        minimum_zone_reserve=0,
        in_transit_supply_credit=0.0,
        maximum_relocation_fraction=0.5,
        maximum_relocation_steps=1,
        relocation_cost_per_vehicle_step=0.5,
        unmet_demand_penalty=10.0,
        fairness_cost_weight=2.0,
    )

    action = result.relocation_action
    assert int(action.sum()) == 5
    # Equal-demand, equal-travel destinations are prioritized by lower
    # historical regional service rate. Zone 4 is outside the travel limit.
    assert action[0, 1] == 5
    assert action[0, 2] == 0
    assert action[0, 3] == 0
    assert (action.sum(axis=1) <= np.asarray([10, 0, 0, 0])).all()
    assert int(action.sum()) <= result.relocation_cap
    assert not ((action > 0) & (travel_steps > 1)).any()
