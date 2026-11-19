from __future__ import annotations

import numpy as np
import pytest

from urbanflow.simulation.fleet_simulator import (
    FleetSimulationError,
    FleetSimulator,
    SimulationCosts,
    allocate_integer_flow,
)


def build_simulator() -> FleetSimulator:
    """Build a small deterministic simulator for tests."""
    return FleetSimulator(
        zone_ids=(
            101,
            102,
            103,
        ),
        initial_vehicles=[
            5,
            3,
            2,
        ],
        passenger_destination_probabilities=[
            [
                0.0,
                0.6,
                0.4,
            ],
            [
                0.5,
                0.0,
                0.5,
            ],
            [
                0.5,
                0.5,
                0.0,
            ],
        ],
        relocation_travel_steps=[
            [
                0,
                1,
                2,
            ],
            [
                1,
                0,
                1,
            ],
            [
                2,
                1,
                0,
            ],
        ],
        passenger_travel_steps=[
            [
                1,
                1,
                2,
            ],
            [
                1,
                1,
                1,
            ],
            [
                2,
                1,
                1,
            ],
        ],
        costs=SimulationCosts(
            unmet_demand_penalty=10.0,
            relocation_cost_per_vehicle_step=0.5,
            idle_vehicle_cost=0.1,
        ),
    )


def test_allocate_integer_flow_preserves_total() -> None:
    allocation = allocate_integer_flow(
        7,
        np.asarray(
            [
                0.2,
                0.5,
                0.3,
            ],
            dtype=np.float64,
        ),
    )

    assert allocation.tolist() == [
        1,
        4,
        2,
    ]

    assert int(
        allocation.sum()
    ) == 7


def test_rejects_probability_rows_not_summing_to_one() -> None:
    with pytest.raises(
        FleetSimulationError,
        match="row must sum to one",
    ):
        FleetSimulator(
            zone_ids=(
                1,
                2,
            ),
            initial_vehicles=[
                2,
                2,
            ],
            passenger_destination_probabilities=[
                [
                    0.5,
                    0.4,
                ],
                [
                    0.5,
                    0.5,
                ],
            ],
            relocation_travel_steps=[
                [
                    0,
                    1,
                ],
                [
                    1,
                    0,
                ],
            ],
            passenger_travel_steps=[
                [
                    1,
                    1,
                ],
                [
                    1,
                    1,
                ],
            ],
            costs=SimulationCosts(
                unmet_demand_penalty=1.0,
                relocation_cost_per_vehicle_step=1.0,
                idle_vehicle_cost=1.0,
            ),
        )


def test_rejects_relocation_above_available_supply() -> None:
    simulator = build_simulator()

    with pytest.raises(
        FleetSimulationError,
        match="exceeds available fleet",
    ):
        simulator.step(
            demand=[
                0,
                0,
                0,
            ],
            relocation_action=[
                [
                    0,
                    6,
                    0,
                ],
                [
                    0,
                    0,
                    0,
                ],
                [
                    0,
                    0,
                    0,
                ],
            ],
        )


def test_serves_demand_and_records_unmet_demand() -> None:
    simulator = build_simulator()

    result = simulator.step(
        demand=[
            7,
            1,
            0,
        ],
        relocation_action=[
            [
                0,
                0,
                0,
            ],
            [
                0,
                0,
                0,
            ],
            [
                0,
                0,
                0,
            ],
        ],
    )

    assert result.served_demand.tolist() == [
        5,
        1,
        0,
    ]

    assert result.unmet_demand.tolist() == [
        2,
        0,
        0,
    ]

    assert result.idle_vehicles.tolist() == [
        0,
        2,
        2,
    ]

    assert (
        result.metrics.served_demand_total
        == 6
    )

    assert (
        result.metrics.unmet_demand_total
        == 2
    )

    assert (
        result.metrics.unmet_demand_cost
        == 20.0
    )


def test_relocation_arrives_at_future_step() -> None:
    simulator = build_simulator()

    first = simulator.step(
        demand=[
            0,
            0,
            0,
        ],
        relocation_action=[
            [
                0,
                1,
                0,
            ],
            [
                0,
                0,
                0,
            ],
            [
                0,
                0,
                0,
            ],
        ],
    )

    assert first.available_vehicles_end.tolist() == [
        4,
        3,
        2,
    ]

    assert first.in_transit_vehicles_end.tolist() == [
        0,
        1,
        0,
    ]

    second = simulator.step(
        demand=[
            0,
            0,
            0,
        ],
        relocation_action=[
            [
                0,
                0,
                0,
            ],
            [
                0,
                0,
                0,
            ],
            [
                0,
                0,
                0,
            ],
        ],
    )

    assert second.arrivals_released.tolist() == [
        0,
        1,
        0,
    ]

    assert second.available_vehicles_end.tolist() == [
        4,
        4,
        2,
    ]


def test_passenger_trips_move_vehicles_between_zones() -> None:
    simulator = build_simulator()

    first = simulator.step(
        demand=[
            5,
            0,
            0,
        ],
        relocation_action=[
            [
                0,
                0,
                0,
            ],
            [
                0,
                0,
                0,
            ],
            [
                0,
                0,
                0,
            ],
        ],
    )

    assert first.passenger_flow.tolist() == [
        [
            0,
            3,
            2,
        ],
        [
            0,
            0,
            0,
        ],
        [
            0,
            0,
            0,
        ],
    ]

    assert first.available_vehicles_end.tolist() == [
        0,
        3,
        2,
    ]

    assert first.in_transit_vehicles_end.tolist() == [
        0,
        3,
        2,
    ]

    second = simulator.step(
        demand=[
            0,
            0,
            0,
        ],
        relocation_action=[
            [
                0,
                0,
                0,
            ],
            [
                0,
                0,
                0,
            ],
            [
                0,
                0,
                0,
            ],
        ],
    )

    assert second.arrivals_released.tolist() == [
        0,
        3,
        0,
    ]

    assert second.available_vehicles_end.tolist() == [
        0,
        6,
        2,
    ]

    third = simulator.step(
        demand=[
            0,
            0,
            0,
        ],
        relocation_action=[
            [
                0,
                0,
                0,
            ],
            [
                0,
                0,
                0,
            ],
            [
                0,
                0,
                0,
            ],
        ],
    )

    assert third.arrivals_released.tolist() == [
        0,
        0,
        2,
    ]

    assert third.available_vehicles_end.tolist() == [
        0,
        6,
        4,
    ]


def test_fleet_is_conserved_across_multiple_steps() -> None:
    simulator = build_simulator()

    scenarios = [
        (
            [
                2,
                1,
                0,
            ],
            [
                [
                    0,
                    1,
                    0,
                ],
                [
                    0,
                    0,
                    0,
                ],
                [
                    0,
                    0,
                    0,
                ],
            ],
        ),
        (
            [
                1,
                2,
                1,
            ],
            [
                [
                    0,
                    0,
                    0,
                ],
                [
                    0,
                    0,
                    0,
                ],
                [
                    0,
                    0,
                    0,
                ],
            ],
        ),
        (
            [
                0,
                1,
                1,
            ],
            [
                [
                    0,
                    0,
                    0,
                ],
                [
                    0,
                    0,
                    0,
                ],
                [
                    0,
                    0,
                    0,
                ],
            ],
        ),
    ]

    for demand, relocation in scenarios:
        result = simulator.step(
            demand=demand,
            relocation_action=relocation,
        )

        assert result.fleet_conservation_passed

        assert (
            int(
                result.available_vehicles_end.sum()
            )
            + int(
                result.in_transit_vehicles_end.sum()
            )
            == simulator.total_fleet
        )


def test_summary_aggregates_completed_steps() -> None:
    simulator = build_simulator()

    simulator.step(
        demand=[
            2,
            1,
            1,
        ],
        relocation_action=[
            [
                0,
                0,
                0,
            ],
            [
                0,
                0,
                0,
            ],
            [
                0,
                0,
                0,
            ],
        ],
    )

    simulator.step(
        demand=[
            3,
            2,
            1,
        ],
        relocation_action=[
            [
                0,
                0,
                0,
            ],
            [
                0,
                0,
                0,
            ],
            [
                0,
                0,
                0,
            ],
        ],
    )

    summary = simulator.summary()

    assert summary[
        "completed_steps"
    ] == 2

    assert summary[
        "demand_total"
    ] == 10

    assert (
        summary[
            "served_demand_total"
        ]
        + summary[
            "unmet_demand_total"
        ]
        == 10
    )

    assert (
        sum(
            summary[
                "snapshot"
            ][
                "available_vehicles"
            ]
        )
        + sum(
            summary[
                "snapshot"
            ][
                "in_transit_vehicles"
            ]
        )
        == simulator.total_fleet
    )

def test_rejects_relocation_across_non_adjacent_zone() -> None:
    simulator = FleetSimulator(
        zone_ids=(101, 102, 103),
        initial_vehicles=[5, 3, 2],
        passenger_destination_probabilities=[
            [0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]
        ],
        relocation_travel_steps=[[0, 1, 2], [1, 0, 1], [2, 1, 0]],
        passenger_travel_steps=[[1, 1, 2], [1, 1, 1], [2, 1, 1]],
        allowed_relocation_edges=[
            [False, True, False], [True, False, True], [False, True, False]
        ],
        costs=SimulationCosts(
            unmet_demand_penalty=10.0,
            relocation_cost_per_vehicle_step=0.5,
            idle_vehicle_cost=0.1,
        ),
    )
    with pytest.raises(FleetSimulationError, match="adjacency constraints"):
        simulator.step(
            demand=[0, 0, 0],
            relocation_action=[[0, 0, 1], [0, 0, 0], [0, 0, 0]],
        )
