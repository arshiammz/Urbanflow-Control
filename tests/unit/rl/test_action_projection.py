"""Synthetic tests run without weather files, training Parquets or Torch."""
from __future__ import annotations

import numpy as np
import pytest

from urbanflow.rl.action_projection import (
    GraphActionProjector, ProjectionError, RelocationConstraints,
)


def projector(*, fraction: float = 1.0, max_steps: int = 2) -> GraphActionProjector:
    adjacency = np.asarray([
        [False, True, False],
        [True, False, True],
        [False, True, False],
    ])
    travel = np.asarray([[0, 1, 4], [1, 0, 2], [4, 2, 0]], dtype=np.int64)
    return GraphActionProjector(adjacency, travel, RelocationConstraints(
        maximum_relocation_steps=max_steps,
        maximum_relocation_fraction=fraction,
        minimum_zone_reserve=0,
        demand_buffer_multiplier=1.0,
        in_transit_supply_credit=0.0,
    ))


def test_projection_moves_only_allowed_edges_with_inventory_conservation() -> None:
    p = projector()
    a = np.zeros(p.action_size, dtype=np.float32)
    index = np.flatnonzero((p.origins == 0) & (p.destinations == 1))[0]
    a[index], a[-1] = 1.0, 1.0
    result = p.project(
        a, available=np.asarray([20, 0, 0], dtype=np.int64),
        in_transit=np.zeros(3, dtype=np.int64),
        current_forecast=np.zeros(3), future_forecasts=np.asarray([[0.0, 15.0, 1.0]]),
    )
    assert int(result.matrix[0, 1]) == 15
    assert int(result.matrix.sum()) == 15
    assert not ((result.matrix > 0) & ~p.allowed).any()
    assert np.all(result.matrix.sum(axis=1) <= [20, 0, 0])
    assert not np.diag(result.matrix).any()


def test_projection_respects_fraction_cap_and_travel_mask() -> None:
    p = projector(fraction=0.10, max_steps=1)
    assert not p.allowed[1, 2]
    a = np.ones(p.action_size, dtype=np.float32)
    output = p.project(
        a, available=np.asarray([20, 0, 0], dtype=np.int64),
        in_transit=np.zeros(3, dtype=np.int64),
        current_forecast=np.zeros(3),
        future_forecasts=np.asarray([[0.0, 20.0, 10.0]]),
    )
    assert int(output.matrix.sum()) == 2
    assert output.maximum_budget == 2
    assert output.matrix[1, 2] == 0


def test_projection_all_negative_action_is_no_relocation() -> None:
    p = projector()
    result = p.project(
        np.full(p.action_size, -1.0, dtype=np.float32),
        available=np.asarray([10, 10, 10]),
        in_transit=np.zeros(3, dtype=np.int64),
        current_forecast=np.zeros(3), future_forecasts=np.ones((1, 3)),
    )
    assert result.vehicles_moved == 0


def test_projection_rejects_invalid_values_and_graph() -> None:
    p = projector()
    bad = np.full(p.action_size, np.nan)
    with pytest.raises(ProjectionError):
        p.project(
            bad, available=np.asarray([4, 4, 4]),
            in_transit=np.zeros(3, dtype=np.int64),
            current_forecast=np.zeros(3), future_forecasts=np.ones((1, 3)),
        )
    with pytest.raises(ProjectionError):
        GraphActionProjector(
            np.asarray([[True, True], [False, False]]),
            np.asarray([[0, 1], [1, 0]]), p.constraints,
        )


def test_projection_never_uses_future_realized_demand() -> None:
    p = projector()
    args = dict(
        raw_action=np.ones(p.action_size, dtype=np.float32),
        available=np.asarray([20, 0, 0], dtype=np.int64),
        in_transit=np.zeros(3, dtype=np.int64),
        current_forecast=np.zeros(3), future_forecasts=np.asarray([[0.0, 20.0, 0.0]]),
    )
    left = p.project(**args).matrix
    right = p.project(**args).matrix
    np.testing.assert_array_equal(left, right)
