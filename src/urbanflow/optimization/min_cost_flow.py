"""Min-cost transportation for constrained fleet rebalancing.

The solver uses a successive-shortest-augmenting-path min-cost-flow algorithm.
Unfilled destination demand is represented by an explicit dummy source with
an unmet-demand penalty. This lets the optimizer decide whether a relocation
is economically justified instead of forcing every deficit to be filled.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

IntegerVector = NDArray[np.int64]
IntegerMatrix = NDArray[np.int64]
FloatVector = NDArray[np.float64]
FloatMatrix = NDArray[np.float64]
BooleanMatrix = NDArray[np.bool_]

_COST_SCALE = 1_000
_INFINITY = 10**30


class MinCostFlowError(ValueError):
    """Raised when a min-cost-flow problem is invalid or infeasible."""


@dataclass(frozen=True)
class FlowAllocation:
    """Relocation matrix and objective diagnostics for one planning step."""

    relocation_action: IntegerMatrix
    unserved_deficit: IntegerVector
    supply_by_origin: IntegerVector
    demand_by_destination: IntegerVector
    relocation_cap: int
    relocated_vehicles: int
    objective_cost: float

    def as_dict(self) -> dict[str, Any]:
        """Return JSON-compatible flow diagnostics."""
        return {
            "relocation_cap": self.relocation_cap,
            "relocated_vehicles": self.relocated_vehicles,
            "unserved_deficit_total": int(self.unserved_deficit.sum()),
            "objective_cost": self.objective_cost,
        }


@dataclass
class _ResidualEdge:
    to: int
    reverse_index: int
    capacity: int
    cost: int
    initial_capacity: int


def _integer_vector(
    values: Any,
    *,
    name: str,
    expected_size: int | None = None,
) -> IntegerVector:
    array = np.asarray(values)
    if array.ndim != 1:
        raise MinCostFlowError(f"{name} must be one-dimensional.")
    if expected_size is not None and array.shape != (expected_size,):
        raise MinCostFlowError(
            f"{name} must have shape ({expected_size},), received {array.shape}."
        )
    if not np.issubdtype(array.dtype, np.number):
        raise MinCostFlowError(f"{name} must be numeric.")
    numeric = array.astype(np.float64, copy=False)
    if not np.isfinite(numeric).all() or (numeric < 0).any():
        raise MinCostFlowError(f"{name} must contain finite non-negative values.")
    if not np.equal(numeric, np.floor(numeric)).all():
        raise MinCostFlowError(f"{name} must contain integer values.")
    return numeric.astype(np.int64, copy=True)


def _integer_matrix(
    values: Any,
    *,
    name: str,
    expected_size: int,
) -> IntegerMatrix:
    array = np.asarray(values)
    expected_shape = (expected_size, expected_size)
    if array.shape != expected_shape:
        raise MinCostFlowError(
            f"{name} must have shape {expected_shape}, received {array.shape}."
        )
    if not np.issubdtype(array.dtype, np.number):
        raise MinCostFlowError(f"{name} must be numeric.")
    numeric = array.astype(np.float64, copy=False)
    if not np.isfinite(numeric).all() or (numeric < 0).any():
        raise MinCostFlowError(f"{name} must contain finite non-negative values.")
    if not np.equal(numeric, np.floor(numeric)).all():
        raise MinCostFlowError(f"{name} must contain integer values.")
    return numeric.astype(np.int64, copy=True)


def _float_vector(
    values: Any,
    *,
    name: str,
    expected_size: int,
) -> FloatVector:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (expected_size,):
        raise MinCostFlowError(
            f"{name} must have shape ({expected_size},), received {array.shape}."
        )
    if not np.isfinite(array).all() or (array < 0).any():
        raise MinCostFlowError(f"{name} must contain finite non-negative values.")
    return array.copy()


def _add_edge(
    graph: list[list[_ResidualEdge]],
    source: int,
    destination: int,
    capacity: int,
    cost: int,
) -> int:
    """Add a forward edge and its reverse residual edge; return forward index."""
    forward_index = len(graph[source])
    reverse_index = len(graph[destination])
    graph[source].append(
        _ResidualEdge(
            to=destination,
            reverse_index=reverse_index,
            capacity=capacity,
            cost=cost,
            initial_capacity=capacity,
        )
    )
    graph[destination].append(
        _ResidualEdge(
            to=source,
            reverse_index=forward_index,
            capacity=0,
            cost=-cost,
            initial_capacity=0,
        )
    )
    return forward_index


def _successive_shortest_augmenting_path(
    graph: list[list[_ResidualEdge]],
    *,
    source: int,
    sink: int,
    required_flow: int,
) -> tuple[int, int]:
    """Send a fixed amount of flow at minimum cost using reduced-cost Dijkstra."""
    node_count = len(graph)
    potential = [0] * node_count
    total_flow = 0
    total_cost = 0

    while total_flow < required_flow:
        distance = [_INFINITY] * node_count
        previous_node = [-1] * node_count
        previous_edge = [-1] * node_count
        distance[source] = 0
        queue: list[tuple[int, int]] = [(0, source)]

        while queue:
            current_distance, node = heapq.heappop(queue)
            if current_distance != distance[node]:
                continue
            for edge_index, edge in enumerate(graph[node]):
                if edge.capacity <= 0:
                    continue
                reduced_cost = edge.cost + potential[node] - potential[edge.to]
                candidate = current_distance + reduced_cost
                if candidate < distance[edge.to]:
                    distance[edge.to] = candidate
                    previous_node[edge.to] = node
                    previous_edge[edge.to] = edge_index
                    heapq.heappush(queue, (candidate, edge.to))

        if distance[sink] == _INFINITY:
            break

        for node in range(node_count):
            if distance[node] < _INFINITY:
                potential[node] += distance[node]

        augmentation = required_flow - total_flow
        node = sink
        while node != source:
            parent = previous_node[node]
            edge_index = previous_edge[node]
            if parent < 0 or edge_index < 0:
                raise MinCostFlowError("Broken predecessor chain in min-cost-flow solver.")
            augmentation = min(augmentation, graph[parent][edge_index].capacity)
            node = parent

        node = sink
        path_cost = 0
        while node != source:
            parent = previous_node[node]
            edge_index = previous_edge[node]
            edge = graph[parent][edge_index]
            path_cost += edge.cost
            edge.capacity -= augmentation
            graph[node][edge.reverse_index].capacity += augmentation
            node = parent

        total_flow += augmentation
        total_cost += augmentation * path_cost

    return total_flow, total_cost


def solve_min_cost_transport(
    *,
    supply: Any,
    demand: Any,
    unit_costs: Any,
    allowed_edges: Any | None,
    relocation_cap: int,
    unserved_unit_costs: Any,
) -> FlowAllocation:
    """Solve a capacitated bipartite min-cost-flow allocation.

    Every unit of destination demand is accounted for by either a real
    origin-to-destination relocation or a dummy unserved-demand flow. The
    dummy source carries the configured unmet-demand penalty, so the solver
    can leave expensive or infeasible deficits unserved.
    """
    supply_vector = _integer_vector(supply, name="supply")
    zone_count = supply_vector.size
    if zone_count < 2:
        raise MinCostFlowError("At least two zones are required.")
    demand_vector = _integer_vector(
        demand,
        name="demand",
        expected_size=zone_count,
    )
    costs = np.asarray(unit_costs, dtype=np.float64)
    if costs.shape != (zone_count, zone_count):
        raise MinCostFlowError(
            "unit_costs must have shape "
            f"({zone_count}, {zone_count}), received {costs.shape}."
        )
    if not np.isfinite(costs).all() or (costs < 0).any():
        raise MinCostFlowError("unit_costs must be finite and non-negative.")

    if allowed_edges is None:
        edge_mask = np.ones((zone_count, zone_count), dtype=bool)
    else:
        edge_mask = np.asarray(allowed_edges, dtype=bool)
        if edge_mask.shape != (zone_count, zone_count):
            raise MinCostFlowError("allowed_edges has an invalid shape.")
    edge_mask = edge_mask.copy()
    np.fill_diagonal(edge_mask, False)

    unserved_costs = _float_vector(
        unserved_unit_costs,
        name="unserved_unit_costs",
        expected_size=zone_count,
    )
    if not isinstance(relocation_cap, int) or relocation_cap < 0:
        raise MinCostFlowError("relocation_cap must be a non-negative integer.")

    total_demand = int(demand_vector.sum())
    action = np.zeros((zone_count, zone_count), dtype=np.int64)
    unserved = demand_vector.copy()
    if total_demand == 0 or relocation_cap == 0 or int(supply_vector.sum()) == 0:
        return FlowAllocation(
            relocation_action=action,
            unserved_deficit=unserved,
            supply_by_origin=supply_vector,
            demand_by_destination=demand_vector,
            relocation_cap=relocation_cap,
            relocated_vehicles=0,
            objective_cost=float(np.dot(unserved, unserved_costs)),
        )

    source = 0
    budget_node = 1
    origin_start = 2
    destination_start = origin_start + zone_count
    dummy_node = destination_start + zone_count
    sink = dummy_node + 1
    graph: list[list[_ResidualEdge]] = [[] for _ in range(sink + 1)]

    _add_edge(graph, source, budget_node, relocation_cap, 0)
    _add_edge(graph, source, dummy_node, total_demand, 0)

    for origin in range(zone_count):
        _add_edge(
            graph,
            budget_node,
            origin_start + origin,
            int(supply_vector[origin]),
            0,
        )

    for destination in range(zone_count):
        _add_edge(
            graph,
            destination_start + destination,
            sink,
            int(demand_vector[destination]),
            0,
        )
        _add_edge(
            graph,
            dummy_node,
            destination_start + destination,
            int(demand_vector[destination]),
            int(round(float(unserved_costs[destination]) * _COST_SCALE)),
        )

    edge_references: dict[tuple[int, int], tuple[int, int]] = {}
    feasible_arcs = edge_mask & (supply_vector[:, None] > 0) & (demand_vector[None, :] > 0)
    for origin_value, destination_value in np.argwhere(feasible_arcs):
        origin = int(origin_value)
        destination = int(destination_value)
        capacity = min(
            int(supply_vector[origin]),
            int(demand_vector[destination]),
            relocation_cap,
        )
        edge_index = _add_edge(
            graph,
            origin_start + origin,
            destination_start + destination,
            capacity,
            int(round(float(costs[origin, destination]) * _COST_SCALE)),
        )
        edge_references[(origin, destination)] = (
            origin_start + origin,
            edge_index,
        )

    sent_flow, total_cost_scaled = _successive_shortest_augmenting_path(
        graph,
        source=source,
        sink=sink,
        required_flow=total_demand,
    )
    if sent_flow != total_demand:
        raise MinCostFlowError(
            "Transportation problem is infeasible even with unserved-demand arcs."
        )

    for (origin, destination), (node, edge_index) in edge_references.items():
        edge = graph[node][edge_index]
        flow = edge.initial_capacity - edge.capacity
        if flow:
            action[origin, destination] = flow

    unserved = demand_vector - action.sum(axis=0)
    relocated = int(action.sum())
    if (unserved < 0).any():
        raise MinCostFlowError("Flow solution exceeds a destination's demand.")
    if (action.sum(axis=1) > supply_vector).any():
        raise MinCostFlowError("Flow solution exceeds origin supply.")
    if relocated > relocation_cap:
        raise MinCostFlowError("Flow solution exceeds the global relocation cap.")

    return FlowAllocation(
        relocation_action=action,
        unserved_deficit=unserved,
        supply_by_origin=supply_vector,
        demand_by_destination=demand_vector,
        relocation_cap=relocation_cap,
        relocated_vehicles=relocated,
        objective_cost=total_cost_scaled / _COST_SCALE,
    )


def build_min_cost_relocation_action(
    *,
    available_vehicles: Any,
    current_forecast: Any,
    target_forecast: Any,
    in_transit_vehicles: Any,
    relocation_travel_steps: Any,
    allowed_edges: Any | None,
    region_index: Any,
    region_service_rates: Any,
    demand_buffer_multiplier: float,
    minimum_zone_reserve: int,
    in_transit_supply_credit: float,
    maximum_relocation_fraction: float,
    maximum_relocation_steps: int,
    relocation_cost_per_vehicle_step: float,
    unmet_demand_penalty: float,
    fairness_cost_weight: float,
) -> FlowAllocation:
    """Build a forecast-aware relocation action using min-cost flow."""
    available = _integer_vector(available_vehicles, name="available_vehicles")
    zone_count = available.size
    current = _float_vector(
        current_forecast,
        name="current_forecast",
        expected_size=zone_count,
    )
    target = _float_vector(
        target_forecast,
        name="target_forecast",
        expected_size=zone_count,
    )
    in_transit = _integer_vector(
        in_transit_vehicles,
        name="in_transit_vehicles",
        expected_size=zone_count,
    )
    travel_steps = _integer_matrix(
        relocation_travel_steps,
        name="relocation_travel_steps",
        expected_size=zone_count,
    )
    if (travel_steps < 0).any() or (np.diag(travel_steps) != 0).any():
        raise MinCostFlowError("Relocation travel-step matrix is invalid.")
    off_diagonal = ~np.eye(zone_count, dtype=bool)
    if (travel_steps[off_diagonal] < 1).any():
        raise MinCostFlowError("Off-diagonal travel steps must be positive.")

    raw_regions = np.asarray(region_index)
    if raw_regions.shape != (zone_count,) or not np.issubdtype(raw_regions.dtype, np.number):
        raise MinCostFlowError("region_index has an invalid shape or values.")
    numeric_regions = raw_regions.astype(np.float64, copy=False)
    if (
        not np.isfinite(numeric_regions).all()
        or (numeric_regions < 0).any()
        or not np.equal(numeric_regions, np.floor(numeric_regions)).all()
    ):
        raise MinCostFlowError("region_index must contain non-negative integers.")
    regions = numeric_regions.astype(np.int64)
    rates = np.asarray(region_service_rates, dtype=np.float64)
    if rates.ndim != 1 or rates.size == 0:
        raise MinCostFlowError("region_service_rates must be a non-empty vector.")
    if not np.isfinite(rates).all() or ((rates < 0) | (rates > 1)).any():
        raise MinCostFlowError("region_service_rates must be finite values in [0, 1].")
    if regions.max(initial=0) >= rates.size:
        raise MinCostFlowError("region_index references an unknown region.")

    controls = {
        "demand_buffer_multiplier": demand_buffer_multiplier,
        "in_transit_supply_credit": in_transit_supply_credit,
        "maximum_relocation_fraction": maximum_relocation_fraction,
        "relocation_cost_per_vehicle_step": relocation_cost_per_vehicle_step,
        "unmet_demand_penalty": unmet_demand_penalty,
        "fairness_cost_weight": fairness_cost_weight,
    }
    if not all(math.isfinite(value) for value in controls.values()):
        raise MinCostFlowError("Optimization controls must be finite.")
    if demand_buffer_multiplier <= 0:
        raise MinCostFlowError("demand_buffer_multiplier must be positive.")
    if not 0 <= in_transit_supply_credit <= 1:
        raise MinCostFlowError("in_transit_supply_credit must be in [0, 1].")
    if not 0 <= maximum_relocation_fraction <= 1:
        raise MinCostFlowError("maximum_relocation_fraction must be in [0, 1].")
    if minimum_zone_reserve < 0 or maximum_relocation_steps < 1:
        raise MinCostFlowError("Reserve and maximum travel steps are invalid.")
    if min(relocation_cost_per_vehicle_step, unmet_demand_penalty, fairness_cost_weight) < 0:
        raise MinCostFlowError("Cost weights must be non-negative.")

    current_reserve = np.ceil(current * demand_buffer_multiplier).astype(np.int64)
    current_reserve += minimum_zone_reserve
    target_inventory = np.ceil(target * demand_buffer_multiplier).astype(np.int64)
    target_inventory += minimum_zone_reserve
    projected_supply = available + np.floor(
        in_transit.astype(np.float64) * in_transit_supply_credit
    ).astype(np.int64)

    origin_supply = np.maximum(available - current_reserve, 0).astype(np.int64)
    destination_deficit = np.maximum(target_inventory - projected_supply, 0).astype(np.int64)
    relocation_cap = int(math.floor(int(available.sum()) * maximum_relocation_fraction))

    if allowed_edges is None:
        edge_mask = np.ones((zone_count, zone_count), dtype=bool)
    else:
        edge_mask = np.asarray(allowed_edges, dtype=bool)
        if edge_mask.shape != (zone_count, zone_count):
            raise MinCostFlowError("allowed_edges has an invalid shape.")
    edge_mask = edge_mask & (travel_steps <= maximum_relocation_steps)
    np.fill_diagonal(edge_mask, False)

    destination_rates = rates[regions]
    unit_costs = (
        travel_steps.astype(np.float64) * relocation_cost_per_vehicle_step
        + fairness_cost_weight * destination_rates[None, :]
    )

    unserved_costs = np.asarray(
        [
            unmet_demand_penalty
            + fairness_cost_weight * (1.0 - float(rates[regions[destination]]))
            for destination in range(zone_count)
        ],
        dtype=np.float64,
    )

    return solve_min_cost_transport(
        supply=origin_supply,
        demand=destination_deficit,
        unit_costs=unit_costs,
        allowed_edges=edge_mask,
        relocation_cap=relocation_cap,
        unserved_unit_costs=unserved_costs,
    )
