from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray


IntegerVector = NDArray[np.int64]
IntegerMatrix = NDArray[np.int64]
FloatMatrix = NDArray[np.float64]


class FleetSimulationError(RuntimeError):
    """Raised when fleet simulation cannot continue safely."""


@dataclass(frozen=True)
class SimulationCosts:
    """Operational cost coefficients for one simulation step."""

    unmet_demand_penalty: float
    relocation_cost_per_vehicle_step: float
    idle_vehicle_cost: float

    def validate(self) -> None:
        """Validate cost coefficients."""
        for name, value in (
            (
                "unmet_demand_penalty",
                self.unmet_demand_penalty,
            ),
            (
                "relocation_cost_per_vehicle_step",
                self.relocation_cost_per_vehicle_step,
            ),
            (
                "idle_vehicle_cost",
                self.idle_vehicle_cost,
            ),
        ):
            if not np.isfinite(value):
                raise FleetSimulationError(
                    f"{name} must be finite."
                )

            if value < 0:
                raise FleetSimulationError(
                    f"{name} must be non-negative."
                )


@dataclass(frozen=True)
class FleetSnapshot:
    """Immutable view of the current fleet state."""

    step: int
    available_vehicles: IntegerVector
    in_transit_vehicles: IntegerVector
    scheduled_arrival_steps: tuple[int, ...]
    total_fleet: int

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible snapshot."""
        return {
            "step": self.step,
            "available_vehicles": (
                self.available_vehicles.tolist()
            ),
            "in_transit_vehicles": (
                self.in_transit_vehicles.tolist()
            ),
            "scheduled_arrival_steps": list(
                self.scheduled_arrival_steps
            ),
            "total_fleet": self.total_fleet,
        }


@dataclass(frozen=True)
class StepMetrics:
    """Aggregated operational metrics from one interval."""

    demand_total: int
    served_demand_total: int
    unmet_demand_total: int
    idle_vehicle_total: int
    relocated_vehicle_total: int
    relocation_vehicle_steps: int
    passenger_trip_total: int
    arrivals_released_total: int

    service_rate: float

    unmet_demand_cost: float
    relocation_cost: float
    idle_vehicle_cost: float
    total_operating_cost: float

    def as_dict(self) -> dict[str, Any]:
        """Return JSON-compatible metrics."""
        return {
            "demand_total": self.demand_total,
            "served_demand_total": (
                self.served_demand_total
            ),
            "unmet_demand_total": (
                self.unmet_demand_total
            ),
            "idle_vehicle_total": (
                self.idle_vehicle_total
            ),
            "relocated_vehicle_total": (
                self.relocated_vehicle_total
            ),
            "relocation_vehicle_steps": (
                self.relocation_vehicle_steps
            ),
            "passenger_trip_total": (
                self.passenger_trip_total
            ),
            "arrivals_released_total": (
                self.arrivals_released_total
            ),
            "service_rate": self.service_rate,
            "unmet_demand_cost": (
                self.unmet_demand_cost
            ),
            "relocation_cost": self.relocation_cost,
            "idle_vehicle_cost": (
                self.idle_vehicle_cost
            ),
            "total_operating_cost": (
                self.total_operating_cost
            ),
        }


@dataclass(frozen=True)
class StepResult:
    """Complete output of one simulator transition."""

    step: int

    arrivals_released: IntegerVector
    demand: IntegerVector
    relocation_action: IntegerMatrix

    served_demand: IntegerVector
    unmet_demand: IntegerVector
    idle_vehicles: IntegerVector

    passenger_flow: IntegerMatrix

    available_vehicles_end: IntegerVector
    in_transit_vehicles_end: IntegerVector

    metrics: StepMetrics

    fleet_conservation_passed: bool

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible transition record."""
        return {
            "step": self.step,
            "arrivals_released": (
                self.arrivals_released.tolist()
            ),
            "demand": self.demand.tolist(),
            "relocation_action": (
                self.relocation_action.tolist()
            ),
            "served_demand": (
                self.served_demand.tolist()
            ),
            "unmet_demand": (
                self.unmet_demand.tolist()
            ),
            "idle_vehicles": (
                self.idle_vehicles.tolist()
            ),
            "passenger_flow": (
                self.passenger_flow.tolist()
            ),
            "available_vehicles_end": (
                self.available_vehicles_end.tolist()
            ),
            "in_transit_vehicles_end": (
                self.in_transit_vehicles_end.tolist()
            ),
            "metrics": self.metrics.as_dict(),
            "fleet_conservation_passed": (
                self.fleet_conservation_passed
            ),
        }


def as_non_negative_integer_vector(
    values: Any,
    *,
    expected_size: int,
    description: str,
) -> IntegerVector:
    """Convert a value to a validated integer vector."""
    array = np.asarray(
        values
    )

    if array.shape != (
        expected_size,
    ):
        raise FleetSimulationError(
            f"{description} must have shape "
            f"({expected_size},), received {array.shape}."
        )

    if not np.issubdtype(
        array.dtype,
        np.number,
    ):
        raise FleetSimulationError(
            f"{description} must be numeric."
        )

    numeric_array = array.astype(
        np.float64,
        copy=False,
    )

    if not np.isfinite(
        numeric_array
    ).all():
        raise FleetSimulationError(
            f"{description} contains non-finite values."
        )

    if (
        numeric_array
        < 0
    ).any():
        raise FleetSimulationError(
            f"{description} contains negative values."
        )

    if not np.equal(
        numeric_array,
        np.floor(
            numeric_array
        ),
    ).all():
        raise FleetSimulationError(
            f"{description} must contain integers."
        )

    return numeric_array.astype(
        np.int64
    )


def as_non_negative_integer_matrix(
    values: Any,
    *,
    expected_size: int,
    description: str,
) -> IntegerMatrix:
    """Convert a value to a validated integer matrix."""
    array = np.asarray(
        values
    )

    expected_shape = (
        expected_size,
        expected_size,
    )

    if array.shape != expected_shape:
        raise FleetSimulationError(
            f"{description} must have shape "
            f"{expected_shape}, received {array.shape}."
        )

    if not np.issubdtype(
        array.dtype,
        np.number,
    ):
        raise FleetSimulationError(
            f"{description} must be numeric."
        )

    numeric_array = array.astype(
        np.float64,
        copy=False,
    )

    if not np.isfinite(
        numeric_array
    ).all():
        raise FleetSimulationError(
            f"{description} contains non-finite values."
        )

    if (
        numeric_array
        < 0
    ).any():
        raise FleetSimulationError(
            f"{description} contains negative values."
        )

    if not np.equal(
        numeric_array,
        np.floor(
            numeric_array
        ),
    ).all():
        raise FleetSimulationError(
            f"{description} must contain integers."
        )

    return numeric_array.astype(
        np.int64
    )


def allocate_integer_flow(
    total: int,
    probabilities: NDArray[np.float64],
) -> IntegerVector:
    """Allocate integer flow using deterministic largest remainder."""
    if total < 0:
        raise FleetSimulationError(
            "Flow total must be non-negative."
        )

    if probabilities.ndim != 1:
        raise FleetSimulationError(
            "Probability input must be one-dimensional."
        )

    if total == 0:
        return np.zeros(
            probabilities.shape[0],
            dtype=np.int64,
        )

    raw_allocation = (
        probabilities
        * total
    )

    allocation = np.floor(
        raw_allocation
    ).astype(
        np.int64
    )

    remaining = (
        total
        - int(
            allocation.sum()
        )
    )

    if remaining > 0:
        fractional_parts = (
            raw_allocation
            - allocation
        )

        priority = np.argsort(
            -fractional_parts,
            kind="stable",
        )

        allocation[
            priority[:remaining]
        ] += 1

    if int(
        allocation.sum()
    ) != total:
        raise FleetSimulationError(
            "Integer flow allocation failed conservation."
        )

    return allocation


class FleetSimulator:
    """Deterministic zone-level supply-demand simulator."""

    def __init__(
        self,
        *,
        zone_ids: tuple[int, ...],
        initial_vehicles: Any,
        passenger_destination_probabilities: Any,
        relocation_travel_steps: Any,
        passenger_travel_steps: Any,
        costs: SimulationCosts,
        allowed_relocation_edges: Any | None = None,
    ) -> None:
        """Initialize and validate simulator state."""
        if not zone_ids:
            raise FleetSimulationError(
                "At least one zone is required."
            )

        if len(
            zone_ids
        ) != len(
            set(
                zone_ids
            )
        ):
            raise FleetSimulationError(
                "Zone identifiers must be unique."
            )

        self._zone_ids = tuple(
            int(
                zone_id
            )
            for zone_id in zone_ids
        )

        self._zone_count = len(
            self._zone_ids
        )

        self._initial_vehicles = (
            as_non_negative_integer_vector(
                initial_vehicles,
                expected_size=self._zone_count,
                description="Initial vehicles",
            )
        )

        self._available_vehicles = (
            self._initial_vehicles.copy()
        )

        self._initial_fleet_total = int(
            self._initial_vehicles.sum()
        )

        if self._initial_fleet_total <= 0:
            raise FleetSimulationError(
                "Initial fleet must contain at least one vehicle."
            )

        self._passenger_probabilities = (
            self._validate_probability_matrix(
                passenger_destination_probabilities
            )
        )

        self._relocation_travel_steps = (
            as_non_negative_integer_matrix(
                relocation_travel_steps,
                expected_size=self._zone_count,
                description="Relocation travel steps",
            )
        )

        self._passenger_travel_steps = (
            as_non_negative_integer_matrix(
                passenger_travel_steps,
                expected_size=self._zone_count,
                description="Passenger travel steps",
            )
        )

        if allowed_relocation_edges is None:
            self._allowed_relocation_edges = None
        else:
            edge_mask = np.asarray(allowed_relocation_edges, dtype=bool)
            expected_shape = (self._zone_count, self._zone_count)
            if edge_mask.shape != expected_shape:
                raise FleetSimulationError(
                    "Allowed relocation edges must have shape "
                    f"{expected_shape}, received {edge_mask.shape}."
                )
            if np.diag(edge_mask).any():
                raise FleetSimulationError("Allowed relocation graph cannot contain self-loops.")
            if not np.array_equal(edge_mask, edge_mask.T):
                raise FleetSimulationError("Allowed relocation graph must be symmetric.")
            edge_mask.setflags(write=False)
            self._allowed_relocation_edges = edge_mask

        self._validate_travel_step_matrices()

        costs.validate()

        self._costs = costs

        self._current_step = 0

        self._arrival_schedule: dict[
            int,
            IntegerVector,
        ] = {}

        self._history: list[
            StepResult
        ] = []

    @property
    def zone_ids(self) -> tuple[int, ...]:
        """Return immutable simulator zone identifiers."""
        return self._zone_ids

    @property
    def current_step(self) -> int:
        """Return the next simulation step."""
        return self._current_step

    @property
    def total_fleet(self) -> int:
        """Return constant fleet size."""
        return self._initial_fleet_total

    @property
    def history(self) -> tuple[StepResult, ...]:
        """Return immutable simulation history."""
        return tuple(
            self._history
        )

    @property
    def pending_arrivals_by_step(self) -> dict[int, IntegerVector]:
        """Return scheduled arrivals keyed by steps from the next transition.

        Offset zero contains vehicles released at the beginning of the next
        call to :meth:`step`. Returned arrays are copies, so callers cannot
        mutate simulator state.
        """
        pending: dict[int, IntegerVector] = {}
        for arrival_step, arrivals in self._arrival_schedule.items():
            offset = arrival_step - self._current_step
            if offset < 0:
                raise FleetSimulationError(
                    "Arrival schedule contains a step in the past."
                )
            pending[offset] = arrivals.copy()
        return pending

    def _validate_probability_matrix(
        self,
        values: Any,
    ) -> FloatMatrix:
        """Validate passenger destination probabilities."""
        matrix = np.asarray(
            values,
            dtype=np.float64,
        )

        expected_shape = (
            self._zone_count,
            self._zone_count,
        )

        if matrix.shape != expected_shape:
            raise FleetSimulationError(
                "Passenger destination probabilities "
                f"must have shape {expected_shape}, "
                f"received {matrix.shape}."
            )

        if not np.isfinite(
            matrix
        ).all():
            raise FleetSimulationError(
                "Passenger destination probabilities "
                "contain non-finite values."
            )

        if (
            matrix
            < 0
        ).any():
            raise FleetSimulationError(
                "Passenger destination probabilities "
                "contain negative values."
            )

        row_sums = matrix.sum(
            axis=1
        )

        if not np.allclose(
            row_sums,
            1.0,
            rtol=0.0,
            atol=1e-9,
        ):
            raise FleetSimulationError(
                "Every passenger destination probability "
                "row must sum to one."
            )

        return (
            matrix
            / row_sums[:, None]
        )

    def _validate_travel_step_matrices(
        self,
    ) -> None:
        """Validate passenger and relocation travel times."""
        diagonal = np.diag(
            self._relocation_travel_steps
        )

        if (
            diagonal
            != 0
        ).any():
            raise FleetSimulationError(
                "Relocation travel-step diagonal must be zero."
            )

        off_diagonal_mask = ~np.eye(
            self._zone_count,
            dtype=bool,
        )

        if (
            self._relocation_travel_steps[
                off_diagonal_mask
            ]
            < 1
        ).any():
            raise FleetSimulationError(
                "Off-diagonal relocation travel steps "
                "must be at least one."
            )

        if (
            self._passenger_travel_steps
            < 1
        ).any():
            raise FleetSimulationError(
                "Passenger travel steps must be at least one."
            )

    def _scheduled_in_transit_by_destination(
        self,
    ) -> IntegerVector:
        """Return all scheduled arrivals grouped by destination."""
        if not self._arrival_schedule:
            return np.zeros(
                self._zone_count,
                dtype=np.int64,
            )

        return np.sum(
            np.stack(
                tuple(
                    self._arrival_schedule.values()
                )
            ),
            axis=0,
            dtype=np.int64,
        )

    def _schedule_arrival(
        self,
        *,
        arrival_step: int,
        destination: int,
        vehicle_count: int,
    ) -> None:
        """Schedule vehicles to arrive at a future step."""
        if vehicle_count == 0:
            return

        if arrival_step <= self._current_step:
            raise FleetSimulationError(
                "Arrival step must be in the future."
            )

        scheduled = self._arrival_schedule.setdefault(
            arrival_step,
            np.zeros(
                self._zone_count,
                dtype=np.int64,
            ),
        )

        scheduled[
            destination
        ] += vehicle_count

    def _schedule_matrix_flows(
        self,
        *,
        flow_matrix: IntegerMatrix,
        travel_steps: IntegerMatrix,
    ) -> None:
        """Schedule every non-zero origin-destination flow."""
        for origin in range(
            self._zone_count
        ):
            for destination in range(
                self._zone_count
            ):
                vehicle_count = int(
                    flow_matrix[
                        origin,
                        destination,
                    ]
                )

                if vehicle_count == 0:
                    continue

                duration = int(
                    travel_steps[
                        origin,
                        destination,
                    ]
                )

                if duration <= 0:
                    raise FleetSimulationError(
                        "Scheduled flow duration must be positive."
                    )

                self._schedule_arrival(
                    arrival_step=(
                        self._current_step
                        + duration
                    ),
                    destination=destination,
                    vehicle_count=vehicle_count,
                )

    def _build_passenger_flow(
        self,
        served_demand: IntegerVector,
    ) -> IntegerMatrix:
        """Convert served pickups into deterministic OD flows."""
        passenger_flow = np.zeros(
            (
                self._zone_count,
                self._zone_count,
            ),
            dtype=np.int64,
        )

        for origin in range(
            self._zone_count
        ):
            passenger_flow[
                origin
            ] = allocate_integer_flow(
                int(
                    served_demand[
                        origin
                    ]
                ),
                self._passenger_probabilities[
                    origin
                ],
            )

        if not np.array_equal(
            passenger_flow.sum(
                axis=1
            ),
            served_demand,
        ):
            raise FleetSimulationError(
                "Passenger-flow row totals do not match "
                "served demand."
            )

        return passenger_flow

    def _assert_fleet_conservation(
        self,
    ) -> None:
        """Ensure no vehicle is created or destroyed."""
        available_total = int(
            self._available_vehicles.sum()
        )

        in_transit_total = int(
            self._scheduled_in_transit_by_destination()
            .sum()
        )

        observed_total = (
            available_total
            + in_transit_total
        )

        if observed_total != self._initial_fleet_total:
            raise FleetSimulationError(
                "Fleet conservation failed: "
                f"expected {self._initial_fleet_total}, "
                f"observed {observed_total}."
            )

    def snapshot(self) -> FleetSnapshot:
        """Return the current simulator state."""
        in_transit = (
            self._scheduled_in_transit_by_destination()
        )

        return FleetSnapshot(
            step=self._current_step,
            available_vehicles=(
                self._available_vehicles.copy()
            ),
            in_transit_vehicles=(
                in_transit.copy()
            ),
            scheduled_arrival_steps=tuple(
                sorted(
                    self._arrival_schedule
                )
            ),
            total_fleet=self._initial_fleet_total,
        )

    def step(
        self,
        *,
        demand: Any,
        relocation_action: Any,
    ) -> StepResult:
        """Advance the fleet simulation by one interval."""
        demand_vector = (
            as_non_negative_integer_vector(
                demand,
                expected_size=self._zone_count,
                description="Demand",
            )
        )

        relocation_matrix = (
            as_non_negative_integer_matrix(
                relocation_action,
                expected_size=self._zone_count,
                description="Relocation action",
            )
        )

        if (
            np.diag(
                relocation_matrix
            )
            != 0
        ).any():
            raise FleetSimulationError(
                "Relocation-action diagonal must be zero."
            )

        if self._allowed_relocation_edges is not None:
            forbidden = (relocation_matrix > 0) & ~self._allowed_relocation_edges
            if forbidden.any():
                origin, destination = np.argwhere(forbidden)[0]
                raise FleetSimulationError(
                    "Relocation action violates zone adjacency constraints: "
                    f"{self._zone_ids[int(origin)]} -> {self._zone_ids[int(destination)]}."
                )

        arrivals_released = (
            self._arrival_schedule.pop(
                self._current_step,
                np.zeros(
                    self._zone_count,
                    dtype=np.int64,
                ),
            )
        )

        self._available_vehicles += (
            arrivals_released
        )

        outbound_relocations = (
            relocation_matrix.sum(
                axis=1
            )
        )

        if (
            outbound_relocations
            > self._available_vehicles
        ).any():
            violating_zones = np.flatnonzero(
                outbound_relocations
                > self._available_vehicles
            )

            zone_details = [
                {
                    "zone_id": self._zone_ids[
                        zone_index
                    ],
                    "available": int(
                        self._available_vehicles[
                            zone_index
                        ]
                    ),
                    "requested_outbound": int(
                        outbound_relocations[
                            zone_index
                        ]
                    ),
                }
                for zone_index in violating_zones
            ]

            raise FleetSimulationError(
                "Relocation action exceeds available fleet: "
                f"{zone_details}"
            )

        available_after_relocation = (
            self._available_vehicles
            - outbound_relocations
        )

        self._schedule_matrix_flows(
            flow_matrix=relocation_matrix,
            travel_steps=(
                self._relocation_travel_steps
            ),
        )

        served_demand = np.minimum(
            available_after_relocation,
            demand_vector,
        )

        unmet_demand = (
            demand_vector
            - served_demand
        )

        idle_vehicles = (
            available_after_relocation
            - served_demand
        )

        passenger_flow = (
            self._build_passenger_flow(
                served_demand
            )
        )

        self._schedule_matrix_flows(
            flow_matrix=passenger_flow,
            travel_steps=(
                self._passenger_travel_steps
            ),
        )

        self._available_vehicles = (
            idle_vehicles.copy()
        )

        relocation_vehicle_steps = int(
            np.sum(
                relocation_matrix
                * self._relocation_travel_steps
            )
        )

        demand_total = int(
            demand_vector.sum()
        )

        served_total = int(
            served_demand.sum()
        )

        unmet_total = int(
            unmet_demand.sum()
        )

        idle_total = int(
            idle_vehicles.sum()
        )

        relocated_total = int(
            relocation_matrix.sum()
        )

        service_rate = (
            served_total
            / demand_total
            if demand_total > 0
            else 1.0
        )

        unmet_cost = (
            unmet_total
            * self._costs.unmet_demand_penalty
        )

        relocation_cost = (
            relocation_vehicle_steps
            * self._costs.relocation_cost_per_vehicle_step
        )

        idle_cost = (
            idle_total
            * self._costs.idle_vehicle_cost
        )

        total_cost = (
            unmet_cost
            + relocation_cost
            + idle_cost
        )

        self._assert_fleet_conservation()

        in_transit_end = (
            self._scheduled_in_transit_by_destination()
        )

        metrics = StepMetrics(
            demand_total=demand_total,
            served_demand_total=served_total,
            unmet_demand_total=unmet_total,
            idle_vehicle_total=idle_total,
            relocated_vehicle_total=relocated_total,
            relocation_vehicle_steps=(
                relocation_vehicle_steps
            ),
            passenger_trip_total=served_total,
            arrivals_released_total=int(
                arrivals_released.sum()
            ),
            service_rate=service_rate,
            unmet_demand_cost=float(
                unmet_cost
            ),
            relocation_cost=float(
                relocation_cost
            ),
            idle_vehicle_cost=float(
                idle_cost
            ),
            total_operating_cost=float(
                total_cost
            ),
        )

        result = StepResult(
            step=self._current_step,
            arrivals_released=(
                arrivals_released.copy()
            ),
            demand=demand_vector.copy(),
            relocation_action=(
                relocation_matrix.copy()
            ),
            served_demand=(
                served_demand.copy()
            ),
            unmet_demand=(
                unmet_demand.copy()
            ),
            idle_vehicles=(
                idle_vehicles.copy()
            ),
            passenger_flow=(
                passenger_flow.copy()
            ),
            available_vehicles_end=(
                self._available_vehicles.copy()
            ),
            in_transit_vehicles_end=(
                in_transit_end.copy()
            ),
            metrics=metrics,
            fleet_conservation_passed=True,
        )

        self._history.append(
            result
        )

        self._current_step += 1

        return result

    def summary(self) -> dict[str, Any]:
        """Return cumulative metrics for all completed steps."""
        if not self._history:
            return {
                "completed_steps": 0,
                "total_fleet": self._initial_fleet_total,
                "demand_total": 0,
                "served_demand_total": 0,
                "unmet_demand_total": 0,
                "service_rate": 1.0,
                "relocated_vehicle_total": 0,
                "total_operating_cost": 0.0,
                "snapshot": self.snapshot().as_dict(),
            }

        demand_total = sum(
            result.metrics.demand_total
            for result in self._history
        )

        served_total = sum(
            result.metrics.served_demand_total
            for result in self._history
        )

        unmet_total = sum(
            result.metrics.unmet_demand_total
            for result in self._history
        )

        relocated_total = sum(
            result.metrics.relocated_vehicle_total
            for result in self._history
        )

        operating_cost = sum(
            result.metrics.total_operating_cost
            for result in self._history
        )

        return {
            "completed_steps": len(
                self._history
            ),
            "total_fleet": self._initial_fleet_total,
            "demand_total": demand_total,
            "served_demand_total": served_total,
            "unmet_demand_total": unmet_total,
            "service_rate": (
                served_total
                / demand_total
                if demand_total > 0
                else 1.0
            ),
            "relocated_vehicle_total": (
                relocated_total
            ),
            "total_operating_cost": float(
                operating_cost
            ),
            "snapshot": self.snapshot().as_dict(),
        }