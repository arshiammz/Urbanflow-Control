"""Deterministic graph/action constraint projection for all RL policies.

No raw neural action reaches the simulator. Availability includes vehicles
scheduled to arrive at the beginning of the next transition, exactly as the
Phase 3 benchmark's decision snapshot does.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


class ProjectionError(ValueError):
    """An invalid policy action or graph was supplied."""


@dataclass(frozen=True)
class RelocationConstraints:
    maximum_relocation_steps: int
    maximum_relocation_fraction: float
    minimum_zone_reserve: int
    demand_buffer_multiplier: float
    in_transit_supply_credit: float

    def __post_init__(self) -> None:
        if self.maximum_relocation_steps < 1 or self.minimum_zone_reserve < 0:
            raise ProjectionError("Invalid travel-step limit or minimum reserve.")
        floats = (
            self.maximum_relocation_fraction,
            self.demand_buffer_multiplier,
            self.in_transit_supply_credit,
        )
        if not all(math.isfinite(v) for v in floats):
            raise ProjectionError("Projection coefficients must be finite.")
        if not 0 <= self.maximum_relocation_fraction <= 1:
            raise ProjectionError("Relocation fraction must be in [0, 1].")
        if self.demand_buffer_multiplier <= 0:
            raise ProjectionError("Demand buffer must be positive.")
        if not 0 <= self.in_transit_supply_credit <= 1:
            raise ProjectionError("In-transit credit must be in [0, 1].")


@dataclass(frozen=True)
class ProjectedAction:
    matrix: NDArray[np.int64]
    eligible_edge_count: int
    vehicles_requested: int
    vehicles_moved: int
    maximum_budget: int


class GraphActionProjector:
    """Projects continuous NN proposals to feasible integer OD decisions."""

    def __init__(
        self,
        allowed_edges: NDArray[np.bool_],
        relocation_travel_steps: NDArray[np.int64],
        constraints: RelocationConstraints,
    ) -> None:
        edges = np.asarray(allowed_edges, dtype=bool)
        times = np.asarray(relocation_travel_steps)
        if edges.ndim != 2 or edges.shape[0] != edges.shape[1] or edges.size == 0:
            raise ProjectionError("Graph must be a nonempty square matrix.")
        if times.shape != edges.shape or not np.issubdtype(times.dtype, np.integer):
            raise ProjectionError("Travel steps must be an integer square matrix.")
        if not np.array_equal(edges, edges.T) or np.diag(edges).any():
            raise ProjectionError("Graph must be symmetric without self edges.")
        if (times < 0).any() or np.diag(times).any() or (times[~np.eye(len(times), dtype=bool)] < 1).any():
            raise ProjectionError("Travel-step matrix must match simulator contract.")
        self.n = int(edges.shape[0])
        self.travel_steps = times.astype(np.int64, copy=True)
        self.constraints = constraints
        self.allowed = edges & (self.travel_steps <= constraints.maximum_relocation_steps)
        self.origins, self.destinations = np.nonzero(self.allowed)
        self.edge_count = len(self.origins)
        self.action_size = self.edge_count + 1

    def project(
        self,
        raw_action: NDArray[np.float32],
        *,
        available: NDArray[np.int64],
        in_transit: NDArray[np.int64],
        current_forecast: NDArray[np.float64],
        future_forecasts: NDArray[np.float64],
    ) -> ProjectedAction:
        """Allocate capped flow by NN edge priority and NN budget proposal.

        The NN controls ranking, intensity, and global fraction, but never
        chooses forbidden links, impossible outflows, or negative vehicles.
        """
        a = np.asarray(raw_action, dtype=np.float64)
        supply = np.asarray(available)
        transit = np.asarray(in_transit)
        current = np.asarray(current_forecast, dtype=np.float64)
        future = np.asarray(future_forecasts, dtype=np.float64)
        if a.shape != (self.action_size,) or not np.isfinite(a).all():
            raise ProjectionError("NN action size/nonfinite values invalid.")
        if (a < -1.000001).any() or (a > 1.000001).any():
            raise ProjectionError("NN action values must lie in [-1, 1].")
        if (supply.shape != (self.n,) or transit.shape != (self.n,)
                or current.shape != (self.n,) or future.ndim != 2
                or future.shape[1] != self.n or future.shape[0] == 0):
            raise ProjectionError("Supply/forecast dimensions do not match network.")
        if (not np.issubdtype(supply.dtype, np.integer)
                or not np.issubdtype(transit.dtype, np.integer)
                or (supply < 0).any() or (transit < 0).any()
                or not np.isfinite(current).all() or not np.isfinite(future).all()
                or (current < 0).any() or (future < 0).any()):
            raise ProjectionError("Supply and forecasts must be finite/nonnegative.")
        c = self.constraints
        reserve = np.ceil(current * c.demand_buffer_multiplier).astype(np.int64)
        reserve += c.minimum_zone_reserve
        donors = np.maximum(supply - reserve, 0).astype(np.int64)
        forecast_target = np.ceil(np.max(future, axis=0) * c.demand_buffer_multiplier)
        forecast_target = forecast_target.astype(np.int64) + c.minimum_zone_reserve
        credited = np.floor(transit * c.in_transit_supply_credit).astype(np.int64)
        deficits = np.maximum(forecast_target - supply - credited, 0).astype(np.int64)

        hard_cap = int(math.floor(int(supply.sum()) * c.maximum_relocation_fraction))
        # The final NN output selects 0..100% of the allowed global budget.
        ratio = float(np.clip((a[-1] + 1.0) / 2.0, 0.0, 1.0))
        proposed_cap = int(math.floor(hard_cap * ratio))
        moves = np.zeros((self.n, self.n), dtype=np.int64)
        remaining = proposed_cap
        # Only positive scores create flow. Stable sort makes replay deterministic.
        priority = np.argsort(-a[:-1], kind="stable")
        for idx in priority:
            score = float(a[idx])
            if remaining <= 0 or score <= 0:
                break
            origin = int(self.origins[idx])
            destination = int(self.destinations[idx])
            if donors[origin] <= 0 or deficits[destination] <= 0:
                continue
            per_edge = max(1, int(math.floor(score * min(
                int(donors[origin]), int(deficits[destination])
            ))))
            flow = min(per_edge, int(donors[origin]), int(deficits[destination]), remaining)
            moves[origin, destination] += flow
            donors[origin] -= flow
            deficits[destination] -= flow
            remaining -= flow

        if (moves.sum(axis=1) > supply).any() or int(moves.sum()) > hard_cap:
            raise ProjectionError("Projection violated supply or global cap.")
        if np.diag(moves).any() or ((moves > 0) & ~self.allowed).any():
            raise ProjectionError("Projection violated zone-graph limits.")
        return ProjectedAction(
            matrix=moves,
            eligible_edge_count=self.edge_count,
            vehicles_requested=proposed_cap,
            vehicles_moved=int(moves.sum()),
            maximum_budget=hard_cap,
        )
