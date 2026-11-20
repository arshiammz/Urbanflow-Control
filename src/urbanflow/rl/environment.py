"""Gymnasium fleet-rebalancing environment backed by Phase 3 FleetSimulator."""
from __future__ import annotations

import math
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray

from urbanflow.rl.action_projection import GraphActionProjector, RelocationConstraints
from urbanflow.rl.episodes import Episode, EpisodeBundle
from urbanflow.simulation.fleet_simulator import FleetSimulator, SimulationCosts, StepResult
from urbanflow.simulation.optimization_benchmark import _decision_snapshot


class RLEnvironmentError(ValueError):
    """Environment settings/state violate Phase 3 constraints."""


class FleetRebalancingEnv(gym.Env[NDArray[np.float32], NDArray[np.float32]]):
    """Day-scale RL episode with identical Phase 3 shared no-relocation warmup.

    Only past demand, present vehicle state, and archived as-of forecasts
    enter the observation; realized current demand enters only step().
    """

    metadata = {"render_modes": []}
    features_per_zone = 10
    global_features = 6

    def __init__(
        self, *, bundle: EpisodeBundle, constraints: RelocationConstraints,
        risk_aversion: float = 0.5, fairness_target_gap: float = 0.25,
        fairness_multiplier: float = 0.5, service_weight: float = 1.0,
        cost_weight: float = 1.0, idle_weight: float = 0.05,
    ) -> None:
        super().__init__()
        bundle.validate()
        weights = (risk_aversion, fairness_target_gap, fairness_multiplier,
                   service_weight, cost_weight, idle_weight)
        if not all(math.isfinite(x) for x in weights):
            raise RLEnvironmentError("Reward and risk coefficients must be finite.")
        if not 0 <= risk_aversion <= 1 or not 0 <= fairness_target_gap <= 1:
            raise RLEnvironmentError("Risk and fairness target must be in [0,1].")
        if any(v < 0 for v in weights[2:]):
            raise RLEnvironmentError("Reward weights and dual multiplier must be nonnegative.")
        self.bundle = bundle
        self.constraints = constraints
        self.projector = GraphActionProjector(
            bundle.allowed_edges, bundle.network.relocation_travel_steps, constraints,
        )
        self.risk_aversion = float(risk_aversion)
        self.fairness_target_gap = float(fairness_target_gap)
        self.fairness_multiplier = float(fairness_multiplier)
        self.service_weight = float(service_weight)
        self.cost_weight = float(cost_weight)
        self.idle_weight = float(idle_weight)
        n = len(bundle.network.zone_ids)
        self.zone_count = n
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(n * self.features_per_zone + self.global_features,), dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.projector.action_size,), dtype=np.float32,
        )
        self._rng = np.random.default_rng(0)
        self._episode: Episode | None = None
        self._simulator: FleetSimulator | None = None
        self._step_index = 0
        self._last_demand = np.zeros(n, dtype=np.int64)
        self._region_demand = np.zeros(len(bundle.region_names), dtype=np.float64)
        self._region_served = np.zeros(len(bundle.region_names), dtype=np.float64)
        self.transitions: list[StepResult] = []
        self.actions: list[NDArray[np.int64]] = []

    def _make_simulator(self) -> FleetSimulator:
        n = self.bundle.network
        b = self.bundle.baseline
        return FleetSimulator(
            zone_ids=n.zone_ids, initial_vehicles=self.bundle.initial_vehicles,
            passenger_destination_probabilities=n.destination_probabilities,
            passenger_travel_steps=n.passenger_travel_steps,
            relocation_travel_steps=n.relocation_travel_steps,
            allowed_relocation_edges=self.bundle.allowed_edges,
            costs=SimulationCosts(
                unmet_demand_penalty=b.unmet_demand_penalty,
                relocation_cost_per_vehicle_step=b.relocation_cost_per_vehicle_step,
                idle_vehicle_cost=b.idle_vehicle_cost,
            ),
        )

    def _update_history(self, result: StepResult) -> None:
        counts = len(self.bundle.region_names)
        idx = self.bundle.region_index
        self._region_demand += np.bincount(idx, weights=result.demand, minlength=counts)
        self._region_served += np.bincount(idx, weights=result.served_demand, minlength=counts)
        self._last_demand = result.demand.copy()

    def _region_rates(self) -> NDArray[np.float64]:
        return np.divide(
            self._region_served, self._region_demand,
            out=np.ones_like(self._region_demand), where=self._region_demand > 0,
        ).clip(0.0, 1.0)

    def _region_gap(self) -> float:
        active = self._region_demand > 0
        rates = self._region_rates()[active]
        return float(np.ptp(rates)) if rates.size >= 2 else 0.0

    def _risk_forecasts(self) -> NDArray[np.float64]:
        assert self._episode is not None
        i = self._step_index
        return ((1.0 - self.risk_aversion) * self._episode.p50[i]
                + self.risk_aversion * self._episode.p90[i])

    def _observe(self) -> NDArray[np.float32]:
        if self._episode is None or self._simulator is None:
            raise RLEnvironmentError("Call reset() before observations.")
        t = self._step_index
        episode = self._episode
        timestamp = episode.timestamps[t]
        adjusted, _ = _decision_snapshot(self._simulator)
        p50, p90, p10 = episode.p50[t], episode.p90[t], episode.p10[t]
        scale = max(float(self._simulator.total_fleet / self.zone_count), 1.0)
        degrees = self.projector.allowed.sum(axis=1) / max(self.zone_count - 1, 1)
        features = np.column_stack([
            adjusted.available_vehicles / scale,
            adjusted.in_transit_vehicles / scale,
            p50[0] / scale,
            p90[0] / scale,
            p50[1:].mean(axis=0) / scale if len(p50) > 1 else p50[0] / scale,
            p90[1:].max(axis=0) / scale if len(p90) > 1 else p90[0] / scale,
            (p90[0] - p10[0]) / scale,
            self._last_demand / scale,
            self._region_rates()[self.bundle.region_index],
            degrees,
        ]).astype(np.float32)
        angle_day = 2 * np.pi * (timestamp.hour * 60 + timestamp.minute) / 1440
        angle_week = 2 * np.pi * timestamp.weekday() / 7
        eval_step = t - episode.warmup_steps
        eval_length = len(episode.timestamps) - episode.warmup_steps
        global_features = np.asarray([
            np.sin(angle_day), np.cos(angle_day), np.sin(angle_week),
            np.cos(angle_week), eval_step / max(eval_length, 1),
            adjusted.available_vehicles.sum() / self._simulator.total_fleet,
        ], dtype=np.float32)
        observation = np.concatenate((features.reshape(-1), global_features))
        if not np.isfinite(observation).all():
            raise RLEnvironmentError("Observation contains nonfinite values.")
        return observation

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None,
    ) -> tuple[NDArray[np.float32], dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        option_index = None if options is None else options.get("episode_index")
        if option_index is None:
            episode_index = int(self._rng.integers(len(self.bundle.episodes)))
        else:
            episode_index = int(option_index)
            if episode_index < 0 or episode_index >= len(self.bundle.episodes):
                raise RLEnvironmentError("Invalid episode_index.")
        self._episode = self.bundle.episodes[episode_index]
        self._simulator = self._make_simulator()
        self._region_demand.fill(0)
        self._region_served.fill(0)
        self._last_demand.fill(0)
        self.transitions = []
        self.actions = []
        zero = np.zeros((self.zone_count, self.zone_count), dtype=np.int64)
        for i in range(self._episode.warmup_steps):
            result = self._simulator.step(demand=self._episode.demand[i], relocation_action=zero)
            if not result.fleet_conservation_passed:
                raise RLEnvironmentError("Fleet conservation failed in shared warmup.")
            self._update_history(result)
        self._step_index = self._episode.warmup_steps
        return self._observe(), {
            "episode_index": episode_index,
            "split": self._episode.split,
            "forecast_provenance": self._episode.forecast_provenance,
            "warmup_steps": self._episode.warmup_steps,
        }

    def project_action(self, action: NDArray[np.float32]):
        """Preview the constraint-safe policy action without mutating state."""
        if self._simulator is None or self._episode is None:
            raise RLEnvironmentError("Call reset() before projecting actions.")
        snapshot, _ = _decision_snapshot(self._simulator)
        forecasts = self._risk_forecasts()
        return self.projector.project(
            np.asarray(action), available=snapshot.available_vehicles,
            in_transit=snapshot.in_transit_vehicles,
            current_forecast=forecasts[0],
            future_forecasts=forecasts[1:] if len(forecasts) > 1 else forecasts[:1],
        )

    def step(
        self, action: NDArray[np.float32],
    ) -> tuple[NDArray[np.float32], float, bool, bool, dict[str, Any]]:
        if self._simulator is None or self._episode is None:
            raise RLEnvironmentError("Call reset() before step().")
        if self._step_index >= len(self._episode.timestamps):
            raise RLEnvironmentError("Episode finished; call reset().")
        decision = self.project_action(action)
        # Only now is the realized demand for the current interval consumed.
        result = self._simulator.step(
            demand=self._episode.demand[self._step_index],
            relocation_action=decision.matrix,
        )
        if not result.fleet_conservation_passed:
            raise RLEnvironmentError("RL transition violated fleet conservation.")
        self._update_history(result)
        self.transitions.append(result)
        self.actions.append(decision.matrix.copy())
        current_gap = self._region_gap()
        violation = max(0.0, current_gap - self.fairness_target_gap)
        metrics = result.metrics
        relative_cost = (metrics.total_operating_cost /
                         max(1.0, metrics.demand_total *
                             self.bundle.baseline.unmet_demand_penalty))
        reward = (
            self.service_weight * metrics.service_rate
            - self.cost_weight * relative_cost
            - self.fairness_multiplier * violation
            - self.idle_weight * metrics.idle_vehicle_total / self._simulator.total_fleet
        )
        info = {
            "decision_time": self._episode.timestamps[self._step_index].isoformat(),
            "split": self._episode.split,
            "demand": metrics.demand_total,
            "served": metrics.served_demand_total,
            "unmet": metrics.unmet_demand_total,
            "service_rate": metrics.service_rate,
            "operating_cost": metrics.total_operating_cost,
            "relocation_vehicle_steps": metrics.relocation_vehicle_steps,
            "relocated_vehicles": metrics.relocated_vehicle_total,
            "region_service_gap": current_gap,
            "fairness_violation": violation,
            "fairness_constraint_signal": current_gap - self.fairness_target_gap,
            "action_budget": decision.maximum_budget,
            "action_requested": decision.vehicles_requested,
            "action_relocated": decision.vehicles_moved,
            "fleet_conservation_passed": result.fleet_conservation_passed,
        }
        self._step_index += 1
        terminated = self._step_index >= len(self._episode.timestamps)
        if terminated:
            final_observation = np.zeros(self.observation_space.shape, dtype=np.float32)
        else:
            final_observation = self._observe()
        return final_observation, float(reward), terminated, False, info
