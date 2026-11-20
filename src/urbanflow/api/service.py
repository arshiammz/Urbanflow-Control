from __future__ import annotations

import os
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np

from urbanflow.api.schemas import PlanRequest, PlanResponse, RelocationMove, SimulationRequest


class ServiceUnavailable(RuntimeError):
    """Required runtime artifacts are not available or are incompatible."""


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


@dataclass(frozen=True)
class ServiceSettings:
    baseline_config: Path
    greedy_config: Path
    forecast_path: Path
    zone_graph_path: Path
    zone_lookup_path: Path
    model_manifest_path: Path
    rl_config_path: Path
    api_key: str | None
    fairness_cost_weight: float

    @classmethod
    def from_env(cls) -> ServiceSettings:
        return cls(
            baseline_config=Path(os.getenv(
                "URBANFLOW_BASELINE_CONFIG",
                "artifacts/generated_configs/final_holdout_july/final-holdout-baseline.yaml",
            )),
            greedy_config=Path(os.getenv(
                "URBANFLOW_GREEDY_CONFIG",
                "artifacts/generated_configs/final_holdout_july/final-holdout-greedy.yaml",
            )),
            forecast_path=Path(os.getenv(
                "URBANFLOW_DECISION_TIME_FORECASTS",
                "data/processed/full/decision_time_quantile_horizon_15m_2024-01_2024-07.parquet",
            )),
            zone_graph_path=Path(os.getenv(
                "URBANFLOW_ZONE_GRAPH",
                "data/processed/spatial/nyc_taxi_zone_graph_july_holdout.json",
            )),
            zone_lookup_path=Path(os.getenv(
                "URBANFLOW_ZONE_LOOKUP",
                "data/reference/taxi_zone_lookup.csv",
            )),
            model_manifest_path=Path(os.getenv(
                "URBANFLOW_MODEL_MANIFEST", "artifacts/production/model_manifest.json"
            )),
            rl_config_path=Path(os.getenv(
                "URBANFLOW_RL_CONFIG",
                "artifacts/generated_configs/final_holdout_july/final-holdout-rl.yaml",
            )),
            api_key=os.getenv("URBANFLOW_API_KEY") or None,
            fairness_cost_weight=float(os.getenv("URBANFLOW_FAIRNESS_COST_WEIGHT", "0.75")),
        )


class UrbanFlowService:
    """Lazy-loaded adapter from HTTP contracts to the Phase 2–4 decision engines."""

    def __init__(self, settings: ServiceSettings | None = None) -> None:
        self.settings = settings or ServiceSettings.from_env()
        self._loaded = False
        self._load_error: str | None = None

    def readiness(self) -> dict[str, Any]:
        required = {
            "baseline_config": self.settings.baseline_config,
            "greedy_config": self.settings.greedy_config,
            "decision_time_forecasts": self.settings.forecast_path,
            "zone_graph": self.settings.zone_graph_path,
            "zone_lookup": self.settings.zone_lookup_path,
        }
        missing = [str(path) for path in required.values() if not path.exists()]
        return {
            "ready": not missing and self._load_error is None,
            "loaded": self._loaded,
            "missing_artifacts": missing,
            "load_error": self._load_error,
        }

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        try:
            import polars as pl
            from urbanflow.geo.zone_graph import ZoneGraph
            from urbanflow.simulation.forecast_guided_heuristic import read_config as read_greedy_config
            from urbanflow.simulation.optimization_benchmark import (
                _load_graph_mask,
                _load_region_index,
            )
            from urbanflow.simulation.real_episode_baseline import (
                load_od_matrices,
                load_zone_index,
                read_config as read_baseline_config,
            )

            for path in (
                self.settings.baseline_config,
                self.settings.greedy_config,
                self.settings.forecast_path,
                self.settings.zone_graph_path,
                self.settings.zone_lookup_path,
            ):
                if not path.exists() or path.stat().st_size == 0:
                    raise ServiceUnavailable(f"Required artifact is missing: {path}")
            self.baseline = read_baseline_config(self.settings.baseline_config)
            self.greedy = read_greedy_config(self.settings.greedy_config)
            _, self.zone_ids, self.fit_shares = load_zone_index(self.baseline)
            self.zone_count = len(self.zone_ids)
            (
                self.destination_probabilities,
                self.passenger_travel_steps,
                self.relocation_travel_steps,
            ) = load_od_matrices(self.baseline, zone_count=self.zone_count)
            self.graph = ZoneGraph.load(self.settings.zone_graph_path)
            self.allowed_edges = _load_graph_mask(self.settings.zone_graph_path, self.zone_ids)
            self.region_labels, self.region_index, self.region_names = _load_region_index(
                self.settings.zone_lookup_path,
                self.zone_ids,
            )
            self.zone_position = {zone_id: index for index, zone_id in enumerate(self.zone_ids)}
            self.region_position = {name: index for index, name in enumerate(self.region_names)}
            graph_position = {zone_id: index for index, zone_id in enumerate(self.graph.zone_ids)}
            self.coordinates = {
                zone_id: {
                    "latitude": float(self.graph.latitudes[graph_position[zone_id]]),
                    "longitude": float(self.graph.longitudes[graph_position[zone_id]]),
                    "borough": self.region_labels[index],
                }
                for index, zone_id in enumerate(self.zone_ids)
            }
            self._polars = pl
            self._loaded = True
            self._load_error = None
        except Exception as error:
            self._load_error = str(error)
            raise ServiceUnavailable(f"UrbanFlow artifacts could not be loaded: {error}") from error

    def zones(self) -> list[dict[str, Any]]:
        self._ensure_loaded()
        return [
            {"zone_id": zone_id, **self.coordinates[zone_id]}
            for zone_id in self.zone_ids
        ]

    def _decision_time(self, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("decision_time must be timezone-aware")
        return value.astimezone(ZoneInfo(self.baseline.timezone))

    def forecast(self, decision_time: datetime, horizon_steps: int) -> dict[str, Any]:
        self._ensure_loaded()
        timestamp = self._decision_time(decision_time)
        pl = self._polars
        frame = (
            pl.scan_parquet(self.settings.forecast_path)
            .filter(pl.col("decision_time") == pl.lit(timestamp))
            .filter(pl.col("horizon_step") <= horizon_steps)
            .select(
                "decision_time", "target_interval_start", "horizon_step", "zone_id", "split",
                "p10_demand", "p50_demand", "p90_demand", "forecast_provenance",
            )
            .collect()
        )
        expected_rows = horizon_steps * self.zone_count
        if frame.height != expected_rows:
            raise ServiceUnavailable(
                "Decision-time forecast is missing or not dense: "
                f"expected={expected_rows}, observed={frame.height}."
            )
        if frame.null_count().to_numpy().sum():
            raise ServiceUnavailable("Decision-time forecast contains null values.")
        if frame.group_by("horizon_step", "zone_id").len().filter(pl.col("len") != 1).height:
            raise ServiceUnavailable("Decision-time forecast contains duplicate keys.")
        zone_values = set(int(value) for value in frame.get_column("zone_id").unique().to_list())
        if zone_values != set(self.zone_ids):
            raise ServiceUnavailable("Decision-time forecast zone IDs differ from the model network.")
        provenance = frame.get_column("forecast_provenance").unique().to_list()
        splits = frame.get_column("split").unique().to_list()
        if len(provenance) != 1 or not provenance[0] or len(splits) != 1:
            raise ServiceUnavailable("Forecast provenance or split is inconsistent.")

        p10 = np.zeros((horizon_steps, self.zone_count), dtype=np.float64)
        p50 = np.zeros_like(p10)
        p90 = np.zeros_like(p10)
        seen: set[tuple[int, int]] = set()
        for row in frame.iter_rows(named=True):
            horizon_index = int(row["horizon_step"]) - 1
            zone_index = self.zone_position[int(row["zone_id"])]
            key = (horizon_index, zone_index)
            if key in seen:
                raise ServiceUnavailable("Duplicate decision-time forecast key.")
            seen.add(key)
            p10[horizon_index, zone_index] = float(row["p10_demand"])
            p50[horizon_index, zone_index] = float(row["p50_demand"])
            p90[horizon_index, zone_index] = float(row["p90_demand"])
        if not (np.isfinite(p10).all() and np.isfinite(p50).all() and np.isfinite(p90).all()):
            raise ServiceUnavailable("Forecast contains non-finite values.")
        if (p10 < 0).any() or (p10 > p50).any() or (p50 > p90).any():
            raise ServiceUnavailable("Forecast quantiles are negative or non-monotonic.")
        target_times = frame.select("horizon_step", "target_interval_start").unique().sort(
            "horizon_step"
        ).get_column("target_interval_start").to_list()
        return {
            "decision_time": timestamp,
            "target_interval_starts": [value.isoformat() for value in target_times],
            "split": str(splits[0]),
            "forecast_provenance": str(provenance[0]),
            "p10": p10,
            "p50": p50,
            "p90": p90,
        }

    def _zone_vector(self, values: dict[int, int], *, name: str) -> np.ndarray:
        unknown = sorted(set(int(key) for key in values).difference(self.zone_position))
        if unknown:
            raise ValueError(f"{name} contains unknown zone IDs: {unknown[:10]}")
        result = np.zeros(self.zone_count, dtype=np.int64)
        for key, value in values.items():
            if not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} values must be non-negative integers.")
            result[self.zone_position[int(key)]] = int(value)
        return result

    def _region_rates(self, values: dict[str, float]) -> np.ndarray:
        unknown = sorted(set(values).difference(self.region_position))
        if unknown:
            raise ValueError(f"Unknown regions in region_service_rates: {unknown}")
        rates = np.ones(len(self.region_names), dtype=np.float64)
        for region, value in values.items():
            numeric = float(value)
            if not np.isfinite(numeric) or not 0.0 <= numeric <= 1.0:
                raise ValueError("Region service rates must be finite values in [0, 1].")
            rates[self.region_position[region]] = numeric
        return rates

    def _pending_arrivals(
        self, raw: dict[int, dict[int, int]], in_transit: np.ndarray
    ) -> dict[int, np.ndarray]:
        pending: dict[int, np.ndarray] = {}
        for offset, zone_values in raw.items():
            if int(offset) < 0:
                raise ValueError("Pending arrival offsets must be non-negative.")
            pending[int(offset)] = self._zone_vector(zone_values, name="pending_arrivals_by_step")
        if pending:
            total = np.sum(np.stack(list(pending.values())), axis=0, dtype=np.int64)
        else:
            total = np.zeros(self.zone_count, dtype=np.int64)
        if not np.array_equal(total, in_transit):
            raise ValueError(
                "pending_arrivals_by_step must sum exactly to in_transit_vehicles for MPC."
            )
        return pending

    def plan(self, request: PlanRequest) -> PlanResponse:
        self._ensure_loaded()
        from urbanflow.simulation.fleet_simulator import FleetSnapshot

        if request.policy == "mpc" and request.horizon_steps < 2:
            raise ValueError("MPC requires at least two forecast steps.")
        forecasts = self.forecast(request.decision_time, request.horizon_steps)
        available = self._zone_vector(request.available_vehicles, name="available_vehicles")
        in_transit = self._zone_vector(request.in_transit_vehicles, name="in_transit_vehicles")
        if int(available.sum() + in_transit.sum()) <= 0:
            raise ValueError("The request must include at least one fleet vehicle.")
        rates = self._region_rates(request.region_service_rates)
        risk = (1.0 - self.greedy.risk_aversion) * forecasts["p50"] + self.greedy.risk_aversion * forecasts["p90"]
        pending = self._pending_arrivals(request.pending_arrivals_by_step, in_transit) if request.policy == "mpc" else {}
        snapshot = FleetSnapshot(
            step=0,
            available_vehicles=available,
            in_transit_vehicles=in_transit,
            scheduled_arrival_steps=tuple(sorted(pending)),
            total_fleet=int(available.sum() + in_transit.sum()),
        )
        diagnostics: dict[str, Any] = {}

        if request.policy == "greedy":
            from urbanflow.simulation.forecast_guided_heuristic import greedy_relocation_action
            decision = greedy_relocation_action(
                snapshot=snapshot,
                current_forecast=risk[0],
                future_forecasts=risk,
                relocation_travel_steps=self.relocation_travel_steps,
                config=self.greedy,
                allowed_relocation_edges=self.allowed_edges,
            )
            action = decision.relocation_action
            diagnostics = decision.as_dict()
        elif request.policy == "min_cost_flow":
            from urbanflow.optimization.min_cost_flow import build_min_cost_relocation_action
            target_window = risk[1 : min(request.horizon_steps, self.greedy.maximum_relocation_steps + 1)]
            if target_window.shape[0] == 0:
                target_window = risk[:1]
            flow = build_min_cost_relocation_action(
                available_vehicles=available,
                current_forecast=risk[0],
                target_forecast=np.max(target_window, axis=0),
                in_transit_vehicles=in_transit,
                relocation_travel_steps=self.relocation_travel_steps,
                allowed_edges=self.allowed_edges,
                region_index=self.region_index,
                region_service_rates=rates,
                demand_buffer_multiplier=self.greedy.demand_buffer_multiplier,
                minimum_zone_reserve=self.greedy.minimum_zone_reserve,
                in_transit_supply_credit=self.greedy.in_transit_supply_credit,
                maximum_relocation_fraction=self.greedy.maximum_relocation_fraction,
                maximum_relocation_steps=self.greedy.maximum_relocation_steps,
                relocation_cost_per_vehicle_step=self.baseline.relocation_cost_per_vehicle_step,
                unmet_demand_penalty=self.baseline.unmet_demand_penalty,
                fairness_cost_weight=self.settings.fairness_cost_weight,
            )
            action = flow.relocation_action
            diagnostics = flow.as_dict()
        elif request.policy == "mpc":
            from urbanflow.optimization.mpc import MPCConfig, build_mpc_action
            mpc_config = MPCConfig(
                demand_buffer_multiplier=self.greedy.demand_buffer_multiplier,
                minimum_zone_reserve=self.greedy.minimum_zone_reserve,
                in_transit_supply_credit=self.greedy.in_transit_supply_credit,
                maximum_relocation_fraction=self.greedy.maximum_relocation_fraction,
                maximum_relocation_steps=self.greedy.maximum_relocation_steps,
                relocation_cost_per_vehicle_step=self.baseline.relocation_cost_per_vehicle_step,
                unmet_demand_penalty=self.baseline.unmet_demand_penalty,
                idle_vehicle_cost=self.baseline.idle_vehicle_cost,
                fairness_cost_weight=self.settings.fairness_cost_weight,
            )
            region_demand = np.ones(len(self.region_names), dtype=np.float64)
            region_served = rates.copy()
            decision = build_mpc_action(
                snapshot=snapshot,
                pending_arrivals_by_step=pending,
                forecast_horizon=risk,
                passenger_destination_probabilities=self.destination_probabilities,
                relocation_travel_steps=self.relocation_travel_steps,
                passenger_travel_steps=self.passenger_travel_steps,
                allowed_edges=self.allowed_edges,
                region_index=self.region_index,
                region_demand_history=region_demand,
                region_served_history=region_served,
                greedy_config=self.greedy,
                config=mpc_config,
            )
            action = decision.relocation_action
            diagnostics = decision.as_dict()
        elif request.policy == "constrained_ppo":
            action, diagnostics = self._rl_action(
                request=request,
                forecasts=forecasts,
                risk_forecasts=risk,
                available=available,
                in_transit=in_transit,
                region_rates=rates,
            )
        else:
            raise ValueError(f"Unsupported policy: {request.policy}")

        action = np.asarray(action, dtype=np.int64)
        self._validate_action(action, available)
        origins, destinations = np.nonzero(action)
        moves = [
            RelocationMove(
                from_zone=int(self.zone_ids[origin]),
                to_zone=int(self.zone_ids[destination]),
                vehicle_count=int(action[origin, destination]),
                travel_steps=int(self.relocation_travel_steps[origin, destination]),
            )
            for origin, destination in zip(origins, destinations, strict=True)
        ]
        return PlanResponse(
            decision_time=forecasts["decision_time"],
            policy=request.policy,
            moves=moves,
            total_fleet=int(available.sum() + in_transit.sum()),
            relocated_vehicles=int(action.sum()),
            relocation_vehicle_steps=int(
                (action * self.relocation_travel_steps).sum()
            ),
            constraints_satisfied=True,
            forecast_provenance=forecasts["forecast_provenance"],
            diagnostics=_json_safe(diagnostics),
        )

    def _validate_action(self, action: np.ndarray, available: np.ndarray) -> None:
        if action.shape != (self.zone_count, self.zone_count):
            raise ServiceUnavailable("Policy returned an action with the wrong shape.")
        if (action < 0).any() or np.diag(action).any():
            raise ServiceUnavailable("Policy returned negative or self-relocation flows.")
        if (action.sum(axis=1) > available).any():
            raise ServiceUnavailable("Policy action exceeds available origin supply.")
        if ((action > 0) & ~self.allowed_edges).any():
            raise ServiceUnavailable("Policy action violates the zone graph.")
        if ((action > 0) & (self.relocation_travel_steps > self.greedy.maximum_relocation_steps)).any():
            raise ServiceUnavailable("Policy action violates maximum relocation travel time.")
        cap = int(np.floor(int(available.sum()) * self.greedy.maximum_relocation_fraction))
        if int(action.sum()) > cap:
            raise ServiceUnavailable("Policy action exceeds the global relocation budget.")

    def _rl_action(
        self,
        *,
        request: PlanRequest,
        forecasts: dict[str, Any],
        risk_forecasts: np.ndarray,
        available: np.ndarray,
        in_transit: np.ndarray,
        region_rates: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        try:
            import torch
            from urbanflow.rl.episodes import Episode, EpisodeBundle
            from urbanflow.rl.workflow import build_constraints, load_checkpoint, make_environment, read_config
            from urbanflow.simulation.real_episode_baseline import NetworkInputs
        except ImportError as error:
            raise ServiceUnavailable("PyTorch RL inference dependencies are not installed.") from error
        if not self.settings.rl_config_path.exists():
            raise ServiceUnavailable(f"RL config is missing: {self.settings.rl_config_path}")
        rl_config = read_config(self.settings.rl_config_path)
        if not rl_config.selected_checkpoint.exists() or not rl_config.training_report.exists():
            raise ServiceUnavailable("Frozen PPO checkpoint or training provenance report is missing.")
        training_report = json.loads(rl_config.training_report.read_text(encoding="utf-8"))
        checkpoint_digest = hashlib.sha256()
        with rl_config.selected_checkpoint.open("rb") as checkpoint_file:
            for chunk in iter(lambda: checkpoint_file.read(1024 * 1024), b""):
                checkpoint_digest.update(chunk)
        digest = checkpoint_digest.hexdigest()
        if (
            training_report.get("selection_uses_train_only") is not True
            or training_report.get("selected_checkpoint_sha256") != digest
        ):
            raise ServiceUnavailable("PPO checkpoint does not match train-only promotion provenance.")
        timestamp = forecasts["decision_time"]
        horizon = request.horizon_steps
        p10 = forecasts["p10"]
        p50 = forecasts["p50"]
        p90 = forecasts["p90"]
        episode = Episode(
            timestamps=(timestamp, timestamp + timedelta(minutes=self.baseline.interval_minutes)),
            demand=np.zeros((2, self.zone_count), dtype=np.int64),
            p10=np.stack([p10, p10]),
            p50=np.stack([p50, p50]),
            p90=np.stack([p90, p90]),
            warmup_steps=0,
            split="inference",
            forecast_provenance=forecasts["forecast_provenance"],
        )
        bundle = EpisodeBundle(
            network=NetworkInputs(
                zone_ids=self.zone_ids,
                fit_pickup_shares=self.fit_shares,
                destination_probabilities=self.destination_probabilities,
                passenger_travel_steps=self.passenger_travel_steps,
                relocation_travel_steps=self.relocation_travel_steps,
            ),
            initial_vehicles=available.copy(),
            allowed_edges=self.allowed_edges,
            region_index=self.region_index,
            region_names=self.region_names,
            baseline=self.baseline,
            episodes=(episode,),
            forecast_horizon_steps=horizon,
            source="inference",
        )
        env = make_environment(bundle, rl_config, build_constraints(rl_config))
        model, metadata = load_checkpoint(rl_config.selected_checkpoint, env)
        if int(metadata["seed"]) != int(training_report.get("selected_seed", -1)):
            raise ServiceUnavailable("PPO checkpoint seed differs from its train-only promotion report.")
        scale = max(float(available.sum() + in_transit.sum()) / self.zone_count, 1.0)
        last_demand = self._zone_vector(request.last_observed_demand, name="last_observed_demand")
        degrees = env.projector.allowed.sum(axis=1) / max(self.zone_count - 1, 1)
        zone_features = np.column_stack([
            available / scale,
            in_transit / scale,
            p50[0] / scale,
            p90[0] / scale,
            p50[1:].mean(axis=0) / scale if horizon > 1 else p50[0] / scale,
            p90[1:].max(axis=0) / scale if horizon > 1 else p90[0] / scale,
            (p90[0] - p10[0]) / scale,
            last_demand / scale,
            region_rates[self.region_index],
            degrees,
        ]).astype(np.float32)
        local = timestamp.astimezone(ZoneInfo(self.baseline.timezone))
        angle_day = 2.0 * np.pi * (local.hour * 60 + local.minute) / 1440.0
        angle_week = 2.0 * np.pi * local.weekday() / 7.0
        global_features = np.asarray([
            np.sin(angle_day), np.cos(angle_day), np.sin(angle_week), np.cos(angle_week),
            0.0, available.sum() / max(int(available.sum() + in_transit.sum()), 1),
        ], dtype=np.float32)
        observation = np.concatenate([zone_features.reshape(-1), global_features])
        with torch.no_grad():
            action, _, _ = model.act(torch.as_tensor(observation[None], dtype=torch.float32), deterministic=True)
        raw_action = action[0].cpu().numpy().astype(np.float32)
        projected = env.projector.project(
            raw_action,
            available=available,
            in_transit=in_transit,
            current_forecast=risk_forecasts[0],
            future_forecasts=(risk_forecasts[1:] if horizon > 1 else risk_forecasts[:1]),
        )
        return projected.matrix, {
            "checkpoint": str(rl_config.selected_checkpoint),
            "checkpoint_seed": int(metadata["seed"]),
            "projection_budget": int(projected.maximum_budget),
            "vehicles_requested": int(projected.vehicles_requested),
            "vehicles_moved": int(projected.vehicles_moved),
            "inference_is_deterministic": True,
        }

    def simulate(self, request: SimulationRequest) -> dict[str, Any]:
        self._ensure_loaded()
        from urbanflow.simulation.fleet_simulator import FleetSimulator, SimulationCosts

        available = self._zone_vector(request.available_vehicles, name="available_vehicles")
        demand = self._zone_vector(request.demand, name="demand")
        if int(available.sum()) <= 0:
            raise ValueError("Simulation preview requires at least one available vehicle.")
        action = np.zeros((self.zone_count, self.zone_count), dtype=np.int64)
        for move in request.moves:
            if move.from_zone not in self.zone_position or move.to_zone not in self.zone_position:
                raise ValueError("Simulation move references an unknown zone.")
            origin = self.zone_position[move.from_zone]
            destination = self.zone_position[move.to_zone]
            action[origin, destination] += int(move.vehicle_count)
        self._validate_action(action, available)
        simulator = FleetSimulator(
            zone_ids=self.zone_ids,
            initial_vehicles=available,
            passenger_destination_probabilities=self.destination_probabilities,
            relocation_travel_steps=self.relocation_travel_steps,
            passenger_travel_steps=self.passenger_travel_steps,
            allowed_relocation_edges=self.allowed_edges,
            costs=SimulationCosts(
                unmet_demand_penalty=self.baseline.unmet_demand_penalty,
                relocation_cost_per_vehicle_step=self.baseline.relocation_cost_per_vehicle_step,
                idle_vehicle_cost=self.baseline.idle_vehicle_cost,
            ),
        )
        result = simulator.step(demand=demand, relocation_action=action)
        return {
            "demand_total": int(result.metrics.demand_total),
            "served_total": int(result.metrics.served_demand_total),
            "unmet_total": int(result.metrics.unmet_demand_total),
            "service_rate": float(result.metrics.service_rate),
            "operating_cost": float(result.metrics.total_operating_cost),
            "idle_vehicles": int(result.metrics.idle_vehicle_total),
            "relocated_vehicles": int(result.metrics.relocated_vehicle_total),
            "fleet_conserved": bool(result.fleet_conservation_passed),
        }

    def model_status(self) -> dict[str, Any]:
        if not self.settings.model_manifest_path.exists():
            return {
                "registered": False,
                "manifest_path": str(self.settings.model_manifest_path),
                "note": "No production model manifest has been promoted yet.",
            }
        from urbanflow.monitoring.model_registry import verify_manifest
        manifest = json.loads(self.settings.model_manifest_path.read_text(encoding="utf-8"))
        verification = verify_manifest(self.settings.model_manifest_path)
        return {**manifest, "verification": verification}
