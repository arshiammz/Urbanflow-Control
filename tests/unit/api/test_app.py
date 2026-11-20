from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from fastapi.testclient import TestClient

from urbanflow.api.app import create_app
from urbanflow.api.schemas import PlanResponse, RelocationMove
from urbanflow.api.service import ServiceSettings


class FakeService:
    def __init__(self) -> None:
        self.settings = ServiceSettings(
            baseline_config=Path("baseline.yaml"),
            greedy_config=Path("greedy.yaml"),
            forecast_path=Path("forecasts.parquet"),
            zone_graph_path=Path("graph.json"),
            zone_lookup_path=Path("lookup.csv"),
            model_manifest_path=Path("manifest.json"),
            rl_config_path=Path("rl.yaml"),
            api_key="secret",
            fairness_cost_weight=0.75,
        )
        self.zone_ids = (1, 2)
        self.zone_position = {1: 0, 2: 1}
        self.coordinates = {
            1: {"borough": "Manhattan", "latitude": 40.7, "longitude": -73.9},
            2: {"borough": "Queens", "latitude": 40.8, "longitude": -73.8},
        }

    def readiness(self):
        return {"ready": True, "loaded": True, "missing_artifacts": [], "load_error": None}

    def _ensure_loaded(self):
        return None

    def zones(self):
        return [{"zone_id": key, **value} for key, value in self.coordinates.items()]

    def model_status(self):
        return {"registered": False}

    def forecast(self, decision_time, horizon_steps):
        import numpy as np
        return {
            "decision_time": decision_time,
            "split": "test",
            "forecast_provenance": "synthetic-test",
            "target_interval_starts": [decision_time.isoformat()] * horizon_steps,
            "p10": np.zeros((horizon_steps, 2)),
            "p50": np.ones((horizon_steps, 2)),
            "p90": np.ones((horizon_steps, 2)) * 2,
        }

    def plan(self, request):
        return PlanResponse(
            decision_time=request.decision_time,
            policy=request.policy,
            moves=[RelocationMove(from_zone=1, to_zone=2, vehicle_count=1, travel_steps=1)],
            total_fleet=2,
            relocated_vehicles=1,
            relocation_vehicle_steps=1,
            constraints_satisfied=True,
            forecast_provenance="synthetic-test",
            diagnostics={},
        )

    def simulate(self, request):
        return {
            "demand_total": 2,
            "served_total": 1,
            "unmet_total": 1,
            "service_rate": 0.5,
            "operating_cost": 10.0,
            "idle_vehicles": 0,
            "relocated_vehicles": 1,
            "fleet_conserved": True,
        }


def test_health_and_auth_contracts() -> None:
    client = TestClient(create_app(FakeService()))
    assert client.get("/health").status_code == 200
    assert client.get("/v1/zones").status_code == 401
    assert client.get("/v1/zones", headers={"X-API-Key": "secret"}).status_code == 200


def test_forecast_plan_and_simulation_routes() -> None:
    client = TestClient(create_app(FakeService()))
    headers = {"X-API-Key": "secret"}
    timestamp = datetime(2024, 7, 15, 18, 0, tzinfo=timezone.utc).isoformat()
    forecast = client.post(
        "/v1/demand/forecast",
        headers=headers,
        json={"decision_time": timestamp, "horizon_steps": 2},
    )
    assert forecast.status_code == 200
    assert len(forecast.json()["horizons"]) == 2

    plan = client.post(
        "/v1/rebalancing/plan",
        headers=headers,
        json={
            "decision_time": timestamp,
            "horizon_steps": 2,
            "available_vehicles": {"1": 1, "2": 1},
            "policy": "min_cost_flow",
        },
    )
    assert plan.status_code == 200
    assert plan.json()["constraints_satisfied"] is True

    simulation = client.post(
        "/v1/scenarios/simulate",
        headers=headers,
        json={"available_vehicles": {"1": 1, "2": 1}, "demand": {"1": 1, "2": 1}},
    )
    assert simulation.status_code == 200
    assert simulation.json()["fleet_conserved"] is True
