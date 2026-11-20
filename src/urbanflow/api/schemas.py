from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ForecastRequest(StrictModel):
    decision_time: datetime
    horizon_steps: int = Field(default=4, ge=1, le=4)

    @field_validator("decision_time")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("decision_time must include a timezone offset")
        return value


class PlanRequest(ForecastRequest):
    available_vehicles: dict[int, int]
    in_transit_vehicles: dict[int, int] = Field(default_factory=dict)
    policy: Literal["greedy", "min_cost_flow", "mpc", "constrained_ppo"] = "min_cost_flow"
    region_service_rates: dict[str, float] = Field(default_factory=dict)
    pending_arrivals_by_step: dict[int, dict[int, int]] = Field(default_factory=dict)
    last_observed_demand: dict[int, int] = Field(default_factory=dict)

    @field_validator("available_vehicles", "in_transit_vehicles", "last_observed_demand")
    @classmethod
    def nonnegative_integer_values(cls, value: dict[int, int]) -> dict[int, int]:
        if any(not isinstance(item, int) or item < 0 for item in value.values()):
            raise ValueError("Vehicle and demand values must be non-negative integers")
        return value


class RelocationMove(StrictModel):
    from_zone: int
    to_zone: int
    vehicle_count: int = Field(ge=1)
    travel_steps: int = Field(ge=1)


class PlanResponse(StrictModel):
    decision_time: datetime
    policy: str
    moves: list[RelocationMove]
    total_fleet: int
    relocated_vehicles: int
    relocation_vehicle_steps: int
    constraints_satisfied: bool
    forecast_provenance: str
    diagnostics: dict[str, object] = Field(default_factory=dict)


class SimulationMove(StrictModel):
    from_zone: int
    to_zone: int
    vehicle_count: int = Field(ge=1)


class SimulationRequest(StrictModel):
    available_vehicles: dict[int, int]
    demand: dict[int, int]
    moves: list[SimulationMove] = Field(default_factory=list)


class SimulationResponse(StrictModel):
    demand_total: int
    served_total: int
    unmet_total: int
    service_rate: float
    operating_cost: float
    idle_vehicles: int
    relocated_vehicles: int
    fleet_conserved: bool


class ZoneForecastStatus(StrictModel):
    zone_id: int
    borough: str
    latitude: float
    longitude: float
    decision_time: datetime
    p10_demand: float
    p50_demand: float
    p90_demand: float
    forecast_provenance: str
