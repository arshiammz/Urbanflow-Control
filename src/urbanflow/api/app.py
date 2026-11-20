from __future__ import annotations

import secrets
import time
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, Query
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

from urbanflow.api.schemas import ForecastRequest, PlanRequest, PlanResponse, SimulationRequest
from urbanflow.api.service import ServiceSettings, ServiceUnavailable, UrbanFlowService


REQUEST_COUNT = Counter(
    "urbanflow_http_requests_total",
    "Total HTTP requests handled by the UrbanFlow API.",
    labelnames=("method", "route", "status"),
)
REQUEST_LATENCY = Histogram(
    "urbanflow_http_request_duration_seconds",
    "HTTP request latency in seconds.",
    labelnames=("method", "route"),
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10),
)
PLAN_COUNT = Counter(
    "urbanflow_rebalancing_plans_total",
    "Number of rebalancing plans by policy and result.",
    labelnames=("policy", "result"),
)
RELOCATED_VEHICLES = Counter(
    "urbanflow_relocated_vehicles_total",
    "Number of vehicles requested to be relocated by API plans.",
    labelnames=("policy",),
)
ARTIFACT_READY = Gauge(
    "urbanflow_artifacts_ready",
    "Whether required serving artifacts are available and loaded.",
)


def create_app(service: UrbanFlowService | None = None) -> FastAPI:
    runtime = service or UrbanFlowService(ServiceSettings.from_env())
    app = FastAPI(
        title="UrbanFlow Control API",
        version="1.0.0",
        description=(
            "Forecast-aware fleet rebalancing API. Simulation metrics are estimates "
            "from the historical OD simulator, not measured real-world wait times."
        ),
    )
    app.state.urbanflow_service = runtime

    def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
        expected = runtime.settings.api_key
        if expected is not None and (
            x_api_key is None or not secrets.compare_digest(x_api_key, expected)
        ):
            raise HTTPException(status_code=401, detail="Invalid or missing API key.")

    @app.middleware("http")
    async def record_metrics(request: Any, call_next: Any) -> Any:
        started = time.perf_counter()
        status_code = 500
        try:
            response = await call_next(request)
            status_code = response.status_code
            return response
        finally:
            elapsed = time.perf_counter() - started
            route_object = request.scope.get("route")
            route = getattr(route_object, "path", "unmatched")
            REQUEST_COUNT.labels(request.method, route, str(status_code)).inc()
            REQUEST_LATENCY.labels(request.method, route).observe(elapsed)

    @app.get("/health")
    def health() -> dict[str, Any]:
        state = runtime.readiness()
        ARTIFACT_READY.set(1 if state["ready"] else 0)
        return {
            "status": "ok",
            "service": "urbanflow-control-api",
            "artifacts": state,
        }

    @app.get("/ready")
    def ready() -> dict[str, Any]:
        try:
            runtime._ensure_loaded()
        except ServiceUnavailable as error:
            ARTIFACT_READY.set(0)
            raise HTTPException(status_code=503, detail=str(error)) from error
        ARTIFACT_READY.set(1)
        return {"ready": True, "zone_count": runtime.zone_count}

    @app.get("/metrics")
    def metrics() -> Response:
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @app.get("/v1/models/current", dependencies=[Depends(require_api_key)])
    def current_model() -> dict[str, Any]:
        try:
            return runtime.model_status()
        except (OSError, ValueError, RuntimeError) as error:
            raise HTTPException(status_code=503, detail=str(error)) from error

    @app.get("/v1/zones", dependencies=[Depends(require_api_key)])
    def zones() -> dict[str, Any]:
        try:
            values = runtime.zones()
        except ServiceUnavailable as error:
            raise HTTPException(status_code=503, detail=str(error)) from error
        return {"zone_count": len(values), "zones": values}

    @app.post("/v1/demand/forecast", dependencies=[Depends(require_api_key)])
    def forecast(request: ForecastRequest) -> dict[str, Any]:
        try:
            result = runtime.forecast(request.decision_time, request.horizon_steps)
        except ServiceUnavailable as error:
            raise HTTPException(status_code=503, detail=str(error)) from error
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        return {
            "decision_time": result["decision_time"].isoformat(),
            "split": result["split"],
            "forecast_provenance": result["forecast_provenance"],
            "horizons": [
                {
                    "horizon_step": index + 1,
                    "target_interval_start": result["target_interval_starts"][index],
                    "zones": [
                        {
                            "zone_id": int(zone_id),
                            "p10": float(result["p10"][index, zone_index]),
                            "p50": float(result["p50"][index, zone_index]),
                            "p90": float(result["p90"][index, zone_index]),
                        }
                        for zone_index, zone_id in enumerate(runtime.zone_ids)
                    ],
                }
                for index in range(request.horizon_steps)
            ],
        }

    @app.get("/v1/zones/{zone_id}/status", dependencies=[Depends(require_api_key)])
    def zone_status(
        zone_id: int,
        decision_time: str = Query(..., description="ISO-8601 timestamp with timezone"),
        horizon_steps: int = Query(1, ge=1, le=4),
    ) -> dict[str, Any]:
        from datetime import datetime
        try:
            runtime._ensure_loaded()
            if zone_id not in runtime.zone_position:
                raise ValueError(f"Unknown zone_id: {zone_id}")
            timestamp = datetime.fromisoformat(decision_time.replace("Z", "+00:00"))
            result = runtime.forecast(timestamp, horizon_steps)
            index = runtime.zone_position[zone_id]
            return {
                "zone_id": zone_id,
                **runtime.coordinates[zone_id],
                "decision_time": result["decision_time"].isoformat(),
                "p10_demand": float(result["p10"][0, index]),
                "p50_demand": float(result["p50"][0, index]),
                "p90_demand": float(result["p90"][0, index]),
                "forecast_provenance": result["forecast_provenance"],
            }
        except ServiceUnavailable as error:
            raise HTTPException(status_code=503, detail=str(error)) from error
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    @app.post(
        "/v1/rebalancing/plan",
        response_model=PlanResponse,
        dependencies=[Depends(require_api_key)],
    )
    def rebalancing_plan(request: PlanRequest) -> PlanResponse:
        try:
            result = runtime.plan(request)
        except ServiceUnavailable as error:
            PLAN_COUNT.labels(request.policy, "unavailable").inc()
            raise HTTPException(status_code=503, detail=str(error)) from error
        except ValueError as error:
            PLAN_COUNT.labels(request.policy, "invalid_request").inc()
            raise HTTPException(status_code=422, detail=str(error)) from error
        except Exception as error:
            PLAN_COUNT.labels(request.policy, "failed").inc()
            raise HTTPException(status_code=500, detail="Policy execution failed safely.") from error
        PLAN_COUNT.labels(request.policy, "success").inc()
        RELOCATED_VEHICLES.labels(request.policy).inc(result.relocated_vehicles)
        return result

    @app.post("/v1/scenarios/simulate", dependencies=[Depends(require_api_key)])
    def simulate(request: SimulationRequest) -> dict[str, Any]:
        try:
            result = runtime.simulate(request)
        except ServiceUnavailable as error:
            raise HTTPException(status_code=503, detail=str(error)) from error
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        return result

    return app


app = create_app()
