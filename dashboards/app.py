from __future__ import annotations

import json
import os
from typing import Any

import pandas as pd
import plotly.express as px
import requests
import streamlit as st


st.set_page_config(page_title="UrbanFlow Control", page_icon="🚕", layout="wide")
st.title("🚕 UrbanFlow Control")
st.caption("Forecast-aware fleet rebalancing · operational decision demo")

with st.sidebar:
    st.header("Connection")
    api_url = st.text_input("API base URL", os.getenv("URBANFLOW_API_URL", "http://localhost:8000")).rstrip("/")
    api_key = st.text_input("API key (optional)", os.getenv("URBANFLOW_API_KEY", ""), type="password")
    headers = {"X-API-Key": api_key} if api_key else {}
    st.divider()
    st.caption("The simulator reports estimated operational outcomes, not measured passenger wait times.")


def api_get(path: str) -> Any:
    response = requests.get(f"{api_url}{path}", headers=headers, timeout=30)
    response.raise_for_status()
    return response.json()


def api_post(path: str, payload: dict[str, Any]) -> Any:
    response = requests.post(f"{api_url}{path}", headers=headers, json=payload, timeout=180)
    response.raise_for_status()
    return response.json()

try:
    health = api_get("/health")
    zones_payload = api_get("/v1/zones")
    zones = zones_payload["zones"]
except requests.RequestException as error:
    st.error(f"Cannot connect to UrbanFlow API: {error}")
    st.info("Start the API with `docker compose up --build api` or run Uvicorn locally.")
    st.stop()

zone_frame = pd.DataFrame(zones)
col_a, col_b, col_c = st.columns(3)
col_a.metric("Eligible zones", f"{len(zones):,}")
col_b.metric("API status", health.get("status", "unknown").upper())
col_c.metric("Artifacts loaded", "Yes" if health.get("artifacts", {}).get("loaded") else "Not yet")

st.subheader("Decision request")
left, right = st.columns([1, 1])
with left:
    decision_time = st.text_input("Decision time (ISO-8601 with timezone)", "2024-07-15T18:00:00-04:00")
    horizon_steps = st.slider("Forecast horizon (15-minute steps)", min_value=1, max_value=4, value=4)
    policy = st.selectbox("Rebalancing policy", ["min_cost_flow", "mpc", "greedy", "constrained_ppo"])
with right:
    st.markdown("**Available fleet by zone**")
    available_json = st.text_area(
        "Paste a JSON object: zone_id → available vehicles",
        value='{"132": 80, "138": 30, "161": 100}',
        height=100,
    )
    transit_json = st.text_area(
        "In-transit vehicles (optional)", value="{}", height=70
    )
    pending_json = st.text_area(
        "MPC arrival schedule (offset → zone_id → vehicles; required when MPC has in-transit vehicles)",
        value="{}", height=90,
    )

try:
    available = {int(key): int(value) for key, value in json.loads(available_json).items()}
    in_transit = {int(key): int(value) for key, value in json.loads(transit_json).items()}
    pending = {
        int(offset): {int(zone): int(count) for zone, count in values.items()}
        for offset, values in json.loads(pending_json).items()
    }
except (ValueError, TypeError, json.JSONDecodeError) as error:
    st.error(f"Invalid fleet JSON: {error}")
    st.stop()

if st.button("Generate rebalancing plan", type="primary"):
    try:
        forecast = api_post("/v1/demand/forecast", {
            "decision_time": decision_time,
            "horizon_steps": horizon_steps,
        })
        plan = api_post("/v1/rebalancing/plan", {
            "decision_time": decision_time,
            "horizon_steps": horizon_steps,
            "available_vehicles": available,
            "in_transit_vehicles": in_transit,
            "pending_arrivals_by_step": pending,
            "policy": policy,
        })
        st.session_state["forecast"] = forecast
        st.session_state["plan"] = plan
        st.session_state["available"] = available
        st.success("Plan generated and hard constraints passed.")
    except requests.RequestException as error:
        st.error(f"Plan request failed: {error}")

if "forecast" in st.session_state and "plan" in st.session_state:
    forecast = st.session_state["forecast"]
    plan = st.session_state["plan"]
    st.subheader("Forecast and fleet rebalancing")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Policy", plan["policy"].replace("_", " ").title())
    k2.metric("Vehicles relocated", f"{plan['relocated_vehicles']:,}")
    k3.metric("Relocation vehicle-steps", f"{plan['relocation_vehicle_steps']:,}")
    k4.metric("Hard constraints", "Passed" if plan["constraints_satisfied"] else "Failed")

    first_horizon = forecast["horizons"][0]
    forecast_frame = pd.DataFrame(first_horizon["zones"])
    map_frame = zone_frame.merge(forecast_frame, on="zone_id", how="inner")
    map_frame["uncertainty_width"] = map_frame["p90"] - map_frame["p10"]
    figure = px.scatter_mapbox(
        map_frame,
        lat="latitude",
        lon="longitude",
        color="p50",
        size="p90",
        hover_name="zone_id",
        hover_data={"borough": True, "p10": ":.2f", "p50": ":.2f", "p90": ":.2f", "latitude": False, "longitude": False},
        color_continuous_scale="Viridis",
        zoom=9,
        height=560,
        title="P50 demand forecast by taxi zone",
    )
    figure.update_layout(mapbox_style="open-street-map", margin={"r": 0, "t": 45, "l": 0, "b": 0})
    st.plotly_chart(figure, use_container_width=True)

    st.markdown("**Recommended vehicle moves**")
    moves = plan.get("moves", [])
    if moves:
        st.dataframe(pd.DataFrame(moves), use_container_width=True, hide_index=True)
    else:
        st.info("The selected policy did not recommend a relocation for this state.")

    with st.expander("Forecast provenance and policy diagnostics"):
        st.json({
            "forecast_provenance": plan["forecast_provenance"],
            "diagnostics": plan.get("diagnostics", {}),
        })

    st.subheader("One-step simulator preview")
    st.caption("This preview initializes the submitted available fleet as on-hand supply; it is a one-step scenario, not a continuation of a live fleet session.")
    if st.button("Simulate this plan against P50 demand"):
        demand_map = {
            int(item["zone_id"]): int(round(float(item["p50"])))
            for item in first_horizon["zones"]
        }
        simulation_payload = {
            "available_vehicles": st.session_state["available"],
            "demand": demand_map,
            "moves": [
                {
                    "from_zone": item["from_zone"],
                    "to_zone": item["to_zone"],
                    "vehicle_count": item["vehicle_count"],
                }
                for item in moves
            ],
        }
        try:
            simulation = api_post("/v1/scenarios/simulate", simulation_payload)
            s1, s2, s3 = st.columns(3)
            s1.metric("Service rate", f"{simulation['service_rate'] * 100:.2f}%")
            s2.metric("Unmet demand", f"{simulation['unmet_total']:,}")
            s3.metric("Operating cost", f"{simulation['operating_cost']:,.2f}")
            st.json(simulation)
        except requests.RequestException as error:
            st.error(f"Simulation failed: {error}")

st.divider()
st.caption("UrbanFlow Control · API-backed operational demo · Use only with validated, decision-time-safe forecast artifacts.")
