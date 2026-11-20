# API Examples

Set:

```bash
export API=http://localhost:8000
export API_KEY='your-local-secret'
```

## Health and readiness

```bash
curl "$API/health"
curl "$API/ready"
```

## Demand forecast

```bash
curl -X POST "$API/v1/demand/forecast" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $API_KEY" \
  -d '{
    "decision_time": "2024-07-15T18:00:00-04:00",
    "horizon_steps": 4
  }'
```

## Min-Cost Flow plan

```bash
curl -X POST "$API/v1/rebalancing/plan" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $API_KEY" \
  -d '{
    "decision_time": "2024-07-15T18:00:00-04:00",
    "horizon_steps": 4,
    "policy": "min_cost_flow",
    "available_vehicles": {"132": 80, "138": 30, "161": 100},
    "in_transit_vehicles": {}
  }'
```

The sample zone/fleet values are illustrative only. A real caller must send a complete, current fleet snapshot. Missing zone keys are interpreted as zero supply.

## MPC with in-transit supply

MPC requires the pending-arrival schedule to sum exactly to the in-transit vector:

```json
{
  "decision_time": "2024-07-15T18:00:00-04:00",
  "horizon_steps": 4,
  "policy": "mpc",
  "available_vehicles": {"132": 80, "138": 30},
  "in_transit_vehicles": {"161": 5},
  "pending_arrivals_by_step": {
    "2": {"161": 5}
  }
}
```

## One-step simulation preview

```bash
curl -X POST "$API/v1/scenarios/simulate" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $API_KEY" \
  -d '{
    "available_vehicles": {"132": 80, "138": 30, "161": 100},
    "demand": {"132": 50, "138": 40, "161": 30},
    "moves": [{"from_zone": 132, "to_zone": 138, "vehicle_count": 5}]
  }'
```

The simulation preview is a one-step scenario with the submitted available fleet treated as on-hand supply. It does not persist state between API calls.
