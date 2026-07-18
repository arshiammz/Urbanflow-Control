# Problem Definition

## Business Problem

A ride-hailing operator must decide how many idle vehicles should be
repositioned between city zones to serve future demand while controlling
empty relocation cost.

## Initial Modeling Problem

The first development phase forecasts pickup demand for the next 60 minutes
for the 20 highest-volume NYC taxi zones.

## Decision Frequency

One decision every 30 minutes during the development phase.

## Forecast Target

Pickup request count for each taxi zone and time interval.

## Future Decision Output

An integer origin-destination vehicle-rebalancing matrix.

## Business Metrics

- Served demand rate
- Unmet demand
- Empty relocation time
- Fleet utilization
- Net operational value
- Worst-zone service rate

## Current Non-goals

- Driver-level assignment
- Dynamic pricing
- Passenger-driver matching
- Real-time cloud deployment
- Online reinforcement learning
- Exact reconstruction of the real NYC fleet