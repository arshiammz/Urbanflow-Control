# HVFHV Trip Data Contract

## Purpose

This document defines the schema, quality rules, row-level flags, and downstream eligibility requirements for NYC TLC High Volume For-Hire Vehicle trip records used by **UrbanFlow Control**.

The source dataset is treated as externally supplied operational data. A record may be valid for one downstream task while being unsuitable for another task.

## Dataset Membership

A monthly HVFHV file is assigned to a month based on `pickup_datetime`.

Therefore:

- A request may occur shortly before the beginning of the month.
- A trip may end shortly after the end of the month.
- These boundary-crossing records are not automatically invalid.
- A record is rejected if its `pickup_datetime` is outside the expected month.

## Required Source Columns

The following columns must exist in every supported dataset version:

- `hvfhs_license_num`
- `request_datetime`
- `pickup_datetime`
- `dropoff_datetime`
- `PULocationID`
- `DOLocationID`
- `trip_miles`
- `trip_time`

If one or more required columns are absent, the pipeline must stop before processing.

## Optional Source Columns

The following fields may contain null values and are not required for the initial forecasting and simulation pipelines:

- `dispatching_base_num`
- `originating_base_num`
- `on_scene_datetime`
- Fare-related fields
- Shared-ride flags
- Wheelchair-accessibility flags

A null value in an optional field does not invalidate a record.

## Reference Data

All pickup and drop-off zone identifiers must be checked against the official **NYC TLC Taxi Zone Lookup**.

The current lookup contains **265 unique identifiers**.

Special identifiers include:

- `LocationID 1`: **Newark Airport / EWR**
- `LocationID 264`: **Unknown**
- `LocationID 265`: **Outside of NYC**

These identifiers are known values and must not be classified as malformed zone IDs. They may, however, be excluded from specific downstream modeling tasks.

## Core Operating Zones

The initial UrbanFlow operating network includes zones whose borough is one of:

- Bronx
- Brooklyn
- Manhattan
- Queens
- Staten Island

The following are known but non-core zones:

- EWR
- Unknown
- Outside of NYC

A record containing a non-core zone may remain in the validated dataset with an explicit flag.

## Derived Columns

The validation pipeline must derive:

- `derived_trip_duration_seconds`
- `request_to_pickup_seconds`
- `pickup_month`
- `flag_missing_required_value`
- `flag_invalid_pickup_zone`
- `flag_invalid_dropoff_zone`
- `flag_pickup_outside_expected_month`
- `flag_non_positive_trip_duration`
- `flag_non_positive_trip_time`
- `flag_request_after_pickup`
- `flag_long_trip_duration`
- `flag_pickup_non_core_zone`
- `flag_dropoff_non_core_zone`
- `row_rejected`
- `row_retained_with_flag`
- `record_status`
- `eligible_for_demand_forecasting`
- `eligible_for_od_modeling`
- `eligible_for_wait_time_analysis`

## Rejection Rules

A record is rejected when any of the following is true:

1. A required value is missing.
2. The pickup zone is absent from the official zone lookup.
3. The drop-off zone is absent from the official zone lookup.
4. The pickup timestamp is outside the expected monthly data window.
5. `pickup_datetime` is greater than or equal to `dropoff_datetime`.
6. `trip_time` is less than or equal to zero.

Rejected records must remain measurable in the validation summary but must not enter downstream modeling datasets.

## Retain-With-Flag Rules

A non-rejected record is retained with a quality flag when one or more of the following is true:

1. `request_datetime` occurs after `pickup_datetime`.
2. Trip duration exceeds the configured long-trip threshold.
3. The pickup zone is known but outside the core operating network.
4. The drop-off zone is known but outside the core operating network.

Flagged records are not automatically removed. Their downstream eligibility is determined separately.

## Downstream Eligibility

### Demand Forecasting

A record is eligible for core-zone pickup-demand forecasting when:

- It is not rejected.
- Its pickup zone belongs to the core operating network.

A non-core drop-off does not prevent the pickup from contributing to pickup-demand forecasting.

### Origin-Destination Modeling

A record is eligible for internal origin-destination modeling when:

- It is not rejected.
- Both pickup and drop-off zones belong to the core operating network.
- It is not marked as an extreme-duration trip.

### Wait-Time Analysis

A record is eligible for request-to-pickup analysis when:

- It is not rejected.
- `request_datetime <= pickup_datetime`.

Records with negative request-to-pickup duration must not be used to estimate passenger wait time.

## Audit Findings for January 2024

The January 2024 raw audit found:

- **19,663,930 records**
- **24 source columns**
- **188,251 records** where request time occurs after pickup time
- **2 records** with non-positive trip duration
- **2 records** with non-positive `trip_time`
- **0 negative trip-mile records**
- **858 requests** beginning before January
- **6,584 trips** ending after January
- **0 pickups** outside the January data window
- Complete agreement between `trip_time` and timestamp-derived duration within **60 seconds**

These findings motivated task-specific eligibility flags rather than broad row deletion.

## Validation Outputs

The validation pipeline must produce:

1. A validated Parquet dataset containing source columns, derived values, flags, status, and eligibility columns.
2. A JSON validation summary containing:
   - Total record count
   - Accepted count
   - Retained-with-flag count
   - Rejected count
   - Count for each validation flag
   - Count for each downstream eligibility group
   - Rejection rate
   - Retained-with-flag rate
3. A machine-readable copy of the effective validation configuration.

## Reproducibility

Validation thresholds and operating-zone definitions must be configuration-driven.

The validation pipeline must not silently change thresholds based on test data.