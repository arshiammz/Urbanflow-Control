"""As-of-safe weather forecast archive ingestion for NYC taxi zones.

The pipeline queries archived individual ECMWF IFS runs, records both the
forecast valid time and a conservative public-availability timestamp, and
selects only forecasts that were available at the demand decision time.
Observed/reanalysis weather is intentionally not used as a backtest feature.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import polars as pl
import yaml


class WeatherIngestionError(RuntimeError):
    """Raised when weather data cannot be ingested safely."""


DEFAULT_ENDPOINT = "https://single-runs-api.open-meteo.com/v1/forecast"
DEFAULT_VARIABLES = (
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation_probability",
    "precipitation",
    "wind_speed_10m",
    "weather_code",
)
WEATHER_OUTPUT_COLUMNS = (
    "weather_temperature_2m",
    "weather_relative_humidity_2m",
    "weather_precipitation_probability",
    "weather_precipitation",
    "weather_wind_speed_10m",
    "weather_code",
    "weather_forecast_lead_hours",
    "weather_forecast_age_hours",
    "weather_missing",
)


@dataclass(frozen=True)
class WeatherIngestionSettings:
    """Validated settings for archived model-run retrieval."""

    endpoint: str
    model: str
    start_date: date
    end_date_exclusive: date
    run_hours_utc: tuple[int, ...]
    forecast_hours: int
    availability_delay_hours: int
    interval_minutes: int
    timezone: str
    request_timeout_seconds: float
    request_delay_seconds: float
    retries: int
    allow_missing_runs: bool
    centroids_path: Path
    output_path: Path
    report_path: Path
    compression: str

    @classmethod
    def from_yaml(cls, path: Path) -> WeatherIngestionSettings:
        if not path.exists():
            raise WeatherIngestionError(f"Weather config does not exist: {path}")
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8"))
            api = raw["api"]
            chronology = raw["chronology"]
            paths = raw["paths"]
            output = raw["output"]
            settings = cls(
                endpoint=str(api.get("endpoint", DEFAULT_ENDPOINT)),
                model=str(api["model"]),
                start_date=date.fromisoformat(str(chronology["start_date"])),
                end_date_exclusive=date.fromisoformat(str(chronology["end_date_exclusive"])),
                run_hours_utc=tuple(int(value) for value in chronology["run_hours_utc"]),
                forecast_hours=int(api["forecast_hours"]),
                availability_delay_hours=int(api["availability_delay_hours"]),
                interval_minutes=int(chronology["interval_minutes"]),
                timezone=str(chronology["timezone"]),
                request_timeout_seconds=float(api["timeout_seconds"]),
                request_delay_seconds=float(api["request_delay_seconds"]),
                retries=int(api["retries"]),
                allow_missing_runs=bool(api["allow_missing_runs"]),
                centroids_path=Path(paths["zone_centroids"]),
                output_path=Path(paths["weather_archive"]),
                report_path=Path(paths["report"]),
                compression=str(output["compression"]),
            )
        except (KeyError, TypeError, ValueError, yaml.YAMLError) as error:
            raise WeatherIngestionError(f"Invalid weather config: {error}") from error
        settings.validate()
        return settings

    def validate(self) -> None:
        if not self.model or not self.endpoint.startswith("https://"):
            raise WeatherIngestionError("Weather model and HTTPS endpoint are required.")
        if self.start_date >= self.end_date_exclusive:
            raise WeatherIngestionError("Weather date range is invalid.")
        if not self.run_hours_utc or any(value not in range(24) for value in self.run_hours_utc):
            raise WeatherIngestionError("run_hours_utc must contain valid UTC hours.")
        if len(set(self.run_hours_utc)) != len(self.run_hours_utc):
            raise WeatherIngestionError("run_hours_utc must be unique.")
        if self.forecast_hours < 6 or self.availability_delay_hours < 0:
            raise WeatherIngestionError("Forecast horizon or availability delay is invalid.")
        if self.interval_minutes <= 0 or 60 % self.interval_minutes != 0:
            raise WeatherIngestionError("interval_minutes must divide one hour.")
        if self.request_timeout_seconds <= 0 or self.request_delay_seconds < 0:
            raise WeatherIngestionError("HTTP timing settings are invalid.")
        if self.retries < 0:
            raise WeatherIngestionError("retries must be non-negative.")
        if not self.centroids_path.exists():
            raise WeatherIngestionError(f"Zone centroids are missing: {self.centroids_path}")


def _parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _response_locations(payload: Any, expected_count: int) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        locations = payload
    elif isinstance(payload, dict) and "hourly" in payload:
        locations = [payload]
    else:
        raise WeatherIngestionError("Open-Meteo response has an unexpected shape.")
    if len(locations) != expected_count:
        raise WeatherIngestionError(
            "Open-Meteo location count mismatch: "
            f"expected={expected_count}, observed={len(locations)}."
        )
    for item in locations:
        if not isinstance(item, dict) or not isinstance(item.get("hourly"), dict):
            raise WeatherIngestionError("Open-Meteo response contains an invalid location block.")
    return locations


def _weather_areas(centroids: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    required = {"zone_id", "borough", "latitude", "longitude"}
    missing = sorted(required.difference(centroids.columns))
    if missing:
        raise WeatherIngestionError("Centroid file is missing: " + ", ".join(missing))
    clean = centroids.select(
        pl.col("zone_id").cast(pl.Int64),
        pl.col("borough").cast(pl.String),
        pl.col("latitude").cast(pl.Float64),
        pl.col("longitude").cast(pl.Float64),
    )
    if clean.null_count().to_numpy().sum():
        raise WeatherIngestionError("Zone centroid data contains null values.")
    if clean.get_column("zone_id").n_unique() != clean.height:
        raise WeatherIngestionError("Zone centroid data contains duplicate zone IDs.")
    clean = clean.with_columns(
        pl.when(
            pl.col("borough").cast(pl.String).is_null()
            | pl.col("borough")
            .cast(pl.String)
            .is_in(["N/A", "Unknown", ""])
        )
        .then(pl.lit("CITYWIDE"))
        .otherwise(pl.col("borough").cast(pl.String))
        .alias("weather_area")
    )
    citywide = clean.select(
        pl.lit("CITYWIDE").alias("weather_area"),
        pl.col("latitude").mean().alias("latitude"),
        pl.col("longitude").mean().alias("longitude"),
    )
    areas = (
        clean.filter(pl.col("weather_area") != "CITYWIDE")
        .group_by("weather_area")
        .agg(
            pl.col("latitude").mean().alias("latitude"),
            pl.col("longitude").mean().alias("longitude"),
        )
    )
    areas = pl.concat([areas, citywide], how="vertical").sort("weather_area")
    zone_area = clean.select("zone_id", "weather_area")
    return areas, zone_area


def _fetch_run(
    client: httpx.Client,
    *,
    settings: WeatherIngestionSettings,
    run_time: datetime,
    areas: pl.DataFrame,
) -> list[dict[str, Any]]:
    latitude = ",".join(f"{value:.6f}" for value in areas.get_column("latitude").to_list())
    longitude = ",".join(f"{value:.6f}" for value in areas.get_column("longitude").to_list())
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": ",".join(DEFAULT_VARIABLES),
        "models": settings.model,
        "run": run_time.strftime("%Y-%m-%dT%H:%M"),
        "forecast_hours": settings.forecast_hours,
        "timezone": "UTC",
        "timeformat": "iso8601",
    }
    response: httpx.Response | None = None
    for attempt in range(settings.retries + 1):
        try:
            response = client.get(settings.endpoint, params=params)
            if response.status_code == 429 or response.status_code >= 500:
                response.raise_for_status()
            response.raise_for_status()
            break
        except httpx.HTTPStatusError as error:
            if error.response.status_code == 404 and settings.allow_missing_runs:
                return []
            retryable = error.response.status_code in {429, 500, 502, 503, 504}
            if attempt >= settings.retries or not retryable:
                raise WeatherIngestionError(
                    f"Weather run {run_time.isoformat()} failed: {error}"
                ) from error
            time.sleep(min(2.0 ** attempt, 20.0))
        except httpx.HTTPError as error:
            if attempt >= settings.retries:
                raise WeatherIngestionError(
                    f"Weather run {run_time.isoformat()} failed: {error}"
                ) from error
            time.sleep(min(2.0 ** attempt, 20.0))
    if response is None:
        raise WeatherIngestionError(f"No response for weather run {run_time.isoformat()}.")

    try:
        locations = _response_locations(response.json(), areas.height)
    except (ValueError, TypeError) as error:
        raise WeatherIngestionError(f"Invalid weather JSON for run {run_time}: {error}") from error

    available_at = run_time + timedelta(hours=settings.availability_delay_hours)
    area_names = areas.get_column("weather_area").to_list()
    rows: list[dict[str, Any]] = []
    for area_name, location in zip(area_names, locations, strict=True):
        hourly = location["hourly"]
        times = hourly.get("time", [])
        if not times:
            continue
        variable_arrays = {name: hourly.get(name, []) for name in DEFAULT_VARIABLES}
        for index, time_value in enumerate(times):
            row: dict[str, Any] = {
                "weather_area": area_name,
                "valid_time": _parse_utc(str(time_value)),
                "run_time": run_time,
                "available_at": available_at,
            }
            for variable in DEFAULT_VARIABLES:
                values = variable_arrays[variable]
                row[variable] = values[index] if index < len(values) else None
            rows.append(row)
    return rows


def build_weather_archive(config_path: Path) -> dict[str, Any]:
    """Fetch individual archived runs and write a borough-level as-of archive."""
    settings = WeatherIngestionSettings.from_yaml(config_path)
    centroids = pl.read_csv(settings.centroids_path)
    areas, zone_area = _weather_areas(centroids)
    area_rows: list[dict[str, Any]] = []
    missing_runs: list[str] = []
    requested_runs = 0

    with httpx.Client(timeout=settings.request_timeout_seconds, follow_redirects=True) as client:
        current = datetime.combine(settings.start_date, datetime.min.time(), tzinfo=UTC)
        end = datetime.combine(settings.end_date_exclusive, datetime.min.time(), tzinfo=UTC)
        while current < end:
            for hour in sorted(settings.run_hours_utc):
                run_time = current.replace(hour=hour)
                if run_time >= end:
                    continue
                requested_runs += 1
                rows = _fetch_run(
                    client,
                    settings=settings,
                    run_time=run_time,
                    areas=areas,
                )
                if not rows:
                    missing_runs.append(run_time.isoformat())
                else:
                    area_rows.extend(rows)
                if settings.request_delay_seconds:
                    time.sleep(settings.request_delay_seconds)
            current += timedelta(days=1)

    if not area_rows:
        raise WeatherIngestionError("No weather forecast runs were retrieved.")

    area_weather = pl.DataFrame(area_rows).unique(
        subset=["weather_area", "valid_time", "run_time"],
        keep="last",
    )
    weather = area_weather.sort(["weather_area", "valid_time", "available_at"])
    if weather.height == 0:
        raise WeatherIngestionError("Weather archive contains no area-level forecasts.")
    if weather.group_by("weather_area", "valid_time", "run_time").len().filter(
        pl.col("len") > 1
    ).height:
        raise WeatherIngestionError("Weather archive contains duplicate area-run keys.")

    settings.output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = settings.output_path.with_name(settings.output_path.name + ".part")
    weather.write_parquet(temporary, compression=settings.compression, statistics=True)
    temporary.replace(settings.output_path)

    report = {
        "version": "open_meteo_single_runs_asof_v1",
        "provider": "Open-Meteo Single Runs API",
        "endpoint": settings.endpoint,
        "model": settings.model,
        "timezone": settings.timezone,
        "requested_run_count": requested_runs,
        "retrieved_run_count": requested_runs - len(missing_runs),
        "missing_run_count": len(missing_runs),
        "missing_runs": missing_runs,
        "availability_delay_hours": settings.availability_delay_hours,
        "run_hours_utc": list(settings.run_hours_utc),
        "weather_areas": areas.get_column("weather_area").to_list(),
        "area_count": areas.height,
        "zone_count": zone_area.height,
        "row_count": weather.height,
        "minimum_valid_time": weather.get_column("valid_time").min().isoformat(),
        "maximum_valid_time": weather.get_column("valid_time").max().isoformat(),
        "uses_observed_weather": False,
        "uses_forecast_run_vintages": True,
        "path": str(settings.output_path),
        "report_path": str(settings.report_path),
        "invariants": {
            "all_rows_have_run_time": weather.get_column("run_time").null_count() == 0,
            "all_rows_have_available_at": weather.get_column("available_at").null_count() == 0,
            "run_available_after_initialization": bool(
                (weather.get_column("available_at") >= weather.get_column("run_time")).all()
            ),
        },
    }
    settings.report_path.parent.mkdir(parents=True, exist_ok=True)
    settings.report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report


def attach_asof_weather(
    frame: pl.DataFrame,
    weather_archive: pl.DataFrame,
    *,
    interval_minutes: int = 15,
    timestamp_column: str = "interval_start",
    zone_column: str = "zone_id",
) -> pl.DataFrame:
    """Attach the latest weather run available before each forecast decision.

    The target interval's weather is matched to the latest model run whose
    conservative `available_at` is no later than the interval's decision time
    (`interval_start`), matching the simulator's Phase 1 decision contract.
    This is an as-of join, not a join to realized weather observations.
    """
    required_left = {timestamp_column, zone_column, "borough"}
    required_right = {
        "weather_area", "valid_time", "run_time", "available_at", *DEFAULT_VARIABLES
    }
    missing_left = sorted(required_left.difference(frame.columns))
    missing_right = sorted(required_right.difference(weather_archive.columns))
    if missing_left or missing_right:
        raise WeatherIngestionError(
            "Weather join columns missing: " + ", ".join([*missing_left, *missing_right])
        )
    if interval_minutes <= 0:
        raise WeatherIngestionError("interval_minutes must be positive.")

    left = frame.with_row_index("_weather_row_order").with_columns(
        pl.when(
            pl.col("borough").cast(pl.String).is_null()
            | pl.col("borough")
            .cast(pl.String)
            .is_in(["N/A", "Unknown", ""])
        )
        .then(pl.lit("CITYWIDE"))
        .otherwise(pl.col("borough").cast(pl.String))
        .alias("weather_area"),
        pl.col(timestamp_column)
        .dt.convert_time_zone("UTC")
        .cast(pl.Datetime("us", "UTC"))
        .alias("_target_utc"),
    ).with_columns(
        pl.col("_target_utc").alias("_decision_time"),
        pl.col("_target_utc").dt.truncate("1h").alias("weather_valid_time"),
    )
    right = weather_archive.select(
        "weather_area",
        pl.col("valid_time")
        .cast(pl.Datetime("us", "UTC"))
        .alias("weather_valid_time"),
        pl.col("available_at").cast(pl.Datetime("us", "UTC")),
        pl.col("run_time").cast(pl.Datetime("us", "UTC")),
        *DEFAULT_VARIABLES,
    )
    left = left.sort(["weather_area", "weather_valid_time", "_decision_time"])
    right = right.sort(["weather_area", "weather_valid_time", "available_at"])
    joined = left.join_asof(
        right,
        left_on="_decision_time",
        right_on="available_at",
        by=["weather_area", "weather_valid_time"],
        strategy="backward",
    )
    if joined.filter(
        pl.col("available_at").is_not_null()
        & (pl.col("available_at") > pl.col("_decision_time"))
    ).height:
        raise WeatherIngestionError("Weather join selected a forecast not yet available.")

    additions = [
        pl.col("temperature_2m").cast(pl.Float64).alias("weather_temperature_2m"),
        pl.col("relative_humidity_2m").cast(pl.Float64).alias("weather_relative_humidity_2m"),
        pl.col("precipitation_probability").cast(pl.Float64).alias(
            "weather_precipitation_probability"
        ),
        pl.col("precipitation").cast(pl.Float64).alias("weather_precipitation"),
        pl.col("wind_speed_10m").cast(pl.Float64).alias("weather_wind_speed_10m"),
        pl.col("weather_code").cast(pl.Float64).alias("weather_code"),
        (
            (pl.col("weather_valid_time") - pl.col("run_time")).dt.total_seconds() / 3600.0
        ).cast(pl.Float64).alias("weather_forecast_lead_hours"),
        (
            (pl.col("_decision_time") - pl.col("available_at")).dt.total_seconds() / 3600.0
        ).cast(pl.Float64).alias("weather_forecast_age_hours"),
        pl.col("available_at").is_null().cast(pl.Int8).alias("weather_missing"),
    ]
    return (
        joined.with_columns(additions)
        .drop(
            "_weather_row_order",
            "_target_utc",
            "_decision_time",
            "weather_area",
            "weather_valid_time",
            "available_at",
            "run_time",
            *DEFAULT_VARIABLES[:-1],
        )
        .sort(timestamp_column, zone_column)
    )
