"""Join as-of weather and geometry-derived graph features to the demand panel."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import polars as pl
import yaml

from urbanflow.data.weather_ingestion import WeatherIngestionError, attach_asof_weather
from urbanflow.geo.zone_graph import ZoneGraph, add_graph_features


def write_parquet(frame: pl.DataFrame, path: Path, compression: str) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    frame.write_parquet(temporary, compression=compression, statistics=True)
    temporary.replace(path)
    return {"path": str(path), "row_count": frame.height, "column_count": frame.width}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/forecasting/spatiotemporal_features.yaml"),
    )
    args = parser.parse_args()
    raw = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    paths = raw["paths"]
    settings = raw["features"]
    output = raw["output"]

    input_path = Path(paths["input"])
    weather_path = Path(paths["weather_archive"])
    zone_index_path = Path(paths["zone_index"])
    graph_path = Path(paths["zone_graph"])
    output_path = Path(paths["output"])
    report_path = Path(paths["report"])
    for path in (input_path, weather_path, zone_index_path, graph_path):
        if not path.exists():
            raise SystemExit(f"Required input is missing: {path}")

    try:
        frame = pl.read_parquet(input_path)
        zone_index = pl.read_parquet(zone_index_path)
        selected_zone_ids = tuple(
            sorted(int(value) for value in zone_index.get_column("zone_id").to_list())
        )
        input_zone_ids = set(int(value) for value in frame.get_column("zone_id").unique().to_list())
        missing_zone_ids = sorted(set(selected_zone_ids).difference(input_zone_ids))
        if missing_zone_ids:
            raise ValueError(
                "Forecast feature panel is missing eligible simulation zones: "
                + ", ".join(str(value) for value in missing_zone_ids[:20])
            )
        frame = frame.filter(pl.col("zone_id").is_in(list(selected_zone_ids))).sort(
            ["interval_start", "zone_id"]
        )
        weather = pl.read_parquet(weather_path)
        graph = ZoneGraph.load(graph_path)
        graph_frame = add_graph_features(
            frame,
            graph=graph,
            source_columns=tuple(settings["graph_source_columns"]),
        )
        enriched = attach_asof_weather(
            graph_frame,
            weather,
            interval_minutes=int(settings["interval_minutes"]),
        )
    except (OSError, ValueError, WeatherIngestionError) as error:
        raise SystemExit(f"Spatiotemporal feature build failed: {error}") from error

    required_new = [
        "graph_neighbor_lag_1",
        "graph_neighbor_lag_4",
        "graph_neighbor_rolling_mean_4",
        "graph_neighbor_count",
        "zone_latitude",
        "zone_longitude",
        "weather_temperature_2m",
        "weather_relative_humidity_2m",
        "weather_precipitation_probability",
        "weather_precipitation",
        "weather_wind_speed_10m",
        "weather_code",
        "weather_forecast_lead_hours",
        "weather_forecast_age_hours",
        "weather_missing",
    ]
    missing = sorted(set(required_new).difference(enriched.columns))
    if missing:
        raise SystemExit("Spatiotemporal output is missing: " + ", ".join(missing))

    artifact = write_parquet(enriched, output_path, str(output["compression"]))
    report = {
        "version": "spatiotemporal_features_v1",
        "input_path": str(input_path),
        "weather_archive_path": str(weather_path),
        "zone_graph_path": str(graph_path),
        "zone_index_path": str(zone_index_path),
        "output": artifact,
        "zone_count": enriched.get_column("zone_id").n_unique(),
        "interval_count": enriched.get_column("interval_start").n_unique(),
        "weather_missing_rows": enriched.filter(pl.col("weather_missing") == 1).height,
        "weather_missing_rate": float(enriched.get_column("weather_missing").mean()),
        "graph_features": [
            "graph_neighbor_lag_1",
            "graph_neighbor_lag_4",
            "graph_neighbor_rolling_mean_4",
            "graph_neighbor_count",
            "zone_latitude",
            "zone_longitude",
        ],
        "weather_features": [
            "weather_temperature_2m",
            "weather_relative_humidity_2m",
            "weather_precipitation_probability",
            "weather_precipitation",
            "weather_wind_speed_10m",
            "weather_code",
            "weather_forecast_lead_hours",
            "weather_forecast_age_hours",
            "weather_missing",
        ],
        "invariants": {
            "row_count_preserved": enriched.height == frame.height,
            "zone_count_preserved": (
                enriched.get_column("zone_id").n_unique()
                == frame.get_column("zone_id").n_unique()
            ),
            "uses_geometry_only_graph": graph.version == "centroid_knn_symmetric_v1",
            "weather_is_asof_joined": True,
        },
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print("Spatiotemporal feature build completed")
    print("Rows:", f'{artifact["row_count"]:,}')
    print("Zones:", report["zone_count"])
    print("Intervals:", report["interval_count"])
    print("Weather missing:", f'{100 * report["weather_missing_rate"]:.2f}%')
    print("Output:", output_path)
    print("Report:", report_path)


if __name__ == "__main__":
    main()
