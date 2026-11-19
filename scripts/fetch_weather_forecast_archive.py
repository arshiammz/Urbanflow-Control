"""Fetch archived, decision-time-safe Open-Meteo forecast runs."""

from __future__ import annotations

import argparse
from pathlib import Path

from urbanflow.data.weather_ingestion import WeatherIngestionError, build_weather_archive


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/weather/open_meteo_asof.yaml"),
    )
    args = parser.parse_args()
    try:
        report = build_weather_archive(args.config)
    except WeatherIngestionError as error:
        raise SystemExit(f"Weather ingestion failed: {error}") from error
    print("As-of weather archive completed")
    print("Model:", report["model"])
    print("Zones:", report["zone_count"])
    print("Rows:", f'{report["row_count"]:,}')
    print("Runs retrieved:", report["retrieved_run_count"])
    print("Runs missing:", report["missing_run_count"])
    print("Availability delay:", report["availability_delay_hours"], "hours")
    print("Archive:", report["path"])
    print("Report:", report["report_path"])


if __name__ == "__main__":
    main()
