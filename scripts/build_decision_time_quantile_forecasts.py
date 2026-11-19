"""Project quantile anchors into a leakage-safe multi-step decision horizon."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import yaml

from urbanflow.forecasting.probabilistic_forecaster import (
    ProbabilisticForecastError,
    fit_train_only_quantile_projector,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/forecasting/decision_time_quantile_horizon.yaml"),
    )
    args = parser.parse_args()
    raw = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    paths, settings = raw["paths"], raw["forecast"]
    predictions_path = Path(paths["predictions"])
    historical_path = Path(paths["historical_demand"])
    zone_index_path = Path(paths["zone_index"])
    output_path = Path(paths["output"])
    report_path = Path(paths["report"])
    horizon_steps = int(settings["horizon_steps"])
    interval_minutes = int(settings["interval_minutes"])
    if horizon_steps < 1 or interval_minutes < 1:
        raise SystemExit("Forecast horizon and interval length must be positive.")

    for path in (predictions_path, historical_path, zone_index_path):
        if not path.exists():
            raise SystemExit(f"Required quantile-horizon input is missing: {path}")

    try:
        predictions = pl.read_parquet(predictions_path)
        historical = pl.read_parquet(historical_path)
        zone_index = pl.read_parquet(zone_index_path).sort("zone_index")
        zone_ids = tuple(int(value) for value in zone_index.get_column("zone_id").to_list())
        if predictions.get_column("zone_id").n_unique() != len(zone_ids):
            raise ProbabilisticForecastError("Prediction zone count differs from all-zone index.")
        prediction_zone_ids = tuple(
            sorted(predictions.get_column("zone_id").unique().to_list())
        )
        if prediction_zone_ids != tuple(sorted(zone_ids)):
            raise ProbabilisticForecastError("Prediction zone IDs differ from all-zone index.")
        predictions = (
            predictions.join(zone_index.select("zone_id", "zone_index"), on="zone_id", how="left")
            .sort(["interval_start", "zone_index"])
        )
        projector = fit_train_only_quantile_projector(
            historical,
            zone_ids=zone_ids,
            interval_minutes=interval_minutes,
        )
    except (OSError, ValueError, ProbabilisticForecastError) as error:
        raise SystemExit(f"Quantile horizon setup failed: {error}") from error

    expected_per_interval = len(zone_ids)
    intervals = predictions.get_column("interval_start").unique().sort().to_list()
    if predictions.height != len(intervals) * expected_per_interval:
        raise SystemExit("One-step quantile predictions are not dense across all zones.")

    chunks: list[pl.DataFrame] = []
    for group in predictions.partition_by("interval_start", maintain_order=True):
        ordered = group.sort("zone_index")
        interval_start = ordered.get_column("interval_start")[0]
        actual_zone_ids = tuple(int(value) for value in ordered.get_column("zone_id").to_list())
        if actual_zone_ids != zone_ids:
            raise SystemExit(f"Zone coverage or ordering changed at {interval_start}.")
        current = ordered.select("p10_demand", "p50_demand", "p90_demand").to_numpy().T
        decision_time = interval_start
        if horizon_steps > 1:
            projected_future = projector.project(
                decision_time=interval_start,
                current_quantiles=current,
                horizon_steps=horizon_steps - 1,
            )
            projected = np.concatenate([current[None, :, :], projected_future], axis=0)
        else:
            projected = current[None, :, :]
        flat = projected.transpose(0, 2, 1).reshape(-1, 3)
        target_times = [
            (
                interval_start.astimezone(UTC)
                + timedelta(minutes=interval_minutes * offset)
            ).astimezone(
                interval_start.tzinfo
            )
            for offset in range(horizon_steps)
        ]
        split_value = str(ordered.get_column("split")[0])
        chunk = pl.DataFrame(
            {
                "decision_time": [decision_time] * (horizon_steps * expected_per_interval),
                "target_interval_start": [value for value in target_times for _ in zone_ids],
                "horizon_step": np.repeat(np.arange(1, horizon_steps + 1), expected_per_interval),
                "zone_id": np.tile(np.asarray(zone_ids, dtype=np.int64), horizon_steps),
                "split": [split_value] * (horizon_steps * expected_per_interval),
                "forecast_provenance": [
                    "quantile_anchor_plus_train_only_weekly_profile"
                ] * (horizon_steps * expected_per_interval),
                "p10_demand": flat[:, 0],
                "p50_demand": flat[:, 1],
                "p90_demand": flat[:, 2],
            }
        )
        chunks.append(chunk)

    if not chunks:
        raise SystemExit("No decision-time quantile forecasts were generated.")
    output = pl.concat(chunks, how="vertical_relaxed").sort(
        ["decision_time", "horizon_step", "zone_id"]
    )
    if output.null_count().to_numpy().sum():
        raise SystemExit("Decision-time quantile output contains nulls.")
    duplicate_count = (
        output.group_by("decision_time", "horizon_step", "zone_id")
        .len()
        .filter(pl.col("len") > 1)
        .height
    )
    if duplicate_count:
        raise SystemExit(f"Decision-time quantile keys are not unique: {duplicate_count}.")
    crossing_count = output.filter(
        (pl.col("p10_demand") > pl.col("p50_demand"))
        | (pl.col("p50_demand") > pl.col("p90_demand"))
    ).height
    if crossing_count:
        raise SystemExit(f"Quantile ordering invariant failed: {crossing_count} rows.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.name + ".part")
    output.write_parquet(temporary, compression=str(raw["output"]["compression"]), statistics=True)
    temporary.replace(output_path)
    report: dict[str, Any] = {
        "version": "decision_time_quantile_projection_v1",
        "input_predictions": str(predictions_path),
        "historical_demand": str(historical_path),
        "horizon_steps": horizon_steps,
        "interval_minutes": interval_minutes,
        "zone_count": len(zone_ids),
        "decision_time_count": len(intervals),
        "output_rows": output.height,
        "split_counts": output.group_by("split").agg(pl.len().alias("rows")).to_dicts(),
        "uses_train_only_profile": True,
        "uses_future_realized_demand": False,
        "uses_future_offline_predictions": False,
        "future_horizon_weather_requery": False,
        "future_horizon_method": "train_only_weekly_profile_from_quantile_anchor",
        "output_path": str(output_path),
        "invariants": {
            "all_zones_present_per_decision_and_horizon": True,
            "decision_horizon_keys_unique": duplicate_count == 0,
            "quantiles_monotonic": crossing_count == 0,
            "predictions_non_negative": bool(
                (output.get_column("p10_demand") >= 0).all()
                and (output.get_column("p50_demand") >= 0).all()
                and (output.get_column("p90_demand") >= 0).all()
            ),
        },
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print("Decision-time quantile horizon completed")
    print("Zones:", len(zone_ids))
    print("Decision times:", len(intervals))
    print("Horizon steps:", horizon_steps)
    print("Output rows:", f"{output.height:,}")
    print("Output:", output_path)
    print("Report:", report_path)


if __name__ == "__main__":
    main()
