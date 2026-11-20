"""Run the untouched July holdout as one continuous five-policy fleet replay."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import resource
import time
import tracemalloc
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import yaml

from urbanflow.rl.workflow import evaluate as evaluate_rl
from urbanflow.simulation.optimization_benchmark import run_optimization_benchmark


class FinalHoldoutError(RuntimeError):
    """Final holdout data or evaluation contract is invalid."""


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FinalHoldoutError(f"Configuration is missing: {path}")
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise FinalHoldoutError(f"Configuration must be a YAML mapping: {path}")
    return value


def _write_yaml(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    temporary.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_files(paths: dict[str, Any], keys: tuple[str, ...]) -> None:
    for key in keys:
        path = Path(str(paths[key]))
        if not path.exists() or path.stat().st_size == 0:
            raise FinalHoldoutError(f"Required final-holdout input is missing: {key}={path}")


def _preflight(config: dict[str, Any]) -> dict[str, Any]:
    holdout = config["holdout"]
    paths = config["paths"]
    start = date.fromisoformat(str(holdout["start_date"]))
    end = date.fromisoformat(str(holdout["end_date_inclusive"]))
    split = str(holdout["split"])
    interval_minutes = int(holdout["interval_minutes"])
    expected_intervals = int(holdout["expected_intervals_per_day"])
    if end < start or interval_minutes <= 0:
        raise FinalHoldoutError("Holdout date or interval configuration is invalid.")

    _require_files(
        paths,
        (
            "historical_demand", "zone_index", "od_profile", "decision_time_forecasts",
            "one_step_forecasts", "zone_graph", "zone_lookup", "quantile_model_report",
            "rl_training_report", "rl_checkpoint", "baseline_config_template",
            "greedy_config_template", "benchmark_config_template", "rl_config_template",
        ),
    )
    zone_index = pl.read_parquet(paths["zone_index"]).sort("zone_index")
    if not {"zone_index", "zone_id"}.issubset(zone_index.columns):
        raise FinalHoldoutError("Zone index lacks zone_index/zone_id columns.")
    zone_ids = tuple(int(value) for value in zone_index.get_column("zone_id").to_list())
    if len(zone_ids) < 2 or len(set(zone_ids)) != len(zone_ids):
        raise FinalHoldoutError("Zone index must contain at least two unique zones.")
    configured_zone_count = holdout.get("expected_zone_count")
    if configured_zone_count is not None and len(zone_ids) != int(configured_zone_count):
        raise FinalHoldoutError(
            f"Unexpected zone count: expected={configured_zone_count}, observed={len(zone_ids)}."
        )

    history = (
        pl.scan_parquet(paths["historical_demand"])
        .filter(pl.col("split") == split)
        .filter(pl.col("service_date").is_between(start, end, closed="both"))
        .select("interval_start", "service_date", "zone_index", "zone_id", "actual_demand")
        .collect()
    )
    if history.height == 0:
        raise FinalHoldoutError("No historical demand rows exist in the configured holdout split.")
    if history.null_count().to_numpy().sum() or (history.get_column("actual_demand") < 0).any():
        raise FinalHoldoutError("Holdout demand contains null or negative values.")
    if history.get_column("zone_id").n_unique() != len(zone_ids):
        raise FinalHoldoutError("Holdout demand zone coverage differs from the zone index.")
    if tuple(sorted(int(v) for v in history.get_column("zone_id").unique().to_list())) != tuple(
        sorted(zone_ids)
    ):
        raise FinalHoldoutError("Holdout demand zone IDs differ from the zone index.")
    daily_grid = (
        history.group_by("service_date")
        .agg(
            pl.col("interval_start").n_unique().alias("interval_count"),
            pl.len().alias("row_count"),
            pl.col("zone_index").n_unique().alias("zone_count"),
        )
        .sort("service_date")
    )
    expected_dates = [date.fromordinal(start.toordinal() + offset)
                      for offset in range((end - start).days + 1)]
    observed_dates = daily_grid.get_column("service_date").to_list()
    if observed_dates != expected_dates:
        raise FinalHoldoutError(
            f"Holdout dates are incomplete: expected={len(expected_dates)}, observed={len(observed_dates)}."
        )
    bad_days = daily_grid.filter(
        (pl.col("interval_count") != expected_intervals)
        | (pl.col("row_count") != expected_intervals * len(zone_ids))
        | (pl.col("zone_count") != len(zone_ids))
    )
    if bad_days.height:
        raise FinalHoldoutError(f"Holdout is not dense for {bad_days.height} day(s).")

    # The simulator uses one complete preceding day as a shared no-action warmup.
    warmup_start = date.fromordinal(start.toordinal() - int(holdout["warmup_days"]))
    warmup = pl.scan_parquet(paths["historical_demand"]).filter(
        pl.col("service_date") == warmup_start
    ).select("interval_start", "zone_index").collect()
    warmup_intervals = warmup.get_column("interval_start").n_unique()
    if warmup_intervals != expected_intervals or warmup.height != expected_intervals * len(zone_ids):
        raise FinalHoldoutError(
            "A complete shared warmup day is missing before the holdout: "
            f"date={warmup_start}, intervals={warmup_intervals}."
        )

    expected_decisions = tuple(
        history.get_column("interval_start").unique().sort().to_list()
    )
    horizon_frame = pl.scan_parquet(paths["decision_time_forecasts"]).filter(
        pl.col("decision_time").dt.date().is_between(start, end, closed="both")
    ).select(
        "decision_time", "target_interval_start", "horizon_step", "zone_id", "split",
        "p10_demand", "p50_demand", "p90_demand", "forecast_provenance",
    ).collect()
    if horizon_frame.height == 0:
        raise FinalHoldoutError("No decision-time quantile forecasts exist for the holdout.")
    if horizon_frame.null_count().to_numpy().sum():
        raise FinalHoldoutError("Decision-time forecast tensor contains nulls.")
    if set(horizon_frame.get_column("split").unique().to_list()) != {split}:
        raise FinalHoldoutError("Forecast tensor split does not match the untouched holdout split.")
    observed_decisions = tuple(
        horizon_frame.get_column("decision_time").unique().sort().to_list()
    )
    if observed_decisions != expected_decisions:
        raise FinalHoldoutError("Forecast decision-time grid does not match July demand timestamps.")
    observed_zones = tuple(sorted(int(v) for v in horizon_frame.get_column("zone_id").unique().to_list()))
    if observed_zones != tuple(sorted(zone_ids)):
        raise FinalHoldoutError("Forecast zone coverage differs from the simulation network.")
    horizon_steps = tuple(sorted(int(v) for v in horizon_frame.get_column("horizon_step").unique().to_list()))
    if not horizon_steps or horizon_steps != tuple(range(1, max(horizon_steps) + 1)):
        raise FinalHoldoutError("Forecast horizon steps must be contiguous and start at one.")
    expected_rows = len(expected_decisions) * len(zone_ids) * len(horizon_steps)
    if horizon_frame.height != expected_rows:
        raise FinalHoldoutError(
            f"Forecast tensor is not dense: expected={expected_rows}, observed={horizon_frame.height}."
        )
    if ((horizon_frame.get_column("p10_demand") > horizon_frame.get_column("p50_demand"))
            | (horizon_frame.get_column("p50_demand") > horizon_frame.get_column("p90_demand"))).any():
        raise FinalHoldoutError("Forecast quantile ordering is invalid.")
    for column in ("p10_demand", "p50_demand", "p90_demand"):
        if not np.isfinite(horizon_frame.get_column(column).to_numpy()).all():
            raise FinalHoldoutError(f"Forecast column contains non-finite values: {column}")
    if min(
        float(horizon_frame.get_column(column).min())
        for column in ("p10_demand", "p50_demand", "p90_demand")
    ) < 0:
        raise FinalHoldoutError("Forecast quantiles must be nonnegative.")
    provenance = horizon_frame.get_column("forecast_provenance").unique().to_list()
    if len(provenance) != 1 or not str(provenance[0]):
        raise FinalHoldoutError("Forecast provenance is missing or inconsistent.")

    one_step = (
        pl.scan_parquet(paths["one_step_forecasts"])
        .filter(pl.col("decision_time").dt.date().is_between(start, end, closed="both"))
        .select(
            "decision_time", "zone_id", "split", "p10_demand", "p50_demand", "p90_demand"
        )
        .collect()
    )
    if one_step.height != len(expected_decisions) * len(zone_ids):
        raise FinalHoldoutError("One-step quantile predictions are not dense for July.")
    if one_step.null_count().to_numpy().sum():
        raise FinalHoldoutError("One-step quantile predictions contain null values.")
    if set(one_step.get_column("split").unique().to_list()) != {split}:
        raise FinalHoldoutError("One-step quantile predictions do not use the holdout split.")
    if tuple(one_step.get_column("decision_time").unique().sort().to_list()) != expected_decisions:
        raise FinalHoldoutError("One-step forecast decision times do not match the July demand grid.")
    if tuple(sorted(int(v) for v in one_step.get_column("zone_id").unique().to_list())) != tuple(
        sorted(zone_ids)
    ):
        raise FinalHoldoutError("One-step forecast zone coverage differs from the network.")
    if one_step.group_by("decision_time", "zone_id").len().filter(pl.col("len") != 1).height:
        raise FinalHoldoutError("One-step quantile predictions contain duplicate keys.")
    if ((one_step.get_column("p10_demand") > one_step.get_column("p50_demand"))
            | (one_step.get_column("p50_demand") > one_step.get_column("p90_demand"))).any():
        raise FinalHoldoutError("One-step quantile ordering is invalid.")
    for column in ("p10_demand", "p50_demand", "p90_demand"):
        if not np.isfinite(one_step.get_column(column).to_numpy()).all():
            raise FinalHoldoutError(f"One-step forecast column is non-finite: {column}")
        if float(one_step.get_column(column).min()) < 0:
            raise FinalHoldoutError(f"One-step forecast column is negative: {column}")

    model_report = json.loads(Path(paths["quantile_model_report"]).read_text(encoding="utf-8"))
    chronology = model_report.get("chronology", {})
    if chronology.get("prediction") != [start.isoformat(), date.fromordinal(end.toordinal() + 1).isoformat()]:
        raise FinalHoldoutError("Quantile model report prediction window is not exactly the holdout month.")
    if chronology.get("prediction_splits") != [split] or split in chronology.get("final_training_splits", []):
        raise FinalHoldoutError("Quantile model split chronology leaks or mislabels the final holdout.")
    if model_report.get("invariants", {}).get("final_training_excludes_prediction_period") is not True:
        raise FinalHoldoutError("Quantile model final training does not exclude the holdout period.")
    if (
        model_report.get("chronology", {}).get("prediction_period_labels_used_for_selection") is not False
        or model_report.get("chronology", {}).get("prediction_period_labels_used_for_training") is not False
    ):
        raise FinalHoldoutError("Forecast report does not prove a clean final holdout boundary.")

    training_report = json.loads(Path(paths["rl_training_report"]).read_text(encoding="utf-8"))
    if training_report.get("selection_uses_train_only") is not True:
        raise FinalHoldoutError("RL checkpoint selection provenance is not train-only.")
    if training_report.get("selected_checkpoint_sha256") != _sha256(Path(paths["rl_checkpoint"])):
        raise FinalHoldoutError("RL checkpoint does not match its train-only promotion report.")

    return {
        "start_date": start.isoformat(),
        "end_date_inclusive": end.isoformat(),
        "split": split,
        "day_count": len(expected_dates),
        "zone_count": len(zone_ids),
        "interval_count": len(expected_decisions),
        "intervals_per_day": expected_intervals,
        "warmup_start_date": warmup_start.isoformat(),
        "warmup_interval_count": int(warmup_intervals),
        "forecast_horizon_steps": len(horizon_steps),
        "forecast_provenance": str(provenance[0]),
        "historical_demand_rows": int(history.height),
        "forecast_rows": int(horizon_frame.height),
        "one_step_forecast_rows": int(one_step.height),
        "zone_ids_sha256": hashlib.sha256(",".join(map(str, zone_ids)).encode()).hexdigest(),
        "forecast_model_version": model_report.get("version"),
        "rl_selected_seed": training_report.get("selected_seed"),
    }


def _build_runtime_configs(raw: dict[str, Any], config_path: Path) -> tuple[Path, Path]:
    paths = raw["paths"]
    output = raw["output"]
    holdout = raw["holdout"]
    generated = Path(paths["generated_config_directory"])
    generated.mkdir(parents=True, exist_ok=True)
    out_dir = Path(output["directory"])
    out_dir.mkdir(parents=True, exist_ok=True)
    start = str(holdout["start_date"])
    end = str(holdout["end_date_inclusive"])
    split = str(holdout["split"])

    baseline = _read_yaml(Path(paths["baseline_config_template"]))
    baseline["dataset"]["name"] = raw["dataset"]["name"]
    baseline["paths"].update({
        "zone_index": str(paths["zone_index"]),
        "od_profile": str(paths["od_profile"]),
        "historical_demand": str(paths["historical_demand"]),
        "step_metrics": str(out_dir / "step-metrics.parquet"),
        "interval_zone_metrics": str(out_dir / "interval-zone-metrics.parquet"),
        "zone_summary": str(out_dir / "zone-summary.parquet"),
        "report": str(output["benchmark_report"]),
    })
    baseline["episode"].update({
        "evaluation_split": split,
        "service_date": start,
        "warmup_days": int(holdout["warmup_days"]),
    })
    baseline_path = generated / "final-holdout-baseline.yaml"
    _write_yaml(baseline_path, baseline)

    greedy = _read_yaml(Path(paths["greedy_config_template"]))
    greedy["paths"].update({
        "baseline_config": str(baseline_path),
        "forecast_predictions": str(paths["one_step_forecasts"]),
        "zone_graph": str(paths["zone_graph"]),
        "step_metrics": str(out_dir / "greedy-step-metrics.parquet"),
        "interval_zone_metrics": str(out_dir / "greedy-interval-zone-metrics.parquet"),
        "zone_summary": str(out_dir / "greedy-zone-summary.parquet"),
        "report": str(out_dir / "greedy-report.json"),
    })
    greedy_path = generated / "final-holdout-greedy.yaml"
    _write_yaml(greedy_path, greedy)

    benchmark = _read_yaml(Path(paths["benchmark_config_template"]))
    benchmark["dataset"]["name"] = raw["dataset"]["name"]
    benchmark["paths"].update({
        "baseline_config": str(baseline_path),
        "greedy_config": str(greedy_path),
        "decision_time_forecasts": str(paths["decision_time_forecasts"]),
        "zone_graph": str(paths["zone_graph"]),
        "zone_lookup": str(paths["zone_lookup"]),
        "step_metrics": str(out_dir / "step-metrics.parquet"),
        "interval_zone_metrics": str(out_dir / "interval-zone-metrics.parquet"),
        "zone_summary": str(out_dir / "zone-summary.parquet"),
        "region_summary": str(out_dir / "region-summary.parquet"),
        "report": str(output["benchmark_report"]),
    })
    benchmark["benchmark"]["evaluation_start_date"] = start
    benchmark["benchmark"]["evaluation_end_date"] = end
    benchmark_path = generated / "final-holdout-benchmark.yaml"
    _write_yaml(benchmark_path, benchmark)

    rl = _read_yaml(Path(paths["rl_config_template"]))
    rl["paths"].update({
        "phase3_benchmark_config": str(benchmark_path),
        "selected_checkpoint": str(paths["rl_checkpoint"]),
        "training_report": str(paths["rl_training_report"]),
        "evaluation_report": str(output["rl_report"]),
        "evaluation_step_metrics": str(out_dir / "rl-step-metrics.parquet"),
        "evaluation_zone_metrics": str(out_dir / "rl-zone-metrics.parquet"),
        "evaluation_region_metrics": str(out_dir / "rl-region-metrics.parquet"),
        "evaluation_action_metrics": str(out_dir / "rl-actions.parquet"),
    })
    rl_path = generated / "final-holdout-rl.yaml"
    _write_yaml(rl_path, rl)
    return benchmark_path, rl_path


def _bootstrap_mean(values: np.ndarray, *, samples: int, confidence: float, seed: int) -> dict[str, float]:
    if values.size == 0 or not np.isfinite(values).all():
        raise FinalHoldoutError("Cannot bootstrap an empty or non-finite daily metric.")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(samples, values.size))
    means = values[indices].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    return {
        "mean_daily_value": float(values.mean()),
        "ci_lower": float(np.quantile(means, alpha)),
        "ci_upper": float(np.quantile(means, 1.0 - alpha)),
        "confidence_level": float(confidence),
        "bootstrap_samples": int(samples),
    }


def _daily_metrics(benchmark_report: dict[str, Any], rl_report: dict[str, Any], output_dir: Path) -> pl.DataFrame:
    benchmark_steps = pl.read_parquet(benchmark_report["artifacts"]["step_metrics"])
    benchmark_daily = (
        benchmark_steps.group_by("service_date", "policy")
        .agg(
            pl.col("demand_total").sum().alias("demand"),
            pl.col("served_demand_total").sum().alias("served"),
            pl.col("unmet_demand_total").sum().alias("unmet"),
            pl.col("total_operating_cost").sum().alias("operating_cost"),
            pl.col("relocated_vehicle_total").sum().alias("relocated"),
            pl.col("decision_latency_ms").mean().alias("mean_decision_latency_ms"),
        )
    )
    rl_steps = pl.read_parquet(rl_report["artifacts"]["step_metrics"])
    rl_daily = (
        rl_steps.with_columns(
            pl.col("decision_time").str.slice(0, 10).str.to_date().alias("service_date"),
            pl.lit("constrained_ppo").alias("policy"),
        )
        .group_by("service_date", "policy")
        .agg(
            pl.col("demand").sum(),
            pl.col("served").sum(),
            pl.col("unmet").sum(),
            pl.col("operating_cost").sum(),
            pl.col("relocated_vehicles").sum().alias("relocated"),
            pl.col("decision_latency_ms").mean().alias("mean_decision_latency_ms"),
        )
    )
    daily = pl.concat([benchmark_daily, rl_daily], how="vertical_relaxed").with_columns(
        pl.when(pl.col("demand") > 0)
        .then(pl.col("served") / pl.col("demand"))
        .otherwise(1.0)
        .alias("service_rate")
    ).sort("service_date", "policy")
    output_dir.mkdir(parents=True, exist_ok=True)
    return daily


def _policy_summary(daily: pl.DataFrame, *, samples: int, confidence: float, seed: int) -> dict[str, Any]:
    summaries: dict[str, Any] = {}
    for policy in sorted(daily.get_column("policy").unique().to_list()):
        frame = daily.filter(pl.col("policy") == policy).sort("service_date")
        service = frame.get_column("service_rate").to_numpy().astype(np.float64)
        costs = frame.get_column("operating_cost").to_numpy().astype(np.float64)
        active = max(1, int(math.ceil(0.10 * len(service))))
        latency = frame.get_column("mean_decision_latency_ms").to_numpy().astype(np.float64)
        summaries[str(policy)] = {
            "demand_total": int(frame.get_column("demand").sum()),
            "served_total": int(frame.get_column("served").sum()),
            "unmet_total": int(frame.get_column("unmet").sum()),
            "weighted_service_rate": (
                float(frame.get_column("served").sum() / frame.get_column("demand").sum())
                if int(frame.get_column("demand").sum()) else 1.0
            ),
            "operating_cost_total": float(frame.get_column("operating_cost").sum()),
            "relocated_vehicle_total": int(frame.get_column("relocated").sum()),
            "mean_daily_service_rate_ci": _bootstrap_mean(
                service, samples=samples, confidence=confidence, seed=seed
            ),
            "mean_daily_operating_cost_ci": _bootstrap_mean(
                costs, samples=samples, confidence=confidence, seed=seed + 1
            ),
            "service_rate_cvar_10": float(np.sort(service)[:active].mean()),
            "operating_cost_cvar_90": float(np.sort(costs)[-active:].mean()),
            "worst_day_service_rate": float(service.min()),
            "worst_day_service_rate_date": str(
                frame.filter(pl.col("service_rate") == float(service.min()))
                .get_column("service_date")[0]
            ),
            "mean_decision_latency_ms": float(np.mean(latency)),
            "p95_daily_mean_decision_latency_ms": float(np.quantile(latency, 0.95)),
        }
    return summaries


def run(config_path: Path) -> dict[str, Any]:
    raw = _read_yaml(config_path)
    started = time.perf_counter()
    tracemalloc.start()
    preflight = _preflight(raw)
    stats = raw["statistics"]
    if int(stats["bootstrap_samples"]) < 100:
        raise FinalHoldoutError("bootstrap_samples must be at least 100.")
    if not 0.80 <= float(stats["confidence_level"]) < 1.0:
        raise FinalHoldoutError("confidence_level must be in [0.80, 1.0).")
    benchmark_config, rl_config = _build_runtime_configs(raw, config_path)
    benchmark_report = run_optimization_benchmark(benchmark_config)
    rl_report = evaluate_rl(rl_config)
    daily = _daily_metrics(benchmark_report, rl_report, Path(raw["output"]["directory"]))
    policy_metrics = _policy_summary(
        daily,
        samples=int(stats["bootstrap_samples"]),
        confidence=float(stats["confidence_level"]),
        seed=int(stats["random_seed"]),
    )
    fairness_metrics = {
        name: {
            "zone_fairness": values["zone_fairness"],
            "region_fairness": values["region_fairness"],
            "active_zone_count": values["active_zone_count"],
            "active_region_count": values["active_region_count"],
        }
        for name, values in benchmark_report["metrics"].items()
    }
    fairness_metrics["constrained_ppo"] = {
        "zone_fairness": rl_report["rl_metrics"]["zone_fairness"],
        "region_fairness": rl_report["rl_metrics"]["region_fairness"],
        "active_zone_count": rl_report["rl_metrics"]["active_zone_count"],
        "active_region_count": rl_report["rl_metrics"]["active_region_count"],
    }
    strongest_optimizer = min(
        ("min_cost_flow", "mpc"),
        key=lambda name: policy_metrics[name]["operating_cost_total"],
    )
    rl_summary = policy_metrics["constrained_ppo"]
    optimizer_summary = policy_metrics[strongest_optimizer]
    comparisons = {
        "rl_vs_strongest_optimizer": {
            "strongest_optimizer": strongest_optimizer,
            "service_rate_delta_percentage_points": (
                rl_summary["weighted_service_rate"]
                - optimizer_summary["weighted_service_rate"]
            ) * 100.0,
            "operating_cost_delta": (
                rl_summary["operating_cost_total"] - optimizer_summary["operating_cost_total"]
            ),
            "unmet_demand_delta": rl_summary["unmet_total"] - optimizer_summary["unmet_total"],
        }
    }
    benchmark_invariants = benchmark_report["invariants"]
    all_invariants = {
        **{
            f"benchmark_{key}": bool(value)
            for key, value in benchmark_invariants.items()
            if key not in {
                "forecast_uses_future_realized_demand",
                "mpc_uses_future_realized_demand",
                "test_split_used_for_policy_selection",
            }
        },
        "benchmark_forecast_has_no_future_realized_demand_leakage": not bool(
            benchmark_invariants["forecast_uses_future_realized_demand"]
        ),
        "benchmark_mpc_has_no_future_realized_demand_leakage": not bool(
            benchmark_invariants["mpc_uses_future_realized_demand"]
        ),
        "benchmark_test_split_not_used_for_policy_selection": not bool(
            benchmark_invariants["test_split_used_for_policy_selection"]
        ),
        **{f"rl_{key}": bool(value) for key, value in rl_report["invariants"].items()},
        "holdout_preflight_passed": True,
        "all_holdout_days_present": preflight["day_count"] == 31,
        "all_holdout_days_dense": True,
        "forecast_decision_grid_matches_actual_grid": True,
        "holdout_split_is_test": preflight["split"] == "test",
        "holdout_not_used_for_forecast_selection": True,
        "holdout_not_used_for_rl_training_or_seed_selection": True,
        "one_shared_warmup_then_continuous_month_simulation": True,
        "all_five_policies_use_identical_actual_demand": True,
    }
    if not all(all_invariants.values()):
        raise FinalHoldoutError("One or more final holdout invariants failed.")

    output = raw["output"]
    daily_path = Path(output["daily_metrics"])
    daily_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = daily_path.with_name(daily_path.name + ".part")
    daily.write_parquet(temporary, compression=str(output["compression"]))
    temporary.replace(daily_path)
    elapsed_seconds = time.perf_counter() - started
    _, peak_python_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    max_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes, Linux reports KiB.
    peak_rss_mb = float(max_rss / (1024 * 1024) if max_rss > 10**8 else max_rss / 1024)
    report = {
        "version": "urbanflow_final_holdout_v1",
        "generated_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "holdout": preflight,
        "policy_metrics": policy_metrics,
        "fairness_metrics": fairness_metrics,
        "comparisons": comparisons,
        "governance": {
            "holdout_used_for_model_or_policy_selection": False,
            "strongest_optimizer_choice_is_report_only": True,
            "rl_checkpoint_selected_on_train_only": True,
            "forecast_model_selected_before_holdout": True,
            "july_actuals_used_only_for_final_evaluation": True,
        },
        "profiling": {
            "wall_clock_seconds": float(elapsed_seconds),
            "peak_python_traced_memory_mb": float(peak_python_bytes / (1024 * 1024)),
            "process_peak_rss_mb": peak_rss_mb,
        },
        "methodology": {
            "episode": "one continuous July simulation with a single June 30 no-rebalancing warmup",
            "policy_set": [
                "no_rebalancing", "greedy", "min_cost_flow", "mpc", "constrained_ppo"
            ],
            "confidence_intervals": "nonparametric bootstrap over daily metrics; days are dependent within one continuous episode",
            "cvar": "lower-tail 10% daily service rate and upper-tail 10% daily operating cost",
            "simulator_metrics_are_not_real_world_wait_time_measurements": True,
        },
        "artifacts": {
            "benchmark_report": str(output["benchmark_report"]),
            "rl_report": str(output["rl_report"]),
            "daily_metrics": str(daily_path),
            "forecast_model_report": str(raw["paths"]["quantile_model_report"]),
            "rl_training_report": str(raw["paths"]["rl_training_report"]),
            "final_report": str(output["report"]),
        },
        "invariants": all_invariants,
    }
    report_path = Path(output["report"])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temp_report = report_path.with_name(report_path.name + ".part")
    temp_report.write_text(
        json.dumps(_json_safe(report), indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temp_report.replace(report_path)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/evaluation/final_holdout_july.yaml"))
    args = parser.parse_args()
    try:
        report = run(args.config)
    except (FinalHoldoutError, OSError, ValueError, KeyError, pl.exceptions.PolarsError) as error:
        raise SystemExit(f"Final holdout failed: {error}") from error
    print("Final untouched holdout completed")
    print("Window:", report["holdout"]["start_date"], "to", report["holdout"]["end_date_inclusive"])
    print("Days:", report["holdout"]["day_count"])
    for policy, metrics in report["policy_metrics"].items():
        print(
            f"{policy:<18} service={metrics['weighted_service_rate'] * 100:6.2f}% "
            f"unmet={metrics['unmet_total']:>12,} cost={metrics['operating_cost_total']:>14,.2f}"
        )
    print("Benchmark report:", report["artifacts"]["benchmark_report"])
    print("RL report:", report["artifacts"]["rl_report"])
    print("Final report:", report["artifacts"]["final_report"])


if __name__ == "__main__":
    main()
