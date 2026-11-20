"""Run deterministic demand, forecast, fleet, travel and graph stress scenarios."""
from __future__ import annotations

import argparse
import copy
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import yaml

from urbanflow.geo.zone_graph import ZoneGraph
from urbanflow.simulation.optimization_benchmark import run_optimization_benchmark


class StressScenarioError(RuntimeError):
    """A stress scenario cannot be applied without breaking data contracts."""


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    temporary.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    temporary.replace(path)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    temporary.write_text(
        json.dumps(_json_safe(payload), indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_parquet(frame: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    frame.write_parquet(temporary, compression="zstd", statistics=True)
    temporary.replace(path)


def _edge_loss(graph_path: Path, output_path: Path, *, fraction: float, seed: int) -> int:
    graph = ZoneGraph.load(graph_path)
    adjacency = graph.adjacency.copy()
    upper = np.argwhere(np.triu(adjacency > 0, k=1))
    rng = np.random.default_rng(seed)
    rng.shuffle(upper)
    target = int(np.floor(len(upper) * fraction))
    degree = (adjacency > 0).sum(axis=1)
    removed = 0
    for origin, destination in upper:
        if removed >= target:
            break
        if degree[origin] <= 1 or degree[destination] <= 1:
            continue
        adjacency[origin, destination] = 0.0
        adjacency[destination, origin] = 0.0
        degree[origin] -= 1
        degree[destination] -= 1
        removed += 1
    if np.any((adjacency > 0).sum(axis=1) == 0):
        raise StressScenarioError("Edge-loss scenario isolated at least one zone.")
    ZoneGraph(
        zone_ids=graph.zone_ids,
        latitudes=graph.latitudes.copy(),
        longitudes=graph.longitudes.copy(),
        adjacency=adjacency,
        k_neighbors=graph.k_neighbors,
        version=f"{graph.version}_stress_edge_loss_v1",
    ).save(output_path)
    return removed


def run(config_path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    base_path = Path(raw["base_benchmark_config"])
    if not base_path.exists():
        raise StressScenarioError(
            "Base final-holdout benchmark config is missing. Run scripts/run_final_holdout.py first."
        )
    base = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    base_paths = base["paths"]
    for key in ("baseline_config", "greedy_config", "decision_time_forecasts", "zone_graph", "zone_lookup"):
        if not Path(base_paths[key]).exists():
            raise StressScenarioError(f"Base benchmark input is missing: {base_paths[key]}")
    output_dir = Path(raw["output_directory"])
    output_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {}
    for index, (name, scenario) in enumerate(raw["scenarios"].items()):
        scenario_dir = output_dir / name
        scenario_dir.mkdir(parents=True, exist_ok=True)
        baseline = yaml.safe_load(Path(base_paths["baseline_config"]).read_text(encoding="utf-8"))
        greedy = yaml.safe_load(Path(base_paths["greedy_config"]).read_text(encoding="utf-8"))
        benchmark = copy.deepcopy(base)
        baseline_paths = baseline["paths"]
        benchmark_paths = benchmark["paths"]
        historical_path = Path(baseline_paths["historical_demand"])
        od_path = Path(baseline_paths["od_profile"])
        forecast_path = Path(benchmark_paths["decision_time_forecasts"])
        graph_path = Path(benchmark_paths["zone_graph"])

        if scenario["kind"] in ("demand_and_forecast_scale",):
            history = pl.read_parquet(historical_path).with_columns(
                pl.when(pl.col("split") == "test")
                .then((pl.col("actual_demand") * float(scenario["actual_demand_multiplier"])).round(0).cast(pl.Int64))
                .otherwise(pl.col("actual_demand"))
                .alias("actual_demand")
            )
            scenario_history = scenario_dir / "historical_demand.parquet"
            _write_parquet(history, scenario_history)
            baseline_paths["historical_demand"] = str(scenario_history)

        if scenario["kind"] in ("demand_and_forecast_scale", "forecast_scale"):
            forecasts = pl.read_parquet(forecast_path).with_columns(
                *[
                    (pl.col(column) * float(scenario["forecast_multiplier"])).alias(column)
                    for column in ("p10_demand", "p50_demand", "p90_demand")
                ]
            )
            scenario_forecasts = scenario_dir / "decision_time_forecasts.parquet"
            _write_parquet(forecasts, scenario_forecasts)
            benchmark_paths["decision_time_forecasts"] = str(scenario_forecasts)

        if scenario["kind"] == "fleet_multiplier":
            baseline["fleet"]["reserve_multiplier"] = (
                float(baseline["fleet"]["reserve_multiplier"])
                * float(scenario["fleet_multiplier"])
            )

        if scenario["kind"] == "travel_time_multiplier":
            od = pl.read_parquet(od_path).with_columns(
                pl.when(pl.col("origin_index") == pl.col("destination_index"))
                .then(0)
                .otherwise(
                    (pl.col("relocation_travel_steps") * float(scenario["travel_multiplier"]))
                    .ceil().clip(1, 12).cast(pl.Int64)
                )
                .alias("relocation_travel_steps"),
                (pl.col("passenger_travel_steps") * float(scenario["travel_multiplier"]))
                .ceil().clip(1, 12).cast(pl.Int64).alias("passenger_travel_steps"),
            )
            scenario_od = scenario_dir / "od_profile.parquet"
            _write_parquet(od, scenario_od)
            baseline_paths["od_profile"] = str(scenario_od)

        if scenario["kind"] == "graph_edge_loss":
            scenario_graph = scenario_dir / "zone_graph.json"
            removed = _edge_loss(
                graph_path,
                scenario_graph,
                fraction=float(scenario["edge_loss_fraction"]),
                seed=int(raw["random_seed"]) + index,
            )
            benchmark_paths["zone_graph"] = str(scenario_graph)
            greedy["paths"]["zone_graph"] = str(scenario_graph)
        else:
            removed = 0

        baseline_path = scenario_dir / "baseline.yaml"
        _write_yaml(baseline_path, baseline)
        greedy["paths"]["baseline_config"] = str(baseline_path)
        greedy["paths"]["step_metrics"] = str(scenario_dir / "greedy-step-metrics.parquet")
        greedy["paths"]["interval_zone_metrics"] = str(scenario_dir / "greedy-zone-interval.parquet")
        greedy["paths"]["zone_summary"] = str(scenario_dir / "greedy-zone-summary.parquet")
        greedy["paths"]["report"] = str(scenario_dir / "greedy-report.json")
        greedy_path = scenario_dir / "greedy.yaml"
        _write_yaml(greedy_path, greedy)
        benchmark_paths["baseline_config"] = str(baseline_path)
        benchmark_paths["greedy_config"] = str(greedy_path)
        benchmark_paths["step_metrics"] = str(scenario_dir / "step-metrics.parquet")
        benchmark_paths["interval_zone_metrics"] = str(scenario_dir / "interval-zone-metrics.parquet")
        benchmark_paths["zone_summary"] = str(scenario_dir / "zone-summary.parquet")
        benchmark_paths["region_summary"] = str(scenario_dir / "region-summary.parquet")
        benchmark_paths["report"] = str(scenario_dir / "benchmark.json")
        benchmark_path = scenario_dir / "benchmark.yaml"
        _write_yaml(benchmark_path, benchmark)
        report = run_optimization_benchmark(benchmark_path)
        results[name] = {
            "scenario": scenario,
            "removed_graph_edges": removed,
            "policy_metrics": report["metrics"],
            "invariants": report["invariants"],
            "benchmark_report": str(benchmark_paths["report"]),
            "rl_evaluated": False,
            "rl_exclusion_reason": (
                "This stress suite is scoped to the four Phase 3 optimizer policies. "
                "PPO stress replay is not included; any graph/OD perturbation requires a "
                "new checkpoint with a matching graph-contract signature."
            ),
        }
        if not all(report["invariants"].values()):
            raise StressScenarioError(f"Scenario {name} violated simulation invariants.")

    summary = {
        "version": "urbanflow_stress_scenarios_v1",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "base_benchmark_config": str(base_path),
        "scenario_count": len(results),
        "policy_scope": ["no_rebalancing", "greedy", "min_cost_flow", "mpc"],
        "rl_policy_scope_note": (
            "This suite does not include PPO stress replay. Any graph/OD perturbation requires "
            "a retrained checkpoint and matching graph-contract verification. Final untouched-holdout "
            "comparison still includes PPO."
        ),
        "scenarios": results,
        "all_simulation_invariants_passed": True,
    }
    report_path = output_dir / "stress-scenarios-report.json"
    _write_json(report_path, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/evaluation/stress_scenarios.yaml"))
    args = parser.parse_args()
    try:
        report = run(args.config)
    except (StressScenarioError, OSError, ValueError, KeyError, pl.exceptions.PolarsError) as error:
        raise SystemExit(f"Stress scenario evaluation failed: {error}") from error
    print("Stress scenarios completed")
    print("Scenarios:", report["scenario_count"])
    output_directory = Path(yaml.safe_load(args.config.read_text(encoding="utf-8"))["output_directory"])
    print("Report:", output_directory / "stress-scenarios-report.json")


if __name__ == "__main__":
    main()
