"""Run the Phase 3 four-policy fleet optimization benchmark."""

from __future__ import annotations

import argparse
from pathlib import Path

from urbanflow.simulation.optimization_benchmark import (
    OptimizationBenchmarkError,
    run_optimization_benchmark,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/simulation/optimization_benchmark.yaml"),
    )
    args = parser.parse_args()

    try:
        report = run_optimization_benchmark(args.config)
    except OptimizationBenchmarkError as error:
        raise SystemExit(f"Optimization benchmark failed: {error}") from error

    metrics = report["metrics"]
    print("Four-policy optimization benchmark completed")
    print()
    print("Evaluation date:", report["episode"]["service_date"])
    print("Zones:", report["network"]["zone_count"])
    print("Regions:", report["network"]["region_count"])
    print("Fleet:", f"{report['network']['total_fleet']:,}")
    print("Evaluation steps:", report["episode"]["evaluation_steps"])
    print()

    for policy in ("no_rebalancing", "greedy", "min_cost_flow", "mpc"):
        result = metrics[policy]
        print(policy.replace("_", " ").title())
        print("  Served:", f"{result['served_demand_total']:,}")
        print("  Unmet:", f"{result['unmet_demand_total']:,}")
        print("  Service rate:", f"{result['service_rate'] * 100:.2f}%")
        print("  Relocated:", f"{result['relocated_vehicle_total']:,}")
        print("  Operating cost:", f"{result['total_operating_cost']:,.2f}")
        worst_region_rate = result["region_fairness"]["minimum_service_rate"] or 0.0
        region_jain = result["region_fairness"]["jain_fairness_index"] or 0.0
        print("  Worst-region service:", f"{worst_region_rate * 100:.2f}%")
        print("  Region Jain fairness:", f"{region_jain:.4f}")
        print("  Mean decision latency:", f"{result['mean_decision_latency_ms']:.3f} ms")
        print()

    print("Improvement vs no rebalancing")
    for policy in ("greedy", "min_cost_flow", "mpc"):
        comparison = report["comparisons"][policy]["vs_no_rebalancing"]
        print(
            f"  {policy}: unmet -{comparison['unmet_demand_reduction']:,} "
            f"({comparison['unmet_demand_reduction_pct']:.2f}%), "
            f"service +{comparison['service_rate_improvement_percentage_points']:.2f} pp, "
            f"cost -{comparison['operating_cost_reduction']:,.2f} "
            f"({comparison['operating_cost_reduction_pct']:.2f}%)"
        )
    print()
    print("Invariants:")
    for name, passed in report["invariants"].items():
        print(f"  {name}: {passed}")
    if not all(report["invariants"].values()):
        raise SystemExit("Optimization benchmark produced a failed invariant.")
    print()
    print("Step metrics:", report["artifacts"]["step_metrics"])
    print("Zone metrics:", report["artifacts"]["interval_zone_metrics"])
    print("Zone summary:", report["artifacts"]["zone_summary"])
    print("Region summary:", report["artifacts"]["region_summary"])
    print("Report:", report["artifacts"]["report"])


if __name__ == "__main__":
    main()
