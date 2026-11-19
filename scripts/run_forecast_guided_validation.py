"""Run continuous May validation with reference or frozen champion config."""

from __future__ import annotations

import argparse
from pathlib import Path

from urbanflow.simulation.forecast_guided_validation import run_forecast_guided_validation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/simulation/forecast_guided_validation.yaml"),
    )
    args = parser.parse_args()
    report = run_forecast_guided_validation(args.config)
    a = report["aggregate_metrics"]
    c = report["comparison"]
    print("Continuous leakage-safe forecast-guided validation completed")
    print("Window:", report["validation_window"]["start_date"], "to", report["validation_window"]["end_date_inclusive"])
    print("Forecast method:", report["methodology"]["asof_method"])
    print("Fleet:", report["fleet"]["total_fleet"])
    print("No rebalancing service rate:", f'{100*a["baseline"]["service_rate"]:.2f}%')
    print("Greedy service rate:", f'{100*a["policy"]["service_rate"]:.2f}%')
    print("Unmet reduction:", f'{100*c["unmet_demand_relative_reduction"]:.2f}%')
    print("Cost reduction:", f'{100*c["operating_cost_relative_reduction"]:.2f}%')
    print("Days improving unmet:", c["days_reducing_unmet_demand"])
    print("Days improving cost:", c["days_beating_baseline_cost"])
    print("All invariants passed:", all(report["invariants"].values()))
    print("Report:", args.config)


if __name__ == "__main__":
    main()
