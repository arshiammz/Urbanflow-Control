from __future__ import annotations

from pathlib import Path

from urbanflow.simulation.fleet_simulator import (
    FleetSimulationError,
)
from urbanflow.simulation.forecast_guided_heuristic import (
    ForecastGuidedPolicyError,
    run_forecast_guided_comparison,
)
from urbanflow.simulation.real_episode_baseline import (
    RealEpisodeBaselineError,
)


CONFIG_PATH = Path(
    "configs/simulation/"
    "forecast_guided_heuristic.yaml"
)


def main() -> None:
    """Run forecast-guided rebalancing comparison."""
    try:
        report = run_forecast_guided_comparison(
            CONFIG_PATH
        )

    except (
        FleetSimulationError,
        ForecastGuidedPolicyError,
        RealEpisodeBaselineError,
    ) as error:
        raise SystemExit(
            "Forecast-guided simulation failed: "
            f"{error}"
        ) from error

    baseline = report[
        "baseline_metrics"
    ]

    policy = report[
        "policy_metrics"
    ]

    comparison = report[
        "comparison"
    ]

    print(
        "Forecast-guided rebalancing completed"
    )

    print()

    print(
        "Evaluation date:",
        report[
            "episode"
        ][
            "service_date"
        ],
    )

    print(
        "Total fleet:",
        f"{report['fleet']['total_fleet']:,}",
    )

    print()

    print(
        "No rebalancing"
    )

    print(
        "Served:",
        f"{baseline['served_demand_total']:,}",
    )

    print(
        "Unmet:",
        f"{baseline['unmet_demand_total']:,}",
    )

    print(
        "Service rate:",
        (
            f"{baseline['service_rate'] * 100:.2f}%"
        ),
    )

    print(
        "Average idle:",
        (
            f"{baseline['average_idle_vehicles']:.2f}"
        ),
    )

    print(
        "Operating cost:",
        (
            f"{baseline['total_operating_cost']:.2f}"
        ),
    )

    print()

    print(
        "Forecast-guided policy"
    )

    print(
        "Served:",
        f"{policy['served_demand_total']:,}",
    )

    print(
        "Unmet:",
        f"{policy['unmet_demand_total']:,}",
    )

    print(
        "Service rate:",
        (
            f"{policy['service_rate'] * 100:.2f}%"
        ),
    )

    print(
        "Average idle:",
        (
            f"{policy['average_idle_vehicles']:.2f}"
        ),
    )

    print(
        "Relocated vehicles:",
        f"{policy['relocated_vehicle_total']:,}",
    )

    print(
        "Relocation vehicle-steps:",
        f"{policy['relocation_vehicle_steps']:,}",
    )

    print(
        "Operating cost:",
        (
            f"{policy['total_operating_cost']:.2f}"
        ),
    )

    print()

    print(
        "Comparison"
    )

    print(
        "Unmet-demand reduction:",
        (
            f"{comparison['unmet_demand_absolute_reduction']:,}"
        ),
        (
            f"({comparison['unmet_demand_relative_reduction'] * 100:.2f}%)"
        ),
    )

    print(
        "Service-rate improvement:",
        (
            f"{comparison['service_rate_absolute_improvement'] * 100:.2f}"
            " percentage points"
        ),
    )

    print(
        "Operating-cost reduction:",
        (
            f"{comparison['operating_cost_absolute_reduction']:.2f}"
        ),
        (
            f"({comparison['operating_cost_relative_reduction'] * 100:.2f}%)"
        ),
    )

    print()

    print(
        "Artifacts:"
    )

    for artifact in report[
        "artifacts"
    ].values():
        print(
            artifact[
                "path"
            ]
        )

    print()

    print(
        "Report:",
        (
            "reports/simulation/"
            "2024-05-15-forecast-guided-heuristic.json"
        ),
    )


if __name__ == "__main__":
    main()