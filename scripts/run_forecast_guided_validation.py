from __future__ import annotations

from pathlib import Path

from urbanflow.simulation.fleet_simulator import (
    FleetSimulationError,
)
from urbanflow.simulation.forecast_guided_heuristic import (
    ForecastGuidedPolicyError,
)
from urbanflow.simulation.forecast_guided_validation import (
    ForecastGuidedValidationError,
    run_forecast_guided_validation,
)
from urbanflow.simulation.real_episode_baseline import (
    RealEpisodeBaselineError,
)


CONFIG_PATH = Path(
    "configs/simulation/"
    "forecast_guided_validation.yaml"
)


def main() -> None:
    """Run continuous multi-day validation."""
    try:
        report = run_forecast_guided_validation(
            CONFIG_PATH
        )

    except (
        FleetSimulationError,
        ForecastGuidedPolicyError,
        ForecastGuidedValidationError,
        RealEpisodeBaselineError,
    ) as error:
        raise SystemExit(
            "Forecast-guided validation failed: "
            f"{error}"
        ) from error

    window = report[
        "validation_window"
    ]

    aggregate = report[
        "aggregate_metrics"
    ]

    baseline = aggregate[
        "baseline"
    ]

    policy = aggregate[
        "policy"
    ]

    comparison = report[
        "comparison"
    ]

    print(
        "Continuous forecast-guided validation completed"
    )

    print()

    print(
        "Validation window:",
        (
            f"{window['start_date']} "
            f"to {window['end_date_inclusive']}"
        ),
    )

    print(
        "Validation days:",
        window[
            "day_count"
        ],
    )

    print(
        "Warmup steps:",
        window[
            "warmup_step_count"
        ],
    )

    print(
        "Evaluation steps:",
        window[
            "evaluation_step_count"
        ],
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
        "Aggregate comparison"
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
        "Daily robustness"
    )

    print(
        "Days reducing unmet demand:",
        (
            f"{comparison['days_reducing_unmet_demand']}"
            f"/{window['day_count']}"
        ),
    )

    print(
        "Days beating baseline cost:",
        (
            f"{comparison['days_beating_baseline_cost']}"
            f"/{window['day_count']}"
        ),
    )

    print(
        "Mean daily service improvement:",
        (
            f"{comparison['mean_daily_service_rate_improvement'] * 100:.2f}"
            " percentage points"
        ),
    )

    print(
        "Median daily service improvement:",
        (
            f"{comparison['median_daily_service_rate_improvement'] * 100:.2f}"
            " percentage points"
        ),
    )

    print(
        "Worst day:",
        comparison[
            "worst_day"
        ][
            "service_date"
        ],
        (
            f"{comparison['worst_day']['service_rate_improvement'] * 100:.2f}"
            " percentage points"
        ),
    )

    print(
        "Best day:",
        comparison[
            "best_day"
        ][
            "service_date"
        ],
        (
            f"{comparison['best_day']['service_rate_improvement'] * 100:.2f}"
            " percentage points"
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
            "2024-05-02_2024-05-30-"
            "forecast-guided-validation.json"
        ),
    )


if __name__ == "__main__":
    main()