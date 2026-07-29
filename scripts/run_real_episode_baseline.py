from __future__ import annotations

from pathlib import Path

from urbanflow.simulation.fleet_simulator import (
    FleetSimulationError,
)
from urbanflow.simulation.real_episode_baseline import (
    RealEpisodeBaselineError,
    run_real_episode_baseline,
)


CONFIG_PATH = Path(
    "configs/simulation/real_episode_baseline.yaml"
)


def main() -> None:
    """Run the real no-rebalancing simulation baseline."""
    try:
        report = run_real_episode_baseline(
            CONFIG_PATH
        )

    except (
        FleetSimulationError,
        RealEpisodeBaselineError,
    ) as error:
        raise SystemExit(
            f"Real episode baseline failed: {error}"
        ) from error

    episode = report[
        "episode"
    ]

    fleet = report[
        "fleet_sizing"
    ]

    metrics = report[
        "evaluation_metrics"
    ]

    print(
        "Real no-rebalancing baseline completed"
    )

    print()

    print(
        "Evaluation date:",
        episode[
            "service_date"
        ],
    )

    print(
        "Warmup steps:",
        episode[
            "warmup_step_count"
        ],
    )

    print(
        "Evaluation steps:",
        episode[
            "evaluation_step_count"
        ],
    )

    print()

    print(
        "Estimated busy vehicles:",
        (
            f"{fleet['estimated_busy_vehicles']:.2f}"
        ),
    )

    print(
        "Reserve multiplier:",
        (
            f"{fleet['reserve_multiplier']:.2f}"
        ),
    )

    print(
        "Total fleet:",
        f"{fleet['total_fleet']:,}",
    )

    print()

    print(
        "Demand:",
        f"{metrics['demand_total']:,}",
    )

    print(
        "Served:",
        f"{metrics['served_demand_total']:,}",
    )

    print(
        "Unmet:",
        f"{metrics['unmet_demand_total']:,}",
    )

    print(
        "Service rate:",
        (
            f"{metrics['service_rate'] * 100:.2f}%"
        ),
    )

    print(
        "Average idle vehicles:",
        (
            f"{metrics['average_idle_vehicles']:.2f}"
        ),
    )

    print(
        "Relocated vehicles:",
        f"{metrics['relocated_vehicle_total']:,}",
    )

    print()

    print(
        "Unmet-demand cost:",
        (
            f"{metrics['unmet_demand_cost']:.2f}"
        ),
    )

    print(
        "Idle-vehicle cost:",
        (
            f"{metrics['idle_vehicle_cost']:.2f}"
        ),
    )

    print(
        "Total operating cost:",
        (
            f"{metrics['total_operating_cost']:.2f}"
        ),
    )

    print()

    print(
        "Top unmet-demand zones:"
    )

    for zone in report[
        "top_unmet_zones"
    ][:5]:
        print(
            f"zone={zone['zone_id']:>3} "
            f"demand={zone['demand_total']:>6,} "
            f"served={zone['served_demand_total']:>6,} "
            f"unmet={zone['unmet_demand_total']:>6,} "
            f"service_rate="
            f"{zone['service_rate'] * 100:>6.2f}%"
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

    print(
        report[
            "artifacts"
        ][
            "step_metrics"
        ][
            "path"
        ]
    )

    print()

    print(
        "Report:",
        (
            "reports/simulation/"
            "2024-05-15-no-rebalancing-baseline.json"
        ),
    )


if __name__ == "__main__":
    main()