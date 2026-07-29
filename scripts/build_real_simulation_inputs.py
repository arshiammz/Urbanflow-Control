from __future__ import annotations

from pathlib import Path

from urbanflow.simulation.real_data_inputs import (
    SimulationInputError,
    build_real_simulation_inputs,
)


CONFIG_PATH = Path(
    "configs/simulation/real_data_inputs.yaml"
)


def main() -> None:
    """Build leakage-safe real-data simulation inputs."""
    try:
        metadata = build_real_simulation_inputs(
            CONFIG_PATH
        )

    except SimulationInputError as error:
        raise SystemExit(
            f"Simulation-input generation failed: {error}"
        ) from error

    zones = metadata[
        "zones"
    ]

    od_profile = metadata[
        "od_profile"
    ]

    historical = metadata[
        "historical_demand"
    ]

    print(
        "Real simulation inputs completed"
    )

    print()

    print(
        "Selected zones:",
        zones[
            "selected_zone_count"
        ],
    )

    print(
        "Fit-period pickups:",
        f"{zones['total_fit_pickups']:,}",
    )

    print()

    print(
        "OD matrix rows:",
        f"{od_profile['matrix_row_count']:,}",
    )

    print(
        "Observed OD pairs:",
        f"{od_profile['observed_pair_count']:,}",
    )

    print(
        "Observed OD coverage:",
        (
            f"{od_profile['observed_pair_coverage'] * 100:.2f}%"
        ),
    )

    print(
        "Fallback OD pairs:",
        f"{od_profile['fallback_pair_count']:,}",
    )

    print(
        "Global median travel time:",
        (
            f"{od_profile['global_median_travel_minutes']:.2f} minutes"
        ),
    )

    print()

    print(
        "Historical demand rows:",
        f"{historical['row_count']:,}",
    )

    print(
        "Historical intervals:",
        f"{historical['interval_count']:,}",
    )

    print()

    for split in historical[
        "splits"
    ]:
        print(
            f"{split['split']:<10} "
            f"rows={split['row_count']:>10,} "
            f"intervals={split['interval_count']:>6,} "
            f"demand={split['demand_total']:>12,}"
        )

    print()

    print(
        "Artifacts:"
    )

    for artifact in metadata[
        "artifacts"
    ].values():
        print(
            artifact[
                "path"
            ]
        )


if __name__ == "__main__":
    main()