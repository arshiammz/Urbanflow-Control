from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from urbanflow.simulation.fleet_simulator import (
    FleetSimulationError,
    FleetSimulator,
    SimulationCosts,
)


CONFIG_PATH = Path(
    "configs/simulation/fleet_simulator_smoke.yaml"
)


class FleetSmokeError(RuntimeError):
    """Raised when the fleet simulator smoke scenario fails."""


def read_config(
    path: Path,
) -> dict[str, Any]:
    """Read the smoke-simulation configuration."""
    if not path.exists():
        raise FleetSmokeError(
            f"Smoke config does not exist: {path}"
        )

    try:
        config = yaml.safe_load(
            path.read_text(
                encoding="utf-8",
            )
        )

    except (
        OSError,
        yaml.YAMLError,
    ) as error:
        raise FleetSmokeError(
            f"Could not read smoke config: {error}"
        ) from error

    if not isinstance(
        config,
        dict,
    ):
        raise FleetSmokeError(
            "Smoke config must contain a YAML mapping."
        )

    return config


def write_json_atomically(
    path: Path,
    payload: dict[str, Any],
) -> None:
    """Write a JSON report atomically."""
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = path.with_name(
        f"{path.name}.part"
    )

    temporary_path.unlink(
        missing_ok=True
    )

    try:
        temporary_path.write_text(
            json.dumps(
                payload,
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )

        temporary_path.replace(
            path
        )

    except (
        OSError,
        TypeError,
        ValueError,
    ) as error:
        temporary_path.unlink(
            missing_ok=True
        )

        raise FleetSmokeError(
            f"Could not write smoke report: {error}"
        ) from error


def build_simulator(
    config: dict[str, Any],
) -> FleetSimulator:
    """Build the fleet simulator from configuration."""
    try:
        simulation = config[
            "simulation"
        ]

        costs = simulation[
            "costs"
        ]

        return FleetSimulator(
            zone_ids=tuple(
                int(
                    zone_id
                )
                for zone_id in simulation[
                    "zone_ids"
                ]
            ),
            initial_vehicles=simulation[
                "initial_vehicles"
            ],
            passenger_destination_probabilities=(
                simulation[
                    "passenger_destination_probabilities"
                ]
            ),
            relocation_travel_steps=simulation[
                "relocation_travel_steps"
            ],
            passenger_travel_steps=simulation[
                "passenger_travel_steps"
            ],
            costs=SimulationCosts(
                unmet_demand_penalty=float(
                    costs[
                        "unmet_demand_penalty"
                    ]
                ),
                relocation_cost_per_vehicle_step=float(
                    costs[
                        "relocation_cost_per_vehicle_step"
                    ]
                ),
                idle_vehicle_cost=float(
                    costs[
                        "idle_vehicle_cost"
                    ]
                ),
            ),
        )

    except (
        KeyError,
        TypeError,
        ValueError,
    ) as error:
        raise FleetSmokeError(
            f"Invalid simulator configuration: {error}"
        ) from error


def main() -> None:
    """Run the deterministic fleet simulator smoke scenario."""
    try:
        config = read_config(
            CONFIG_PATH
        )

        simulator = build_simulator(
            config
        )

        scenario = config[
            "scenario"
        ]

        demands = scenario[
            "demands"
        ]

        relocation_actions = scenario[
            "relocation_actions"
        ]

        if len(
            demands
        ) != len(
            relocation_actions
        ):
            raise FleetSmokeError(
                "Demand and relocation scenarios "
                "must have equal lengths."
            )

        transitions = []

        print(
            "Fleet simulator smoke scenario"
        )

        print(
            "Zones:",
            simulator.zone_ids,
        )

        print(
            "Initial fleet:",
            simulator.total_fleet,
        )

        print()

        for demand, relocation_action in zip(
            demands,
            relocation_actions,
            strict=True,
        ):
            result = simulator.step(
                demand=demand,
                relocation_action=(
                    relocation_action
                ),
            )

            transitions.append(
                result.as_dict()
            )

            print(
                f"Step {result.step}"
            )

            print(
                "Demand:",
                result.metrics.demand_total,
            )

            print(
                "Served:",
                result.metrics.served_demand_total,
            )

            print(
                "Unmet:",
                result.metrics.unmet_demand_total,
            )

            print(
                "Relocated:",
                result.metrics.relocated_vehicle_total,
            )

            print(
                "Idle:",
                result.metrics.idle_vehicle_total,
            )

            print(
                "Service rate:",
                (
                    f"{result.metrics.service_rate * 100:.2f}%"
                ),
            )

            print(
                "Operating cost:",
                (
                    f"{result.metrics.total_operating_cost:.2f}"
                ),
            )

            print(
                "Available end:",
                (
                    result.available_vehicles_end.tolist()
                ),
            )

            print(
                "In transit end:",
                (
                    result.in_transit_vehicles_end.tolist()
                ),
            )

            print()

        summary = simulator.summary()

        report = {
            "simulation_version": "0.1.0",
            "config": str(
                CONFIG_PATH
            ),
            "zone_ids": list(
                simulator.zone_ids
            ),
            "initial_fleet": (
                simulator.total_fleet
            ),
            "transitions": transitions,
            "summary": summary,
            "invariants": {
                "all_steps_preserve_fleet": all(
                    transition[
                        "fleet_conservation_passed"
                    ]
                    for transition in transitions
                ),
                "completed_step_count_matches": (
                    summary[
                        "completed_steps"
                    ]
                    == len(
                        demands
                    )
                ),
                "final_fleet_is_conserved": (
                    sum(
                        summary[
                            "snapshot"
                        ][
                            "available_vehicles"
                        ]
                    )
                    + sum(
                        summary[
                            "snapshot"
                        ][
                            "in_transit_vehicles"
                        ]
                    )
                    == simulator.total_fleet
                ),
            },
        }

        report_path = Path(
            config[
                "output"
            ][
                "report"
            ]
        )

        write_json_atomically(
            report_path,
            report,
        )

    except (
        FleetSimulationError,
        FleetSmokeError,
        KeyError,
    ) as error:
        raise SystemExit(
            f"Fleet simulator smoke scenario failed: {error}"
        ) from error

    print(
        "Fleet simulator smoke scenario completed"
    )

    print()

    print(
        "Completed steps:",
        summary[
            "completed_steps"
        ],
    )

    print(
        "Total demand:",
        summary[
            "demand_total"
        ],
    )

    print(
        "Total served:",
        summary[
            "served_demand_total"
        ],
    )

    print(
        "Total unmet:",
        summary[
            "unmet_demand_total"
        ],
    )

    print(
        "Overall service rate:",
        (
            f"{summary['service_rate'] * 100:.2f}%"
        ),
    )

    print(
        "Total operating cost:",
        (
            f"{summary['total_operating_cost']:.2f}"
        ),
    )

    print(
        "Report:",
        report_path,
    )


if __name__ == "__main__":
    main()