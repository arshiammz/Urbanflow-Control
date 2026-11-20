"""Compare frozen PPO to all four Phase 3 policies on identical May inputs."""
from __future__ import annotations

import argparse
from pathlib import Path

from urbanflow.rl.workflow import evaluate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/rl/constrained_ppo.yaml"),
    )
    args = parser.parse_args()
    report = evaluate(args.config)
    print("Constrained PPO benchmark completed (May validation)")
    print("Strongest Phase 3 optimizer:", report["strongest_phase3_optimizer"])
    print("PPO service rate:", f"{report['rl_metrics']['service_rate'] * 100:.2f}%")
    print("PPO operating cost:", f"{report['rl_metrics']['total_operating_cost']:.2f}")
    print("RL vs strongest optimizer:", report["rl_vs_strongest_optimizer"])
    print("Invariants:", report["invariants"])


if __name__ == "__main__":
    main()
