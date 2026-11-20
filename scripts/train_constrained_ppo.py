"""Train constrained PPO on train-only demand and store three checkpoints."""
from __future__ import annotations

import argparse
from pathlib import Path

from urbanflow.rl.workflow import train


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/rl/constrained_ppo.yaml"),
    )
    args = parser.parse_args()
    result = train(args.config)
    print("Train-only constrained PPO completed")
    print("Seeds:", result["seed_count"])
    print("Train days:", result["training_days"])
    print("Selected seed (TRAIN ONLY):", result["selected_seed"])
    print("Checkpoint:", result["selected_checkpoint"])


if __name__ == "__main__":
    main()
