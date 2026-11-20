"""Promote a verified quantile-forecast + constrained-PPO bundle to a local manifest."""
from __future__ import annotations

import argparse
from pathlib import Path

from urbanflow.monitoring.model_registry import ModelRegistryError, create_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quantile-model-directory", type=Path, default=Path(
        "artifacts/models/urbanflow-quantile-graph-weather-july-holdout"
    ))
    parser.add_argument("--quantile-metadata", type=Path, default=Path(
        "artifacts/models/urbanflow-quantile-graph-weather-july-holdout/metadata.json"
    ))
    parser.add_argument("--rl-checkpoint", type=Path, default=Path(
        "artifacts/models/constrained_ppo/champion_train_selected.pt"
    ))
    parser.add_argument("--rl-training-report", type=Path, default=Path("reports/rl/constrained_ppo_training.json"))
    parser.add_argument("--zone-graph", type=Path, default=Path(
        "data/processed/spatial/nyc_taxi_zone_graph_july_holdout.json"
    ))
    parser.add_argument("--final-evaluation-report", type=Path, default=Path(
        "reports/evaluation/2024-07-final-holdout.json"
    ))
    parser.add_argument("--output", type=Path, default=Path("artifacts/production/model_manifest.json"))
    args = parser.parse_args()
    try:
        manifest = create_manifest(
            quantile_model_directory=args.quantile_model_directory,
            quantile_metadata=args.quantile_metadata,
            rl_checkpoint=args.rl_checkpoint,
            rl_training_report=args.rl_training_report,
            zone_graph=args.zone_graph,
            final_evaluation_report=args.final_evaluation_report,
            output_path=args.output,
        )
    except (ModelRegistryError, OSError, ValueError, KeyError) as error:
        raise SystemExit(f"Production promotion failed: {error}") from error
    print("Production model manifest created")
    print("Quantile models:", len(manifest["models"]["demand_quantiles"]["models"]))
    print("RL seed:", manifest["models"]["rebalancing_policy"]["selected_seed"])
    print("Manifest:", args.output)


if __name__ == "__main__":
    main()
