from __future__ import annotations

import hashlib
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class ModelRegistryError(RuntimeError):
    """Production model manifest is invalid or cannot be promoted."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def create_manifest(
    *,
    quantile_model_directory: Path,
    quantile_metadata: Path,
    rl_checkpoint: Path,
    rl_training_report: Path,
    zone_graph: Path,
    final_evaluation_report: Path,
    output_path: Path,
) -> dict[str, Any]:
    quantile_names = ("P10.ubj", "P50.ubj", "P90.ubj")
    required = [
        quantile_metadata, rl_checkpoint, rl_training_report, zone_graph,
        final_evaluation_report,
    ]
    required.extend(quantile_model_directory / name for name in quantile_names)
    missing = [str(path) for path in required if not path.exists() or path.stat().st_size == 0]
    if missing:
        raise ModelRegistryError("Cannot promote an incomplete model bundle: " + ", ".join(missing))
    training = json.loads(rl_training_report.read_text(encoding="utf-8"))
    if training.get("selection_uses_train_only") is not True:
        raise ModelRegistryError("RL model provenance is not train-only.")
    if training.get("selected_checkpoint_sha256") != sha256(rl_checkpoint):
        raise ModelRegistryError("RL checkpoint hash differs from its promotion report.")
    evaluation = json.loads(final_evaluation_report.read_text(encoding="utf-8"))
    if (
        evaluation.get("holdout", {}).get("split") != "test"
        or evaluation.get("governance", {}).get("holdout_used_for_model_or_policy_selection") is not False
        or not all(evaluation.get("invariants", {}).values())
    ):
        raise ModelRegistryError("Final untouched holdout report is missing or has failed governance invariants.")
    quantile_metadata_payload = json.loads(quantile_metadata.read_text(encoding="utf-8"))
    chronology = quantile_metadata_payload.get("chronology", {})
    if chronology.get("prediction_period_labels_used_for_training") is not False:
        raise ModelRegistryError(
            "Quantile model metadata does not prove that holdout labels were excluded."
        )
    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "environment": "local-production-candidate",
        "models": {
            "demand_quantiles": {
                "type": "xgboost_quantile_graph_weather",
                "models": {
                    name.removesuffix(".ubj").lower(): {
                        "path": str(quantile_model_directory / name),
                        "sha256": sha256(quantile_model_directory / name),
                    }
                    for name in quantile_names
                },
                "metadata_path": str(quantile_metadata),
                "metadata_sha256": sha256(quantile_metadata),
            },
            "rebalancing_policy": {
                "type": "constrained_ppo",
                "path": str(rl_checkpoint),
                "sha256": sha256(rl_checkpoint),
                "training_report_path": str(rl_training_report),
                "training_report_sha256": sha256(rl_training_report),
                "selected_seed": training.get("selected_seed"),
            },
        },
        "zone_graph": {"path": str(zone_graph), "sha256": sha256(zone_graph)},
        "final_evaluation": {
            "path": str(final_evaluation_report),
            "sha256": sha256(final_evaluation_report),
        },
        "promotion_guardrails": {
            "forecast_holdout_labels_used_for_training": False,
            "rl_holdout_labels_used_for_training_or_seed_selection": False,
            "checkpoint_hash_verified": True,
            "all_model_files_hash_recorded": True,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        previous = output_path.with_name(output_path.stem + ".previous" + output_path.suffix)
        shutil.copy2(output_path, previous)
    temporary = output_path.with_name(output_path.name + ".part")
    temporary.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(output_path)
    return manifest


def verify_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ModelRegistryError(f"Model manifest is missing: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    failures: list[str] = []
    for name, item in manifest.get("models", {}).get("demand_quantiles", {}).get("models", {}).items():
        model_path = Path(item["path"])
        if not model_path.exists() or sha256(model_path) != item["sha256"]:
            failures.append(f"demand_quantiles.{name}")
    for key in ("metadata_path",):
        model = manifest.get("models", {}).get("demand_quantiles", {})
        path_value = model.get(key)
        expected = model.get("metadata_sha256")
        if not path_value or not Path(path_value).exists() or sha256(Path(path_value)) != expected:
            failures.append(f"demand_quantiles.{key}")
    rl = manifest.get("models", {}).get("rebalancing_policy", {})
    for path_key, hash_key in (("path", "sha256"), ("training_report_path", "training_report_sha256")):
        path_value = rl.get(path_key)
        if not path_value or not Path(path_value).exists() or sha256(Path(path_value)) != rl.get(hash_key):
            failures.append(f"rebalancing_policy.{path_key}")
    graph = manifest.get("zone_graph", {})
    graph_path = Path(graph.get("path", ""))
    if not graph_path.exists() or sha256(graph_path) != graph.get("sha256"):
        failures.append("zone_graph")
    evaluation = manifest.get("final_evaluation", {})
    evaluation_path = Path(evaluation.get("path", ""))
    if not evaluation_path.exists() or sha256(evaluation_path) != evaluation.get("sha256"):
        failures.append("final_evaluation")
    if failures:
        raise ModelRegistryError("Manifest verification failed: " + ", ".join(failures))
    return {"verified": True, "model_count": 4, "manifest": str(path)}
