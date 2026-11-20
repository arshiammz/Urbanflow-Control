from __future__ import annotations

import json
from pathlib import Path

from urbanflow.monitoring.model_registry import (
    ModelRegistryError,
    create_manifest,
    verify_manifest,
)


def test_manifest_promotion_and_hash_verification(tmp_path: Path) -> None:
    quantile_dir = tmp_path / "quantiles"
    quantile_dir.mkdir()
    for name in ("P10.ubj", "P50.ubj", "P90.ubj"):
        (quantile_dir / name).write_bytes(name.encode("utf-8"))
    metadata = quantile_dir / "metadata.json"
    metadata.write_text(
        json.dumps({"chronology": {"prediction_period_labels_used_for_training": False}}),
        encoding="utf-8",
    )
    checkpoint = tmp_path / "champion.pt"
    checkpoint.write_bytes(b"checkpoint")
    training = tmp_path / "training.json"
    import hashlib
    training.write_text(json.dumps({
        "selection_uses_train_only": True,
        "selected_checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "selected_seed": 23,
    }), encoding="utf-8")
    graph = tmp_path / "graph.json"
    graph.write_text("{}", encoding="utf-8")
    final_evaluation = tmp_path / "final-evaluation.json"
    final_evaluation.write_text(json.dumps({
        "holdout": {"split": "test"},
        "governance": {"holdout_used_for_model_or_policy_selection": False},
        "invariants": {"all_passed": True},
    }), encoding="utf-8")
    manifest_path = tmp_path / "production" / "model_manifest.json"

    create_manifest(
        quantile_model_directory=quantile_dir,
        quantile_metadata=metadata,
        rl_checkpoint=checkpoint,
        rl_training_report=training,
        zone_graph=graph,
        final_evaluation_report=final_evaluation,
        output_path=manifest_path,
    )
    result = verify_manifest(manifest_path)
    assert result["verified"] is True

    checkpoint.write_bytes(b"tampered")
    try:
        verify_manifest(manifest_path)
    except ModelRegistryError as error:
        assert "verification failed" in str(error).lower()
    else:
        raise AssertionError("Manifest verification should detect a modified checkpoint.")
