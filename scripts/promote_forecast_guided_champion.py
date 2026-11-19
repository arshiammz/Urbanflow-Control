"""Freeze the May-selected Greedy champion and its input/code provenance.

Run ONLY after validation-only tuning succeeds. This script does not evaluate
June/test data, does not retune, and fails if a frozen artifact already exists.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from urbanflow.simulation.forecast_guided_heuristic import write_json_atomically


ROOT = Path(".")
REPORT = ROOT / "reports/simulation/2024-05-02_2024-05-30-forecast-guided-tuning.json"
REFERENCE = ROOT / "configs/simulation/forecast_guided_heuristic.yaml"
VALIDATION = ROOT / "configs/simulation/forecast_guided_validation.yaml"
CHAMPION = ROOT / "configs/simulation/forecast_guided_champion.yaml"
CHAMPION_VALIDATION = ROOT / "configs/simulation/forecast_guided_champion_validation.yaml"
MANIFEST = ROOT / "reports/simulation/forecast-guided-champion-freeze.json"

PARAMETERS = (
    "forecast_horizon_steps",
    "demand_buffer_multiplier",
    "minimum_zone_reserve",
    "maximum_relocation_steps",
    "maximum_relocation_fraction",
    "in_transit_supply_credit",
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_yaml(path: Path) -> dict[str, Any]:
    result = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(result, dict):
        raise ValueError(f"Invalid YAML mapping: {path}")
    return result


def main() -> None:
    for dest in (CHAMPION, CHAMPION_VALIDATION, MANIFEST):
        if dest.exists():
            raise SystemExit(
                f"Freeze refused: {dest} exists. Preserve immutable champion artifacts."
            )
    if not REPORT.is_file():
        raise SystemExit(f"Tuning report missing: {REPORT}. Run tuning first.")
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    expected_method = "asof_single_step_anchor_train_only_weekly_profile_v1"
    method = report.get("methodology", {})
    champion = report.get("champion_candidate", {})
    invariants = report.get("invariants", {})
    if (
        method.get("selection_split") != "validation"
        or method.get("asof_forecasting_method") != expected_method
        or method.get("future_offline_predictions_used") is not False
        or method.get("test_evaluation_performed") is not False
        or not invariants
        or not all(value is True for value in invariants.values())
        or champion.get("feasible") is not True
        or report.get("search_grid", {}).get("candidate_count") != 108
    ):
        raise SystemExit("Tuning report is not a complete leakage-safe validation result.")

    frozen = load_yaml(REFERENCE)
    validation = load_yaml(VALIDATION)
    for field in PARAMETERS:
        if field not in champion:
            raise SystemExit(f"Champion parameter missing: {field}")
        frozen["policy"][field] = champion[field]
    frozen["dataset"]["name"] = "nyc_hvfhv_forecast_guided_frozen_champion"
    frozen["paths"].update({
        "step_metrics": "reports/simulation/frozen_champion/2024-05-15-step-metrics.parquet",
        "interval_zone_metrics": "reports/simulation/frozen_champion/2024-05-15-interval-zone-metrics.parquet",
        "zone_summary": "reports/simulation/frozen_champion/2024-05-15-zone-summary.parquet",
        "report": "reports/simulation/2024-05-15-frozen-champion.json",
    })
    validation["dataset"]["name"] = "nyc_hvfhv_frozen_champion_validation"
    validation["paths"]["heuristic_config"] = str(CHAMPION)
    validation["paths"].update({
        "step_metrics": "reports/simulation/frozen_champion_validation/step-metrics.parquet",
        "daily_metrics": "reports/simulation/frozen_champion_validation/daily-metrics.parquet",
        "daily_zone_metrics": "reports/simulation/frozen_champion_validation/daily-zone-metrics.parquet",
        "zone_summary": "reports/simulation/frozen_champion_validation/zone-summary.parquet",
        "report": "reports/simulation/frozen-champion-may-validation.json",
    })
    if validation["validation"]["split"] != "validation":
        raise SystemExit("Champion validation config must be May validation only.")

    asof_metadata_path = Path("reports/simulation/asof-validation-forecast-metadata.json")
    if not asof_metadata_path.is_file():
        raise SystemExit("Pre-May as-of forecast metadata missing. Build it first.")
    asof_metadata = json.loads(asof_metadata_path.read_text(encoding="utf-8"))
    timeline = asof_metadata.get("chronology", {})
    if (
        timeline.get("may_labels_used_for_training") is not False
        or timeline.get("may_labels_used_for_model_selection") is not False
        or timeline.get("june_data_used") is not False
    ):
        raise SystemExit("May as-of model provenance is not causally valid.")
    forecast_source = Path(frozen["paths"]["forecast_predictions"])
    if forecast_source.name != "asof_validation_predictions.parquet":
        raise SystemExit("Frozen champion may use only the pre-May as-of forecasts.")
    if frozen["forecast"]["prediction_column"] != "asof_prediction":
        raise SystemExit("Frozen champion is not configured for as-of predictions.")
    dependent_files = [
        REFERENCE,
        Path("configs/simulation/asof_validation_forecast.yaml"),
        Path("configs/forecasting/xgboost.yaml"),
        Path("scripts/build_asof_validation_forecasts.py"),
        asof_metadata_path,
        VALIDATION,
        Path("configs/simulation/real_episode_baseline.yaml"),
        Path("src/urbanflow/simulation/decision_time_forecasting.py"),
        Path("src/urbanflow/simulation/forecast_guided_heuristic.py"),
        Path("src/urbanflow/simulation/forecast_guided_validation.py"),
        Path("src/urbanflow/simulation/forecast_guided_tuning.py"),
        Path("src/urbanflow/simulation/fleet_simulator.py"),
        REPORT,
    ]
    artifact_paths = [
        Path("artifacts/models/xgboost-asof-may/model.ubj"),
        Path("data/processed/full/forecast_features_15m_2024-01_2024-06.parquet"),
        forecast_source,
        Path(load_yaml(Path(frozen["paths"]["baseline_config"]))["paths"]["historical_demand"]),
        Path(load_yaml(Path(frozen["paths"]["baseline_config"]))["paths"]["zone_index"]),
        Path(load_yaml(Path(frozen["paths"]["baseline_config"]))["paths"]["od_profile"]),
    ]
    for path in dependent_files + artifact_paths:
        if not path.is_file():
            raise SystemExit(f"Cannot freeze: required provenance input missing: {path}")
    provenance = {str(path): sha256(path) for path in dependent_files + artifact_paths}
    if asof_metadata.get("prediction_sha256") != sha256(forecast_source):
        raise SystemExit("As-of forecast file hash does not match its creation metadata.")
    if asof_metadata.get("model_sha256") != sha256(Path(asof_metadata["model_path"])):
        raise SystemExit("As-of model hash does not match its creation metadata.")
    CHAMPION.parent.mkdir(parents=True, exist_ok=True)
    for path, payload in ((CHAMPION, frozen), (CHAMPION_VALIDATION, validation)):
        tmp = path.with_name(path.name + ".part")
        tmp.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
        tmp.replace(path)
    write_json_atomically(MANIFEST, {
        "version": "phase1-champion-freeze-v1",
        "frozen_at": datetime.now(UTC),
        "selection_split": "validation",
        "evaluation_split_used_for_selection": "validation",
        "test_used_for_selection": False,
        "multi_horizon_method": expected_method,
        "champion_candidate_id": champion["candidate_id"],
        "champion_parameters": {key: champion[key] for key in PARAMETERS},
        "reference_candidate_id": report["reference_candidate"]["candidate_id"],
        "candidate_count": 108,
        "source_and_dataset_sha256": provenance,
        "frozen_config_sha256": sha256(CHAMPION),
        "frozen_validation_config_sha256": sha256(CHAMPION_VALIDATION),
        "paths": {
            "champion": str(CHAMPION),
            "champion_validation": str(CHAMPION_VALIDATION),
            "manifest": str(MANIFEST),
        },
    })
    print("Champion frozen without using test data.")
    print("Champion ID:", champion["candidate_id"])
    for path in (CHAMPION, CHAMPION_VALIDATION, MANIFEST):
        print(path)
    print("Validation replay:")
    print("python scripts/run_forecast_guided_validation.py --config", CHAMPION_VALIDATION)


if __name__ == "__main__":
    main()
