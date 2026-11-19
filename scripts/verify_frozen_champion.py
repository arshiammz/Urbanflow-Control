"""Verify frozen source/data hashes and reproduce May champion metrics.

No test data is loaded. This is a validation-only replay after selection.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from urbanflow.simulation.forecast_guided_validation import run_forecast_guided_validation

from promote_forecast_guided_champion import CHAMPION, CHAMPION_VALIDATION, MANIFEST, sha256


def main() -> None:
    if not all(path.is_file() for path in (CHAMPION, CHAMPION_VALIDATION, MANIFEST)):
        raise SystemExit("Missing frozen artifacts. Run promote_forecast_guided_champion.py first.")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if (
        manifest.get("selection_split") != "validation"
        or manifest.get("test_used_for_selection") is not False
    ):
        raise SystemExit("Freeze manifest failed temporal integrity checks.")
    for filename, digest in manifest["source_and_dataset_sha256"].items():
        path = Path(filename)
        if not path.is_file() or sha256(path) != digest:
            raise SystemExit(f"Frozen source or dataset changed: {path}")
    if sha256(CHAMPION) != manifest["frozen_config_sha256"]:
        raise SystemExit("Frozen champion config was modified.")
    if sha256(CHAMPION_VALIDATION) != manifest["frozen_validation_config_sha256"]:
        raise SystemExit("Frozen validation config was modified.")

    replay = run_forecast_guided_validation(CHAMPION_VALIDATION)
    if not all(replay["invariants"].values()):
        raise SystemExit("Frozen replay failed simulation invariants.")
    tune_report = json.loads(Path(
        "reports/simulation/2024-05-02_2024-05-30-forecast-guided-tuning.json"
    ).read_text(encoding="utf-8"))
    selected = tune_report["champion_candidate"]
    metrics = replay["aggregate_metrics"]["policy"]
    if metrics["unmet_demand_total"] != selected["policy_unmet_demand_total"]:
        raise SystemExit("Frozen replay unmet demand differs from tuning champion.")
    if not math.isclose(
        metrics["total_operating_cost"], selected["policy_total_operating_cost"],
        rel_tol=1e-9, abs_tol=0.05,
    ):
        raise SystemExit("Frozen replay operating cost differs from tuning champion.")
    print("Frozen champion verified: hashes, May metrics, and invariants all match.")
    print("Candidate ID:", selected["candidate_id"])
    print("Cost:", metrics["total_operating_cost"])
    print("Unmet demand:", metrics["unmet_demand_total"])


if __name__ == "__main__":
    main()
