"""Compare a current demand batch with the train-only monitoring reference."""
from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from urbanflow.monitoring.drift import distribution_summary, psi_from_reference, write_monitoring_report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=Path("artifacts/monitoring/demand-reference.json"))
    parser.add_argument("--input", type=Path, default=Path(
        "data/processed/simulation/nyc_2024-01_2024-07_all_zones/historical_demand.parquet"
    ))
    parser.add_argument("--split", default="test")
    parser.add_argument("--output", type=Path, default=Path("reports/monitoring/latest-demand-drift.json"))
    parser.add_argument("--warning-psi", type=float, default=0.10)
    parser.add_argument("--alert-psi", type=float, default=0.25)
    args = parser.parse_args()
    if not args.reference.exists() or not args.input.exists():
        raise SystemExit("Monitoring reference or current input is missing.")
    reference = json.loads(args.reference.read_text(encoding="utf-8"))
    current = pl.scan_parquet(args.input).filter(pl.col("split") == args.split).select(
        "actual_demand"
    ).collect().get_column("actual_demand").to_numpy()
    psi = psi_from_reference(reference, current)
    status = "alert" if psi >= args.alert_psi else "warning" if psi >= args.warning_psi else "stable"
    report = {
        "version": "urbanflow_batch_drift_report_v1",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "reference_path": str(args.reference),
        "current_path": str(args.input),
        "reference_split": reference["reference_split"],
        "current_split": args.split,
        "current_summary": distribution_summary(current),
        "population_stability_index": float(psi),
        "thresholds": {"warning": args.warning_psi, "alert": args.alert_psi},
        "status": status,
        "recommended_action": (
            "investigate demand drift and run challenger evaluation"
            if status == "alert"
            else "review recent segment metrics"
            if status == "warning"
            else "continue routine monitoring"
        ),
    }
    write_monitoring_report(args.output, report)
    print("Demand monitoring completed")
    print("PSI:", f"{psi:.4f}")
    print("Status:", status)
    print("Report:", args.output)


if __name__ == "__main__":
    main()
