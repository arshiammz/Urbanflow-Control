"""Log an existing JSON experiment/evaluation report to local or remote MLflow."""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any


def _flatten(value: Any, prefix: str = "") -> dict[str, float]:
    output: dict[str, float] = {}
    if isinstance(value, dict):
        for key, item in value.items():
            output.update(_flatten(item, f"{prefix}.{key}" if prefix else str(key)))
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        number = float(value)
        if number == number and abs(number) != float("inf"):
            output[re.sub(r"[^A-Za-z0-9_./-]", "_", prefix)[:240]] = number
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--experiment", default="urbanflow-control")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--tracking-uri", default=os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns"))
    args = parser.parse_args()
    if not args.report.exists():
        raise SystemExit(f"Report is missing: {args.report}")
    try:
        import mlflow
    except ImportError as error:
        raise SystemExit("Install MLflow with `pip install -e '.[mlops]'`.") from error
    report = json.loads(args.report.read_text(encoding="utf-8"))
    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment)
    run_name = args.run_name or args.report.stem
    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("report_path", str(args.report))
        mlflow.log_param("report_version", str(report.get("version", report.get("benchmark_version", "unknown"))))
        metrics = _flatten(report)
        if metrics:
            mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(args.report))
    print("MLflow run logged")
    print("Experiment:", args.experiment)
    print("Run name:", run_name)
    print("Tracking URI:", args.tracking_uri)


if __name__ == "__main__":
    main()
