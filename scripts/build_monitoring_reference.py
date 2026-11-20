"""Build a train-only demand-drift reference for production monitoring."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl

from urbanflow.monitoring.drift import build_reference


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path(
        "data/processed/simulation/nyc_2024-01_2024-07_all_zones/historical_demand.parquet"
    ))
    parser.add_argument("--output", type=Path, default=Path(
        "artifacts/monitoring/demand-reference.json"
    ))
    parser.add_argument("--split", default="train")
    args = parser.parse_args()
    if not args.input.exists():
        raise SystemExit(f"Historical demand file is missing: {args.input}")
    frame = pl.scan_parquet(args.input).filter(pl.col("split") == args.split).select(
        "actual_demand"
    ).collect()
    values = frame.get_column("actual_demand").to_numpy()
    report = build_reference(values, split=args.split, source_path=str(args.input))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".part")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print("Train-only demand reference created")
    print("Rows:", report["summary"]["count"])
    print("Output:", args.output)


if __name__ == "__main__":
    main()
