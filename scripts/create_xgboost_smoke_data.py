from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl


INPUT_PATH = Path(
    "data/processed/full/"
    "forecast_features_15m_2024-01_2024-06.parquet"
)

OUTPUT_PATH = Path(
    "data/processed/dev/"
    "forecast_features_xgboost_smoke.parquet"
)

REPORT_PATH = Path(
    "reports/data_quality/"
    "xgboost-smoke-dataset.json"
)

INTERVAL_LIMITS = {
    "train": 192,
    "validation": 48,
    "test": 48,
}

KEY_COLUMNS = (
    "interval_start",
    "zone_id",
)

REQUIRED_COLUMNS = {
    "interval_start",
    "zone_id",
    "split",
    "target_pickup_count",
}


class SmokeDatasetError(RuntimeError):
    """Raised when the XGBoost smoke dataset cannot be built."""


def select_split_intervals(
    source: pl.LazyFrame,
    *,
    split: str,
    interval_limit: int,
) -> pl.DataFrame:
    """Select the earliest configured intervals from one split."""
    intervals = (
        source.filter(
            pl.col("split") == split
        )
        .select(
            "interval_start"
        )
        .unique()
        .sort(
            "interval_start"
        )
        .limit(
            interval_limit
        )
        .collect()
    )

    if intervals.height != interval_limit:
        raise SmokeDatasetError(
            f"Split {split!r} contains only "
            f"{intervals.height:,} intervals; "
            f"{interval_limit:,} were requested."
        )

    try:
        frame = (
            source.filter(
                pl.col("split") == split
            )
            .join(
                intervals.lazy(),
                on="interval_start",
                how="inner",
            )
            .collect()
            .sort(
                *KEY_COLUMNS
            )
        )
    except Exception as error:
        raise SmokeDatasetError(
            f"Could not materialize split {split!r}: {error}"
        ) from error

    return frame


def summarize_split(
    frame: pl.DataFrame,
    *,
    split: str,
) -> dict[str, Any]:
    """Validate and summarize one smoke-data split."""
    row = (
        frame.select(
            pl.len().alias(
                "row_count"
            ),
            pl.col("interval_start")
            .n_unique()
            .alias(
                "interval_count"
            ),
            pl.col("zone_id")
            .n_unique()
            .alias(
                "zone_count"
            ),
            pl.col("interval_start")
            .min()
            .alias(
                "first_interval"
            ),
            pl.col("interval_start")
            .max()
            .alias(
                "last_interval"
            ),
            pl.col("target_pickup_count")
            .min()
            .alias(
                "target_minimum"
            ),
            pl.col("target_pickup_count")
            .max()
            .alias(
                "target_maximum"
            ),
            pl.col("target_pickup_count")
            .mean()
            .alias(
                "target_mean"
            ),
        )
        .row(
            0,
            named=True,
        )
    )

    row_count = int(
        row["row_count"]
    )

    interval_count = int(
        row["interval_count"]
    )

    zone_count = int(
        row["zone_count"]
    )

    expected_row_count = (
        interval_count
        * zone_count
    )

    if row_count != expected_row_count:
        raise SmokeDatasetError(
            f"Split {split!r} is not a dense grid: "
            f"expected {expected_row_count:,} rows, "
            f"received {row_count:,}."
        )

    if int(
        row["target_minimum"]
    ) < 0:
        raise SmokeDatasetError(
            f"Split {split!r} contains a negative target."
        )

    first_interval = row[
        "first_interval"
    ]

    last_interval = row[
        "last_interval"
    ]

    return {
        "row_count": row_count,
        "interval_count": interval_count,
        "zone_count": zone_count,
        "first_interval": (
            first_interval.isoformat()
        ),
        "last_interval": (
            last_interval.isoformat()
        ),
        "target_minimum": int(
            row["target_minimum"]
        ),
        "target_maximum": int(
            row["target_maximum"]
        ),
        "target_mean": round(
            float(
                row["target_mean"]
            ),
            6,
        ),
    }


def build_smoke_dataset() -> dict[str, Any]:
    """Build a small dense forecasting dataset for XGBoost testing."""
    if not INPUT_PATH.exists():
        raise SmokeDatasetError(
            f"Input feature dataset does not exist: {INPUT_PATH}"
        )

    if INPUT_PATH.stat().st_size == 0:
        raise SmokeDatasetError(
            f"Input feature dataset is empty: {INPUT_PATH}"
        )

    try:
        source = pl.scan_parquet(
            INPUT_PATH
        )

        schema = source.collect_schema()

    except Exception as error:
        raise SmokeDatasetError(
            f"Could not read input feature dataset: {error}"
        ) from error

    missing_columns = sorted(
        REQUIRED_COLUMNS.difference(
            schema.names()
        )
    )

    if missing_columns:
        raise SmokeDatasetError(
            "Input feature dataset is missing columns: "
            + ", ".join(
                missing_columns
            )
        )

    split_frames: list[pl.DataFrame] = []
    split_summaries: dict[str, dict[str, Any]] = {}

    for split, interval_limit in INTERVAL_LIMITS.items():
        frame = select_split_intervals(
            source,
            split=split,
            interval_limit=interval_limit,
        )

        split_frames.append(
            frame
        )

        split_summaries[split] = summarize_split(
            frame,
            split=split,
        )

    zone_counts = {
        summary["zone_count"]
        for summary in split_summaries.values()
    }

    if len(zone_counts) != 1:
        raise SmokeDatasetError(
            "Train, validation, and test do not contain "
            "the same number of zones."
        )

    smoke_frame = (
        pl.concat(
            split_frames,
            how="vertical",
        )
        .sort(
            *KEY_COLUMNS
        )
    )

    duplicate_key_count = int(
        smoke_frame.group_by(
            *KEY_COLUMNS
        )
        .agg(
            pl.len().alias(
                "key_count"
            )
        )
        .filter(
            pl.col("key_count") > 1
        )
        .height
    )

    null_value_count = int(
        sum(
            int(value)
            for value in (
                smoke_frame
                .null_count()
                .row(0)
            )
        )
    )

    if duplicate_key_count:
        raise SmokeDatasetError(
            "Smoke dataset contains duplicate prediction keys."
        )

    if null_value_count:
        raise SmokeDatasetError(
            "Smoke dataset contains null values."
        )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_output = OUTPUT_PATH.with_name(
        f"{OUTPUT_PATH.name}.part"
    )

    temporary_output.unlink(
        missing_ok=True
    )

    try:
        smoke_frame.write_parquet(
            temporary_output,
            compression="zstd",
            statistics=True,
        )

        temporary_output.replace(
            OUTPUT_PATH
        )

    except Exception as error:
        temporary_output.unlink(
            missing_ok=True
        )

        raise SmokeDatasetError(
            f"Could not write smoke dataset: {error}"
        ) from error

    report: dict[str, Any] = {
        "smoke_dataset_version": "0.1.0",
        "generated_at": datetime.now(
            UTC
        ).isoformat(),
        "input": {
            "path": str(
                INPUT_PATH
            ),
            "size_bytes": (
                INPUT_PATH.stat().st_size
            ),
        },
        "output": {
            "path": str(
                OUTPUT_PATH
            ),
            "size_bytes": (
                OUTPUT_PATH.stat().st_size
            ),
            "row_count": (
                smoke_frame.height
            ),
            "column_count": (
                smoke_frame.width
            ),
        },
        "interval_limits": INTERVAL_LIMITS,
        "splits": split_summaries,
        "invariants": {
            "zone_counts_match": True,
            "duplicate_key_count": (
                duplicate_key_count
            ),
            "null_value_count": (
                null_value_count
            ),
            "dense_grid_per_split": True,
        },
    }

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_PATH.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    return report


def main() -> None:
    """Build and print the smoke-dataset summary."""
    report = build_smoke_dataset()

    output = report[
        "output"
    ]

    splits = report[
        "splits"
    ]

    print("XGBoost smoke dataset created")
    print(
        f"Rows: {output['row_count']:,}"
    )
    print(
        f"Columns: {output['column_count']:,}"
    )

    for split in (
        "train",
        "validation",
        "test",
    ):
        summary = splits[
            split
        ]

        print(
            f"{split.title()}: "
            f"{summary['row_count']:,} rows, "
            f"{summary['interval_count']:,} intervals, "
            f"{summary['zone_count']:,} zones"
        )

    print(
        f"Output: {output['path']}"
    )
    print(
        f"Report: {REPORT_PATH}"
    )


if __name__ == "__main__":
    main()