from __future__ import annotations

import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl
import yaml


CONFIG_PATH = Path(
    "configs/forecasting/final_model_comparison.yaml"
)


TABLE_SPECS: dict[str, tuple[str, ...]] = {
    "overall_metrics": (
        "split",
    ),
    "zone_metrics": (
        "split",
        "zone_id",
        "zone_name",
        "borough",
        "service_zone",
    ),
    "borough_metrics": (
        "split",
        "borough",
    ),
    "service_zone_metrics": (
        "split",
        "service_zone",
    ),
    "hour_metrics": (
        "split",
        "hour",
    ),
    "day_of_week_metrics": (
        "split",
        "day_of_week",
    ),
    "weekend_metrics": (
        "split",
        "is_weekend",
    ),
    "peak_period_metrics": (
        "split",
        "peak_period",
    ),
    "demand_band_metrics": (
        "split",
        "demand_band",
    ),
}


COMPARISON_METRICS = (
    "mae",
    "rmse",
    "wape",
    "smape",
)


MODEL_METRICS = (
    "prediction_sum",
    "mae",
    "rmse",
    "wape",
    "smape",
    "mean_bias",
)


class FinalModelComparisonError(RuntimeError):
    """Raised when final model comparison cannot be completed."""


def read_config(
    path: Path,
) -> dict[str, Any]:
    """Read and validate final-comparison configuration."""
    if not path.exists():
        raise FinalModelComparisonError(
            f"Comparison config does not exist: {path}"
        )

    try:
        raw = yaml.safe_load(
            path.read_text(
                encoding="utf-8",
            )
        )

        if not isinstance(
            raw,
            dict,
        ):
            raise FinalModelComparisonError(
                "Comparison config must contain a YAML mapping."
            )

        paths = raw["paths"]
        analysis = raw["analysis"]
        output = raw["output"]

        config = {
            "dataset_name": str(
                raw["dataset"]["name"]
            ),
            "raw_tables_directory": Path(
                paths["raw_tables_directory"]
            ),
            "champion_tables_directory": Path(
                paths["champion_tables_directory"]
            ),
            "output_tables_directory": Path(
                paths["output_tables_directory"]
            ),
            "report_path": Path(
                paths["report"]
            ),
            "evaluation_split": str(
                analysis["evaluation_split"]
            ),
            "floating_tolerance": float(
                analysis["floating_tolerance"]
            ),
            "top_n_zones": int(
                analysis["top_n_zones"]
            ),
            "compression": str(
                output["compression"]
            ),
        }

    except FinalModelComparisonError:
        raise

    except (
        OSError,
        KeyError,
        TypeError,
        ValueError,
        yaml.YAMLError,
    ) as error:
        raise FinalModelComparisonError(
            f"Invalid comparison config: {error}"
        ) from error

    for directory_key in (
        "raw_tables_directory",
        "champion_tables_directory",
    ):
        directory = config[
            directory_key
        ]

        if not directory.exists():
            raise FinalModelComparisonError(
                f"Input directory does not exist: {directory}"
            )

        if not directory.is_dir():
            raise FinalModelComparisonError(
                f"Input path is not a directory: {directory}"
            )

    if not config["dataset_name"]:
        raise FinalModelComparisonError(
            "Dataset name must not be empty."
        )

    if not config["evaluation_split"]:
        raise FinalModelComparisonError(
            "Evaluation split must not be empty."
        )

    if (
        not math.isfinite(
            config["floating_tolerance"]
        )
        or config["floating_tolerance"] < 0
    ):
        raise FinalModelComparisonError(
            "Floating tolerance must be finite and non-negative."
        )

    if config["top_n_zones"] <= 0:
        raise FinalModelComparisonError(
            "top_n_zones must be positive."
        )

    if not config["compression"]:
        raise FinalModelComparisonError(
            "Output compression must not be empty."
        )

    return config


def read_table(
    directory: Path,
    *,
    table_name: str,
    key_columns: tuple[str, ...],
) -> pl.DataFrame:
    """Read and validate one segmented metrics table."""
    path = directory / f"{table_name}.parquet"

    if not path.exists():
        raise FinalModelComparisonError(
            f"Metrics table does not exist: {path}"
        )

    if path.stat().st_size == 0:
        raise FinalModelComparisonError(
            f"Metrics table is empty: {path}"
        )

    try:
        frame = pl.read_parquet(
            path
        )

    except Exception as error:
        raise FinalModelComparisonError(
            f"Could not read metrics table {path}: {error}"
        ) from error

    required_columns = {
        *key_columns,
        "row_count",
        "actual_sum",
        "actual_mean",
        *(
            f"model_{metric}"
            for metric in MODEL_METRICS
        ),
        *(
            f"baseline_{metric}"
            for metric in MODEL_METRICS
        ),
    }

    missing_columns = sorted(
        required_columns.difference(
            frame.columns
        )
    )

    if missing_columns:
        raise FinalModelComparisonError(
            f"{path} is missing columns: "
            + ", ".join(
                missing_columns
            )
        )

    duplicate_key_count = (
        frame.group_by(
            *key_columns
        )
        .agg(
            pl.len().alias(
                "key_count"
            )
        )
        .filter(
            pl.col(
                "key_count"
            )
            > 1
        )
        .height
    )

    if duplicate_key_count:
        raise FinalModelComparisonError(
            f"{path} contains duplicate segment keys."
        )

    return frame


def finite_or_null_expression(
    column: str,
) -> pl.Expr:
    """Replace non-finite metric values with null."""
    finite_value = (
        pl.col(
            column
        )
        .cast(
            pl.Float64
        )
    )

    return (
        pl.when(
            finite_value.is_finite()
        )
        .then(
            finite_value
        )
        .otherwise(
            None
        )
        .alias(
            column
        )
    )


def prepare_raw_table(
    frame: pl.DataFrame,
    *,
    key_columns: tuple[str, ...],
) -> pl.DataFrame:
    """Prepare raw-XGBoost and lag-1 columns."""
    selected_columns = [
        *key_columns,
        "row_count",
        "actual_sum",
        "actual_mean",
        *(
            f"model_{metric}"
            for metric in MODEL_METRICS
        ),
        *(
            f"baseline_{metric}"
            for metric in MODEL_METRICS
        ),
    ]

    rename_mapping = {
        "row_count": "raw_row_count",
        "actual_sum": "raw_actual_sum",
        "actual_mean": "raw_actual_mean",
    }

    rename_mapping.update(
        {
            f"model_{metric}": f"raw_{metric}"
            for metric in MODEL_METRICS
        }
    )

    rename_mapping.update(
        {
            f"baseline_{metric}": f"lag1_{metric}"
            for metric in MODEL_METRICS
        }
    )

    return (
        frame.select(
            selected_columns
        )
        .rename(
            rename_mapping
        )
    )


def prepare_champion_table(
    frame: pl.DataFrame,
    *,
    key_columns: tuple[str, ...],
) -> pl.DataFrame:
    """Prepare operational-champion columns."""
    selected_columns = [
        *key_columns,
        "row_count",
        "actual_sum",
        "actual_mean",
        *(
            f"model_{metric}"
            for metric in MODEL_METRICS
        ),
        *(
            f"baseline_{metric}"
            for metric in MODEL_METRICS
        ),
    ]

    rename_mapping = {
        "row_count": "champion_row_count",
        "actual_sum": "champion_actual_sum",
        "actual_mean": "champion_actual_mean",
    }

    rename_mapping.update(
        {
            f"model_{metric}": f"champion_{metric}"
            for metric in MODEL_METRICS
        }
    )

    rename_mapping.update(
        {
            f"baseline_{metric}": (
                f"champion_lag1_{metric}"
            )
            for metric in MODEL_METRICS
        }
    )

    return (
        frame.select(
            selected_columns
        )
        .rename(
            rename_mapping
        )
    )


def relative_improvement_expression(
    *,
    reference_column: str,
    candidate_column: str,
    alias: str,
) -> pl.Expr:
    """Calculate a null-safe relative metric improvement."""
    reference = pl.col(
        reference_column
    )

    candidate = pl.col(
        candidate_column
    )

    return (
        pl.when(
            reference.is_not_null()
            & candidate.is_not_null()
            & (
                reference
                != 0
            )
        )
        .then(
            (
                reference
                - candidate
            )
            / reference.abs()
        )
        .otherwise(
            None
        )
        .alias(
            alias
        )
    )


def absolute_improvement_expression(
    *,
    reference_column: str,
    candidate_column: str,
    alias: str,
) -> pl.Expr:
    """Calculate a null-safe absolute metric improvement."""
    return (
        pl.when(
            pl.col(
                reference_column
            ).is_not_null()
            & pl.col(
                candidate_column
            ).is_not_null()
        )
        .then(
            pl.col(
                reference_column
            )
            - pl.col(
                candidate_column
            )
        )
        .otherwise(
            None
        )
        .alias(
            alias
        )
    )


def validate_joined_table(
    frame: pl.DataFrame,
    *,
    tolerance: float,
    table_name: str,
) -> dict[str, int]:
    """Validate equivalence of raw and champion segment inputs."""
    row_count_mismatch = (
        frame.filter(
            pl.col(
                "raw_row_count"
            )
            != pl.col(
                "champion_row_count"
            )
        )
        .height
    )

    actual_sum_mismatch = (
        frame.filter(
            (
                pl.col(
                    "raw_actual_sum"
                )
                - pl.col(
                    "champion_actual_sum"
                )
            )
            .abs()
            > tolerance
        )
        .height
    )

    actual_mean_mismatch = (
        frame.filter(
            (
                pl.col(
                    "raw_actual_mean"
                )
                - pl.col(
                    "champion_actual_mean"
                )
            )
            .abs()
            > tolerance
        )
        .height
    )

    lag1_mismatch = 0

    for metric in MODEL_METRICS:
        raw_column = (
            f"lag1_{metric}"
        )

        champion_column = (
            f"champion_lag1_{metric}"
        )

        raw_finite = (
            pl.col(
                raw_column
            )
            .cast(
                pl.Float64
            )
            .is_finite()
            .fill_null(
                False
            )
        )

        champion_finite = (
            pl.col(
                champion_column
            )
            .cast(
                pl.Float64
            )
            .is_finite()
            .fill_null(
                False
            )
        )

        finite_mismatch = (
            frame.filter(
                raw_finite
                & champion_finite
                & (
                    (
                        pl.col(
                            raw_column
                        )
                        .cast(
                            pl.Float64
                        )
                        - pl.col(
                            champion_column
                        )
                        .cast(
                            pl.Float64
                        )
                    )
                    .abs()
                    > tolerance
                )
            )
            .height
        )

        finite_state_mismatch = (
            frame.filter(
                raw_finite
                != champion_finite
            )
            .height
        )

        lag1_mismatch += (
            finite_mismatch
            + finite_state_mismatch
        )

    validation = {
        "row_count_mismatch_count": (
            row_count_mismatch
        ),
        "actual_sum_mismatch_count": (
            actual_sum_mismatch
        ),
        "actual_mean_mismatch_count": (
            actual_mean_mismatch
        ),
        "lag1_metric_mismatch_count": (
            lag1_mismatch
        ),
    }

    if any(
        validation.values()
    ):
        raise FinalModelComparisonError(
            f"Input mismatch in {table_name}: {validation}"
        )

    return validation


def build_comparison_table(
    raw_frame: pl.DataFrame,
    champion_frame: pl.DataFrame,
    *,
    key_columns: tuple[str, ...],
    tolerance: float,
    table_name: str,
) -> tuple[
    pl.DataFrame,
    dict[str, int],
]:
    """Build one lag-1/raw/champion comparison table."""
    prepared_raw = prepare_raw_table(
        raw_frame,
        key_columns=key_columns,
    )

    prepared_champion = prepare_champion_table(
        champion_frame,
        key_columns=key_columns,
    )

    joined = prepared_raw.join(
        prepared_champion,
        on=list(
            key_columns
        ),
        how="inner",
    )

    if (
        joined.height
        != prepared_raw.height
        or joined.height
        != prepared_champion.height
    ):
        raise FinalModelComparisonError(
            f"Segment-key join mismatch in {table_name}."
        )

    validation = validate_joined_table(
        joined,
        tolerance=tolerance,
        table_name=table_name,
    )

    output = (
        joined.with_columns(
            pl.col(
                "raw_row_count"
            )
            .alias(
                "row_count"
            ),

            pl.col(
                "raw_actual_sum"
            )
            .alias(
                "actual_sum"
            ),

            pl.col(
                "raw_actual_mean"
            )
            .alias(
                "actual_mean"
            ),
        )
        .drop(
            "raw_row_count",
            "champion_row_count",
            "raw_actual_sum",
            "champion_actual_sum",
            "raw_actual_mean",
            "champion_actual_mean",
            *(
                f"champion_lag1_{metric}"
                for metric in MODEL_METRICS
            ),
        )
    )

    finite_metric_columns: list[str] = []

    for model_name in (
        "lag1",
        "raw",
        "champion",
    ):
        for metric in MODEL_METRICS:
            finite_metric_columns.append(
                f"{model_name}_{metric}"
            )

    output = output.with_columns(
        *(
            finite_or_null_expression(
                column
            )
            for column in finite_metric_columns
        )
    )

    comparison_expressions: list[
        pl.Expr
    ] = []

    for metric in COMPARISON_METRICS:
        comparison_expressions.extend(
            [
                absolute_improvement_expression(
                    reference_column=(
                        f"lag1_{metric}"
                    ),
                    candidate_column=(
                        f"raw_{metric}"
                    ),
                    alias=(
                        f"raw_vs_lag1_"
                        f"{metric}_absolute_improvement"
                    ),
                ),

                relative_improvement_expression(
                    reference_column=(
                        f"lag1_{metric}"
                    ),
                    candidate_column=(
                        f"raw_{metric}"
                    ),
                    alias=(
                        f"raw_vs_lag1_"
                        f"{metric}_relative_improvement"
                    ),
                ),

                absolute_improvement_expression(
                    reference_column=(
                        f"lag1_{metric}"
                    ),
                    candidate_column=(
                        f"champion_{metric}"
                    ),
                    alias=(
                        f"champion_vs_lag1_"
                        f"{metric}_absolute_improvement"
                    ),
                ),

                relative_improvement_expression(
                    reference_column=(
                        f"lag1_{metric}"
                    ),
                    candidate_column=(
                        f"champion_{metric}"
                    ),
                    alias=(
                        f"champion_vs_lag1_"
                        f"{metric}_relative_improvement"
                    ),
                ),

                absolute_improvement_expression(
                    reference_column=(
                        f"raw_{metric}"
                    ),
                    candidate_column=(
                        f"champion_{metric}"
                    ),
                    alias=(
                        f"champion_vs_raw_"
                        f"{metric}_absolute_improvement"
                    ),
                ),

                relative_improvement_expression(
                    reference_column=(
                        f"raw_{metric}"
                    ),
                    candidate_column=(
                        f"champion_{metric}"
                    ),
                    alias=(
                        f"champion_vs_raw_"
                        f"{metric}_relative_improvement"
                    ),
                ),
            ]
        )

    output = (
        output.with_columns(
            comparison_expressions
        )
        .with_columns(
            pl.col(
                "raw_mean_bias"
            )
            .abs()
            .alias(
                "raw_absolute_bias"
            ),

            pl.col(
                "champion_mean_bias"
            )
            .abs()
            .alias(
                "champion_absolute_bias"
            ),

            (
                pl.col(
                    "champion_mean_bias"
                )
                - pl.col(
                    "raw_mean_bias"
                )
            )
            .alias(
                "champion_vs_raw_bias_change"
            ),

            (
                pl.col(
                    "champion_mae"
                )
                < pl.col(
                    "raw_mae"
                )
            )
            .alias(
                "champion_beats_raw_mae"
            ),

            (
                pl.col(
                    "champion_rmse"
                )
                < pl.col(
                    "raw_rmse"
                )
            )
            .alias(
                "champion_beats_raw_rmse"
            ),

            (
                pl.col(
                    "champion_smape"
                )
                < pl.col(
                    "raw_smape"
                )
            )
            .alias(
                "champion_beats_raw_smape"
            ),

            (
                pl.col(
                    "champion_mean_bias"
                )
                .abs()
                < pl.col(
                    "raw_mean_bias"
                )
                .abs()
            )
            .alias(
                "champion_reduces_absolute_bias"
            ),
        )
        .sort(
            *key_columns
        )
    )

    return (
        output,
        validation,
    )


def write_parquet_atomically(
    frame: pl.DataFrame,
    *,
    path: Path,
    compression: str,
) -> dict[str, Any]:
    """Write one comparison table atomically."""
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = path.with_name(
        f"{path.name}.part"
    )

    temporary_path.unlink(
        missing_ok=True
    )

    try:
        frame.write_parquet(
            temporary_path,
            compression=compression,
            statistics=True,
        )

        temporary_path.replace(
            path
        )

    except Exception as error:
        temporary_path.unlink(
            missing_ok=True
        )

        raise FinalModelComparisonError(
            f"Could not write comparison table {path}: {error}"
        ) from error

    return {
        "path": str(
            path
        ),
        "size_bytes": (
            path.stat().st_size
        ),
        "row_count": frame.height,
        "column_count": frame.width,
    }


def sanitize_json_value(
    value: Any,
) -> Any:
    """Replace non-finite floats with JSON null."""
    if isinstance(
        value,
        float,
    ):
        if not math.isfinite(
            value
        ):
            return None

        return value

    if isinstance(
        value,
        dict,
    ):
        return {
            str(key): sanitize_json_value(
                item
            )
            for key, item in value.items()
        }

    if isinstance(
        value,
        (
            list,
            tuple,
        ),
    ):
        return [
            sanitize_json_value(
                item
            )
            for item in value
        ]

    return value


def write_json_atomically(
    path: Path,
    payload: dict[str, Any],
) -> None:
    """Write standards-compliant JSON atomically."""
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = path.with_name(
        f"{path.name}.part"
    )

    temporary_path.unlink(
        missing_ok=True
    )

    sanitized_payload = sanitize_json_value(
        payload
    )

    try:
        temporary_path.write_text(
            json.dumps(
                sanitized_payload,
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )

        temporary_path.replace(
            path
        )

    except (
        OSError,
        TypeError,
        ValueError,
    ) as error:
        temporary_path.unlink(
            missing_ok=True
        )

        raise FinalModelComparisonError(
            f"Could not write final comparison report: {error}"
        ) from error


def table_summary(
    frame: pl.DataFrame,
    *,
    evaluation_split: str,
) -> dict[str, Any]:
    """Summarize champion trade-offs for one segmented table."""
    evaluation = frame.filter(
        pl.col(
            "split"
        )
        == evaluation_split
    )

    if evaluation.height == 0:
        raise FinalModelComparisonError(
            f"Evaluation split is absent: {evaluation_split}"
        )

    return {
        "segment_count": evaluation.height,
        "champion_beats_raw_mae_count": (
            evaluation.filter(
                pl.col(
                    "champion_beats_raw_mae"
                )
            )
            .height
        ),
        "champion_beats_raw_rmse_count": (
            evaluation.filter(
                pl.col(
                    "champion_beats_raw_rmse"
                )
            )
            .height
        ),
        "champion_beats_raw_smape_count": (
            evaluation.filter(
                pl.col(
                    "champion_beats_raw_smape"
                )
            )
            .height
        ),
        "champion_reduces_absolute_bias_count": (
            evaluation.filter(
                pl.col(
                    "champion_reduces_absolute_bias"
                )
            )
            .height
        ),
    }


def records(
    frame: pl.DataFrame,
    *,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Convert a frame to JSON-compatible records."""
    selected = (
        frame.head(
            limit
        )
        if limit is not None
        else frame
    )

    return selected.to_dicts()


def build_final_comparison(
    config: dict[str, Any],
) -> dict[str, Any]:
    """Build final lag-1/raw/champion comparison artifacts."""
    comparison_tables: dict[
        str,
        pl.DataFrame
    ] = {}

    artifact_summaries: dict[
        str,
        dict[str, Any]
    ] = {}

    validation_summaries: dict[
        str,
        dict[str, int]
    ] = {}

    segment_summaries: dict[
        str,
        dict[str, Any]
    ] = {}

    for table_name, key_columns in TABLE_SPECS.items():
        raw_table = read_table(
            config[
                "raw_tables_directory"
            ],
            table_name=table_name,
            key_columns=key_columns,
        )

        champion_table = read_table(
            config[
                "champion_tables_directory"
            ],
            table_name=table_name,
            key_columns=key_columns,
        )

        (
            comparison_table,
            validation,
        ) = build_comparison_table(
            raw_table,
            champion_table,
            key_columns=key_columns,
            tolerance=config[
                "floating_tolerance"
            ],
            table_name=table_name,
        )

        output_path = (
            config[
                "output_tables_directory"
            ]
            / f"{table_name}.parquet"
        )

        artifact_summaries[
            table_name
        ] = write_parquet_atomically(
            comparison_table,
            path=output_path,
            compression=config[
                "compression"
            ],
        )

        comparison_tables[
            table_name
        ] = comparison_table

        validation_summaries[
            table_name
        ] = validation

        segment_summaries[
            table_name
        ] = table_summary(
            comparison_table,
            evaluation_split=config[
                "evaluation_split"
            ],
        )

    overall = comparison_tables[
        "overall_metrics"
    ]

    evaluation_overall = overall.filter(
        pl.col(
            "split"
        )
        == config[
            "evaluation_split"
        ]
    )

    if evaluation_overall.height != 1:
        raise FinalModelComparisonError(
            "Expected exactly one overall evaluation record."
        )

    zone_metrics = (
        comparison_tables[
            "zone_metrics"
        ]
        .filter(
            pl.col(
                "split"
            )
            == config[
                "evaluation_split"
            ]
        )
    )

    best_champion_zone_improvements = (
        zone_metrics.filter(
            pl.col(
                "champion_vs_raw_mae_relative_improvement"
            )
            .is_not_null()
        )
        .sort(
            "champion_vs_raw_mae_relative_improvement",
            descending=True,
        )
        .select(
            "zone_id",
            "zone_name",
            "borough",
            "actual_sum",
            "actual_mean",
            "raw_mae",
            "champion_mae",
            "champion_vs_raw_mae_relative_improvement",
            "raw_mean_bias",
            "champion_mean_bias",
        )
        .head(
            config[
                "top_n_zones"
            ]
        )
    )

    worst_champion_zone_changes = (
        zone_metrics.filter(
            pl.col(
                "champion_vs_raw_mae_relative_improvement"
            )
            .is_not_null()
        )
        .sort(
            "champion_vs_raw_mae_relative_improvement"
        )
        .select(
            "zone_id",
            "zone_name",
            "borough",
            "actual_sum",
            "actual_mean",
            "raw_mae",
            "champion_mae",
            "champion_vs_raw_mae_relative_improvement",
            "raw_mean_bias",
            "champion_mean_bias",
        )
        .head(
            config[
                "top_n_zones"
            ]
        )
    )

    demand_band_metrics = (
        comparison_tables[
            "demand_band_metrics"
        ]
        .filter(
            pl.col(
                "split"
            )
            == config[
                "evaluation_split"
            ]
        )
        .sort(
            "actual_mean"
        )
    )

    peak_period_metrics = (
        comparison_tables[
            "peak_period_metrics"
        ]
        .filter(
            pl.col(
                "split"
            )
            == config[
                "evaluation_split"
            ]
        )
        .sort(
            "peak_period"
        )
    )

    borough_metrics = (
        comparison_tables[
            "borough_metrics"
        ]
        .filter(
            pl.col(
                "split"
            )
            == config[
                "evaluation_split"
            ]
        )
        .sort(
            "borough"
        )
    )

    all_validation_counts_zero = all(
        all(
            count == 0
            for count in validation.values()
        )
        for validation in validation_summaries.values()
    )

    report: dict[str, Any] = {
        "comparison_report_version": "1.0.0",
        "generated_at": datetime.now(
            UTC
        ).isoformat(),
        "dataset_name": config[
            "dataset_name"
        ],
        "models": {
            "baseline": "lag_1",
            "core_model": "raw_xgboost",
            "operational_champion": (
                "xgboost_with_constrained_"
                "low_demand_calibration"
            ),
        },
        "evaluation_split": config[
            "evaluation_split"
        ],
        "overall_metrics": records(
            overall
        ),
        "segment_summaries": (
            segment_summaries
        ),
        "key_findings": {
            "best_champion_zone_improvements": records(
                best_champion_zone_improvements
            ),
            "worst_champion_zone_changes": records(
                worst_champion_zone_changes
            ),
            "demand_band_metrics": records(
                demand_band_metrics
            ),
            "peak_period_metrics": records(
                peak_period_metrics
            ),
            "borough_metrics": records(
                borough_metrics
            ),
        },
        "artifacts": artifact_summaries,
        "input_validations": (
            validation_summaries
        ),
        "invariants": {
            "raw_and_champion_use_identical_segments": (
                all_validation_counts_zero
            ),
            "raw_and_champion_use_identical_actuals": (
                all_validation_counts_zero
            ),
            "raw_and_champion_use_identical_lag1_baseline": (
                all_validation_counts_zero
            ),
            "all_comparison_tables_created": (
                len(
                    artifact_summaries
                )
                == len(
                    TABLE_SPECS
                )
            ),
        },
    }

    write_json_atomically(
        config[
            "report_path"
        ],
        report,
    )

    return report


def main() -> None:
    """Build and print the final model comparison."""
    try:
        config = read_config(
            CONFIG_PATH
        )

        report = build_final_comparison(
            config
        )

    except FinalModelComparisonError as error:
        raise SystemExit(
            f"Final model comparison failed: {error}"
        ) from error

    evaluation_split = report[
        "evaluation_split"
    ]

    overall_records = report[
        "overall_metrics"
    ]

    overall = next(
        record
        for record in overall_records
        if record[
            "split"
        ]
        == evaluation_split
    )

    print(
        "Final forecasting model comparison completed"
    )

    print()

    print(
        "Evaluation split:",
        evaluation_split,
    )

    print()

    print(
        "MAE"
    )

    print(
        "Lag-1:",
        f"{overall['lag1_mae']:.6f}",
    )

    print(
        "Raw XGBoost:",
        f"{overall['raw_mae']:.6f}",
    )

    print(
        "Operational champion:",
        f"{overall['champion_mae']:.6f}",
    )

    print(
        "Champion vs lag-1:",
        (
            f"{overall['champion_vs_lag1_mae_relative_improvement'] * 100:.2f}%"
        ),
    )

    print(
        "Champion vs raw:",
        (
            f"{overall['champion_vs_raw_mae_relative_improvement'] * 100:.3f}%"
        ),
    )

    print()

    print(
        "RMSE"
    )

    print(
        "Raw XGBoost:",
        f"{overall['raw_rmse']:.6f}",
    )

    print(
        "Operational champion:",
        f"{overall['champion_rmse']:.6f}",
    )

    print(
        "Champion vs raw:",
        (
            f"{overall['champion_vs_raw_rmse_relative_improvement'] * 100:.3f}%"
        ),
    )

    print()

    print(
        "WAPE"
    )

    print(
        "Lag-1:",
        f"{overall['lag1_wape'] * 100:.2f}%",
    )

    print(
        "Raw XGBoost:",
        f"{overall['raw_wape'] * 100:.2f}%",
    )

    print(
        "Operational champion:",
        f"{overall['champion_wape'] * 100:.2f}%",
    )

    print()

    print(
        "SMAPE"
    )

    print(
        "Raw XGBoost:",
        f"{overall['raw_smape'] * 100:.2f}%",
    )

    print(
        "Operational champion:",
        f"{overall['champion_smape'] * 100:.2f}%",
    )

    print()

    print(
        "Mean bias"
    )

    print(
        "Raw XGBoost:",
        f"{overall['raw_mean_bias']:.6f}",
    )

    print(
        "Operational champion:",
        f"{overall['champion_mean_bias']:.6f}",
    )

    print()

    print(
        "Report:",
        config[
            "report_path"
        ],
    )

    print(
        "Tables:",
        config[
            "output_tables_directory"
        ],
    )


if __name__ == "__main__":
    main()