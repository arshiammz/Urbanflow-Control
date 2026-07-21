from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import polars as pl
import yaml


PREDICTION_COLUMNS = {
    "zero": "prediction_zero",
    "zone_historical_mean": (
        "prediction_zone_historical_mean"
    ),
    "lag_1": "prediction_lag_1",
    "seasonal_lag_96": (
        "prediction_seasonal_lag_96"
    ),
}

OUTPUT_COLUMNS = (
    "interval_start",
    "time_step",
    "service_date",
    "zone_id",
    "borough",
    "zone_name",
    "service_zone",
    "split",
    "fit_scope",
    "actual_pickup_count",
    *PREDICTION_COLUMNS.values(),
)


class BaselineEvaluationError(RuntimeError):
    """Raised when baseline evaluation cannot be completed."""


@dataclass(frozen=True)
class BaselineEvaluationSettings:
    """Configuration for forecasting baseline evaluation."""

    timezone: str

    input_path: Path
    predictions_path: Path
    report_path: Path

    target_column: str
    split_column: str
    zone_column: str
    lag_1_column: str
    seasonal_lag_column: str

    validation_fit_splits: tuple[str, ...]
    test_fit_splits: tuple[str, ...]

    compression: str

    @classmethod
    def from_yaml(
        cls,
        config_path: Path,
    ) -> BaselineEvaluationSettings:
        """Load baseline settings from YAML."""
        if not config_path.exists():
            raise BaselineEvaluationError(
                f"Baseline config does not exist: {config_path}"
            )

        try:
            raw_config = yaml.safe_load(
                config_path.read_text(
                    encoding="utf-8",
                )
            )
        except (OSError, yaml.YAMLError) as error:
            raise BaselineEvaluationError(
                f"Could not read baseline config: {error}"
            ) from error

        if not isinstance(raw_config, dict):
            raise BaselineEvaluationError(
                "Baseline config must contain a YAML mapping."
            )

        try:
            dataset_config = raw_config["dataset"]
            paths_config = raw_config["paths"]
            evaluation_config = raw_config[
                "evaluation"
            ]
            output_config = raw_config["output"]

            settings = cls(
                timezone=str(
                    dataset_config["timezone"]
                ),
                input_path=Path(
                    paths_config["input"]
                ),
                predictions_path=Path(
                    paths_config["predictions"]
                ),
                report_path=Path(
                    paths_config["report"]
                ),
                target_column=str(
                    evaluation_config[
                        "target_column"
                    ]
                ),
                split_column=str(
                    evaluation_config[
                        "split_column"
                    ]
                ),
                zone_column=str(
                    evaluation_config[
                        "zone_column"
                    ]
                ),
                lag_1_column=str(
                    evaluation_config[
                        "lag_1_column"
                    ]
                ),
                seasonal_lag_column=str(
                    evaluation_config[
                        "seasonal_lag_column"
                    ]
                ),
                validation_fit_splits=tuple(
                    str(value)
                    for value in evaluation_config[
                        "validation_fit_splits"
                    ]
                ),
                test_fit_splits=tuple(
                    str(value)
                    for value in evaluation_config[
                        "test_fit_splits"
                    ]
                ),
                compression=str(
                    output_config["compression"]
                ),
            )

        except (
            KeyError,
            TypeError,
            ValueError,
        ) as error:
            raise BaselineEvaluationError(
                f"Invalid baseline configuration: {error}"
            ) from error

        settings.validate()

        return settings

    @property
    def required_columns(self) -> set[str]:
        """Return columns required from the feature dataset."""
        return {
            "interval_start",
            "time_step",
            "service_date",
            "borough",
            "zone_name",
            "service_zone",
            self.zone_column,
            self.split_column,
            self.target_column,
            self.lag_1_column,
            self.seasonal_lag_column,
        }

    def validate(self) -> None:
        """Validate baseline configuration."""
        try:
            ZoneInfo(self.timezone)
        except ZoneInfoNotFoundError as error:
            raise BaselineEvaluationError(
                f"Unknown timezone: {self.timezone}"
            ) from error

        if not self.validation_fit_splits:
            raise BaselineEvaluationError(
                "validation_fit_splits must not be empty."
            )

        if not self.test_fit_splits:
            raise BaselineEvaluationError(
                "test_fit_splits must not be empty."
            )

        if "validation" in self.validation_fit_splits:
            raise BaselineEvaluationError(
                "Validation baselines cannot fit on "
                "the validation split."
            )

        if "test" in self.validation_fit_splits:
            raise BaselineEvaluationError(
                "Validation baselines cannot fit on test data."
            )

        if "test" in self.test_fit_splits:
            raise BaselineEvaluationError(
                "Test baselines cannot fit on test data."
            )

        if not self.compression:
            raise BaselineEvaluationError(
                "Output compression must not be empty."
            )

    def as_report_dict(self) -> dict[str, Any]:
        """Return effective configuration as JSON data."""
        return {
            "dataset": {
                "timezone": self.timezone,
            },
            "paths": {
                "input": str(self.input_path),
                "predictions": str(
                    self.predictions_path
                ),
                "report": str(self.report_path),
            },
            "evaluation": {
                "target_column": self.target_column,
                "split_column": self.split_column,
                "zone_column": self.zone_column,
                "lag_1_column": self.lag_1_column,
                "seasonal_lag_column": (
                    self.seasonal_lag_column
                ),
                "validation_fit_splits": list(
                    self.validation_fit_splits
                ),
                "test_fit_splits": list(
                    self.test_fit_splits
                ),
            },
            "output": {
                "compression": self.compression,
            },
        }


def validate_feature_input(
    settings: BaselineEvaluationSettings,
) -> dict[str, int]:
    """Validate the forecasting feature input."""
    if not settings.input_path.exists():
        raise BaselineEvaluationError(
            f"Feature dataset does not exist: {settings.input_path}"
        )

    if settings.input_path.stat().st_size == 0:
        raise BaselineEvaluationError(
            f"Feature dataset is empty: {settings.input_path}"
        )

    try:
        source = pl.scan_parquet(
            settings.input_path
        )
        schema = source.collect_schema()
    except Exception as error:
        raise BaselineEvaluationError(
            f"Could not read feature dataset: {error}"
        ) from error

    missing_columns = sorted(
        settings.required_columns.difference(
            schema.names()
        )
    )

    if missing_columns:
        raise BaselineEvaluationError(
            "Feature dataset is missing columns: "
            + ", ".join(missing_columns)
        )

    interval_dtype = schema["interval_start"]

    if settings.timezone not in str(
        interval_dtype
    ):
        raise BaselineEvaluationError(
            "interval_start does not use the configured "
            f"timezone {settings.timezone}: {interval_dtype}"
        )

    null_row = (
        source.select(
            [
                pl.col(column)
                .null_count()
                .alias(column)
                for column in sorted(
                    settings.required_columns
                )
            ]
        )
        .collect()
        .row(0)
    )

    null_value_count = sum(
        int(value)
        for value in null_row
    )

    if null_value_count:
        raise BaselineEvaluationError(
            "Feature dataset contains null values "
            "in required columns."
        )

    split_counts = (
        source.group_by(
            settings.split_column
        )
        .agg(
            pl.len().alias("row_count")
        )
        .collect()
    )

    counts = {
        str(row[settings.split_column]): int(
            row["row_count"]
        )
        for row in split_counts.to_dicts()
    }

    for required_split in (
        "train",
        "validation",
        "test",
    ):
        if counts.get(required_split, 0) <= 0:
            raise BaselineEvaluationError(
                f"Feature dataset has no {required_split} rows."
            )

    minimum_target = int(
        source.select(
            pl.col(settings.target_column)
            .min()
        )
        .collect()
        .item()
    )

    if minimum_target < 0:
        raise BaselineEvaluationError(
            "Feature dataset contains negative targets."
        )

    return {
        "row_count": sum(counts.values()),
        "train_rows": counts["train"],
        "validation_rows": counts[
            "validation"
        ],
        "test_rows": counts["test"],
        "null_value_count": null_value_count,
    }


def build_split_predictions(
    *,
    settings: BaselineEvaluationSettings,
    evaluation_split: str,
    fit_splits: tuple[str, ...],
) -> pl.LazyFrame:
    """Build baseline predictions for one evaluation split."""
    source = pl.scan_parquet(
        settings.input_path
    )

    fit_filter = pl.col(
        settings.split_column
    ).is_in(
        list(fit_splits)
    )

    global_mean = (
        source.filter(
            fit_filter
        )
        .select(
            pl.col(settings.target_column)
            .mean()
            .alias("global_mean")
        )
        .collect()
        .item()
    )

    if global_mean is None:
        raise BaselineEvaluationError(
            f"No fitting rows found for {evaluation_split}."
        )

    zone_means = (
        source.filter(
            fit_filter
        )
        .group_by(
            settings.zone_column
        )
        .agg(
            pl.col(settings.target_column)
            .mean()
            .alias(
                "zone_historical_mean"
            )
        )
    )

    evaluation = (
        source.filter(
            pl.col(settings.split_column)
            == evaluation_split
        )
        .select(
            "interval_start",
            "time_step",
            "service_date",
            pl.col(
                settings.zone_column
            ).alias("zone_id"),
            "borough",
            "zone_name",
            "service_zone",
            pl.col(
                settings.split_column
            ).alias("split"),
            pl.col(
                settings.target_column
            )
            .cast(pl.Float64)
            .alias(
                "actual_pickup_count"
            ),
            pl.col(
                settings.lag_1_column
            )
            .cast(pl.Float64)
            .alias(
                "prediction_lag_1"
            ),
            pl.col(
                settings.seasonal_lag_column
            )
            .cast(pl.Float64)
            .alias(
                "prediction_seasonal_lag_96"
            ),
        )
    )

    fit_scope = "+".join(
        fit_splits
    )

    return (
        evaluation.join(
            zone_means,
            left_on="zone_id",
            right_on=settings.zone_column,
            how="left",
        )
        .with_columns(
            pl.lit(
                fit_scope
            ).alias("fit_scope"),

            pl.lit(
                0.0,
                dtype=pl.Float64,
            ).alias(
                "prediction_zero"
            ),

            pl.coalesce(
                [
                    pl.col(
                        "zone_historical_mean"
                    ),
                    pl.lit(
                        float(global_mean),
                        dtype=pl.Float64,
                    ),
                ]
            )
            .cast(pl.Float64)
            .alias(
                "prediction_zone_historical_mean"
            ),
        )
        .select(
            *OUTPUT_COLUMNS
        )
    )


def build_prediction_frame(
    settings: BaselineEvaluationSettings,
) -> pl.LazyFrame:
    """Build validation and test predictions."""
    validation_predictions = (
        build_split_predictions(
            settings=settings,
            evaluation_split="validation",
            fit_splits=(
                settings.validation_fit_splits
            ),
        )
    )

    test_predictions = (
        build_split_predictions(
            settings=settings,
            evaluation_split="test",
            fit_splits=(
                settings.test_fit_splits
            ),
        )
    )

    return (
        pl.concat(
            [
                validation_predictions,
                test_predictions,
            ],
            how="vertical_relaxed",
        )
        .sort(
            "interval_start",
            "zone_id",
        )
    )


def calculate_baseline_metrics(
    *,
    predictions: pl.LazyFrame,
    split: str,
    baseline_name: str,
    prediction_column: str,
) -> dict[str, int | float]:
    """Calculate count-forecast metrics."""
    actual = pl.col(
        "actual_pickup_count"
    )

    prediction = pl.col(
        prediction_column
    )

    absolute_error = (
        prediction - actual
    ).abs()

    squared_error = (
        prediction - actual
    ).pow(2)

    smape_denominator = (
        prediction.abs()
        + actual.abs()
    )

    smape = (
        pl.when(
            smape_denominator > 0
        )
        .then(
            2.0
            * absolute_error
            / smape_denominator
        )
        .otherwise(0.0)
    )

    row = (
        predictions.filter(
            pl.col("split") == split
        )
        .select(
            pl.len().alias(
                "row_count"
            ),
            actual.sum().alias(
                "actual_sum"
            ),
            prediction.sum().alias(
                "prediction_sum"
            ),
            absolute_error.sum().alias(
                "absolute_error_sum"
            ),
            absolute_error.mean().alias(
                "mae"
            ),
            squared_error.mean().alias(
                "mse"
            ),
            (
                prediction - actual
            )
            .mean()
            .alias(
                "mean_bias"
            ),
            smape.mean().alias(
                "smape"
            ),
        )
        .collect()
        .row(
            0,
            named=True,
        )
    )

    row_count = int(
        row["row_count"]
    )

    if row_count == 0:
        raise BaselineEvaluationError(
            f"No rows found for split {split}."
        )

    actual_sum = float(
        row["actual_sum"]
    )

    absolute_error_sum = float(
        row["absolute_error_sum"]
    )

    wape = (
        absolute_error_sum
        / actual_sum
        if actual_sum > 0
        else 0.0
    )

    rmse = float(
        row["mse"]
    ) ** 0.5

    return {
        "baseline": baseline_name,
        "row_count": row_count,
        "actual_sum": int(
            actual_sum
        ),
        "prediction_sum": round(
            float(row["prediction_sum"]),
            6,
        ),
        "mae": round(
            float(row["mae"]),
            6,
        ),
        "rmse": round(
            rmse,
            6,
        ),
        "wape": round(
            wape,
            8,
        ),
        "smape": round(
            float(row["smape"]),
            8,
        ),
        "mean_bias": round(
            float(row["mean_bias"]),
            6,
        ),
    }


def collect_all_metrics(
    predictions_path: Path,
) -> dict[str, dict[str, dict[str, int | float]]]:
    """Calculate metrics for every split and baseline."""
    predictions = pl.scan_parquet(
        predictions_path
    )

    metrics: dict[
        str,
        dict[
            str,
            dict[str, int | float],
        ],
    ] = {}

    for split in (
        "validation",
        "test",
    ):
        split_metrics: dict[
            str,
            dict[str, int | float],
        ] = {}

        for (
            baseline_name,
            prediction_column,
        ) in PREDICTION_COLUMNS.items():
            split_metrics[
                baseline_name
            ] = calculate_baseline_metrics(
                predictions=predictions,
                split=split,
                baseline_name=baseline_name,
                prediction_column=(
                    prediction_column
                ),
            )

        metrics[split] = split_metrics

    return metrics


def collect_prediction_summary(
    predictions_path: Path,
) -> dict[str, Any]:
    """Collect prediction-output invariants."""
    predictions = pl.scan_parquet(
        predictions_path
    )

    schema = predictions.collect_schema()

    row = (
        predictions.select(
            pl.len().alias(
                "row_count"
            ),
            (
                pl.col("split")
                == "validation"
            )
            .sum()
            .alias(
                "validation_rows"
            ),
            (
                pl.col("split")
                == "test"
            )
            .sum()
            .alias(
                "test_rows"
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
        )
        .collect()
        .row(
            0,
            named=True,
        )
    )

    duplicate_key_count = int(
        predictions.group_by(
            "split",
            "interval_start",
            "zone_id",
        )
        .agg(
            pl.len().alias(
                "key_count"
            )
        )
        .filter(
            pl.col("key_count") > 1
        )
        .select(
            pl.len().alias(
                "duplicate_key_count"
            )
        )
        .collect()
        .item()
    )

    null_row = (
        predictions.select(
            [
                pl.col(column)
                .null_count()
                .alias(column)
                for column in schema.names()
            ]
        )
        .collect()
        .row(0)
    )

    null_value_count = sum(
        int(value)
        for value in null_row
    )

    negative_prediction_count = int(
        predictions.select(
            pl.sum_horizontal(
                [
                    (
                        pl.col(column) < 0
                    ).cast(pl.Int64)
                    for column in (
                        PREDICTION_COLUMNS.values()
                    )
                ]
            )
            .sum()
            .alias(
                "negative_prediction_count"
            )
        )
        .collect()
        .item()
    )

    return {
        "row_count": int(
            row["row_count"]
        ),
        "validation_rows": int(
            row["validation_rows"]
        ),
        "test_rows": int(
            row["test_rows"]
        ),
        "first_interval": row[
            "first_interval"
        ].isoformat(),
        "last_interval": row[
            "last_interval"
        ].isoformat(),
        "duplicate_key_count": (
            duplicate_key_count
        ),
        "null_value_count": (
            null_value_count
        ),
        "negative_prediction_count": (
            negative_prediction_count
        ),
    }


def rank_baselines(
    split_metrics: dict[
        str,
        dict[str, int | float],
    ],
) -> list[str]:
    """Rank baseline names from lowest to highest MAE."""
    return sorted(
        split_metrics,
        key=lambda baseline: float(
            split_metrics[baseline]["mae"]
        ),
    )


def write_baseline_report(
    *,
    settings: BaselineEvaluationSettings,
    input_summary: dict[str, int],
    prediction_summary: dict[str, Any],
    metrics: dict[
        str,
        dict[
            str,
            dict[str, int | float],
        ],
    ],
) -> dict[str, Any]:
    """Write baseline metrics and invariants."""
    validation_ranking = rank_baselines(
        metrics["validation"]
    )

    test_ranking = rank_baselines(
        metrics["test"]
    )

    report: dict[str, Any] = {
        "baseline_pipeline_version": "0.1.0",
        "generated_at": datetime.now(
            UTC
        ).isoformat(),
        "effective_config": (
            settings.as_report_dict()
        ),
        "input": {
            "path": str(settings.input_path),
            "size_bytes": (
                settings.input_path.stat().st_size
            ),
            **input_summary,
        },
        "predictions": {
            "path": str(
                settings.predictions_path
            ),
            "size_bytes": (
                settings.predictions_path
                .stat()
                .st_size
            ),
            **prediction_summary,
        },
        "metrics": metrics,
        "ranking_by_mae": {
            "validation": validation_ranking,
            "test": test_ranking,
        },
        "best_baseline_by_mae": {
            "validation": (
                validation_ranking[0]
            ),
            "test": test_ranking[0],
        },
        "invariants": {
            "validation_uses_train_only": True,
            "test_uses_train_and_validation": True,
            "duplicate_key_count": (
                prediction_summary[
                    "duplicate_key_count"
                ]
            ),
            "null_value_count": (
                prediction_summary[
                    "null_value_count"
                ]
            ),
            "negative_prediction_count": (
                prediction_summary[
                    "negative_prediction_count"
                ]
            ),
        },
    }

    settings.report_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    settings.report_path.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    return report


def evaluate_forecasting_baselines(
    config_path: Path,
) -> dict[str, Any]:
    """Generate and evaluate forecasting baselines."""
    settings = (
        BaselineEvaluationSettings.from_yaml(
            config_path
        )
    )

    input_summary = validate_feature_input(
        settings
    )

    predictions = build_prediction_frame(
        settings
    )

    settings.predictions_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_output = (
        settings.predictions_path.with_name(
            f"{settings.predictions_path.name}.part"
        )
    )

    temporary_output.unlink(
        missing_ok=True
    )

    try:
        predictions.sink_parquet(
            temporary_output,
            compression=settings.compression,
            statistics=True,
        )

        temporary_output.replace(
            settings.predictions_path
        )

    except Exception as error:
        temporary_output.unlink(
            missing_ok=True
        )

        raise BaselineEvaluationError(
            "Could not write baseline predictions: "
            f"{error}"
        ) from error

    prediction_summary = (
        collect_prediction_summary(
            settings.predictions_path
        )
    )

    expected_prediction_rows = (
        input_summary["validation_rows"]
        + input_summary["test_rows"]
    )

    if (
        prediction_summary["row_count"]
        != expected_prediction_rows
    ):
        raise BaselineEvaluationError(
            "Unexpected prediction row count: "
            f"expected {expected_prediction_rows:,}, "
            f"received "
            f"{prediction_summary['row_count']:,}"
        )

    if prediction_summary[
        "duplicate_key_count"
    ]:
        raise BaselineEvaluationError(
            "Baseline predictions contain duplicate keys."
        )

    if prediction_summary[
        "null_value_count"
    ]:
        raise BaselineEvaluationError(
            "Baseline predictions contain null values."
        )

    if prediction_summary[
        "negative_prediction_count"
    ]:
        raise BaselineEvaluationError(
            "Baseline predictions contain negative values."
        )

    metrics = collect_all_metrics(
        settings.predictions_path
    )

    return write_baseline_report(
        settings=settings,
        input_summary=input_summary,
        prediction_summary=(
            prediction_summary
        ),
        metrics=metrics,
    )