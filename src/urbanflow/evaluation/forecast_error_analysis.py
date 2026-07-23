from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import polars as pl
import yaml


class ForecastErrorAnalysisError(RuntimeError):
    """Raised when forecast error analysis cannot be completed."""


@dataclass(frozen=True)
class ForecastErrorAnalysisSettings:
    """Configuration for segmented forecast error analysis."""

    dataset_name: str
    timezone: str

    features_path: Path
    predictions_path: Path
    report_path: Path
    tables_directory: Path

    key_columns: tuple[str, ...]
    split_column: str
    target_column: str
    prediction_column: str
    baseline_column: str
    metadata_columns: tuple[str, ...]

    analysis_splits: tuple[str, ...]
    top_n_worst_predictions: int
    top_n_zones: int
    minimum_zone_rows: int
    demand_band_upper_bounds: tuple[int, ...]

    morning_peak_start: int
    morning_peak_end_exclusive: int
    evening_peak_start: int
    evening_peak_end_exclusive: int

    compression: str

    @classmethod
    def from_yaml(
        cls,
        config_path: Path,
    ) -> ForecastErrorAnalysisSettings:
        """Load error-analysis settings from YAML."""
        if not config_path.exists():
            raise ForecastErrorAnalysisError(
                f"Error-analysis config does not exist: {config_path}"
            )

        try:
            raw_config = yaml.safe_load(
                config_path.read_text(
                    encoding="utf-8",
                )
            )
        except (OSError, yaml.YAMLError) as error:
            raise ForecastErrorAnalysisError(
                f"Could not read error-analysis config: {error}"
            ) from error

        if not isinstance(raw_config, dict):
            raise ForecastErrorAnalysisError(
                "Error-analysis config must contain a YAML mapping."
            )

        try:
            dataset_config = raw_config["dataset"]
            paths_config = raw_config["paths"]
            schema_config = raw_config["schema"]
            analysis_config = raw_config["analysis"]
            output_config = raw_config["output"]

            morning_peak = analysis_config["morning_peak"]
            evening_peak = analysis_config["evening_peak"]

            settings = cls(
                dataset_name=str(
                    dataset_config["name"]
                ),
                timezone=str(
                    dataset_config["timezone"]
                ),
                features_path=Path(
                    paths_config["features"]
                ),
                predictions_path=Path(
                    paths_config["predictions"]
                ),
                report_path=Path(
                    paths_config["report"]
                ),
                tables_directory=Path(
                    paths_config["tables_directory"]
                ),
                key_columns=tuple(
                    str(column)
                    for column in schema_config["key_columns"]
                ),
                split_column=str(
                    schema_config["split_column"]
                ),
                target_column=str(
                    schema_config["target_column"]
                ),
                prediction_column=str(
                    schema_config["prediction_column"]
                ),
                baseline_column=str(
                    schema_config["baseline_column"]
                ),
                metadata_columns=tuple(
                    str(column)
                    for column in schema_config[
                        "metadata_columns"
                    ]
                ),
                analysis_splits=tuple(
                    str(split)
                    for split in analysis_config["splits"]
                ),
                top_n_worst_predictions=int(
                    analysis_config[
                        "top_n_worst_predictions"
                    ]
                ),
                top_n_zones=int(
                    analysis_config["top_n_zones"]
                ),
                minimum_zone_rows=int(
                    analysis_config["minimum_zone_rows"]
                ),
                demand_band_upper_bounds=tuple(
                    int(value)
                    for value in analysis_config[
                        "demand_band_upper_bounds"
                    ]
                ),
                morning_peak_start=int(
                    morning_peak["start_hour"]
                ),
                morning_peak_end_exclusive=int(
                    morning_peak["end_hour_exclusive"]
                ),
                evening_peak_start=int(
                    evening_peak["start_hour"]
                ),
                evening_peak_end_exclusive=int(
                    evening_peak["end_hour_exclusive"]
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
            raise ForecastErrorAnalysisError(
                f"Invalid error-analysis configuration: {error}"
            ) from error

        settings.validate()

        return settings

    def validate(self) -> None:
        """Validate all error-analysis settings."""
        if not self.dataset_name:
            raise ForecastErrorAnalysisError(
                "Dataset name must not be empty."
            )

        try:
            ZoneInfo(self.timezone)
        except ZoneInfoNotFoundError as error:
            raise ForecastErrorAnalysisError(
                f"Unknown timezone: {self.timezone}"
            ) from error

        for path, description in (
            (
                self.features_path,
                "feature dataset",
            ),
            (
                self.predictions_path,
                "prediction dataset",
            ),
        ):
            if not path.exists():
                raise ForecastErrorAnalysisError(
                    f"{description.capitalize()} does not exist: {path}"
                )

            if path.stat().st_size == 0:
                raise ForecastErrorAnalysisError(
                    f"{description.capitalize()} is empty: {path}"
                )

        if not self.key_columns:
            raise ForecastErrorAnalysisError(
                "At least one key column must be configured."
            )

        if len(
            self.key_columns
        ) != len(
            set(self.key_columns)
        ):
            raise ForecastErrorAnalysisError(
                "Key columns must be unique."
            )

        if not self.metadata_columns:
            raise ForecastErrorAnalysisError(
                "At least one metadata column must be configured."
            )

        if len(
            self.metadata_columns
        ) != len(
            set(self.metadata_columns)
        ):
            raise ForecastErrorAnalysisError(
                "Metadata columns must be unique."
            )

        if not self.analysis_splits:
            raise ForecastErrorAnalysisError(
                "At least one analysis split must be configured."
            )

        if len(
            self.analysis_splits
        ) != len(
            set(self.analysis_splits)
        ):
            raise ForecastErrorAnalysisError(
                "Analysis splits must be unique."
            )

        if self.top_n_worst_predictions <= 0:
            raise ForecastErrorAnalysisError(
                "top_n_worst_predictions must be positive."
            )

        if self.top_n_zones <= 0:
            raise ForecastErrorAnalysisError(
                "top_n_zones must be positive."
            )

        if self.minimum_zone_rows <= 0:
            raise ForecastErrorAnalysisError(
                "minimum_zone_rows must be positive."
            )

        if not self.demand_band_upper_bounds:
            raise ForecastErrorAnalysisError(
                "At least one demand-band upper bound is required."
            )

        if any(
            value < 0
            for value in self.demand_band_upper_bounds
        ):
            raise ForecastErrorAnalysisError(
                "Demand-band upper bounds must be non-negative."
            )

        if tuple(
            sorted(
                set(
                    self.demand_band_upper_bounds
                )
            )
        ) != self.demand_band_upper_bounds:
            raise ForecastErrorAnalysisError(
                "Demand-band upper bounds must be sorted and unique."
            )

        for name, value in (
            (
                "morning_peak_start",
                self.morning_peak_start,
            ),
            (
                "morning_peak_end_exclusive",
                self.morning_peak_end_exclusive,
            ),
            (
                "evening_peak_start",
                self.evening_peak_start,
            ),
            (
                "evening_peak_end_exclusive",
                self.evening_peak_end_exclusive,
            ),
        ):
            if not 0 <= value <= 24:
                raise ForecastErrorAnalysisError(
                    f"{name} must be between 0 and 24."
                )

        if not (
            self.morning_peak_start
            < self.morning_peak_end_exclusive
        ):
            raise ForecastErrorAnalysisError(
                "Morning peak start must be before its end."
            )

        if not (
            self.evening_peak_start
            < self.evening_peak_end_exclusive
        ):
            raise ForecastErrorAnalysisError(
                "Evening peak start must be before its end."
            )

        if not self.compression:
            raise ForecastErrorAnalysisError(
                "Output compression must not be empty."
            )

    @property
    def required_feature_columns(
        self,
    ) -> tuple[str, ...]:
        """Return columns required from the feature dataset."""
        return ordered_unique(
            (
                *self.key_columns,
                self.split_column,
                self.target_column,
                self.baseline_column,
                *self.metadata_columns,
            )
        )

    @property
    def required_prediction_columns(
        self,
    ) -> tuple[str, ...]:
        """Return columns required from the prediction dataset."""
        return ordered_unique(
            (
                *self.key_columns,
                self.split_column,
                self.target_column,
                self.prediction_column,
            )
        )

    def as_report_dict(
        self,
    ) -> dict[str, Any]:
        """Return effective configuration as JSON-compatible data."""
        return {
            "dataset": {
                "name": self.dataset_name,
                "timezone": self.timezone,
            },
            "paths": {
                "features": str(
                    self.features_path
                ),
                "predictions": str(
                    self.predictions_path
                ),
                "report": str(
                    self.report_path
                ),
                "tables_directory": str(
                    self.tables_directory
                ),
            },
            "schema": {
                "key_columns": list(
                    self.key_columns
                ),
                "split_column": self.split_column,
                "target_column": self.target_column,
                "prediction_column": (
                    self.prediction_column
                ),
                "baseline_column": (
                    self.baseline_column
                ),
                "metadata_columns": list(
                    self.metadata_columns
                ),
            },
            "analysis": {
                "splits": list(
                    self.analysis_splits
                ),
                "top_n_worst_predictions": (
                    self.top_n_worst_predictions
                ),
                "top_n_zones": self.top_n_zones,
                "minimum_zone_rows": (
                    self.minimum_zone_rows
                ),
                "demand_band_upper_bounds": list(
                    self.demand_band_upper_bounds
                ),
                "morning_peak": {
                    "start_hour": (
                        self.morning_peak_start
                    ),
                    "end_hour_exclusive": (
                        self.morning_peak_end_exclusive
                    ),
                },
                "evening_peak": {
                    "start_hour": (
                        self.evening_peak_start
                    ),
                    "end_hour_exclusive": (
                        self.evening_peak_end_exclusive
                    ),
                },
            },
            "output": {
                "compression": self.compression,
            },
        }


def ordered_unique(
    values: tuple[str, ...],
) -> tuple[str, ...]:
    """Remove duplicates while preserving order."""
    return tuple(
        dict.fromkeys(
            values
        )
    )


def validate_source_schema(
    path: Path,
    *,
    required_columns: tuple[str, ...],
    timezone: str,
    interval_column: str,
    description: str,
) -> dict[str, Any]:
    """Validate one Parquet source before analysis."""
    try:
        source = pl.scan_parquet(
            path
        )

        schema = source.collect_schema()

    except Exception as error:
        raise ForecastErrorAnalysisError(
            f"Could not read {description}: {error}"
        ) from error

    missing_columns = sorted(
        set(
            required_columns
        ).difference(
            schema.names()
        )
    )

    if missing_columns:
        raise ForecastErrorAnalysisError(
            f"{description.capitalize()} is missing columns: "
            + ", ".join(
                missing_columns
            )
        )

    interval_dtype = schema[
        interval_column
    ]

    if timezone not in str(
        interval_dtype
    ):
        raise ForecastErrorAnalysisError(
            f"{description.capitalize()} interval column "
            f"does not use timezone {timezone}: "
            f"{interval_dtype}"
        )

    summary = (
        source.select(
            pl.len().alias(
                "row_count"
            ),
            pl.col(interval_column)
            .min()
            .alias(
                "first_interval"
            ),
            pl.col(interval_column)
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

    return {
        "path": str(
            path
        ),
        "size_bytes": (
            path.stat().st_size
        ),
        "row_count": int(
            summary["row_count"]
        ),
        "first_interval": (
            summary["first_interval"]
            .isoformat()
        ),
        "last_interval": (
            summary["last_interval"]
            .isoformat()
        ),
    }


def demand_band_expression(
    *,
    target_column: str,
    upper_bounds: tuple[int, ...],
) -> pl.Expr:
    """Build deterministic target-demand bands."""
    expression: pl.Expr | None = None
    previous_bound: int | None = None

    for upper_bound in upper_bounds:
        if previous_bound is None:
            if upper_bound == 0:
                label = "zero"
            else:
                label = (
                    f"0-{upper_bound:02d}"
                )
        else:
            label = (
                f"{previous_bound + 1:02d}-"
                f"{upper_bound:02d}"
            )

        condition = (
            pl.col(
                target_column
            )
            <= upper_bound
        )

        if expression is None:
            expression = (
                pl.when(
                    condition
                )
                .then(
                    pl.lit(
                        label
                    )
                )
            )
        else:
            expression = (
                expression
                .when(
                    condition
                )
                .then(
                    pl.lit(
                        label
                    )
                )
            )

        previous_bound = upper_bound

    if expression is None:
        raise ForecastErrorAnalysisError(
            "Could not build demand-band expression."
        )

    final_label = (
        f">{upper_bounds[-1]:02d}"
    )

    return (
        expression
        .otherwise(
            pl.lit(
                final_label
            )
        )
        .alias(
            "demand_band"
        )
    )


def load_analysis_frame(
    settings: ForecastErrorAnalysisSettings,
) -> tuple[
    pl.DataFrame,
    dict[str, Any],
]:
    """Join model predictions with feature metadata and baseline values."""
    feature_summary = validate_source_schema(
        settings.features_path,
        required_columns=(
            settings.required_feature_columns
        ),
        timezone=settings.timezone,
        interval_column=settings.key_columns[0],
        description="feature dataset",
    )

    prediction_summary = validate_source_schema(
        settings.predictions_path,
        required_columns=(
            settings.required_prediction_columns
        ),
        timezone=settings.timezone,
        interval_column=settings.key_columns[0],
        description="prediction dataset",
    )

    try:
        prediction_source = (
            pl.scan_parquet(
                settings.predictions_path
            )
            .filter(
                pl.col(
                    settings.split_column
                ).is_in(
                    list(
                        settings.analysis_splits
                    )
                )
            )
            .select(
                *settings.required_prediction_columns
            )
            .rename(
                {
                    settings.split_column: (
                        "prediction_split"
                    ),
                    settings.target_column: (
                        "prediction_target"
                    ),
                    settings.prediction_column: (
                        "model_prediction"
                    ),
                }
            )
        )

        feature_source = (
            pl.scan_parquet(
                settings.features_path
            )
            .filter(
                pl.col(
                    settings.split_column
                ).is_in(
                    list(
                        settings.analysis_splits
                    )
                )
            )
            .select(
                *settings.required_feature_columns
            )
        )

        frame = (
            feature_source.join(
                prediction_source,
                on=list(
                    settings.key_columns
                ),
                how="inner",
            )
            .collect()
            .sort(
                *settings.key_columns
            )
        )

    except Exception as error:
        raise ForecastErrorAnalysisError(
            f"Could not join features and predictions: {error}"
        ) from error

    if frame.height == 0:
        raise ForecastErrorAnalysisError(
            "Joined error-analysis dataset is empty."
        )

    expected_prediction_rows = int(
        pl.scan_parquet(
            settings.predictions_path
        )
        .filter(
            pl.col(
                settings.split_column
            ).is_in(
                list(
                    settings.analysis_splits
                )
            )
        )
        .select(
            pl.len()
        )
        .collect()
        .item()
    )

    if frame.height != expected_prediction_rows:
        raise ForecastErrorAnalysisError(
            "Joined row count does not match prediction row count: "
            f"expected {expected_prediction_rows:,}, "
            f"received {frame.height:,}."
        )

    duplicate_key_count = (
        frame.group_by(
            *settings.key_columns
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

    null_value_count = int(
        sum(
            int(value)
            for value in (
                frame
                .null_count()
                .row(0)
            )
        )
    )

    split_mismatch_count = (
        frame.filter(
            pl.col(
                settings.split_column
            )
            != pl.col(
                "prediction_split"
            )
        )
        .height
    )

    target_mismatch_count = (
        frame.filter(
            pl.col(
                settings.target_column
            )
            != pl.col(
                "prediction_target"
            )
        )
        .height
    )

    negative_prediction_count = (
        frame.filter(
            pl.col(
                "model_prediction"
            )
            < 0
        )
        .height
    )

    if duplicate_key_count:
        raise ForecastErrorAnalysisError(
            "Joined dataset contains duplicate keys."
        )

    if null_value_count:
        raise ForecastErrorAnalysisError(
            "Joined dataset contains null values."
        )

    if split_mismatch_count:
        raise ForecastErrorAnalysisError(
            "Feature and prediction split labels do not match."
        )

    if target_mismatch_count:
        raise ForecastErrorAnalysisError(
            "Feature and prediction targets do not match."
        )

    if negative_prediction_count:
        raise ForecastErrorAnalysisError(
            "Prediction dataset contains negative predictions."
        )

    frame = (
        frame.with_columns(
            pl.col(
                settings.target_column
            )
            .cast(
                pl.Float64
            ),

            pl.col(
                "model_prediction"
            )
            .cast(
                pl.Float64
            ),

            pl.col(
                settings.baseline_column
            )
            .cast(
                pl.Float64
            )
            .alias(
                "baseline_prediction"
            ),
        )
        .with_columns(
            (
                pl.col(
                    "model_prediction"
                )
                - pl.col(
                    settings.target_column
                )
            )
            .alias(
                "model_residual"
            ),

            (
                pl.col(
                    "baseline_prediction"
                )
                - pl.col(
                    settings.target_column
                )
            )
            .alias(
                "baseline_residual"
            ),

            demand_band_expression(
                target_column=(
                    settings.target_column
                ),
                upper_bounds=(
                    settings.demand_band_upper_bounds
                ),
            ),

            pl.when(
                (
                    pl.col(
                        "hour"
                    )
                    >= settings.morning_peak_start
                )
                & (
                    pl.col(
                        "hour"
                    )
                    < settings.morning_peak_end_exclusive
                )
            )
            .then(
                pl.lit(
                    "morning_peak"
                )
            )
            .when(
                (
                    pl.col(
                        "hour"
                    )
                    >= settings.evening_peak_start
                )
                & (
                    pl.col(
                        "hour"
                    )
                    < settings.evening_peak_end_exclusive
                )
            )
            .then(
                pl.lit(
                    "evening_peak"
                )
            )
            .otherwise(
                pl.lit(
                    "off_peak"
                )
            )
            .alias(
                "peak_period"
            ),
        )
        .with_columns(
            pl.col(
                "model_residual"
            )
            .abs()
            .alias(
                "model_absolute_error"
            ),

            pl.col(
                "baseline_residual"
            )
            .abs()
            .alias(
                "baseline_absolute_error"
            ),
        )
    )

    join_summary = {
        "row_count": frame.height,
        "duplicate_key_count": (
            duplicate_key_count
        ),
        "null_value_count": (
            null_value_count
        ),
        "split_mismatch_count": (
            split_mismatch_count
        ),
        "target_mismatch_count": (
            target_mismatch_count
        ),
        "negative_prediction_count": (
            negative_prediction_count
        ),
    }

    return frame, {
        "features": feature_summary,
        "predictions": prediction_summary,
        "joined": join_summary,
    }


def metric_expressions(
    *,
    target_column: str,
    prediction_column: str,
    prefix: str,
) -> list[pl.Expr]:
    """Return grouped regression metric expressions."""
    residual = (
        pl.col(
            prediction_column
        )
        - pl.col(
            target_column
        )
    )

    absolute_error = (
        residual.abs()
    )

    denominator = (
        pl.col(
            target_column
        )
        .abs()
        + pl.col(
            prediction_column
        )
        .abs()
    )

    smape_term = (
        pl.when(
            denominator
            > 0
        )
        .then(
            (
                2.0
                * absolute_error
            )
            / denominator
        )
        .otherwise(
            0.0
        )
    )

    return [
        pl.col(
            prediction_column
        )
        .sum()
        .alias(
            f"{prefix}_prediction_sum"
        ),

        absolute_error
        .mean()
        .alias(
            f"{prefix}_mae"
        ),

        (
            residual
            .pow(2)
            .mean()
            .sqrt()
        )
        .alias(
            f"{prefix}_rmse"
        ),

        (
            absolute_error
            .sum()
            / pl.col(
                target_column
            )
            .abs()
            .sum()
        )
        .alias(
            f"{prefix}_wape"
        ),

        smape_term
        .mean()
        .alias(
            f"{prefix}_smape"
        ),

        residual
        .mean()
        .alias(
            f"{prefix}_mean_bias"
        ),
    ]


def add_improvement_columns(
    frame: pl.DataFrame,
) -> pl.DataFrame:
    """Add absolute and relative baseline improvement columns."""
    expressions: list[pl.Expr] = []

    for metric in (
        "mae",
        "rmse",
        "wape",
        "smape",
    ):
        model_column = (
            f"model_{metric}"
        )

        baseline_column = (
            f"baseline_{metric}"
        )

        absolute_column = (
            f"{metric}_absolute_improvement"
        )

        relative_column = (
            f"{metric}_relative_improvement"
        )

        expressions.extend(
            [
                (
                    pl.col(
                        baseline_column
                    )
                    - pl.col(
                        model_column
                    )
                )
                .alias(
                    absolute_column
                ),

                pl.when(
                    pl.col(
                        baseline_column
                    )
                    > 0
                )
                .then(
                    (
                        pl.col(
                            baseline_column
                        )
                        - pl.col(
                            model_column
                        )
                    )
                    / pl.col(
                        baseline_column
                    )
                )
                .otherwise(
                    0.0
                )
                .alias(
                    relative_column
                ),
            ]
        )

    expressions.append(
        (
            pl.col(
                "model_mae"
            )
            < pl.col(
                "baseline_mae"
            )
        )
        .alias(
            "model_beats_baseline"
        )
    )

    return frame.with_columns(
        expressions
    )


def aggregate_metrics(
    frame: pl.DataFrame,
    *,
    settings: ForecastErrorAnalysisSettings,
    group_columns: tuple[str, ...],
) -> pl.DataFrame:
    """Aggregate model and baseline metrics for one segmentation."""
    aggregation = (
        frame.group_by(
            *group_columns
        )
        .agg(
            pl.len().alias(
                "row_count"
            ),

            pl.col(
                settings.target_column
            )
            .sum()
            .alias(
                "actual_sum"
            ),

            pl.col(
                settings.target_column
            )
            .mean()
            .alias(
                "actual_mean"
            ),

            *metric_expressions(
                target_column=(
                    settings.target_column
                ),
                prediction_column=(
                    "model_prediction"
                ),
                prefix="model",
            ),

            *metric_expressions(
                target_column=(
                    settings.target_column
                ),
                prediction_column=(
                    "baseline_prediction"
                ),
                prefix="baseline",
            ),
        )
        .sort(
            *group_columns
        )
    )

    return add_improvement_columns(
        aggregation
    )


def build_worst_predictions(
    frame: pl.DataFrame,
    *,
    settings: ForecastErrorAnalysisSettings,
) -> pl.DataFrame:
    """Return the largest model errors from each analysis split."""
    split_frames: list[
        pl.DataFrame
    ] = []

    selected_columns = ordered_unique(
        (
            *settings.key_columns,
            settings.split_column,
            "zone_name",
            "borough",
            "service_zone",
            "day_of_week",
            "hour",
            "minute",
            "peak_period",
            "demand_band",
            settings.target_column,
            "model_prediction",
            "baseline_prediction",
            "model_residual",
            "model_absolute_error",
            "baseline_absolute_error",
        )
    )

    for split in settings.analysis_splits:
        split_frame = (
            frame.filter(
                pl.col(
                    settings.split_column
                )
                == split
            )
            .sort(
                "model_absolute_error",
                descending=True,
            )
            .head(
                settings.top_n_worst_predictions
            )
            .select(
                *selected_columns
            )
        )

        split_frames.append(
            split_frame
        )

    return (
        pl.concat(
            split_frames,
            how="vertical",
        )
        .sort(
            settings.split_column,
            "model_absolute_error",
            descending=[
                False,
                True,
            ],
        )
    )


def parquet_path(
    settings: ForecastErrorAnalysisSettings,
    *,
    name: str,
) -> Path:
    """Return the output path for one analysis table."""
    return (
        settings.tables_directory
        / f"{name}.parquet"
    )


def write_parquet_atomically(
    frame: pl.DataFrame,
    *,
    path: Path,
    compression: str,
) -> dict[str, Any]:
    """Write one Parquet analysis table atomically."""
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

        raise ForecastErrorAnalysisError(
            f"Could not write analysis table {path}: {error}"
        ) from error

    return {
        "path": str(
            path
        ),
        "size_bytes": (
            path.stat().st_size
        ),
        "row_count": (
            frame.height
        ),
        "column_count": (
            frame.width
        ),
    }


def write_json_atomically(
    path: Path,
    payload: dict[str, Any],
) -> None:
    """Write one JSON report atomically."""
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
        temporary_path.write_text(
            json.dumps(
                payload,
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )

        temporary_path.replace(
            path
        )

    except OSError as error:
        temporary_path.unlink(
            missing_ok=True
        )

        raise ForecastErrorAnalysisError(
            f"Could not write error-analysis report: {error}"
        ) from error


def frame_to_records(
    frame: pl.DataFrame,
    *,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Convert a DataFrame to JSON-compatible records."""
    selected = (
        frame.head(
            limit
        )
        if limit is not None
        else frame
    )

    return selected.to_dicts()


def analyze_forecast_errors(
    config_path: Path,
) -> dict[str, Any]:
    """Run segmented XGBoost forecast error analysis."""
    settings = (
        ForecastErrorAnalysisSettings
        .from_yaml(
            config_path
        )
    )

    frame, input_summary = (
        load_analysis_frame(
            settings
        )
    )

    overall_metrics = aggregate_metrics(
        frame,
        settings=settings,
        group_columns=(
            settings.split_column,
        ),
    )

    zone_metrics = aggregate_metrics(
        frame,
        settings=settings,
        group_columns=(
            settings.split_column,
            "zone_id",
            "zone_name",
            "borough",
            "service_zone",
        ),
    )

    borough_metrics = aggregate_metrics(
        frame,
        settings=settings,
        group_columns=(
            settings.split_column,
            "borough",
        ),
    )

    service_zone_metrics = aggregate_metrics(
        frame,
        settings=settings,
        group_columns=(
            settings.split_column,
            "service_zone",
        ),
    )

    hour_metrics = aggregate_metrics(
        frame,
        settings=settings,
        group_columns=(
            settings.split_column,
            "hour",
        ),
    )

    day_of_week_metrics = aggregate_metrics(
        frame,
        settings=settings,
        group_columns=(
            settings.split_column,
            "day_of_week",
        ),
    )

    weekend_metrics = aggregate_metrics(
        frame,
        settings=settings,
        group_columns=(
            settings.split_column,
            "is_weekend",
        ),
    )

    peak_period_metrics = aggregate_metrics(
        frame,
        settings=settings,
        group_columns=(
            settings.split_column,
            "peak_period",
        ),
    )

    demand_band_metrics = aggregate_metrics(
        frame,
        settings=settings,
        group_columns=(
            settings.split_column,
            "demand_band",
        ),
    )

    worst_predictions = build_worst_predictions(
        frame,
        settings=settings,
    )

    tables = {
        "overall_metrics": overall_metrics,
        "zone_metrics": zone_metrics,
        "borough_metrics": borough_metrics,
        "service_zone_metrics": (
            service_zone_metrics
        ),
        "hour_metrics": hour_metrics,
        "day_of_week_metrics": (
            day_of_week_metrics
        ),
        "weekend_metrics": weekend_metrics,
        "peak_period_metrics": (
            peak_period_metrics
        ),
        "demand_band_metrics": (
            demand_band_metrics
        ),
        "worst_predictions": (
            worst_predictions
        ),
    }

    table_artifacts: dict[
        str,
        dict[str, Any]
    ] = {}

    for table_name, table_frame in tables.items():
        table_artifacts[
            table_name
        ] = write_parquet_atomically(
            table_frame,
            path=parquet_path(
                settings,
                name=table_name,
            ),
            compression=settings.compression,
        )

    eligible_zones = (
        zone_metrics.filter(
            pl.col(
                "row_count"
            )
            >= settings.minimum_zone_rows
        )
    )

    worst_zones = (
        eligible_zones.sort(
            [
                settings.split_column,
                "model_wape",
            ],
            descending=[
                False,
                True,
            ],
        )
        .group_by(
            settings.split_column,
            maintain_order=True,
        )
        .head(
            settings.top_n_zones
        )
    )

    best_improvement_zones = (
        eligible_zones.sort(
            [
                settings.split_column,
                "mae_relative_improvement",
            ],
            descending=[
                False,
                True,
            ],
        )
        .group_by(
            settings.split_column,
            maintain_order=True,
        )
        .head(
            settings.top_n_zones
        )
    )

    zones_where_baseline_wins = (
        eligible_zones.filter(
            ~pl.col(
                "model_beats_baseline"
            )
        )
        .sort(
            [
                settings.split_column,
                "mae_relative_improvement",
            ],
            descending=[
                False,
                False,
            ],
        )
    )

    report: dict[str, Any] = {
        "error_analysis_version": "0.1.0",
        "generated_at": datetime.now(
            UTC
        ).isoformat(),
        "effective_config": (
            settings.as_report_dict()
        ),
        "input": input_summary,
        "overall_metrics": (
            frame_to_records(
                overall_metrics
            )
        ),
        "key_findings": {
            "worst_zones_by_model_wape": (
                frame_to_records(
                    worst_zones
                )
            ),
            "best_zones_by_mae_improvement": (
                frame_to_records(
                    best_improvement_zones
                )
            ),
            "zones_where_baseline_wins": (
                frame_to_records(
                    zones_where_baseline_wins,
                    limit=(
                        settings.top_n_zones
                    ),
                )
            ),
        },
        "artifacts": table_artifacts,
        "invariants": {
            "joined_row_count_matches_predictions": True,
            "duplicate_key_count": (
                input_summary[
                    "joined"
                ][
                    "duplicate_key_count"
                ]
            ),
            "null_value_count": (
                input_summary[
                    "joined"
                ][
                    "null_value_count"
                ]
            ),
            "split_mismatch_count": (
                input_summary[
                    "joined"
                ][
                    "split_mismatch_count"
                ]
            ),
            "target_mismatch_count": (
                input_summary[
                    "joined"
                ][
                    "target_mismatch_count"
                ]
            ),
            "negative_prediction_count": (
                input_summary[
                    "joined"
                ][
                    "negative_prediction_count"
                ]
            ),
        },
    }

    write_json_atomically(
        settings.report_path,
        report,
    )

    return report