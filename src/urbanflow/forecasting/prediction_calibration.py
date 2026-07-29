from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import polars as pl
import yaml


class PredictionCalibrationError(RuntimeError):
    """Raised when prediction calibration cannot be completed."""


@dataclass(frozen=True)
class PredictionCalibrationSettings:
    """Configuration for leakage-safe residual calibration."""

    dataset_name: str
    timezone: str

    features_path: Path
    predictions_path: Path
    calibrated_predictions_path: Path
    calibration_artifact_path: Path
    report_path: Path

    key_columns: tuple[str, ...]
    split_column: str
    target_column: str
    prediction_column: str
    baseline_column: str

    fit_split: str
    apply_splits: tuple[str, ...]
    prediction_band_upper_bounds: tuple[float, ...]
    minimum_rows_per_band: int
    shrinkage_candidates: tuple[float, ...]
    selection_metric: str
    high_demand_threshold: float

    compression: str

    @classmethod
    def from_yaml(
        cls,
        config_path: Path,
    ) -> PredictionCalibrationSettings:
        """Load prediction-calibration settings from YAML."""
        if not config_path.exists():
            raise PredictionCalibrationError(
                f"Calibration config does not exist: {config_path}"
            )

        try:
            raw_config = yaml.safe_load(
                config_path.read_text(
                    encoding="utf-8",
                )
            )
        except (OSError, yaml.YAMLError) as error:
            raise PredictionCalibrationError(
                f"Could not read calibration config: {error}"
            ) from error

        if not isinstance(raw_config, dict):
            raise PredictionCalibrationError(
                "Calibration config must contain a YAML mapping."
            )

        try:
            dataset_config = raw_config["dataset"]
            paths_config = raw_config["paths"]
            schema_config = raw_config["schema"]
            calibration_config = raw_config["calibration"]
            output_config = raw_config["output"]

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
                calibrated_predictions_path=Path(
                    paths_config["calibrated_predictions"]
                ),
                calibration_artifact_path=Path(
                    paths_config["calibration_artifact"]
                ),
                report_path=Path(
                    paths_config["report"]
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
                fit_split=str(
                    calibration_config["fit_split"]
                ),
                apply_splits=tuple(
                    str(split)
                    for split in calibration_config["apply_splits"]
                ),
                prediction_band_upper_bounds=tuple(
                    float(value)
                    for value in calibration_config[
                        "prediction_band_upper_bounds"
                    ]
                ),
                minimum_rows_per_band=int(
                    calibration_config[
                        "minimum_rows_per_band"
                    ]
                ),
                shrinkage_candidates=tuple(
                    float(value)
                    for value in calibration_config[
                        "shrinkage_candidates"
                    ]
                ),
                selection_metric=str(
                    calibration_config["selection_metric"]
                ),
                high_demand_threshold=float(
                    calibration_config[
                        "high_demand_threshold"
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
            raise PredictionCalibrationError(
                f"Invalid calibration configuration: {error}"
            ) from error

        settings.validate()

        return settings

    def validate(self) -> None:
        """Validate calibration settings."""
        if not self.dataset_name:
            raise PredictionCalibrationError(
                "Dataset name must not be empty."
            )

        try:
            ZoneInfo(
                self.timezone
            )
        except ZoneInfoNotFoundError as error:
            raise PredictionCalibrationError(
                f"Unknown timezone: {self.timezone}"
            ) from error

        for path, description in (
            (
                self.features_path,
                "Feature dataset",
            ),
            (
                self.predictions_path,
                "Prediction dataset",
            ),
        ):
            if not path.exists():
                raise PredictionCalibrationError(
                    f"{description} does not exist: {path}"
                )

            if path.stat().st_size == 0:
                raise PredictionCalibrationError(
                    f"{description} is empty: {path}"
                )

        if not self.key_columns:
            raise PredictionCalibrationError(
                "At least one key column must be configured."
            )

        if len(
            self.key_columns
        ) != len(
            set(
                self.key_columns
            )
        ):
            raise PredictionCalibrationError(
                "Key columns must be unique."
            )

        if not self.apply_splits:
            raise PredictionCalibrationError(
                "At least one apply split must be configured."
            )

        if self.fit_split not in self.apply_splits:
            raise PredictionCalibrationError(
                "fit_split must be included in apply_splits."
            )

        if len(
            self.apply_splits
        ) != len(
            set(
                self.apply_splits
            )
        ):
            raise PredictionCalibrationError(
                "Apply splits must be unique."
            )

        bounds = self.prediction_band_upper_bounds

        if not bounds:
            raise PredictionCalibrationError(
                "At least one prediction-band bound is required."
            )

        if any(
            not math.isfinite(
                value
            )
            or value < 0
            for value in bounds
        ):
            raise PredictionCalibrationError(
                "Prediction-band bounds must be finite "
                "and non-negative."
            )

        if tuple(
            sorted(
                set(
                    bounds
                )
            )
        ) != bounds:
            raise PredictionCalibrationError(
                "Prediction-band bounds must be sorted and unique."
            )

        if self.minimum_rows_per_band <= 0:
            raise PredictionCalibrationError(
                "minimum_rows_per_band must be positive."
            )

        candidates = self.shrinkage_candidates

        if not candidates:
            raise PredictionCalibrationError(
                "At least one shrinkage candidate is required."
            )

        if any(
            not math.isfinite(
                value
            )
            or not 0.0 <= value <= 1.0
            for value in candidates
        ):
            raise PredictionCalibrationError(
                "Shrinkage candidates must be between 0 and 1."
            )

        if tuple(
            sorted(
                set(
                    candidates
                )
            )
        ) != candidates:
            raise PredictionCalibrationError(
                "Shrinkage candidates must be sorted and unique."
            )

        if 0.0 not in candidates:
            raise PredictionCalibrationError(
                "Shrinkage candidates must include 0.0 "
                "as a no-op fallback."
            )

        if self.selection_metric != "mae":
            raise PredictionCalibrationError(
                "Only MAE selection is currently supported."
            )

        if self.high_demand_threshold <= 0:
            raise PredictionCalibrationError(
                "high_demand_threshold must be positive."
            )

        if not self.compression:
            raise PredictionCalibrationError(
                "Output compression must not be empty."
            )

    def as_report_dict(
        self,
    ) -> dict[str, Any]:
        """Return effective settings as JSON-compatible data."""
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
                "calibrated_predictions": str(
                    self.calibrated_predictions_path
                ),
                "calibration_artifact": str(
                    self.calibration_artifact_path
                ),
                "report": str(
                    self.report_path
                ),
            },
            "schema": {
                "key_columns": list(
                    self.key_columns
                ),
                "split_column": self.split_column,
                "target_column": self.target_column,
                "prediction_column": self.prediction_column,
                "baseline_column": self.baseline_column,
            },
            "calibration": {
                "fit_split": self.fit_split,
                "apply_splits": list(
                    self.apply_splits
                ),
                "prediction_band_upper_bounds": list(
                    self.prediction_band_upper_bounds
                ),
                "minimum_rows_per_band": (
                    self.minimum_rows_per_band
                ),
                "shrinkage_candidates": list(
                    self.shrinkage_candidates
                ),
                "selection_metric": self.selection_metric,
                "high_demand_threshold": (
                    self.high_demand_threshold
                ),
            },
            "output": {
                "compression": self.compression,
            },
        }


def ordered_unique(
    values: tuple[str, ...],
) -> tuple[str, ...]:
    """Remove duplicate strings while preserving order."""
    return tuple(
        dict.fromkeys(
            values
        )
    )


def format_bound(
    value: float,
) -> str:
    """Format a numeric prediction-band bound."""
    if value.is_integer():
        return str(
            int(
                value
            )
        )

    return format(
        value,
        "g",
    )


def band_labels(
    upper_bounds: tuple[float, ...],
) -> tuple[str, ...]:
    """Return deterministic ordered prediction-band labels."""
    labels: list[str] = []

    previous_bound: float | None = None

    for upper_bound in upper_bounds:
        formatted_upper = format_bound(
            upper_bound
        )

        if previous_bound is None:
            labels.append(
                f"<={formatted_upper}"
            )
        else:
            labels.append(
                f"{format_bound(previous_bound)}-"
                f"{formatted_upper}"
            )

        previous_bound = upper_bound

    labels.append(
        f">{format_bound(upper_bounds[-1])}"
    )

    return tuple(
        labels
    )


def prediction_band_expression(
    *,
    prediction_column: str,
    upper_bounds: tuple[float, ...],
) -> pl.Expr:
    """Create inference-safe bands using predictions only."""
    labels = band_labels(
        upper_bounds
    )

    expression: pl.Expr | None = None

    for upper_bound, label in zip(
        upper_bounds,
        labels[:-1],
        strict=True,
    ):
        condition = (
            pl.col(
                prediction_column
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

    if expression is None:
        raise PredictionCalibrationError(
            "Could not build prediction-band expression."
        )

    return (
        expression
        .otherwise(
            pl.lit(
                labels[-1]
            )
        )
        .alias(
            "prediction_band"
        )
    )


def validate_parquet_schema(
    path: Path,
    *,
    required_columns: tuple[str, ...],
    interval_column: str,
    timezone: str,
    description: str,
) -> dict[str, Any]:
    """Validate a Parquet source and return its summary."""
    try:
        source = pl.scan_parquet(
            path
        )

        schema = source.collect_schema()

    except Exception as error:
        raise PredictionCalibrationError(
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
        raise PredictionCalibrationError(
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
        raise PredictionCalibrationError(
            f"{description.capitalize()} interval column "
            f"does not use timezone {timezone}: "
            f"{interval_dtype}"
        )

    summary = (
        source.select(
            pl.len().alias(
                "row_count"
            ),
            pl.col(
                interval_column
            )
            .min()
            .alias(
                "first_interval"
            ),
            pl.col(
                interval_column
            )
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
        "size_bytes": path.stat().st_size,
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


def load_calibration_frame(
    settings: PredictionCalibrationSettings,
) -> tuple[
    pl.DataFrame,
    dict[str, Any],
]:
    """Join model predictions with lag-1 baseline values."""
    prediction_columns = ordered_unique(
        (
            *settings.key_columns,
            settings.split_column,
            settings.target_column,
            settings.prediction_column,
        )
    )

    feature_columns = ordered_unique(
        (
            *settings.key_columns,
            settings.split_column,
            settings.target_column,
            settings.baseline_column,
        )
    )

    prediction_summary = validate_parquet_schema(
        settings.predictions_path,
        required_columns=prediction_columns,
        interval_column=settings.key_columns[0],
        timezone=settings.timezone,
        description="prediction dataset",
    )

    feature_summary = validate_parquet_schema(
        settings.features_path,
        required_columns=feature_columns,
        interval_column=settings.key_columns[0],
        timezone=settings.timezone,
        description="feature dataset",
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
                        settings.apply_splits
                    )
                )
            )
            .select(
                *prediction_columns
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
                        "raw_prediction"
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
                        settings.apply_splits
                    )
                )
            )
            .select(
                *feature_columns
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
        raise PredictionCalibrationError(
            f"Could not join calibration inputs: {error}"
        ) from error

    if frame.height == 0:
        raise PredictionCalibrationError(
            "Calibration dataset is empty."
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
                    settings.apply_splits
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
        raise PredictionCalibrationError(
            "Joined row count does not match selected predictions: "
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
            for value in frame.null_count().row(0)
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

    non_finite_prediction_count = (
        frame.filter(
            ~pl.col(
                "raw_prediction"
            )
            .cast(
                pl.Float64
            )
            .is_finite()
        )
        .height
    )

    negative_prediction_count = (
        frame.filter(
            pl.col(
                "raw_prediction"
            )
            < 0
        )
        .height
    )

    if duplicate_key_count:
        raise PredictionCalibrationError(
            "Calibration dataset contains duplicate keys."
        )

    if null_value_count:
        raise PredictionCalibrationError(
            "Calibration dataset contains null values."
        )

    if split_mismatch_count:
        raise PredictionCalibrationError(
            "Feature and prediction split labels do not match."
        )

    if target_mismatch_count:
        raise PredictionCalibrationError(
            "Feature and prediction targets do not match."
        )

    if non_finite_prediction_count:
        raise PredictionCalibrationError(
            "Raw predictions contain non-finite values."
        )

    if negative_prediction_count:
        raise PredictionCalibrationError(
            "Raw predictions contain negative values."
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
                "raw_prediction"
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
            prediction_band_expression(
                prediction_column="raw_prediction",
                upper_bounds=(
                    settings.prediction_band_upper_bounds
                ),
            )
        )
    )

    return frame, {
        "features": feature_summary,
        "predictions": prediction_summary,
        "joined": {
            "row_count": frame.height,
            "duplicate_key_count": duplicate_key_count,
            "null_value_count": null_value_count,
            "split_mismatch_count": split_mismatch_count,
            "target_mismatch_count": target_mismatch_count,
            "non_finite_prediction_count": (
                non_finite_prediction_count
            ),
            "negative_prediction_count": (
                negative_prediction_count
            ),
        },
    }


def fit_band_corrections(
    validation_frame: pl.DataFrame,
    *,
    target_column: str,
    upper_bounds: tuple[float, ...],
    minimum_rows_per_band: int,
) -> pl.DataFrame:
    """Fit median residual corrections using validation only."""
    observed_mapping = (
        validation_frame.group_by(
            "prediction_band"
        )
        .agg(
            pl.len().alias(
                "row_count"
            ),

            pl.col(
                target_column
            )
            .mean()
            .alias(
                "actual_mean"
            ),

            pl.col(
                "raw_prediction"
            )
            .mean()
            .alias(
                "raw_prediction_mean"
            ),

            (
                pl.col(
                    target_column
                )
                - pl.col(
                    "raw_prediction"
                )
            )
            .median()
            .alias(
                "median_residual"
            ),

            (
                pl.col(
                    target_column
                )
                - pl.col(
                    "raw_prediction"
                )
            )
            .mean()
            .alias(
                "mean_residual"
            ),

            (
                pl.col(
                    target_column
                )
                - pl.col(
                    "raw_prediction"
                )
            )
            .abs()
            .mean()
            .alias(
                "raw_mae"
            ),
        )
    )

    labels = band_labels(
        upper_bounds
    )

    label_frame = pl.DataFrame(
        {
            "prediction_band": list(
                labels
            ),
            "band_order": list(
                range(
                    len(
                        labels
                    )
                )
            ),
        }
    )

    return (
        label_frame.join(
            observed_mapping,
            on="prediction_band",
            how="left",
        )
        .with_columns(
            pl.col(
                "row_count"
            )
            .fill_null(
                0
            )
            .cast(
                pl.UInt64
            ),

            pl.col(
                "actual_mean"
            )
            .cast(
                pl.Float64
            ),

            pl.col(
                "raw_prediction_mean"
            )
            .cast(
                pl.Float64
            ),

            pl.col(
                "median_residual"
            )
            .cast(
                pl.Float64
            ),

            pl.col(
                "mean_residual"
            )
            .cast(
                pl.Float64
            ),

            pl.col(
                "raw_mae"
            )
            .cast(
                pl.Float64
            ),
        )
        .with_columns(
            (
                pl.col(
                    "row_count"
                )
                >= minimum_rows_per_band
            )
            .alias(
                "eligible_for_calibration"
            ),

            pl.when(
                pl.col(
                    "row_count"
                )
                >= minimum_rows_per_band
            )
            .then(
                pl.col(
                    "median_residual"
                )
                .fill_null(
                    0.0
                )
            )
            .otherwise(
                0.0
            )
            .alias(
                "base_correction"
            ),
        )
        .sort(
            "band_order"
        )
    )


def add_calibrated_prediction(
    frame: pl.DataFrame,
    *,
    mapping: pl.DataFrame,
    shrinkage: float,
) -> pl.DataFrame:
    """Apply one fixed residual-calibration mapping."""
    joined = frame.join(
        mapping.select(
            "prediction_band",
            "base_correction",
            "eligible_for_calibration",
        ),
        on="prediction_band",
        how="left",
    )

    if (
        joined.get_column(
            "base_correction"
        )
        .null_count()
        > 0
    ):
        raise PredictionCalibrationError(
            "Calibration mapping does not cover "
            "every prediction band."
        )

    adjustment = (
        pl.col(
            "base_correction"
        )
        * pl.lit(
            shrinkage
        )
    )

    unbounded_prediction = (
        pl.col(
            "raw_prediction"
        )
        + adjustment
    )

    return joined.with_columns(
        adjustment.alias(
            "calibration_adjustment"
        ),

        pl.when(
            unbounded_prediction
            < 0.0
        )
        .then(
            0.0
        )
        .otherwise(
            unbounded_prediction
        )
        .alias(
            "calibrated_prediction"
        ),

        pl.lit(
            shrinkage
        )
        .alias(
            "calibration_shrinkage"
        ),
    )


def regression_metrics(
    frame: pl.DataFrame,
    *,
    target_column: str,
    prediction_column: str,
) -> dict[str, Any]:
    """Calculate regression metrics with zero-safe WAPE."""
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

    smape_denominator = (
        pl.col(
            target_column
        )
        .abs()
        + pl.col(
            prediction_column
        )
        .abs()
    )

    values = (
        frame.select(
            pl.len().alias(
                "row_count"
            ),

            pl.col(
                target_column
            )
            .sum()
            .alias(
                "actual_sum"
            ),

            pl.col(
                prediction_column
            )
            .sum()
            .alias(
                "prediction_sum"
            ),

            absolute_error
            .sum()
            .alias(
                "absolute_error_sum"
            ),

            absolute_error
            .mean()
            .alias(
                "mae"
            ),

            residual
            .pow(2)
            .mean()
            .sqrt()
            .alias(
                "rmse"
            ),

            residual
            .mean()
            .alias(
                "mean_bias"
            ),

            pl.when(
                smape_denominator
                > 0
            )
            .then(
                (
                    2.0
                    * absolute_error
                )
                / smape_denominator
            )
            .otherwise(
                0.0
            )
            .mean()
            .alias(
                "smape"
            ),
        )
        .row(
            0,
            named=True,
        )
    )

    actual_sum = float(
        values[
            "actual_sum"
        ]
    )

    absolute_error_sum = float(
        values[
            "absolute_error_sum"
        ]
    )

    return {
        "row_count": int(
            values[
                "row_count"
            ]
        ),
        "actual_sum": actual_sum,
        "prediction_sum": float(
            values[
                "prediction_sum"
            ]
        ),
        "mae": float(
            values[
                "mae"
            ]
        ),
        "rmse": float(
            values[
                "rmse"
            ]
        ),
        "wape": (
            absolute_error_sum
            / abs(
                actual_sum
            )
            if actual_sum != 0.0
            else None
        ),
        "smape": float(
            values[
                "smape"
            ]
        ),
        "mean_bias": float(
            values[
                "mean_bias"
            ]
        ),
    }


def select_shrinkage(
    validation_frame: pl.DataFrame,
    *,
    mapping: pl.DataFrame,
    target_column: str,
    candidates: tuple[float, ...],
) -> tuple[
    float,
    list[dict[str, Any]],
]:
    """Select calibration strength using validation MAE only."""
    candidate_results: list[
        dict[str, Any]
    ] = []

    for candidate in candidates:
        calibrated_frame = add_calibrated_prediction(
            validation_frame,
            mapping=mapping,
            shrinkage=candidate,
        )

        metrics = regression_metrics(
            calibrated_frame,
            target_column=target_column,
            prediction_column="calibrated_prediction",
        )

        candidate_results.append(
            {
                "shrinkage": candidate,
                **metrics,
            }
        )

    best_mae = min(
        float(
            candidate[
                "mae"
            ]
        )
        for candidate in candidate_results
    )

    tie_tolerance = 1e-12

    selected_shrinkage = min(
        float(
            candidate[
                "shrinkage"
            ]
        )
        for candidate in candidate_results
        if float(
            candidate[
                "mae"
            ]
        )
        <= best_mae + tie_tolerance
    )

    return (
        selected_shrinkage,
        candidate_results,
    )


def metric_comparison(
    *,
    raw_metrics: dict[str, Any],
    calibrated_metrics: dict[str, Any],
    baseline_metrics: dict[str, Any],
) -> dict[str, Any]:
    """Compare calibrated predictions with raw and baseline."""
    raw_mae = float(
        raw_metrics[
            "mae"
        ]
    )

    calibrated_mae = float(
        calibrated_metrics[
            "mae"
        ]
    )

    baseline_mae = float(
        baseline_metrics[
            "mae"
        ]
    )

    return {
        "raw": raw_metrics,
        "calibrated": calibrated_metrics,
        "baseline": baseline_metrics,
        "calibrated_vs_raw": {
            "mae_absolute_improvement": (
                raw_mae
                - calibrated_mae
            ),
            "mae_relative_improvement": (
                (
                    raw_mae
                    - calibrated_mae
                )
                / raw_mae
                if raw_mae > 0
                else 0.0
            ),
            "calibration_improves_mae": (
                calibrated_mae
                < raw_mae
            ),
        },
        "calibrated_vs_baseline": {
            "mae_absolute_improvement": (
                baseline_mae
                - calibrated_mae
            ),
            "mae_relative_improvement": (
                (
                    baseline_mae
                    - calibrated_mae
                )
                / baseline_mae
                if baseline_mae > 0
                else 0.0
            ),
            "calibrated_beats_baseline": (
                calibrated_mae
                < baseline_mae
            ),
        },
    }


def operational_risk_summary(
    frame: pl.DataFrame,
    *,
    target_column: str,
    high_demand_threshold: float,
) -> dict[str, Any]:
    """Compare calibration risk on zero and high demand."""
    zero_demand_frame = frame.filter(
        pl.col(
            target_column
        )
        == 0
    )

    high_demand_frame = frame.filter(
        pl.col(
            target_column
        )
        > high_demand_threshold
    )

    def segment_metrics(
        segment: pl.DataFrame,
    ) -> dict[str, Any]:
        if segment.height == 0:
            return {
                "row_count": 0,
            }

        return {
            "row_count": segment.height,
            "row_share": (
                segment.height
                / frame.height
            ),
            "actual_mean": float(
                segment.get_column(
                    target_column
                ).mean()
            ),
            "raw": regression_metrics(
                segment,
                target_column=target_column,
                prediction_column="raw_prediction",
            ),
            "calibrated": regression_metrics(
                segment,
                target_column=target_column,
                prediction_column="calibrated_prediction",
            ),
            "baseline": regression_metrics(
                segment,
                target_column=target_column,
                prediction_column="baseline_prediction",
            ),
        }

    return {
        "zero_demand": segment_metrics(
            zero_demand_frame
        ),
        "high_demand": segment_metrics(
            high_demand_frame
        ),
    }


def sanitize_json_value(
    value: Any,
) -> Any:
    """Replace non-finite floating-point values with JSON null."""
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

        raise PredictionCalibrationError(
            f"Could not write JSON artifact {path}: {error}"
        ) from error


def write_parquet_atomically(
    frame: pl.DataFrame,
    *,
    path: Path,
    compression: str,
) -> dict[str, Any]:
    """Write calibrated predictions atomically."""
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

        raise PredictionCalibrationError(
            f"Could not write calibrated predictions: {error}"
        ) from error

    return {
        "path": str(
            path
        ),
        "size_bytes": path.stat().st_size,
        "row_count": frame.height,
        "column_count": frame.width,
    }


def calibrate_xgboost_predictions(
    config_path: Path,
) -> dict[str, Any]:
    """Fit validation-only calibration and apply it to test."""
    settings = (
        PredictionCalibrationSettings
        .from_yaml(
            config_path
        )
    )

    frame, input_summary = (
        load_calibration_frame(
            settings
        )
    )

    validation_frame = frame.filter(
        pl.col(
            settings.split_column
        )
        == settings.fit_split
    )

    if validation_frame.height == 0:
        raise PredictionCalibrationError(
            f"Fit split is empty: {settings.fit_split}"
        )

    mapping = fit_band_corrections(
        validation_frame,
        target_column=settings.target_column,
        upper_bounds=(
            settings.prediction_band_upper_bounds
        ),
        minimum_rows_per_band=(
            settings.minimum_rows_per_band
        ),
    )

    (
        selected_shrinkage,
        candidate_metrics,
    ) = select_shrinkage(
        validation_frame,
        mapping=mapping,
        target_column=settings.target_column,
        candidates=settings.shrinkage_candidates,
    )

    calibrated_frame = (
        add_calibrated_prediction(
            frame,
            mapping=mapping,
            shrinkage=selected_shrinkage,
        )
        .sort(
            *settings.key_columns
        )
    )

    negative_prediction_count = (
        calibrated_frame.filter(
            pl.col(
                "calibrated_prediction"
            )
            < 0
        )
        .height
    )

    non_finite_prediction_count = (
        calibrated_frame.filter(
            ~pl.col(
                "calibrated_prediction"
            )
            .is_finite()
        )
        .height
    )

    null_prediction_count = (
        calibrated_frame.get_column(
            "calibrated_prediction"
        )
        .null_count()
    )

    if negative_prediction_count:
        raise PredictionCalibrationError(
            "Calibrated predictions contain negative values."
        )

    if non_finite_prediction_count:
        raise PredictionCalibrationError(
            "Calibrated predictions contain non-finite values."
        )

    if null_prediction_count:
        raise PredictionCalibrationError(
            "Calibrated predictions contain null values."
        )

    split_reports: dict[
        str,
        Any
    ] = {}

    for split in settings.apply_splits:
        split_frame = calibrated_frame.filter(
            pl.col(
                settings.split_column
            )
            == split
        )

        if split_frame.height == 0:
            raise PredictionCalibrationError(
                f"Apply split is empty: {split}"
            )

        raw_metrics = regression_metrics(
            split_frame,
            target_column=settings.target_column,
            prediction_column="raw_prediction",
        )

        calibrated_metrics = regression_metrics(
            split_frame,
            target_column=settings.target_column,
            prediction_column="calibrated_prediction",
        )

        baseline_metrics = regression_metrics(
            split_frame,
            target_column=settings.target_column,
            prediction_column="baseline_prediction",
        )

        split_reports[
            split
        ] = {
            "metrics": metric_comparison(
                raw_metrics=raw_metrics,
                calibrated_metrics=calibrated_metrics,
                baseline_metrics=baseline_metrics,
            ),
            "operational_risk": (
                operational_risk_summary(
                    split_frame,
                    target_column=settings.target_column,
                    high_demand_threshold=(
                        settings.high_demand_threshold
                    ),
                )
            ),
        }

    output_columns = ordered_unique(
        (
            *settings.key_columns,
            settings.split_column,
            settings.target_column,
            "raw_prediction",
            "calibrated_prediction",
            "baseline_prediction",
            "prediction_band",
            "base_correction",
            "calibration_adjustment",
            "calibration_shrinkage",
            "eligible_for_calibration",
        )
    )

    output_frame = calibrated_frame.select(
        *output_columns
    )

    prediction_artifact = write_parquet_atomically(
        output_frame,
        path=settings.calibrated_predictions_path,
        compression=settings.compression,
    )

    generated_at = datetime.now(
        UTC
    ).isoformat()

    calibration_artifact = {
        "calibration_version": "0.1.0",
        "generated_at": generated_at,
        "dataset_name": settings.dataset_name,
        "fit_split": settings.fit_split,
        "test_was_used_for_selection": False,
        "selection_metric": settings.selection_metric,
        "selected_shrinkage": selected_shrinkage,
        "prediction_band_upper_bounds": list(
            settings.prediction_band_upper_bounds
        ),
        "minimum_rows_per_band": (
            settings.minimum_rows_per_band
        ),
        "mapping": mapping.to_dicts(),
        "source_prediction_path": str(
            settings.predictions_path
        ),
    }

    write_json_atomically(
        settings.calibration_artifact_path,
        calibration_artifact,
    )

    report: dict[str, Any] = {
        "calibration_report_version": "0.1.0",
        "generated_at": generated_at,
        "effective_config": (
            settings.as_report_dict()
        ),
        "input": input_summary,
        "model_selection": {
            "fit_split": settings.fit_split,
            "selection_metric": (
                settings.selection_metric
            ),
            "selected_shrinkage": (
                selected_shrinkage
            ),
            "candidate_metrics": (
                candidate_metrics
            ),
            "mapping": mapping.to_dicts(),
        },
        "splits": split_reports,
        "artifacts": {
            "calibrated_predictions": (
                prediction_artifact
            ),
            "calibration_artifact": {
                "path": str(
                    settings.calibration_artifact_path
                ),
                "size_bytes": (
                    settings.calibration_artifact_path
                    .stat()
                    .st_size
                ),
            },
            "report": {
                "path": str(
                    settings.report_path
                ),
            },
        },
        "invariants": {
            "calibration_fit_uses_validation_only": True,
            "test_is_not_used_for_calibration_selection": True,
            "mapping_is_fixed_before_test_application": True,
            "output_row_count_matches_input": (
                output_frame.height
                == frame.height
            ),
            "negative_calibrated_prediction_count": (
                negative_prediction_count
            ),
            "non_finite_calibrated_prediction_count": (
                non_finite_prediction_count
            ),
            "null_calibrated_prediction_count": (
                null_prediction_count
            ),
        },
    }

    write_json_atomically(
        settings.report_path,
        report,
    )

    return report