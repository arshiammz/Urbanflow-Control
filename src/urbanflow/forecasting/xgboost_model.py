from __future__ import annotations

import gc
import hashlib
import json
import math
import platform
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import numpy as np
import polars as pl
import sklearn
import xgboost as xgb
import yaml


class XGBoostTrainingError(RuntimeError):
    """Raised when XGBoost forecasting training cannot be completed."""


@dataclass(frozen=True)
class XGBoostSettings:
    """Configuration for global XGBoost demand forecasting."""

    dataset_name: str
    timezone: str

    input_path: Path
    model_path: Path
    metadata_path: Path
    predictions_path: Path
    report_path: Path
    feature_importance_path: Path
    baseline_report_path: Path

    target_column: str
    split_column: str
    key_columns: tuple[str, ...]
    categorical_columns: tuple[str, ...]
    feature_columns: tuple[str, ...]

    training_splits: tuple[str, ...]
    validation_splits: tuple[str, ...]
    final_training_splits: tuple[str, ...]
    test_splits: tuple[str, ...]

    objective: str
    eval_metric: str
    tree_method: str
    enable_categorical: bool
    n_estimators: int
    learning_rate: float
    max_depth: int
    min_child_weight: float
    subsample: float
    colsample_bytree: float
    reg_alpha: float
    reg_lambda: float
    max_bin: int
    max_cat_to_onehot: int
    max_cat_threshold: int
    early_stopping_rounds: int
    random_state: int
    n_jobs: int
    verbosity: int

    minimum_prediction: float
    compression: str

    @classmethod
    def from_yaml(cls, config_path: Path) -> XGBoostSettings:
        """Load XGBoost settings from YAML."""
        if not config_path.exists():
            raise XGBoostTrainingError(
                f"XGBoost config does not exist: {config_path}"
            )

        try:
            raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as error:
            raise XGBoostTrainingError(
                f"Could not read XGBoost config: {error}"
            ) from error

        if not isinstance(raw, dict):
            raise XGBoostTrainingError(
                "XGBoost config must contain a YAML mapping."
            )

        try:
            dataset = raw["dataset"]
            paths = raw["paths"]
            schema = raw["schema"]
            splits = raw["splits"]
            model = raw["model"]
            prediction = raw["prediction"]
            output = raw["output"]

            settings = cls(
                dataset_name=str(dataset["name"]),
                timezone=str(dataset["timezone"]),
                input_path=Path(paths["input"]),
                model_path=Path(paths["model"]),
                metadata_path=Path(paths["metadata"]),
                predictions_path=Path(paths["predictions"]),
                report_path=Path(paths["report"]),
                feature_importance_path=Path(paths["feature_importance"]),
                baseline_report_path=Path(paths["baseline_report"]),
                target_column=str(schema["target_column"]),
                split_column=str(schema["split_column"]),
                key_columns=tuple(str(v) for v in schema["key_columns"]),
                categorical_columns=tuple(
                    str(v) for v in schema["categorical_columns"]
                ),
                feature_columns=tuple(
                    str(v) for v in schema["feature_columns"]
                ),
                training_splits=tuple(
                    str(v) for v in splits["training"]
                ),
                validation_splits=tuple(
                    str(v) for v in splits["validation"]
                ),
                final_training_splits=tuple(
                    str(v) for v in splits["final_training"]
                ),
                test_splits=tuple(str(v) for v in splits["test"]),
                objective=str(model["objective"]),
                eval_metric=str(model["eval_metric"]),
                tree_method=str(model["tree_method"]),
                enable_categorical=bool(model["enable_categorical"]),
                n_estimators=int(model["n_estimators"]),
                learning_rate=float(model["learning_rate"]),
                max_depth=int(model["max_depth"]),
                min_child_weight=float(model["min_child_weight"]),
                subsample=float(model["subsample"]),
                colsample_bytree=float(model["colsample_bytree"]),
                reg_alpha=float(model["reg_alpha"]),
                reg_lambda=float(model["reg_lambda"]),
                max_bin=int(model["max_bin"]),
                max_cat_to_onehot=int(model["max_cat_to_onehot"]),
                max_cat_threshold=int(model["max_cat_threshold"]),
                early_stopping_rounds=int(
                    model["early_stopping_rounds"]
                ),
                random_state=int(model["random_state"]),
                n_jobs=int(model["n_jobs"]),
                verbosity=int(model["verbosity"]),
                minimum_prediction=float(
                    prediction["minimum_prediction"]
                ),
                compression=str(output["compression"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise XGBoostTrainingError(
                f"Invalid XGBoost configuration: {error}"
            ) from error

        settings.validate()
        return settings

    def validate(self) -> None:
        """Validate configuration and leakage boundaries."""
        if not self.dataset_name:
            raise XGBoostTrainingError("Dataset name must not be empty.")

        try:
            ZoneInfo(self.timezone)
        except ZoneInfoNotFoundError as error:
            raise XGBoostTrainingError(
                f"Unknown timezone: {self.timezone}"
            ) from error

        if not self.input_path.exists() or self.input_path.stat().st_size == 0:
            raise XGBoostTrainingError(
                f"Feature dataset is missing or empty: {self.input_path}"
            )

        if not self.baseline_report_path.exists():
            raise XGBoostTrainingError(
                "Baseline report does not exist: "
                f"{self.baseline_report_path}"
            )

        if self.model_path.suffix.lower() not in {".json", ".ubj"}:
            raise XGBoostTrainingError(
                "Model path must end with .json or .ubj."
            )

        for name, values in (
            ("key columns", self.key_columns),
            ("categorical columns", self.categorical_columns),
            ("feature columns", self.feature_columns),
            ("training splits", self.training_splits),
            ("validation splits", self.validation_splits),
            ("final-training splits", self.final_training_splits),
            ("test splits", self.test_splits),
        ):
            if not values:
                raise XGBoostTrainingError(
                    f"At least one {name} value is required."
                )
            if len(values) != len(set(values)):
                raise XGBoostTrainingError(
                    f"{name.capitalize()} must be unique."
                )

        missing_categorical = sorted(
            set(self.categorical_columns).difference(self.feature_columns)
        )
        if missing_categorical:
            raise XGBoostTrainingError(
                "Categorical columns must also be feature columns: "
                + ", ".join(missing_categorical)
            )

        if set(self.training_splits).intersection(self.validation_splits):
            raise XGBoostTrainingError(
                "Training and validation splits must not overlap."
            )

        if set(self.final_training_splits) != set(
            (*self.training_splits, *self.validation_splits)
        ):
            raise XGBoostTrainingError(
                "final_training must equal training + validation."
            )

        if set(self.final_training_splits).intersection(self.test_splits):
            raise XGBoostTrainingError(
                "Final-training and test splits must not overlap."
            )

        if self.objective != "reg:squarederror":
            raise XGBoostTrainingError(
                "objective must be 'reg:squarederror'."
            )
        if self.eval_metric != "mae":
            raise XGBoostTrainingError("eval_metric must be 'mae'.")
        if self.tree_method != "hist":
            raise XGBoostTrainingError("tree_method must be 'hist'.")
        if not self.enable_categorical:
            raise XGBoostTrainingError(
                "enable_categorical must be true."
            )
        if self.n_estimators <= 0:
            raise XGBoostTrainingError("n_estimators must be positive.")
        if not 0.0 < self.learning_rate <= 1.0:
            raise XGBoostTrainingError(
                "learning_rate must be in (0, 1]."
            )
        if self.max_depth <= 0:
            raise XGBoostTrainingError("max_depth must be positive.")
        if self.min_child_weight < 0.0:
            raise XGBoostTrainingError(
                "min_child_weight must be non-negative."
            )
        if not 0.0 < self.subsample <= 1.0:
            raise XGBoostTrainingError("subsample must be in (0, 1].")
        if not 0.0 < self.colsample_bytree <= 1.0:
            raise XGBoostTrainingError(
                "colsample_bytree must be in (0, 1]."
            )
        if self.reg_alpha < 0.0 or self.reg_lambda < 0.0:
            raise XGBoostTrainingError(
                "Regularization values must be non-negative."
            )
        if self.max_bin < 2:
            raise XGBoostTrainingError("max_bin must be at least 2.")
        if self.max_cat_to_onehot < 1 or self.max_cat_threshold < 1:
            raise XGBoostTrainingError(
                "Categorical thresholds must be positive."
            )
        if self.early_stopping_rounds <= 0:
            raise XGBoostTrainingError(
                "early_stopping_rounds must be positive."
            )
        if self.minimum_prediction < 0.0:
            raise XGBoostTrainingError(
                "minimum_prediction must be non-negative."
            )
        if not self.compression:
            raise XGBoostTrainingError(
                "Output compression must not be empty."
            )

    @property
    def required_columns(self) -> tuple[str, ...]:
        """Return required input columns in deterministic order."""
        return ordered_unique(
            (
                *self.key_columns,
                self.split_column,
                self.target_column,
                *self.feature_columns,
            )
        )

    def base_model_parameters(self) -> dict[str, Any]:
        """Return parameters shared by both training stages."""
        return {
            "objective": self.objective,
            "eval_metric": self.eval_metric,
            "tree_method": self.tree_method,
            "enable_categorical": self.enable_categorical,
            "learning_rate": self.learning_rate,
            "max_depth": self.max_depth,
            "min_child_weight": self.min_child_weight,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "reg_alpha": self.reg_alpha,
            "reg_lambda": self.reg_lambda,
            "max_bin": self.max_bin,
            "max_cat_to_onehot": self.max_cat_to_onehot,
            "max_cat_threshold": self.max_cat_threshold,
            "random_state": self.random_state,
            "n_jobs": self.n_jobs,
            "verbosity": self.verbosity,
        }

    def selection_model_parameters(self) -> dict[str, Any]:
        """Return parameters for validation-based model selection."""
        return {
            **self.base_model_parameters(),
            "n_estimators": self.n_estimators,
            "early_stopping_rounds": self.early_stopping_rounds,
        }

    def final_model_parameters(
        self,
        *,
        selected_n_estimators: int,
    ) -> dict[str, Any]:
        """Return parameters for the final train-plus-validation refit."""
        return {
            **self.base_model_parameters(),
            "n_estimators": selected_n_estimators,
        }

    def as_report_dict(self) -> dict[str, Any]:
        """Return effective settings as JSON-compatible data."""
        return {
            "dataset": {
                "name": self.dataset_name,
                "timezone": self.timezone,
            },
            "paths": {
                "input": str(self.input_path),
                "model": str(self.model_path),
                "metadata": str(self.metadata_path),
                "predictions": str(self.predictions_path),
                "report": str(self.report_path),
                "feature_importance": str(self.feature_importance_path),
                "baseline_report": str(self.baseline_report_path),
            },
            "schema": {
                "target_column": self.target_column,
                "split_column": self.split_column,
                "key_columns": list(self.key_columns),
                "categorical_columns": list(self.categorical_columns),
                "feature_columns": list(self.feature_columns),
            },
            "splits": {
                "training": list(self.training_splits),
                "validation": list(self.validation_splits),
                "final_training": list(self.final_training_splits),
                "test": list(self.test_splits),
            },
            "model": {
                **self.base_model_parameters(),
                "n_estimators": self.n_estimators,
                "early_stopping_rounds": self.early_stopping_rounds,
            },
            "prediction": {
                "minimum_prediction": self.minimum_prediction,
            },
            "output": {"compression": self.compression},
        }


@dataclass(frozen=True)
class PreparedDataset:
    """Materialized feature matrix, labels, and original prediction keys."""

    keys: pl.DataFrame
    features: pl.DataFrame
    target: np.ndarray

    @property
    def row_count(self) -> int:
        """Return the number of materialized rows."""
        return self.features.height


def ordered_unique(values: tuple[str, ...]) -> tuple[str, ...]:
    """Remove duplicate strings while preserving order."""
    return tuple(dict.fromkeys(values))


def load_json_mapping(path: Path, *, description: str) -> dict[str, Any]:
    """Read a JSON file and require a top-level mapping."""
    if not path.exists():
        raise XGBoostTrainingError(
            f"{description.capitalize()} does not exist: {path}"
        )

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise XGBoostTrainingError(
            f"Could not read {description}: {error}"
        ) from error

    if not isinstance(value, dict):
        raise XGBoostTrainingError(
            f"{description.capitalize()} must contain a JSON object."
        )
    return value


def validate_input_dataset(settings: XGBoostSettings) -> dict[str, Any]:
    """Validate the feature dataset and temporal split coverage."""
    try:
        source = pl.scan_parquet(settings.input_path)
        schema = source.collect_schema()
    except Exception as error:
        raise XGBoostTrainingError(
            f"Could not read feature dataset: {error}"
        ) from error

    missing = sorted(
        set(settings.required_columns).difference(schema.names())
    )
    if missing:
        raise XGBoostTrainingError(
            "Feature dataset is missing columns: " + ", ".join(missing)
        )

    interval_column = settings.key_columns[0]
    interval_dtype = schema[interval_column]
    if settings.timezone not in str(interval_dtype):
        raise XGBoostTrainingError(
            f"{interval_column} does not use {settings.timezone}: "
            f"{interval_dtype}"
        )

    summary = (
        source.select(
            pl.len().alias("row_count"),
            pl.col(interval_column).n_unique().alias("interval_count"),
            pl.col("zone_id").n_unique().alias("zone_count"),
            pl.col(interval_column).min().alias("first_interval"),
            pl.col(interval_column).max().alias("last_interval"),
            pl.col(settings.target_column).min().alias("target_minimum"),
            pl.col(settings.target_column).max().alias("target_maximum"),
            pl.col(settings.target_column).mean().alias("target_mean"),
        )
        .collect()
        .row(0, named=True)
    )

    split_frame = (
        source.group_by(settings.split_column)
        .agg(pl.len().alias("row_count"))
        .sort(settings.split_column)
        .collect()
    )
    split_counts = {
        str(split): int(count)
        for split, count in split_frame.iter_rows()
    }

    configured_splits = set(
        (
            *settings.training_splits,
            *settings.validation_splits,
            *settings.test_splits,
        )
    )
    missing_splits = sorted(
        split for split in configured_splits if split_counts.get(split, 0) <= 0
    )
    if missing_splits:
        raise XGBoostTrainingError(
            "Configured splits contain no rows: "
            + ", ".join(missing_splits)
        )

    unexpected_splits = sorted(
        set(split_counts).difference(configured_splits)
    )
    if unexpected_splits:
        raise XGBoostTrainingError(
            "Feature dataset contains unexpected splits: "
            + ", ".join(unexpected_splits)
        )

    duplicate_key_count = int(
        source.group_by(*settings.key_columns)
        .agg(pl.len().alias("key_count"))
        .filter(pl.col("key_count") > 1)
        .select(pl.len().alias("duplicate_key_count"))
        .collect()
        .item()
    )

    required_null_count = int(
        sum(
            int(value)
            for value in (
                source.select(
                    [
                        pl.col(column).null_count().alias(column)
                        for column in settings.required_columns
                    ]
                )
                .collect()
                .row(0)
            )
        )
    )

    if duplicate_key_count:
        raise XGBoostTrainingError(
            "Feature dataset contains duplicate prediction keys."
        )
    if required_null_count:
        raise XGBoostTrainingError(
            "Feature dataset contains nulls in required columns."
        )
    if int(summary["target_minimum"]) < 0:
        raise XGBoostTrainingError(
            "Feature dataset contains negative target values."
        )

    return {
        "path": str(settings.input_path),
        "size_bytes": settings.input_path.stat().st_size,
        "row_count": int(summary["row_count"]),
        "interval_count": int(summary["interval_count"]),
        "zone_count": int(summary["zone_count"]),
        "first_interval": summary["first_interval"].isoformat(),
        "last_interval": summary["last_interval"].isoformat(),
        "target_minimum": int(summary["target_minimum"]),
        "target_maximum": int(summary["target_maximum"]),
        "target_mean": float(summary["target_mean"]),
        "split_counts": split_counts,
        "duplicate_key_count": duplicate_key_count,
        "required_null_count": required_null_count,
    }


def build_categorical_contract(
    settings: XGBoostSettings,
) -> dict[str, tuple[str, ...]]:
    """Build fixed Polars Enum vocabularies for all model stages."""
    source = pl.scan_parquet(settings.input_path).filter(
        pl.col(settings.split_column).is_in(
            list(settings.final_training_splits)
        )
    )
    contract: dict[str, tuple[str, ...]] = {}

    for column in settings.categorical_columns:
        values = (
            source.select(pl.col(column).cast(pl.String).alias(column))
            .unique()
            .drop_nulls()
            .sort(column)
            .collect()
            .get_column(column)
            .to_list()
        )
        categories = tuple(str(value) for value in values)
        if not categories:
            raise XGBoostTrainingError(
                f"Categorical column has no values: {column}"
            )
        contract[column] = categories

    return contract


def apply_categorical_contract(
    frame: pl.DataFrame,
    *,
    contract: dict[str, tuple[str, ...]],
) -> pl.DataFrame:
    """Cast model categoricals to deterministic Polars Enum types."""
    try:
        return frame.with_columns(
            [
                pl.col(column)
                .cast(pl.String)
                .cast(pl.Enum(list(categories)))
                .alias(column)
                for column, categories in contract.items()
            ]
        )
    except Exception as error:
        raise XGBoostTrainingError(
            f"Could not apply categorical contract: {error}"
        ) from error


def load_prepared_dataset(
    settings: XGBoostSettings,
    *,
    splits: tuple[str, ...],
    categorical_contract: dict[str, tuple[str, ...]],
) -> PreparedDataset:
    """Materialize one split group as Polars features plus NumPy labels."""
    try:
        frame = (
            pl.scan_parquet(settings.input_path)
            .filter(pl.col(settings.split_column).is_in(list(splits)))
            .select(*settings.required_columns)
            .sort(*settings.key_columns)
            .collect()
        )
    except Exception as error:
        raise XGBoostTrainingError(
            f"Could not materialize splits {list(splits)}: {error}"
        ) from error

    if frame.height == 0:
        raise XGBoostTrainingError(
            f"Model split contains no rows: {list(splits)}"
        )

    keys = frame.select(
        *ordered_unique(
            (
                *settings.key_columns,
                settings.split_column,
                settings.target_column,
            )
        )
    )
    features = apply_categorical_contract(
        frame.select(*settings.feature_columns),
        contract=categorical_contract,
    )
    target = (
        frame.get_column(settings.target_column)
        .cast(pl.Float32)
        .to_numpy()
    )

    if not np.isfinite(target).all():
        raise XGBoostTrainingError(
            f"Target contains non-finite values: {list(splits)}"
        )

    return PreparedDataset(keys=keys, features=features, target=target)


def calculate_regression_metrics(
    actual: np.ndarray,
    prediction: np.ndarray,
) -> dict[str, Any]:
    """Calculate metrics identical to the baseline evaluation contract."""
    actual_array = np.asarray(actual, dtype=np.float64)
    prediction_array = np.asarray(prediction, dtype=np.float64)

    if actual_array.shape != prediction_array.shape:
        raise XGBoostTrainingError(
            "Actual and prediction arrays have different shapes."
        )
    if actual_array.size == 0:
        raise XGBoostTrainingError(
            "Cannot calculate metrics for an empty dataset."
        )
    if not np.isfinite(actual_array).all():
        raise XGBoostTrainingError("Actual values are not finite.")
    if not np.isfinite(prediction_array).all():
        raise XGBoostTrainingError("Predictions are not finite.")

    residual = prediction_array - actual_array
    absolute_error = np.abs(residual)
    denominator = np.abs(actual_array) + np.abs(prediction_array)
    smape_terms = np.divide(
        2.0 * absolute_error,
        denominator,
        out=np.zeros_like(absolute_error),
        where=denominator > 0.0,
    )
    actual_absolute_sum = float(np.abs(actual_array).sum())

    return {
        "row_count": int(actual_array.size),
        "actual_sum": round(float(actual_array.sum()), 6),
        "prediction_sum": round(float(prediction_array.sum()), 6),
        "mae": round(float(np.mean(absolute_error)), 6),
        "rmse": round(float(np.sqrt(np.mean(np.square(residual)))), 6),
        "wape": round(
            (
                float(absolute_error.sum() / actual_absolute_sum)
                if actual_absolute_sum > 0.0
                else 0.0
            ),
            8,
        ),
        "smape": round(float(np.mean(smape_terms)), 8),
        "mean_bias": round(float(np.mean(residual)), 6),
    }


def clip_predictions(
    prediction: np.ndarray,
    *,
    minimum_prediction: float,
) -> np.ndarray:
    """Clip predictions to a valid non-negative demand range."""
    clipped = np.maximum(
        np.asarray(prediction, dtype=np.float64),
        minimum_prediction,
    )
    if not np.isfinite(clipped).all():
        raise XGBoostTrainingError(
            "Model produced non-finite predictions."
        )
    return clipped.astype(np.float32)


def fit_selection_model(
    settings: XGBoostSettings,
    *,
    training: PreparedDataset,
    validation: PreparedDataset,
) -> tuple[xgb.XGBRegressor, dict[str, Any]]:
    """Select the boosting-round count with May validation data."""
    model = xgb.XGBRegressor(**settings.selection_model_parameters())

    try:
        model.fit(
            training.features,
            training.target,
            eval_set=[
                (training.features, training.target),
                (validation.features, validation.target),
            ],
            verbose=False,
        )
    except Exception as error:
        raise XGBoostTrainingError(
            f"XGBoost model selection failed: {error}"
        ) from error

    history = model.evals_result()

    try:
        best_iteration = int(model.best_iteration)
        best_score = float(model.best_score)
    except (AttributeError, TypeError, ValueError):
        values = history.get("validation_1", {}).get(
            settings.eval_metric,
            [],
        )
        if not values:
            raise XGBoostTrainingError(
                "Validation MAE history is missing."
            )
        best_iteration = int(np.argmin(np.asarray(values)))
        best_score = float(values[best_iteration])

    selected_n_estimators = best_iteration + 1
    if not 1 <= selected_n_estimators <= settings.n_estimators:
        raise XGBoostTrainingError(
            "Selected estimator count is outside the configured range."
        )

    return model, {
        "best_iteration": best_iteration,
        "selected_n_estimators": selected_n_estimators,
        "best_score": round(best_score, 8),
        "history": history,
    }


def fit_final_model(
    settings: XGBoostSettings,
    *,
    final_training: PreparedDataset,
    selected_n_estimators: int,
) -> xgb.XGBRegressor:
    """Refit from scratch on January through May."""
    model = xgb.XGBRegressor(
        **settings.final_model_parameters(
            selected_n_estimators=selected_n_estimators
        )
    )
    try:
        model.fit(
            final_training.features,
            final_training.target,
            verbose=False,
        )
    except Exception as error:
        raise XGBoostTrainingError(
            f"Final XGBoost refit failed: {error}"
        ) from error
    return model


def build_prediction_frame(
    *,
    dataset: PreparedDataset,
    prediction: np.ndarray,
    model_stage: str,
    minimum_prediction: float,
) -> pl.DataFrame:
    """Build prediction rows with residuals and stage metadata."""
    clipped = clip_predictions(
        prediction,
        minimum_prediction=minimum_prediction,
    )
    if clipped.shape[0] != dataset.row_count:
        raise XGBoostTrainingError(
            "Prediction count does not match dataset row count."
        )

    return (
        dataset.keys.with_columns(
            pl.Series("prediction", clipped, dtype=pl.Float32),
            pl.lit(model_stage).alias("model_stage"),
        )
        .with_columns(
            (
                pl.col("prediction") - pl.col("target_pickup_count")
            )
            .cast(pl.Float32)
            .alias("residual"),
            (
                pl.col("prediction") - pl.col("target_pickup_count")
            )
            .abs()
            .cast(pl.Float32)
            .alias("absolute_error"),
        )
    )


def write_predictions(
    settings: XGBoostSettings,
    *,
    validation_predictions: pl.DataFrame,
    test_predictions: pl.DataFrame,
) -> dict[str, Any]:
    """Write validation and test predictions atomically."""
    predictions = (
        pl.concat(
            [validation_predictions, test_predictions],
            how="vertical",
        )
        .sort(*settings.key_columns)
    )

    prediction_key_columns = ordered_unique(
        (*settings.key_columns, settings.split_column)
    )
    duplicate_key_count = int(
        predictions.group_by(*prediction_key_columns)
        .agg(pl.len().alias("key_count"))
        .filter(pl.col("key_count") > 1)
        .height
    )
    null_value_count = int(
        sum(int(value) for value in predictions.null_count().row(0))
    )
    negative_prediction_count = int(
        predictions.filter(
            pl.col("prediction") < settings.minimum_prediction
        ).height
    )

    if duplicate_key_count:
        raise XGBoostTrainingError(
            "Prediction output contains duplicate keys."
        )
    if null_value_count:
        raise XGBoostTrainingError(
            "Prediction output contains null values."
        )
    if negative_prediction_count:
        raise XGBoostTrainingError(
            "Prediction output violates its lower bound."
        )

    settings.predictions_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = settings.predictions_path.with_name(
        f"{settings.predictions_path.name}.part"
    )
    temporary.unlink(missing_ok=True)

    try:
        predictions.write_parquet(
            temporary,
            compression=settings.compression,
            statistics=True,
        )
        temporary.replace(settings.predictions_path)
    except Exception as error:
        temporary.unlink(missing_ok=True)
        raise XGBoostTrainingError(
            f"Could not write XGBoost predictions: {error}"
        ) from error

    split_counts = {
        str(split): int(count)
        for split, count in (
            predictions.group_by(settings.split_column)
            .agg(pl.len().alias("row_count"))
            .sort(settings.split_column)
            .iter_rows()
        )
    }

    return {
        "path": str(settings.predictions_path),
        "size_bytes": settings.predictions_path.stat().st_size,
        "row_count": predictions.height,
        "split_counts": split_counts,
        "duplicate_key_count": duplicate_key_count,
        "null_value_count": null_value_count,
        "negative_prediction_count": negative_prediction_count,
    }


def save_model_atomically(
    model: xgb.XGBRegressor,
    *,
    model_path: Path,
) -> dict[str, Any]:
    """Save XGBoost in JSON/UBJSON while preserving feature metadata."""
    model_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = model_path.with_name(
        f"{model_path.stem}.part{model_path.suffix}"
    )
    temporary.unlink(missing_ok=True)

    try:
        model.save_model(temporary)
        temporary.replace(model_path)
    except Exception as error:
        temporary.unlink(missing_ok=True)
        raise XGBoostTrainingError(
            f"Could not save XGBoost model: {error}"
        ) from error

    return {
        "path": str(model_path),
        "size_bytes": model_path.stat().st_size,
        "sha256": sha256_file(model_path),
    }


def collect_feature_importance(
    model: xgb.XGBRegressor,
    *,
    feature_columns: tuple[str, ...],
) -> dict[str, Any]:
    """Collect gain-based importance for every configured feature."""
    booster = model.get_booster()
    gain = booster.get_score(importance_type="gain")
    total_gain = booster.get_score(importance_type="total_gain")
    weight = booster.get_score(importance_type="weight")

    total = float(
        sum(float(total_gain.get(feature, 0.0)) for feature in feature_columns)
    )
    records = []

    for feature in feature_columns:
        feature_total = float(total_gain.get(feature, 0.0))
        records.append(
            {
                "feature": feature,
                "gain": round(float(gain.get(feature, 0.0)), 8),
                "total_gain": round(feature_total, 8),
                "weight": int(weight.get(feature, 0.0)),
                "normalized_total_gain": round(
                    feature_total / total if total > 0.0 else 0.0,
                    8,
                ),
            }
        )

    records.sort(
        key=lambda record: (
            record["normalized_total_gain"],
            record["weight"],
        ),
        reverse=True,
    )

    return {
        "importance_type": "total_gain",
        "feature_count": len(feature_columns),
        "features": records,
    }


def compare_with_baseline(
    settings: XGBoostSettings,
    *,
    validation_metrics: dict[str, Any],
    test_metrics: dict[str, Any],
) -> dict[str, Any]:
    """Compare model metrics with the strongest MAE baseline."""
    baseline = load_json_mapping(
        settings.baseline_report_path,
        description="baseline report",
    )

    try:
        validation_name = str(
            baseline["best_baseline_by_mae"]["validation"]
        )
        test_name = str(baseline["best_baseline_by_mae"]["test"])
        validation_baseline = baseline["metrics"]["validation"][
            validation_name
        ]
        test_baseline = baseline["metrics"]["test"][test_name]
    except (KeyError, TypeError) as error:
        raise XGBoostTrainingError(
            f"Baseline report has an invalid structure: {error}"
        ) from error

    def one_split(
        model_metrics: dict[str, Any],
        baseline_metrics: dict[str, Any],
        baseline_name: str,
    ) -> dict[str, Any]:
        comparison: dict[str, Any] = {
            "baseline": baseline_name,
            "baseline_metrics": baseline_metrics,
            "model_metrics": model_metrics,
        }
        for metric_name in ("mae", "rmse", "wape", "smape"):
            old = float(baseline_metrics[metric_name])
            new = float(model_metrics[metric_name])
            improvement = old - new
            comparison[f"{metric_name}_absolute_improvement"] = round(
                improvement,
                8,
            )
            comparison[f"{metric_name}_relative_improvement"] = round(
                improvement / old if old != 0.0 else 0.0,
                8,
            )

        comparison["beats_baseline_by_mae"] = (
            float(model_metrics["mae"])
            < float(baseline_metrics["mae"])
        )
        return comparison

    return {
        "validation": one_split(
            validation_metrics,
            validation_baseline,
            validation_name,
        ),
        "test": one_split(test_metrics, test_baseline, test_name),
    }


def write_json_atomically(path: Path, payload: dict[str, Any]) -> None:
    """Write a JSON artifact atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.part")
    temporary.unlink(missing_ok=True)

    try:
        temporary.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    except OSError as error:
        temporary.unlink(missing_ok=True)
        raise XGBoostTrainingError(
            f"Could not write JSON artifact {path}: {error}"
        ) from error


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Calculate a file SHA-256 digest."""
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while chunk := handle.read(chunk_size):
                digest.update(chunk)
    except OSError as error:
        raise XGBoostTrainingError(
            f"Could not hash file {path}: {error}"
        ) from error
    return digest.hexdigest()


def library_versions() -> dict[str, str]:
    """Return runtime versions for reproducibility."""
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "polars": pl.__version__,
        "numpy": np.__version__,
        "xgboost": xgb.__version__,
        "scikit_learn": sklearn.__version__,
    }


def validate_metric_values(
    metrics: dict[str, Any],
    *,
    split_name: str,
) -> None:
    """Reject empty or non-finite evaluation results."""
    if int(metrics["row_count"]) <= 0:
        raise XGBoostTrainingError(
            f"{split_name} metrics contain no rows."
        )

    for metric_name in ("mae", "rmse", "wape", "smape", "mean_bias"):
        if not math.isfinite(float(metrics[metric_name])):
            raise XGBoostTrainingError(
                f"{split_name} metric is not finite: {metric_name}"
            )


def train_xgboost_forecaster(config_path: Path) -> dict[str, Any]:
    """Select, refit, evaluate, and persist a global XGBoost model."""
    settings = XGBoostSettings.from_yaml(config_path)
    input_summary = validate_input_dataset(settings)
    categorical_contract = build_categorical_contract(settings)

    training = load_prepared_dataset(
        settings,
        splits=settings.training_splits,
        categorical_contract=categorical_contract,
    )
    validation = load_prepared_dataset(
        settings,
        splits=settings.validation_splits,
        categorical_contract=categorical_contract,
    )

    selection_model, selection = fit_selection_model(
        settings,
        training=training,
        validation=validation,
    )
    validation_prediction = clip_predictions(
        selection_model.predict(validation.features),
        minimum_prediction=settings.minimum_prediction,
    )
    validation_metrics = calculate_regression_metrics(
        validation.target,
        validation_prediction,
    )
    validate_metric_values(validation_metrics, split_name="validation")
    validation_prediction_frame = build_prediction_frame(
        dataset=validation,
        prediction=validation_prediction,
        model_stage="selection",
        minimum_prediction=settings.minimum_prediction,
    )

    del selection_model, training
    gc.collect()

    final_training = load_prepared_dataset(
        settings,
        splits=settings.final_training_splits,
        categorical_contract=categorical_contract,
    )
    test = load_prepared_dataset(
        settings,
        splits=settings.test_splits,
        categorical_contract=categorical_contract,
    )

    final_model = fit_final_model(
        settings,
        final_training=final_training,
        selected_n_estimators=selection["selected_n_estimators"],
    )
    test_prediction = clip_predictions(
        final_model.predict(test.features),
        minimum_prediction=settings.minimum_prediction,
    )
    test_metrics = calculate_regression_metrics(
        test.target,
        test_prediction,
    )
    validate_metric_values(test_metrics, split_name="test")
    test_prediction_frame = build_prediction_frame(
        dataset=test,
        prediction=test_prediction,
        model_stage="final",
        minimum_prediction=settings.minimum_prediction,
    )

    predictions_artifact = write_predictions(
        settings,
        validation_predictions=validation_prediction_frame,
        test_predictions=test_prediction_frame,
    )
    model_artifact = save_model_atomically(
        final_model,
        model_path=settings.model_path,
    )

    feature_importance = collect_feature_importance(
        final_model,
        feature_columns=settings.feature_columns,
    )
    write_json_atomically(
        settings.feature_importance_path,
        feature_importance,
    )

    baseline_comparison = compare_with_baseline(
        settings,
        validation_metrics=validation_metrics,
        test_metrics=test_metrics,
    )

    metadata = {
        "metadata_version": "0.1.0",
        "generated_at": datetime.now(UTC).isoformat(),
        "dataset_name": settings.dataset_name,
        "timezone": settings.timezone,
        "model_format": settings.model_path.suffix.lower().lstrip("."),
        "model_artifact": model_artifact,
        "selected_n_estimators": selection["selected_n_estimators"],
        "feature_columns": list(settings.feature_columns),
        "categorical_columns": list(settings.categorical_columns),
        "categorical_contract": {
            column: list(categories)
            for column, categories in categorical_contract.items()
        },
        "minimum_prediction": settings.minimum_prediction,
        "model_parameters": settings.final_model_parameters(
            selected_n_estimators=selection["selected_n_estimators"]
        ),
        "top_features": feature_importance["features"][:15],
        "library_versions": library_versions(),
    }
    write_json_atomically(settings.metadata_path, metadata)

    report: dict[str, Any] = {
        "xgboost_pipeline_version": "0.1.0",
        "generated_at": datetime.now(UTC).isoformat(),
        "effective_config": settings.as_report_dict(),
        "library_versions": library_versions(),
        "input": input_summary,
        "categorical_contract": {
            column: {
                "category_count": len(categories),
                "categories": list(categories),
            }
            for column, categories in categorical_contract.items()
        },
        "model_selection": selection,
        "final_training": {
            "row_count": final_training.row_count,
            "selected_n_estimators": selection["selected_n_estimators"],
        },
        "metrics": {
            "validation": validation_metrics,
            "test": test_metrics,
        },
        "baseline_comparison": baseline_comparison,
        "artifacts": {
            "model": model_artifact,
            "metadata": {
                "path": str(settings.metadata_path),
                "size_bytes": settings.metadata_path.stat().st_size,
            },
            "predictions": predictions_artifact,
            "feature_importance": {
                "path": str(settings.feature_importance_path),
                "size_bytes": (
                    settings.feature_importance_path.stat().st_size
                ),
            },
        },
        "invariants": {
            "validation_uses_training_only": True,
            "test_is_not_used_for_model_selection": True,
            "final_training_uses_train_and_validation": True,
            "categorical_contract_is_fixed": True,
            "prediction_duplicate_key_count": (
                predictions_artifact["duplicate_key_count"]
            ),
            "prediction_null_value_count": (
                predictions_artifact["null_value_count"]
            ),
            "negative_prediction_count": (
                predictions_artifact["negative_prediction_count"]
            ),
            "validation_metric_rows_match": (
                validation_metrics["row_count"]
                == validation_prediction_frame.height
            ),
            "test_metric_rows_match": (
                test_metrics["row_count"]
                == test_prediction_frame.height
            ),
        },
    }

    write_json_atomically(settings.report_path, report)

    del final_training, test, final_model
    gc.collect()

    return report