"""Leakage-aware global graph/weather quantile demand forecasting with XGBoost."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
try:
    import polars as pl
except ImportError:  # Pure metric/projector utilities remain importable without Polars.
    pl = None  # type: ignore[assignment]

import xgboost as xgb
import yaml

from urbanflow.simulation.decision_time_forecasting import DecisionTimeProjector


class ProbabilisticForecastError(RuntimeError):
    """Raised when quantile forecasting violates its data or model contract."""


QUANTILES = (0.10, 0.50, 0.90)
QUANTILE_NAMES = ("p10_demand", "p50_demand", "p90_demand")


def pinball_loss(actual: np.ndarray, prediction: np.ndarray, quantile: float) -> float:
    """Calculate mean pinball loss for one quantile."""
    y = np.asarray(actual, dtype=np.float64)
    p = np.asarray(prediction, dtype=np.float64)
    if y.shape != p.shape or y.size == 0:
        raise ProbabilisticForecastError("Pinball inputs must have equal non-empty shapes.")
    if not 0.0 < quantile < 1.0:
        raise ProbabilisticForecastError("quantile must be between zero and one.")
    if np.any(~np.isfinite(y)) or np.any(~np.isfinite(p)):
        raise ProbabilisticForecastError("Pinball inputs must be finite.")
    error = y - p
    return float(np.maximum(quantile * error, (quantile - 1.0) * error).mean())


def enforce_quantile_order(
    predictions: np.ndarray,
    *,
    minimum_prediction: float = 0.0,
) -> tuple[np.ndarray, int]:
    """Clip negative predictions and monotonically reorder quantiles per row."""
    values = np.asarray(predictions, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ProbabilisticForecastError("Predictions must have shape (rows, 3).")
    if np.any(~np.isfinite(values)):
        raise ProbabilisticForecastError("Quantile predictions must be finite.")
    if not np.isfinite(minimum_prediction) or minimum_prediction < 0:
        raise ProbabilisticForecastError("minimum_prediction must be finite and non-negative.")
    clipped = np.maximum(values, minimum_prediction)
    crossing_count = int(np.any(np.diff(clipped, axis=1) < 0, axis=1).sum())
    ordered = np.sort(clipped, axis=1)
    return ordered, crossing_count


def quantile_metrics(
    actual: np.ndarray,
    predictions: np.ndarray,
) -> dict[str, float | int]:
    """Return proper quantile and central-interval diagnostics."""
    y = np.asarray(actual, dtype=np.float64)
    values = np.asarray(predictions, dtype=np.float64)
    if values.shape != (y.size, 3) or y.size == 0:
        raise ProbabilisticForecastError("Metric inputs must have shapes (n,) and (n, 3).")
    if np.any(~np.isfinite(y)) or np.any(~np.isfinite(values)):
        raise ProbabilisticForecastError("Metric inputs must be finite.")
    p10, p50, p90 = values.T
    denominator = float(np.abs(y).sum())
    return {
        "row_count": int(y.size),
        "p10_pinball": pinball_loss(y, p10, 0.10),
        "p50_pinball": pinball_loss(y, p50, 0.50),
        "p90_pinball": pinball_loss(y, p90, 0.90),
        "mean_pinball": float(
            np.mean(
                [
                    pinball_loss(y, p10, 0.10),
                    pinball_loss(y, p50, 0.50),
                    pinball_loss(y, p90, 0.90),
                ]
            )
        ),
        "p50_mae": float(np.abs(y - p50).mean()),
        "p50_wape": float(np.abs(y - p50).sum() / denominator) if denominator > 0 else 0.0,
        "p10_empirical_coverage": float(np.mean(y <= p10)),
        "p90_empirical_coverage": float(np.mean(y <= p90)),
        "central_80_coverage": float(np.mean((y >= p10) & (y <= p90))),
        "mean_interval_width": float(np.mean(p90 - p10)),
        "negative_prediction_count": int(np.sum(values < 0)),
        "quantile_crossing_count": int(np.sum((p10 > p50) | (p50 > p90))),
    }


@dataclass(frozen=True)
class ProbabilisticForecastSettings:
    """Settings for chronological quantile-model selection, refit and prediction."""

    input_path: Path
    model_directory: Path
    metadata_path: Path
    predictions_path: Path
    report_path: Path
    target_column: str
    split_column: str
    key_columns: tuple[str, ...]
    categorical_columns: tuple[str, ...]
    feature_columns: tuple[str, ...]
    train_start: date
    selection_train_end_exclusive: date
    selection_validation_start: date
    selection_validation_end_exclusive: date
    prediction_start: date
    prediction_end_exclusive: date
    selection_validation_split: str
    final_training_splits: tuple[str, ...]
    prediction_splits: tuple[str, ...]
    estimator_candidates: tuple[int, ...]
    learning_rate: float
    max_depth: int
    min_child_weight: float
    subsample: float
    colsample_bytree: float
    reg_lambda: float
    max_bin: int
    max_cat_to_onehot: int
    max_cat_threshold: int
    random_state: int
    n_jobs: int
    verbosity: int
    minimum_prediction: float
    compression: str

    @classmethod
    def from_yaml(cls, path: Path) -> ProbabilisticForecastSettings:
        if not path.exists():
            raise ProbabilisticForecastError(f"Quantile config does not exist: {path}")
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8"))
            paths, schema = raw["paths"], raw["schema"]
            chronology, model, output = raw["chronology"], raw["model"], raw["output"]
            settings = cls(
                input_path=Path(paths["input"]),
                model_directory=Path(paths["model_directory"]),
                metadata_path=Path(paths["metadata"]),
                predictions_path=Path(paths["predictions"]),
                report_path=Path(paths["report"]),
                target_column=str(schema["target_column"]),
                split_column=str(schema["split_column"]),
                key_columns=tuple(str(v) for v in schema["key_columns"]),
                categorical_columns=tuple(str(v) for v in schema["categorical_columns"]),
                feature_columns=tuple(str(v) for v in schema["feature_columns"]),
                train_start=date.fromisoformat(str(chronology["train_start"])),
                selection_train_end_exclusive=date.fromisoformat(
                    str(chronology["selection_train_end_exclusive"])
                ),
                selection_validation_start=date.fromisoformat(
                    str(chronology["selection_validation_start"])
                ),
                selection_validation_end_exclusive=date.fromisoformat(
                    str(chronology["selection_validation_end_exclusive"])
                ),
                prediction_start=date.fromisoformat(str(chronology["prediction_start"])),
                prediction_end_exclusive=date.fromisoformat(
                    str(chronology["prediction_end_exclusive"])
                ),
                selection_validation_split=str(
                    chronology.get("selection_validation_split", "train")
                ),
                final_training_splits=tuple(
                    str(v) for v in chronology.get("final_training_splits", ["train"])
                ),
                prediction_splits=tuple(
                    str(v)
                    for v in chronology.get("prediction_splits", ["validation", "test"])
                ),
                estimator_candidates=tuple(int(v) for v in model["estimator_candidates"]),
                learning_rate=float(model["learning_rate"]),
                max_depth=int(model["max_depth"]),
                min_child_weight=float(model["min_child_weight"]),
                subsample=float(model["subsample"]),
                colsample_bytree=float(model["colsample_bytree"]),
                reg_lambda=float(model["reg_lambda"]),
                max_bin=int(model["max_bin"]),
                max_cat_to_onehot=int(model["max_cat_to_onehot"]),
                max_cat_threshold=int(model["max_cat_threshold"]),
                random_state=int(model["random_state"]),
                n_jobs=int(model["n_jobs"]),
                verbosity=int(model["verbosity"]),
                minimum_prediction=float(model["minimum_prediction"]),
                compression=str(output["compression"]),
            )
        except (KeyError, TypeError, ValueError, yaml.YAMLError) as error:
            raise ProbabilisticForecastError(f"Invalid quantile config: {error}") from error
        settings.validate()
        return settings

    def validate(self) -> None:
        if not self.input_path.exists() or self.input_path.stat().st_size == 0:
            raise ProbabilisticForecastError(f"Feature dataset is missing: {self.input_path}")
        if not self.key_columns or not self.feature_columns:
            raise ProbabilisticForecastError("Key and feature columns must be configured.")
        if not set(self.categorical_columns).issubset(self.feature_columns):
            raise ProbabilisticForecastError("Categorical columns must be model features.")
        if tuple(sorted(set(self.estimator_candidates))) != self.estimator_candidates:
            raise ProbabilisticForecastError("Estimator candidates must be unique and ascending.")
        if not self.estimator_candidates or self.estimator_candidates[0] < 1:
            raise ProbabilisticForecastError("Estimator candidates must be positive.")
        if (
            not self.selection_validation_split
            or not self.final_training_splits
            or not self.prediction_splits
        ):
            raise ProbabilisticForecastError("Chronology split selections must not be empty.")
        if "test" in self.final_training_splits:
            raise ProbabilisticForecastError("The test split cannot be used for final model training.")
        if set(self.final_training_splits).intersection(self.prediction_splits):
            raise ProbabilisticForecastError(
                "Prediction splits must be disjoint from final-training splits."
            )
        if not (self.train_start < self.selection_train_end_exclusive):
            raise ProbabilisticForecastError("Selection training window is invalid.")
        if not (
            self.selection_train_end_exclusive
            == self.selection_validation_start
            < self.selection_validation_end_exclusive
            == self.prediction_start
            < self.prediction_end_exclusive
        ):
            raise ProbabilisticForecastError(
                "Chronological windows must be contiguous and ordered."
            )
        if self.learning_rate <= 0 or self.max_depth <= 0 or self.max_bin < 2:
            raise ProbabilisticForecastError("Invalid XGBoost hyperparameters.")
        if not 0 < self.subsample <= 1 or not 0 < self.colsample_bytree <= 1:
            raise ProbabilisticForecastError("Sampling fractions must be in (0, 1].")
        if self.minimum_prediction < 0:
            raise ProbabilisticForecastError("minimum_prediction must be non-negative.")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _model_parameters(
    settings: ProbabilisticForecastSettings,
    quantile: float,
    trees: int,
) -> dict[str, Any]:
    return {
        "objective": "reg:quantileerror",
        "quantile_alpha": quantile,
        "tree_method": "hist",
        "enable_categorical": True,
        "n_estimators": trees,
        "learning_rate": settings.learning_rate,
        "max_depth": settings.max_depth,
        "min_child_weight": settings.min_child_weight,
        "subsample": settings.subsample,
        "colsample_bytree": settings.colsample_bytree,
        "reg_lambda": settings.reg_lambda,
        "max_bin": settings.max_bin,
        "max_cat_to_onehot": settings.max_cat_to_onehot,
        "max_cat_threshold": settings.max_cat_threshold,
        "random_state": settings.random_state,
        "n_jobs": settings.n_jobs,
        "verbosity": settings.verbosity,
    }


def _require_polars() -> Any:
    if pl is None:
        raise ProbabilisticForecastError(
            "Polars is required for dataset loading and model training. "
            "Install project dependencies."
        )
    return pl


def _read_slice(
    settings: ProbabilisticForecastSettings,
    *,
    start: date,
    end: date,
    split_values: tuple[str, ...],
) -> pl.DataFrame:
    polars = _require_polars()
    columns = list(
        dict.fromkeys(
            [
                *settings.key_columns,
                settings.split_column,
                settings.target_column,
                "service_date",
                "borough",
                *settings.feature_columns,
            ]
        )
    )
    frame = (
        polars.scan_parquet(settings.input_path)
        .filter(polars.col("service_date") >= start)
        .filter(polars.col("service_date") < end)
        .filter(polars.col(settings.split_column).is_in(list(split_values)))
        .select(columns)
        .sort(*settings.key_columns)
        .collect()
    )
    if frame.height == 0:
        raise ProbabilisticForecastError(
            f"No rows in {start.isoformat()} to {end.isoformat()} for {split_values}."
        )
    target = frame.get_column(settings.target_column).cast(pl.Float64).to_numpy()
    if np.any(~np.isfinite(target)) or np.any(target < 0):
        raise ProbabilisticForecastError("Target contains invalid demand values.")
    return frame


def _categorical_contract(
    frame: pl.DataFrame,
    categorical_columns: tuple[str, ...],
) -> dict[str, tuple[str, ...]]:
    contract: dict[str, tuple[str, ...]] = {}
    for column in categorical_columns:
        categories = tuple(
            str(value)
            for value in (
                frame.get_column(column)
                .cast(pl.String)
                .drop_nulls()
                .unique()
                .sort()
                .to_list()
            )
        )
        if not categories:
            raise ProbabilisticForecastError(f"No training categories for {column}.")
        contract[column] = categories
    return contract


def _features(
    frame: pl.DataFrame,
    settings: ProbabilisticForecastSettings,
    contract: dict[str, tuple[str, ...]],
) -> pl.DataFrame:
    missing = sorted(set(settings.feature_columns).difference(frame.columns))
    if missing:
        raise ProbabilisticForecastError("Missing model features: " + ", ".join(missing))
    from urbanflow.forecasting.xgboost_model import apply_categorical_contract

    return apply_categorical_contract(
        frame.select(*settings.feature_columns),
        contract=contract,
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def train_probabilistic_forecaster(config_path: Path) -> dict[str, Any]:
    """Select quantile model complexity on a past window and forecast a later period."""
    settings = ProbabilisticForecastSettings.from_yaml(config_path)
    train = _read_slice(
        settings,
        start=settings.train_start,
        end=settings.selection_train_end_exclusive,
        split_values=("train",),
    )
    validation = _read_slice(
        settings,
        start=settings.selection_validation_start,
        end=settings.selection_validation_end_exclusive,
        split_values=(settings.selection_validation_split,),
    )
    final_training = _read_slice(
        settings,
        start=settings.train_start,
        end=settings.prediction_start,
        split_values=settings.final_training_splits,
    )
    prediction = _read_slice(
        settings,
        start=settings.prediction_start,
        end=settings.prediction_end_exclusive,
        split_values=settings.prediction_splits,
    )
    if train.get_column("service_date").max() >= settings.selection_train_end_exclusive:
        raise ProbabilisticForecastError("Selection training extends beyond its configured window.")
    if validation.get_column("service_date").min() < settings.selection_validation_start:
        raise ProbabilisticForecastError("Selection validation contains pre-validation rows.")
    if final_training.get_column("service_date").max() >= settings.prediction_start:
        raise ProbabilisticForecastError("Final training includes prediction-period labels.")
    if prediction.get_column("service_date").min() < settings.prediction_start:
        raise ProbabilisticForecastError("Prediction window contains pre-prediction rows.")

    contract = _categorical_contract(train, settings.categorical_columns)
    x_train = _features(train, settings, contract)
    x_validation = _features(validation, settings, contract)
    y_train = train.get_column(settings.target_column).cast(pl.Float32).to_numpy()
    y_validation = validation.get_column(settings.target_column).cast(pl.Float32).to_numpy()

    selection: dict[str, Any] = {}
    selected_estimators: dict[str, int] = {}
    for quantile, name in zip(QUANTILES, QUANTILE_NAMES, strict=True):
        candidate_results: list[dict[str, Any]] = []
        for trees in settings.estimator_candidates:
            candidate = xgb.XGBRegressor(
                **_model_parameters(settings, quantile, trees)
            )
            candidate.fit(x_train, y_train, verbose=False)
            values = np.maximum(
                candidate.predict(x_validation),
                settings.minimum_prediction,
            )
            loss = pinball_loss(y_validation, values, quantile)
            candidate_results.append(
                {"n_estimators": trees, "validation_pinball_loss": loss}
            )
            del candidate
        winner = min(
            candidate_results,
            key=lambda item: (float(item["validation_pinball_loss"]), int(item["n_estimators"])),
        )
        selected_estimators[name] = int(winner["n_estimators"])
        selection[name] = {
            "selected_n_estimators": int(winner["n_estimators"]),
            "best_validation_pinball_loss": float(winner["validation_pinball_loss"]),
            "candidate_metrics": candidate_results,
        }

    x_final = _features(final_training, settings, contract)
    y_final = final_training.get_column(settings.target_column).cast(pl.Float32).to_numpy()
    x_prediction = _features(prediction, settings, contract)
    model_directory = settings.model_directory
    model_directory.mkdir(parents=True, exist_ok=True)
    raw_columns: list[np.ndarray] = []
    for quantile, name in zip(QUANTILES, QUANTILE_NAMES, strict=True):
        model = xgb.XGBRegressor(
            **_model_parameters(settings, quantile, selected_estimators[name])
        )
        model.fit(x_final, y_final, verbose=False)
        model_path = model_directory / f"{name}.ubj"
        temporary = model_path.with_name(model_path.name + ".part.ubj")
        model.save_model(temporary)
        temporary.replace(model_path)
        raw_columns.append(model.predict(x_prediction))

    raw_predictions = np.column_stack(raw_columns)
    ordered_predictions, crossing_count = enforce_quantile_order(
        raw_predictions,
        minimum_prediction=settings.minimum_prediction,
    )
    prediction_output = prediction.select(
        *settings.key_columns,
        settings.split_column,
        "borough",
    ).with_columns(
        pl.col("interval_start").alias("decision_time"),
        pl.Series("p10_demand", ordered_predictions[:, 0]),
        pl.Series("p50_demand", ordered_predictions[:, 1]),
        pl.Series("p90_demand", ordered_predictions[:, 2]),
        pl.lit("xgboost_quantile_graph_weather_asof_v1").alias("forecast_provenance"),
    ).sort(*settings.key_columns)
    if prediction_output.null_count().to_numpy().sum():
        raise ProbabilisticForecastError("Quantile prediction output contains nulls.")
    if prediction_output.group_by(*settings.key_columns).len().filter(pl.col("len") > 1).height:
        raise ProbabilisticForecastError("Quantile prediction output contains duplicate keys.")

    settings.predictions_path.parent.mkdir(parents=True, exist_ok=True)
    prediction_tmp = settings.predictions_path.with_name(settings.predictions_path.name + ".part")
    prediction_output.write_parquet(
        prediction_tmp,
        compression=settings.compression,
        statistics=True,
    )
    prediction_tmp.replace(settings.predictions_path)

    report: dict[str, Any] = {
        "version": "urbanflow_global_graph_weather_quantile_v1",
        "quantiles": list(QUANTILES),
        "chronology": {
            "selection_training": [
                settings.train_start.isoformat(),
                settings.selection_train_end_exclusive.isoformat(),
            ],
            "selection_validation": [
                settings.selection_validation_start.isoformat(),
                settings.selection_validation_end_exclusive.isoformat(),
            ],
            "final_training": [
                settings.train_start.isoformat(),
                settings.prediction_start.isoformat(),
            ],
            "prediction": [
                settings.prediction_start.isoformat(),
                settings.prediction_end_exclusive.isoformat(),
            ],
            "selection_validation_split": settings.selection_validation_split,
            "final_training_splits": list(settings.final_training_splits),
            "prediction_splits": list(settings.prediction_splits),
            "prediction_period_labels_used_for_selection": False,
            "prediction_period_labels_used_for_training": False,
            "weather_observations_used_as_features": False,
        },
        "selection": selection,
        "selected_estimators": selected_estimators,
        "training_rows": train.height,
        "validation_rows": validation.height,
        "final_training_rows": final_training.height,
        "prediction_rows": prediction.height,
        "zone_count": prediction.get_column("zone_id").n_unique(),
        "pre_correction_crossing_rows": crossing_count,
        "validation_metrics": {
            "selection_pinball_by_quantile": {
                name: selection[name]["best_validation_pinball_loss"]
                for name in QUANTILE_NAMES
            }
        },
        "test_metrics": {},
        "models": {},
        "predictions_path": str(settings.predictions_path),
        "invariants": {
            "train_precedes_validation": train.get_column("service_date").max()
            < validation.get_column("service_date").min(),
            "final_training_excludes_prediction_period": (
                final_training.get_column("service_date").max()
                < prediction.get_column("service_date").min()
            ),
            "prediction_period_excluded_from_model_selection": (
                validation.get_column("service_date").max()
                < prediction.get_column("service_date").min()
            ),
            "predictions_are_non_negative": bool(np.all(ordered_predictions >= 0)),
            "quantiles_are_monotonic": bool(np.all(np.diff(ordered_predictions, axis=1) >= 0)),
            "prediction_row_count_preserved": prediction_output.height == prediction.height,
            "prediction_keys_unique": True,
        },
    }
    target = prediction.get_column(settings.target_column).cast(pl.Float64).to_numpy()
    split_values = prediction.get_column(settings.split_column).to_list()
    test_mask = np.asarray(split_values) == "test"
    if test_mask.any():
        report["test_metrics"] = quantile_metrics(target[test_mask], ordered_predictions[test_mask])
    for name in QUANTILE_NAMES:
        model_path = model_directory / f"{name}.ubj"
        report["models"][name] = {
            "path": str(model_path),
            "sha256": _sha256(model_path),
            "n_estimators": selected_estimators[name],
        }
    metadata = {
        "version": report["version"],
        "feature_columns": list(settings.feature_columns),
        "categorical_contract": {key: list(values) for key, values in contract.items()},
        "quantiles": list(QUANTILES),
        "selected_estimators": selected_estimators,
        "chronology": report["chronology"],
        "models": report["models"],
        "config_sha256": _sha256(config_path),
        "input_sha256": _sha256(settings.input_path),
    }
    _write_json(settings.metadata_path, metadata)
    _write_json(settings.report_path, report)
    return report


@dataclass(frozen=True)
class DecisionTimeQuantileProjector:
    """Leakage-safe multi-step quantile projection around a train-only profile."""

    point_projector: DecisionTimeProjector

    def project(
        self,
        *,
        decision_time: datetime,
        current_quantiles: np.ndarray,
        horizon_steps: int,
    ) -> np.ndarray:
        """Project shape (3, zones) anchors to (horizon, 3, zones)."""
        anchors = np.asarray(current_quantiles, dtype=np.float64)
        if anchors.shape != (3, len(self.point_projector.zone_ids)):
            raise ProbabilisticForecastError("current_quantiles must have shape (3, zone_count).")
        if np.any(~np.isfinite(anchors)) or np.any(anchors < 0):
            raise ProbabilisticForecastError("Quantile anchors must be finite and non-negative.")
        projected = np.stack(
            [
                self.point_projector.project(
                    decision_time=decision_time,
                    current_prediction=anchors[index],
                    horizon_steps=horizon_steps,
                )
                for index in range(3)
            ],
            axis=1,
        )
        return np.sort(np.maximum(projected, 0.0), axis=1)


def fit_train_only_quantile_projector(
    historical_demand: Any,
    *,
    zone_ids: tuple[int, ...],
    interval_minutes: int = 15,
) -> DecisionTimeQuantileProjector:
    """Fit horizon shape from train split only; May/June labels are rejected."""
    polars = _require_polars()
    training = historical_demand.filter(polars.col("split") == "train")
    if training.height == 0:
        raise ProbabilisticForecastError("Train-only demand history is empty.")
    if historical_demand.filter(polars.col("split").is_in(["validation", "test"])).height == 0:
        raise ProbabilisticForecastError("Expected validation/test rows for leakage guard.")
    try:
        projector = DecisionTimeProjector.fit(
            zone_ids=zone_ids,
            timestamps=training.get_column("interval_start").to_list(),
            zone_indices=training.get_column("zone_index").to_numpy().astype(np.int64),
            demand=training.get_column("actual_demand").to_numpy().astype(np.int64),
            split_labels=training.get_column("split").to_list(),
            interval_minutes=interval_minutes,
            training_split="train",
        )
    except ValueError as error:
        raise ProbabilisticForecastError(
            f"Could not fit quantile horizon profile: {error}"
        ) from error
    return DecisionTimeQuantileProjector(projector)
