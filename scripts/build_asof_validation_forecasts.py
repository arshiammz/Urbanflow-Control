"""Train a genuinely pre-May XGBoost model and produce ex-ante May nowcasts.

January–March: early-stopping fit; April: internal model selection;
January–April: final refit; May: predictions only, never model selection.
May labels are not used until after the model and forecast are fixed.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import xgboost as xgb
import yaml

from urbanflow.forecasting.xgboost_model import (
    XGBoostSettings,
    apply_categorical_contract,
    clip_predictions,
)
from urbanflow.simulation.forecast_guided_heuristic import write_json_atomically, write_parquet_atomically


CONFIG = Path("configs/simulation/asof_validation_forecast.yaml")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def get_frame(
    *, settings: XGBoostSettings, split: str,
    start: date | None, end: date,
) -> pl.DataFrame:
    """Read just one chronological date slice, including its known features."""
    query = (
        pl.scan_parquet(settings.input_path)
        .filter(pl.col(settings.split_column) == split)
        .filter(pl.col("service_date") < end)
    )
    if start is not None:
        query = query.filter(pl.col("service_date") >= start)
    return query.select(
        *(list(settings.required_columns) + ["service_date"])
    ).sort(*settings.key_columns).collect()


def main() -> None:
    raw = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    paths, periods = raw["paths"], raw["chronology"]
    settings = XGBoostSettings.from_yaml(Path(paths["xgboost_config"]))
    selection_train_end = date.fromisoformat(periods["internal_selection_train_end_exclusive"])
    selection_val_end = date.fromisoformat(periods["internal_selection_validation_end_exclusive"])
    may_start = date.fromisoformat(periods["forecasting_start"])
    may_end = date.fromisoformat(periods["forecasting_end_exclusive"])
    if not (selection_train_end == date(2024, 4, 1)
            and selection_val_end == may_start == date(2024, 5, 1)
            and may_end == date(2024, 6, 1)):
        raise ValueError("As-of validation builder is strictly locked to Jan–Apr fit / May forecast.")
    if settings.training_splits != ("train",) or settings.validation_splits != ("validation",):
        raise ValueError("Unexpected XGBoost split configuration.")

    # Categorical vocabularies are never learned from May or June.
    training_query = pl.scan_parquet(settings.input_path).filter(
        pl.col(settings.split_column) == "train"
    )
    contract = {}
    for name in settings.categorical_columns:
        values = (
            training_query.select(pl.col(name).cast(pl.String).alias(name))
            .unique().drop_nulls().sort(name).collect().get_column(name).to_list()
        )
        if not values:
            raise ValueError(f"Empty train categorical vocabulary: {name}")
        contract[name] = tuple(map(str, values))

    jan_mar = get_frame(settings=settings, split="train", start=None, end=selection_train_end)
    april = get_frame(
        settings=settings, split="train", start=selection_train_end, end=selection_val_end
    )
    may = get_frame(settings=settings, split="validation", start=may_start, end=may_end)
    for name, frame in (("Jan-Mar", jan_mar), ("April", april), ("May", may)):
        if frame.height == 0:
            raise ValueError(f"Empty chronological slice: {name}")
    if jan_mar.get_column("service_date").max() >= selection_train_end:
        raise ValueError("Jan–Mar selection train leaked into April.")
    if april.get_column("service_date").min() < selection_train_end:
        raise ValueError("Internal validation leaked into Jan–Mar.")
    if may.get_column("service_date").min() < may_start:
        raise ValueError("May forecast contains pre-May records.")

    def features(frame: pl.DataFrame) -> pl.DataFrame:
        return apply_categorical_contract(
            frame.select(*settings.feature_columns), contract=contract
        )

    selection = xgb.XGBRegressor(**settings.selection_model_parameters())
    selection.fit(
        features(jan_mar), jan_mar.get_column(settings.target_column).to_numpy(),
        eval_set=[(features(april), april.get_column(settings.target_column).to_numpy())],
        verbose=False,
    )
    best_iteration = int(selection.best_iteration)
    chosen_trees = best_iteration + 1
    if not 1 <= chosen_trees <= settings.n_estimators:
        raise ValueError("Selected estimator count is invalid.")
    print(f"Internal April early stopping: selected {chosen_trees} trees", flush=True)
    del selection

    jan_apr = pl.concat((jan_mar, april), how="vertical")
    del jan_mar, april
    final = xgb.XGBRegressor(
        **settings.final_model_parameters(selected_n_estimators=chosen_trees)
    )
    final.fit(
        features(jan_apr), jan_apr.get_column(settings.target_column).to_numpy(),
        verbose=False,
    )
    del jan_apr
    raw_predictions = final.predict(features(may))
    predictions = clip_predictions(
        raw_predictions, minimum_prediction=settings.minimum_prediction
    )
    out = may.select(*settings.key_columns).with_columns(
        pl.Series("asof_prediction", predictions.astype(np.float64)),
        pl.lit("validation").alias("split"),
        pl.lit("jan_apr_refit_april_selection").alias("forecast_provenance"),
    ).sort(*settings.key_columns)
    if out.height != may.height or out.group_by(*settings.key_columns).len().filter(
        pl.col("len") > 1
    ).height or out.null_count().to_numpy().sum():
        raise ValueError("Invalid as-of predictions output keys/nulls.")
    if not np.isfinite(predictions).all() or np.any(predictions < 0):
        raise ValueError("Invalid as-of prediction values.")

    model_path = Path(paths["model"])
    model_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = model_path.with_name(model_path.name + ".part.ubj")
    final.save_model(tmp)
    tmp.replace(model_path)
    prediction_artifact = write_parquet_atomically(
        out, path=Path(paths["predictions"]), compression=raw["output"]["compression"]
    )
    metadata: dict[str, Any] = {
        "version": "phase1-pre-may-validation-v1",
        "created_at": datetime.now(UTC),
        "chronology": {
            "internal_selection_train": "2024-01-08 to 2024-03-31",
            "internal_selection_validation": "2024-04-01 to 2024-04-30",
            "final_refit": "2024-01-08 to 2024-04-30",
            "prediction_period": "2024-05-01 to 2024-05-31",
            "may_labels_used_for_training": False,
            "may_labels_used_for_model_selection": False,
            "june_data_used": False,
        },
        "selected_trees": chosen_trees,
        "selection_best_iteration": best_iteration,
        "training_categorical_contract": {
            key: list(values) for key, values in contract.items()
        },
        "features": list(settings.feature_columns),
        "artifact": prediction_artifact,
        "model_path": str(model_path),
        "model_sha256": sha256(model_path),
        "config_sha256": sha256(CONFIG),
        "upstream_xgboost_config_sha256": sha256(Path(paths["xgboost_config"])),
        "prediction_sha256": sha256(Path(paths["predictions"])),
    }
    write_json_atomically(Path(paths["metadata"]), metadata)
    print("As-of May forecasts created:", prediction_artifact["path"], flush=True)
    print("Model and metadata:", model_path, paths["metadata"], flush=True)


if __name__ == "__main__":
    main()
