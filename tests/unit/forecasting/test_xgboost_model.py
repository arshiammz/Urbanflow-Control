from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from urbanflow.forecasting.xgboost_model import (
    PreparedDataset,
    XGBoostSettings,
    XGBoostTrainingError,
    apply_categorical_contract,
    build_categorical_contract,
    calculate_regression_metrics,
    clip_predictions,
    fit_final_model,
    fit_selection_model,
    load_prepared_dataset,
    ordered_unique,
    validate_input_dataset,
)


TIMEZONE = "America/New_York"


def make_feature_frame() -> pl.DataFrame:
    """Build a small train-validation-test forecasting dataset."""
    interval_starts: list[datetime] = []
    zone_ids: list[int] = []
    boroughs: list[str] = []
    service_zones: list[str] = []
    splits: list[str] = []
    targets: list[int] = []
    hours: list[int] = []
    lag_1_values: list[int] = []

    start = datetime(
        year=2024,
        month=1,
        day=1,
        hour=0,
        minute=0,
    )

    for interval_index in range(12):
        interval_start = start + timedelta(
            minutes=15 * interval_index
        )

        if interval_index < 6:
            split = "train"
        elif interval_index < 9:
            split = "validation"
        else:
            split = "test"

        for zone_id in (1, 2):
            target = (
                interval_index * 2
                + zone_id
            )

            previous_target = max(
                0,
                (interval_index - 1) * 2
                + zone_id,
            )

            interval_starts.append(
                interval_start
            )
            zone_ids.append(
                zone_id
            )
            boroughs.append(
                (
                    "Manhattan"
                    if zone_id == 1
                    else "Queens"
                )
            )
            service_zones.append(
                (
                    "Yellow Zone"
                    if zone_id == 1
                    else "Boro Zone"
                )
            )
            splits.append(
                split
            )
            targets.append(
                target
            )
            hours.append(
                interval_start.hour
            )
            lag_1_values.append(
                previous_target
            )

    return (
        pl.DataFrame(
            {
                "interval_start": interval_starts,
                "zone_id": zone_ids,
                "borough": boroughs,
                "service_zone": service_zones,
                "split": splits,
                "target_pickup_count": targets,
                "hour": hours,
                "pickup_count_lag_1": lag_1_values,
            }
        )
        .with_columns(
            pl.col("interval_start")
            .dt.replace_time_zone(
                TIMEZONE
            )
        )
        .sort(
            "interval_start",
            "zone_id",
        )
    )


def make_settings(
    tmp_path: Path,
    *,
    frame: pl.DataFrame | None = None,
) -> XGBoostSettings:
    """Create valid temporary XGBoost settings."""
    if frame is None:
        frame = make_feature_frame()

    input_path = (
        tmp_path
        / "forecast_features.parquet"
    )

    baseline_report_path = (
        tmp_path
        / "baseline_metrics.json"
    )

    frame.write_parquet(
        input_path
    )

    baseline_report_path.write_text(
        "{}\n",
        encoding="utf-8",
    )

    return XGBoostSettings(
        dataset_name="test_global_xgboost",
        timezone=TIMEZONE,
        input_path=input_path,
        model_path=(
            tmp_path
            / "model.ubj"
        ),
        metadata_path=(
            tmp_path
            / "metadata.json"
        ),
        predictions_path=(
            tmp_path
            / "predictions.parquet"
        ),
        report_path=(
            tmp_path
            / "metrics.json"
        ),
        feature_importance_path=(
            tmp_path
            / "feature_importance.json"
        ),
        baseline_report_path=(
            baseline_report_path
        ),
        target_column=(
            "target_pickup_count"
        ),
        split_column="split",
        key_columns=(
            "interval_start",
            "zone_id",
        ),
        categorical_columns=(
            "zone_id",
            "borough",
            "service_zone",
        ),
        feature_columns=(
            "zone_id",
            "borough",
            "service_zone",
            "hour",
            "pickup_count_lag_1",
        ),
        training_splits=(
            "train",
        ),
        validation_splits=(
            "validation",
        ),
        final_training_splits=(
            "train",
            "validation",
        ),
        test_splits=(
            "test",
        ),
        objective=(
            "reg:squarederror"
        ),
        eval_metric="mae",
        tree_method="hist",
        enable_categorical=True,
        n_estimators=20,
        learning_rate=0.1,
        max_depth=3,
        min_child_weight=1.0,
        subsample=1.0,
        colsample_bytree=1.0,
        reg_alpha=0.0,
        reg_lambda=1.0,
        max_bin=64,
        max_cat_to_onehot=4,
        max_cat_threshold=32,
        early_stopping_rounds=5,
        random_state=42,
        n_jobs=1,
        verbosity=0,
        minimum_prediction=0.0,
        compression="zstd",
    )


def test_ordered_unique_preserves_order() -> None:
    """Duplicate columns should be removed without reordering."""
    result = ordered_unique(
        (
            "interval_start",
            "zone_id",
            "zone_id",
            "split",
            "interval_start",
        )
    )

    assert result == (
        "interval_start",
        "zone_id",
        "split",
    )


def test_calculate_regression_metrics() -> None:
    """Regression metrics should follow the baseline contract."""
    actual = np.array(
        [0.0, 2.0, 4.0],
        dtype=np.float32,
    )

    prediction = np.array(
        [0.0, 1.0, 6.0],
        dtype=np.float32,
    )

    metrics = calculate_regression_metrics(
        actual,
        prediction,
    )

    assert metrics["row_count"] == 3
    assert metrics["actual_sum"] == 6.0
    assert metrics["prediction_sum"] == 7.0
    assert metrics["mae"] == pytest.approx(
        1.0
    )
    assert metrics["rmse"] == pytest.approx(
        1.290994,
        abs=1e-6,
    )
    assert metrics["wape"] == pytest.approx(
        0.5
    )
    assert metrics["smape"] == pytest.approx(
        0.35555556,
        abs=1e-8,
    )
    assert metrics["mean_bias"] == pytest.approx(
        0.333333,
        abs=1e-6,
    )


def test_calculate_regression_metrics_rejects_shape_mismatch() -> None:
    """Actual and prediction arrays must have equal shapes."""
    with pytest.raises(
        XGBoostTrainingError,
        match="different shapes",
    ):
        calculate_regression_metrics(
            np.array(
                [1.0, 2.0],
                dtype=np.float32,
            ),
            np.array(
                [1.0],
                dtype=np.float32,
            ),
        )


def test_clip_predictions_enforces_lower_bound() -> None:
    """Negative demand predictions should be clipped to zero."""
    prediction = np.array(
        [-3.0, 0.0, 2.5],
        dtype=np.float64,
    )

    clipped = clip_predictions(
        prediction,
        minimum_prediction=0.0,
    )

    np.testing.assert_allclose(
        clipped,
        np.array(
            [0.0, 0.0, 2.5],
            dtype=np.float32,
        ),
    )

    assert clipped.dtype == np.float32


def test_apply_categorical_contract_uses_enum() -> None:
    """Configured categorical columns should use fixed Enum types."""
    frame = pl.DataFrame(
        {
            "zone_id": [1, 2, 1],
            "borough": [
                "Manhattan",
                "Queens",
                "Manhattan",
            ],
        }
    )

    converted = apply_categorical_contract(
        frame,
        contract={
            "zone_id": (
                "1",
                "2",
            ),
            "borough": (
                "Manhattan",
                "Queens",
            ),
        },
    )

    assert str(
        converted.schema["zone_id"]
    ).startswith("Enum")

    assert str(
        converted.schema["borough"]
    ).startswith("Enum")

    assert (
        converted.get_column("zone_id")
        .cast(pl.String)
        .to_list()
        == ["1", "2", "1"]
    )


def test_apply_categorical_contract_rejects_unknown_category() -> None:
    """Unseen categories must fail rather than change encoding."""
    frame = pl.DataFrame(
        {
            "zone_id": [
                1,
                999,
            ],
        }
    )

    with pytest.raises(
        XGBoostTrainingError,
        match="categorical contract",
    ):
        apply_categorical_contract(
            frame,
            contract={
                "zone_id": (
                    "1",
                    "2",
                ),
            },
        )


def test_build_categorical_contract_uses_train_and_validation(
    tmp_path: Path,
) -> None:
    """Test-only categories must not influence model preparation."""
    frame = make_feature_frame().with_columns(
        pl.when(
            pl.col("split") == "test"
        )
        .then(
            pl.lit(999)
        )
        .otherwise(
            pl.col("zone_id")
        )
        .alias("zone_id"),

        pl.when(
            pl.col("split") == "test"
        )
        .then(
            pl.lit("Test Borough")
        )
        .otherwise(
            pl.col("borough")
        )
        .alias("borough"),

        pl.when(
            pl.col("split") == "test"
        )
        .then(
            pl.lit("Test Zone")
        )
        .otherwise(
            pl.col("service_zone")
        )
        .alias("service_zone"),
    )

    settings = make_settings(
        tmp_path,
        frame=frame,
    )

    contract = build_categorical_contract(
        settings
    )

    assert contract["zone_id"] == (
        "1",
        "2",
    )

    assert contract["borough"] == (
        "Manhattan",
        "Queens",
    )

    assert contract["service_zone"] == (
        "Boro Zone",
        "Yellow Zone",
    )

    assert "999" not in contract["zone_id"]
    assert (
        "Test Borough"
        not in contract["borough"]
    )
    assert (
        "Test Zone"
        not in contract["service_zone"]
    )


def test_validate_input_dataset_returns_summary(
    tmp_path: Path,
) -> None:
    """A valid feature panel should produce deterministic counts."""
    settings = make_settings(
        tmp_path
    )

    settings.validate()

    summary = validate_input_dataset(
        settings
    )

    assert summary["row_count"] == 24
    assert summary["interval_count"] == 12
    assert summary["zone_count"] == 2

    assert summary["split_counts"] == {
        "test": 6,
        "train": 12,
        "validation": 6,
    }

    assert summary[
        "duplicate_key_count"
    ] == 0

    assert summary[
        "required_null_count"
    ] == 0

    assert summary[
        "target_minimum"
    ] >= 0


def test_validate_input_dataset_rejects_duplicate_keys(
    tmp_path: Path,
) -> None:
    """The prediction key must be unique across the dataset."""
    frame = make_feature_frame()

    duplicated = pl.concat(
        [
            frame,
            frame.slice(
                0,
                1,
            ),
        ],
        how="vertical",
    )

    settings = make_settings(
        tmp_path,
        frame=duplicated,
    )

    with pytest.raises(
        XGBoostTrainingError,
        match="duplicate prediction keys",
    ):
        validate_input_dataset(
            settings
        )


def test_settings_reject_split_leakage(
    tmp_path: Path,
) -> None:
    """Training and validation scopes must never overlap."""
    settings = make_settings(
        tmp_path
    )

    invalid_settings = replace(
        settings,
        validation_splits=(
            "train",
        ),
    )

    with pytest.raises(
        XGBoostTrainingError,
        match="must not overlap",
    ):
        invalid_settings.validate()


def test_load_prepared_dataset_applies_contract(
    tmp_path: Path,
) -> None:
    """Prepared datasets should preserve keys and fixed categories."""
    settings = make_settings(
        tmp_path
    )

    contract = build_categorical_contract(
        settings
    )

    prepared = load_prepared_dataset(
        settings,
        splits=(
            "train",
        ),
        categorical_contract=contract,
    )

    assert isinstance(
        prepared,
        PreparedDataset,
    )

    assert prepared.row_count == 12

    assert prepared.features.columns == list(
        settings.feature_columns
    )

    assert prepared.keys.columns == [
        "interval_start",
        "zone_id",
        "split",
        "target_pickup_count",
    ]

    assert str(
        prepared.features.schema["zone_id"]
    ).startswith("Enum")

    assert str(
        prepared.features.schema["borough"]
    ).startswith("Enum")

    assert prepared.target.dtype == np.float32

    assert (
        prepared.keys
        .get_column("interval_start")
        .is_sorted()
    )


def test_selection_and_final_refit_smoke(
    tmp_path: Path,
) -> None:
    """XGBoost should train with Enum features in both stages."""
    settings = make_settings(
        tmp_path
    )

    settings.validate()

    contract = build_categorical_contract(
        settings
    )

    training = load_prepared_dataset(
        settings,
        splits=settings.training_splits,
        categorical_contract=contract,
    )

    validation = load_prepared_dataset(
        settings,
        splits=settings.validation_splits,
        categorical_contract=contract,
    )

    selection_model, selection_summary = (
        fit_selection_model(
            settings,
            training=training,
            validation=validation,
        )
    )

    selected_n_estimators = int(
        selection_summary[
            "selected_n_estimators"
        ]
    )

    assert (
        1
        <= selected_n_estimators
        <= settings.n_estimators
    )

    validation_prediction = (
        selection_model.predict(
            validation.features
        )
    )

    assert validation_prediction.shape == (
        validation.row_count,
    )

    assert np.isfinite(
        validation_prediction
    ).all()

    final_training = load_prepared_dataset(
        settings,
        splits=settings.final_training_splits,
        categorical_contract=contract,
    )

    test = load_prepared_dataset(
        settings,
        splits=settings.test_splits,
        categorical_contract=contract,
    )

    final_model = fit_final_model(
        settings,
        final_training=final_training,
        selected_n_estimators=(
            selected_n_estimators
        ),
    )

    test_prediction = final_model.predict(
        test.features
    )

    assert test_prediction.shape == (
        test.row_count,
    )

    assert np.isfinite(
        test_prediction
    ).all()