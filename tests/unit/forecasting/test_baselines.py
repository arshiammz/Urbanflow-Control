from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import polars as pl
import pytest
import yaml

from urbanflow.forecasting.baselines import (
    BaselineEvaluationError,
    BaselineEvaluationSettings,
    evaluate_forecasting_baselines,
)


TIMEZONE = "America/New_York"


def write_feature_dataset(
    path: Path,
) -> None:
    """Write a small deterministic forecasting feature dataset."""
    timezone_info = ZoneInfo(
        TIMEZONE
    )

    timestamps = [
        datetime(
            2024,
            1,
            1,
            0,
            0,
            tzinfo=timezone_info,
        )
        + timedelta(hours=index)
        for index in range(6)
    ]

    splits = [
        "train",
        "train",
        "validation",
        "validation",
        "test",
        "test",
    ]

    zone_targets = {
        1: [
            10,
            14,
            18,
            22,
            26,
            30,
        ],
        2: [
            20,
            24,
            28,
            32,
            36,
            40,
        ],
    }

    zone_lag_1 = {
        1: [
            8,
            10,
            14,
            18,
            22,
            26,
        ],
        2: [
            18,
            20,
            24,
            28,
            32,
            36,
        ],
    }

    zone_seasonal_lag = {
        1: [
            6,
            8,
            10,
            14,
            18,
            22,
        ],
        2: [
            16,
            18,
            20,
            24,
            28,
            32,
        ],
    }

    interval_values: list[datetime] = []
    time_step_values: list[int] = []
    service_date_values = []
    zone_id_values: list[int] = []
    borough_values: list[str] = []
    zone_name_values: list[str] = []
    service_zone_values: list[str] = []
    split_values: list[str] = []
    target_values: list[int] = []
    lag_1_values: list[int] = []
    seasonal_lag_values: list[int] = []

    for time_step, timestamp in enumerate(
        timestamps
    ):
        for zone_id in (
            1,
            2,
        ):
            interval_values.append(
                timestamp
            )
            time_step_values.append(
                time_step
            )
            service_date_values.append(
                timestamp.date()
            )
            zone_id_values.append(
                zone_id
            )
            split_values.append(
                splits[time_step]
            )

            if zone_id == 1:
                borough_values.append(
                    "Manhattan"
                )
                zone_name_values.append(
                    "Test Zone One"
                )
                service_zone_values.append(
                    "Yellow Zone"
                )
            else:
                borough_values.append(
                    "Queens"
                )
                zone_name_values.append(
                    "Test Zone Two"
                )
                service_zone_values.append(
                    "Boro Zone"
                )

            target_values.append(
                zone_targets[
                    zone_id
                ][time_step]
            )

            lag_1_values.append(
                zone_lag_1[
                    zone_id
                ][time_step]
            )

            seasonal_lag_values.append(
                zone_seasonal_lag[
                    zone_id
                ][time_step]
            )

    frame = pl.DataFrame(
        {
            "interval_start": pl.Series(
                name="interval_start",
                values=interval_values,
                dtype=pl.Datetime(
                    time_unit="ns",
                    time_zone=TIMEZONE,
                ),
            ),
            "time_step": pl.Series(
                time_step_values,
                dtype=pl.Int32,
            ),
            "service_date": service_date_values,
            "zone_id": pl.Series(
                zone_id_values,
                dtype=pl.Int32,
            ),
            "borough": borough_values,
            "zone_name": zone_name_values,
            "service_zone": service_zone_values,
            "split": split_values,
            "target_pickup_count": pl.Series(
                target_values,
                dtype=pl.Int32,
            ),
            "pickup_count_lag_1": pl.Series(
                lag_1_values,
                dtype=pl.Int32,
            ),
            "pickup_count_lag_96": pl.Series(
                seasonal_lag_values,
                dtype=pl.Int32,
            ),
        }
    ).sort(
        "interval_start",
        "zone_id",
    )

    frame.write_parquet(
        path
    )


def write_baseline_config(
    tmp_path: Path,
    *,
    timezone: str = TIMEZONE,
    validation_fit_splits: list[str] | None = None,
    test_fit_splits: list[str] | None = None,
) -> Path:
    """Write a temporary baseline configuration."""
    config: dict[str, Any] = {
        "dataset": {
            "name": "test_forecasting_baselines",
            "timezone": timezone,
        },
        "paths": {
            "input": str(
                tmp_path / "features.parquet"
            ),
            "predictions": str(
                tmp_path / "predictions.parquet"
            ),
            "report": str(
                tmp_path / "baseline-report.json"
            ),
        },
        "evaluation": {
            "target_column": (
                "target_pickup_count"
            ),
            "split_column": "split",
            "zone_column": "zone_id",
            "lag_1_column": (
                "pickup_count_lag_1"
            ),
            "seasonal_lag_column": (
                "pickup_count_lag_96"
            ),
            "validation_fit_splits": (
                validation_fit_splits
                if validation_fit_splits
                is not None
                else ["train"]
            ),
            "test_fit_splits": (
                test_fit_splits
                if test_fit_splits
                is not None
                else [
                    "train",
                    "validation",
                ]
            ),
        },
        "output": {
            "compression": "zstd",
        },
    }

    config_path = (
        tmp_path / "baselines.yaml"
    )

    config_path.write_text(
        yaml.safe_dump(
            config,
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    return config_path


def prepare_baseline_case(
    tmp_path: Path,
) -> Path:
    """Create the synthetic dataset and baseline config."""
    write_feature_dataset(
        tmp_path / "features.parquet"
    )

    return write_baseline_config(
        tmp_path
    )


def test_evaluate_baselines_writes_predictions_and_report(
    tmp_path: Path,
) -> None:
    """Generate complete validation and test predictions."""
    config_path = prepare_baseline_case(
        tmp_path
    )

    report = (
        evaluate_forecasting_baselines(
            config_path
        )
    )

    predictions_path = (
        tmp_path / "predictions.parquet"
    )

    report_path = (
        tmp_path / "baseline-report.json"
    )

    predictions = pl.read_parquet(
        predictions_path
    )

    assert predictions_path.exists()
    assert report_path.exists()

    assert report["input"][
        "row_count"
    ] == 12

    assert report["input"][
        "train_rows"
    ] == 4

    assert report["input"][
        "validation_rows"
    ] == 4

    assert report["input"][
        "test_rows"
    ] == 4

    assert report["predictions"][
        "row_count"
    ] == 8

    assert report["predictions"][
        "validation_rows"
    ] == 4

    assert report["predictions"][
        "test_rows"
    ] == 4

    assert report["invariants"][
        "validation_uses_train_only"
    ] is True

    assert report["invariants"][
        "test_uses_train_and_validation"
    ] is True

    assert report["invariants"][
        "duplicate_key_count"
    ] == 0

    assert report["invariants"][
        "null_value_count"
    ] == 0

    assert report["invariants"][
        "negative_prediction_count"
    ] == 0

    assert predictions.height == 8

    assert set(
        predictions[
            "split"
        ].unique()
    ) == {
        "validation",
        "test",
    }

    assert (
        predictions.null_count()
        .row(0)
        == (0,) * predictions.width
    )


def test_validation_mean_uses_train_only(
    tmp_path: Path,
) -> None:
    """Fit validation historical means using only train rows."""
    config_path = prepare_baseline_case(
        tmp_path
    )

    evaluate_forecasting_baselines(
        config_path
    )

    predictions = pl.read_parquet(
        tmp_path / "predictions.parquet"
    )

    zone_one = (
        predictions.filter(
            (pl.col("split") == "validation")
            & (pl.col("zone_id") == 1)
        )
    )

    zone_two = (
        predictions.filter(
            (pl.col("split") == "validation")
            & (pl.col("zone_id") == 2)
        )
    )

    assert zone_one[
        "fit_scope"
    ].unique().to_list() == [
        "train"
    ]

    assert zone_two[
        "fit_scope"
    ].unique().to_list() == [
        "train"
    ]

    assert zone_one[
        "prediction_zone_historical_mean"
    ].to_list() == [
        12.0,
        12.0,
    ]

    assert zone_two[
        "prediction_zone_historical_mean"
    ].to_list() == [
        22.0,
        22.0,
    ]


def test_test_mean_uses_train_and_validation(
    tmp_path: Path,
) -> None:
    """Refit test historical means using train and validation."""
    config_path = prepare_baseline_case(
        tmp_path
    )

    evaluate_forecasting_baselines(
        config_path
    )

    predictions = pl.read_parquet(
        tmp_path / "predictions.parquet"
    )

    zone_one = predictions.filter(
        (pl.col("split") == "test")
        & (pl.col("zone_id") == 1)
    )

    zone_two = predictions.filter(
        (pl.col("split") == "test")
        & (pl.col("zone_id") == 2)
    )

    assert zone_one[
        "fit_scope"
    ].unique().to_list() == [
        "train+validation"
    ]

    assert zone_two[
        "fit_scope"
    ].unique().to_list() == [
        "train+validation"
    ]

    assert zone_one[
        "prediction_zone_historical_mean"
    ].to_list() == [
        16.0,
        16.0,
    ]

    assert zone_two[
        "prediction_zone_historical_mean"
    ].to_list() == [
        26.0,
        26.0,
    ]


def test_lag_predictions_are_copied_from_feature_columns(
    tmp_path: Path,
) -> None:
    """Preserve configured lag features as predictions."""
    config_path = prepare_baseline_case(
        tmp_path
    )

    evaluate_forecasting_baselines(
        config_path
    )

    source = (
        pl.read_parquet(
            tmp_path / "features.parquet"
        )
        .filter(
            pl.col("split").is_in(
                [
                    "validation",
                    "test",
                ]
            )
        )
        .select(
            "interval_start",
            "zone_id",
            "pickup_count_lag_1",
            "pickup_count_lag_96",
        )
    )

    predictions = (
        pl.read_parquet(
            tmp_path / "predictions.parquet"
        )
        .select(
            "interval_start",
            "zone_id",
            "prediction_lag_1",
            "prediction_seasonal_lag_96",
        )
    )

    comparison = source.join(
        predictions,
        on=[
            "interval_start",
            "zone_id",
        ],
        how="inner",
    )

    assert comparison.height == 8

    assert (
        comparison[
            "pickup_count_lag_1"
        ].cast(pl.Float64)
        == comparison[
            "prediction_lag_1"
        ]
    ).all()

    assert (
        comparison[
            "pickup_count_lag_96"
        ].cast(pl.Float64)
        == comparison[
            "prediction_seasonal_lag_96"
        ]
    ).all()


def test_baseline_metrics_are_calculated_correctly(
    tmp_path: Path,
) -> None:
    """Verify MAE, RMSE, WAPE, bias, and ranking."""
    config_path = prepare_baseline_case(
        tmp_path
    )

    report = (
        evaluate_forecasting_baselines(
            config_path
        )
    )

    validation_lag = report[
        "metrics"
    ]["validation"]["lag_1"]

    test_lag = report[
        "metrics"
    ]["test"]["lag_1"]

    validation_zero = report[
        "metrics"
    ]["validation"]["zero"]

    assert validation_lag[
        "row_count"
    ] == 4

    assert validation_lag[
        "actual_sum"
    ] == 100

    assert validation_lag[
        "prediction_sum"
    ] == 84.0

    assert validation_lag[
        "mae"
    ] == pytest.approx(
        4.0
    )

    assert validation_lag[
        "rmse"
    ] == pytest.approx(
        4.0
    )

    assert validation_lag[
        "wape"
    ] == pytest.approx(
        0.16
    )

    assert validation_lag[
        "mean_bias"
    ] == pytest.approx(
        -4.0
    )

    assert test_lag[
        "actual_sum"
    ] == 132

    assert test_lag[
        "mae"
    ] == pytest.approx(
        4.0
    )

    assert test_lag[
        "rmse"
    ] == pytest.approx(
        4.0
    )

    assert test_lag[
        "wape"
    ] == pytest.approx(
        16 / 132
    )

    assert validation_zero[
        "mae"
    ] == pytest.approx(
        25.0
    )

    assert validation_zero[
        "wape"
    ] == pytest.approx(
        1.0
    )

    assert validation_zero[
        "mean_bias"
    ] == pytest.approx(
        -25.0
    )

    assert report[
        "best_baseline_by_mae"
    ]["validation"] == "lag_1"

    assert report[
        "best_baseline_by_mae"
    ]["test"] == "lag_1"


def test_prediction_keys_do_not_overlap_or_duplicate(
    tmp_path: Path,
) -> None:
    """Keep validation and test prediction keys unique."""
    config_path = prepare_baseline_case(
        tmp_path
    )

    evaluate_forecasting_baselines(
        config_path
    )

    predictions = pl.read_parquet(
        tmp_path / "predictions.parquet"
    )

    duplicate_count = (
        predictions.group_by(
            "split",
            "interval_start",
            "zone_id",
        )
        .len()
        .filter(
            pl.col("len") > 1
        )
        .height
    )

    validation_keys = set(
        predictions.filter(
            pl.col("split")
            == "validation"
        )
        .select(
            "interval_start",
            "zone_id",
        )
        .iter_rows()
    )

    test_keys = set(
        predictions.filter(
            pl.col("split")
            == "test"
        )
        .select(
            "interval_start",
            "zone_id",
        )
        .iter_rows()
    )

    assert duplicate_count == 0
    assert validation_keys.isdisjoint(
        test_keys
    )


@pytest.mark.parametrize(
    (
        "timezone",
        "validation_fit_splits",
        "test_fit_splits",
        "expected_message",
    ),
    [
        (
            "Mars/Olympus_Mons",
            ["train"],
            [
                "train",
                "validation",
            ],
            "Unknown timezone",
        ),
        (
            TIMEZONE,
            [],
            [
                "train",
                "validation",
            ],
            (
                "validation_fit_splits "
                "must not be empty"
            ),
        ),
        (
            TIMEZONE,
            [
                "train",
                "validation",
            ],
            [
                "train",
                "validation",
            ],
            (
                "Validation baselines "
                "cannot fit on the validation split"
            ),
        ),
        (
            TIMEZONE,
            ["train"],
            [
                "train",
                "test",
            ],
            (
                "Test baselines cannot "
                "fit on test data"
            ),
        ),
    ],
)
def test_invalid_baseline_configuration_is_rejected(
    tmp_path: Path,
    timezone: str,
    validation_fit_splits: list[str],
    test_fit_splits: list[str],
    expected_message: str,
) -> None:
    """Reject configurations that permit temporal leakage."""
    config_path = write_baseline_config(
        tmp_path,
        timezone=timezone,
        validation_fit_splits=(
            validation_fit_splits
        ),
        test_fit_splits=(
            test_fit_splits
        ),
    )

    with pytest.raises(
        BaselineEvaluationError,
        match=expected_message,
    ):
        BaselineEvaluationSettings.from_yaml(
            config_path
        )


def test_missing_required_column_is_rejected(
    tmp_path: Path,
) -> None:
    """Reject an incomplete feature dataset."""
    config_path = prepare_baseline_case(
        tmp_path
    )

    frame = pl.read_parquet(
        tmp_path / "features.parquet"
    ).drop(
        "pickup_count_lag_96"
    )

    frame.write_parquet(
        tmp_path / "features.parquet"
    )

    with pytest.raises(
        BaselineEvaluationError,
        match=(
            "Feature dataset is missing columns"
        ),
    ):
        evaluate_forecasting_baselines(
            config_path
        )


def test_null_required_value_is_rejected(
    tmp_path: Path,
) -> None:
    """Reject null values in baseline input columns."""
    config_path = prepare_baseline_case(
        tmp_path
    )

    frame = (
        pl.read_parquet(
            tmp_path / "features.parquet"
        )
        .with_row_index(
            "row_id"
        )
        .with_columns(
            pl.when(
                pl.col("row_id") == 0
            )
            .then(
                pl.lit(
                    None,
                    dtype=pl.Int32,
                )
            )
            .otherwise(
                pl.col(
                    "pickup_count_lag_1"
                )
            )
            .alias(
                "pickup_count_lag_1"
            )
        )
        .drop(
            "row_id"
        )
    )

    frame.write_parquet(
        tmp_path / "features.parquet"
    )

    with pytest.raises(
        BaselineEvaluationError,
        match=(
            "contains null values"
        ),
    ):
        evaluate_forecasting_baselines(
            config_path
        )


def test_negative_prediction_is_rejected(
    tmp_path: Path,
) -> None:
    """Reject negative lag-based demand predictions."""
    config_path = prepare_baseline_case(
        tmp_path
    )

    frame = (
        pl.read_parquet(
            tmp_path / "features.parquet"
        )
        .with_columns(
            pl.when(
                (pl.col("split") == "validation")
                & (pl.col("zone_id") == 1)
                & (pl.col("time_step") == 2)
            )
            .then(
                pl.lit(-1)
            )
            .otherwise(
                pl.col(
                    "pickup_count_lag_1"
                )
            )
            .cast(pl.Int32)
            .alias(
                "pickup_count_lag_1"
            )
        )
    )

    frame.write_parquet(
        tmp_path / "features.parquet"
    )

    with pytest.raises(
        BaselineEvaluationError,
        match=(
            "predictions contain negative values"
        ),
    ):
        evaluate_forecasting_baselines(
            config_path
        )


def test_duplicate_prediction_key_is_rejected(
    tmp_path: Path,
) -> None:
    """Reject duplicate evaluation keys."""
    config_path = prepare_baseline_case(
        tmp_path
    )

    frame = pl.read_parquet(
        tmp_path / "features.parquet"
    )

    duplicate_row = (
        frame.filter(
            (pl.col("split") == "validation")
            & (pl.col("zone_id") == 1)
        )
        .head(1)
    )

    frame = pl.concat(
        [
            frame,
            duplicate_row,
        ],
        how="vertical",
    )

    frame.write_parquet(
        tmp_path / "features.parquet"
    )

    with pytest.raises(
        BaselineEvaluationError,
        match=(
            "predictions contain duplicate keys"
        ),
    ):
        evaluate_forecasting_baselines(
            config_path
        )