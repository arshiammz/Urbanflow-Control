from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl
import yaml

from urbanflow.forecasting.prediction_calibration import (
    regression_metrics,
    write_json_atomically,
    write_parquet_atomically,
)


CONFIG_PATH = Path(
    "configs/forecasting/"
    "promote_constrained_calibration.yaml"
)


class CalibrationPromotionError(RuntimeError):
    """Raised when constrained calibration cannot be promoted."""


def read_config(
    path: Path,
) -> dict[str, Any]:
    """Read constrained-calibration promotion settings."""
    if not path.exists():
        raise CalibrationPromotionError(
            f"Promotion config does not exist: {path}"
        )

    try:
        raw = yaml.safe_load(
            path.read_text(
                encoding="utf-8",
            )
        )

        paths = raw["paths"]
        schema = raw["schema"]
        contract = raw["contract"]
        output = raw["output"]

        config = {
            "dataset_name": str(
                raw["dataset"]["name"]
            ),
            "experiment_report": Path(
                paths["experiment_report"]
            ),
            "experiment_predictions": Path(
                paths["experiment_predictions"]
            ),
            "output_predictions": Path(
                paths["output_predictions"]
            ),
            "artifact": Path(
                paths["artifact"]
            ),
            "report": Path(
                paths["report"]
            ),
            "key_columns": tuple(
                str(column)
                for column in schema["key_columns"]
            ),
            "split_column": str(
                schema["split_column"]
            ),
            "target_column": str(
                schema["target_column"]
            ),
            "raw_prediction_column": str(
                schema["raw_prediction_column"]
            ),
            "baseline_prediction_column": str(
                schema[
                    "baseline_prediction_column"
                ]
            ),
            "prediction_band_column": str(
                schema["prediction_band_column"]
            ),
            "correction_column": str(
                schema["correction_column"]
            ),
            "fit_split": str(
                contract["fit_split"]
            ),
            "evaluation_split": str(
                contract["evaluation_split"]
            ),
            "ordered_bands": tuple(
                str(value)
                for value in contract[
                    "ordered_prediction_bands"
                ]
            ),
            "high_demand_threshold": float(
                contract[
                    "high_demand_threshold"
                ]
            ),
            "compression": str(
                output["compression"]
            ),
        }

    except (
        OSError,
        KeyError,
        TypeError,
        ValueError,
        yaml.YAMLError,
    ) as error:
        raise CalibrationPromotionError(
            f"Invalid promotion config: {error}"
        ) from error

    for key in (
        "experiment_report",
        "experiment_predictions",
    ):
        input_path = config[key]

        if not input_path.exists():
            raise CalibrationPromotionError(
                f"Required input does not exist: {input_path}"
            )

        if input_path.stat().st_size == 0:
            raise CalibrationPromotionError(
                f"Required input is empty: {input_path}"
            )

    if not config["key_columns"]:
        raise CalibrationPromotionError(
            "At least one key column is required."
        )

    if len(
        config["key_columns"]
    ) != len(
        set(
            config["key_columns"]
        )
    ):
        raise CalibrationPromotionError(
            "Key columns must be unique."
        )

    if not config["ordered_bands"]:
        raise CalibrationPromotionError(
            "At least one prediction band is required."
        )

    if len(
        config["ordered_bands"]
    ) != len(
        set(
            config["ordered_bands"]
        )
    ):
        raise CalibrationPromotionError(
            "Ordered prediction bands must be unique."
        )

    if config["high_demand_threshold"] <= 0:
        raise CalibrationPromotionError(
            "High-demand threshold must be positive."
        )

    if not config["compression"]:
        raise CalibrationPromotionError(
            "Output compression must not be empty."
        )

    return config


def read_experiment_report(
    config: dict[str, Any],
) -> tuple[
    dict[str, Any],
    dict[str, Any],
]:
    """Read and verify the validation-selected experiment."""
    experiment_path = config[
        "experiment_report"
    ]

    try:
        report = json.loads(
            experiment_path.read_text(
                encoding="utf-8",
            )
        )

        selection_contract = report[
            "selection_contract"
        ]

        selected_candidate = report[
            "selected_candidate"
        ]

    except (
        OSError,
        KeyError,
        TypeError,
        json.JSONDecodeError,
    ) as error:
        raise CalibrationPromotionError(
            f"Invalid experiment report: {error}"
        ) from error

    if (
        selection_contract["fit_split"]
        != config["fit_split"]
    ):
        raise CalibrationPromotionError(
            "Experiment fit split does not match "
            "the promotion config."
        )

    if (
        selection_contract["evaluation_split"]
        != config["evaluation_split"]
    ):
        raise CalibrationPromotionError(
            "Experiment evaluation split does not match "
            "the promotion config."
        )

    if bool(
        selection_contract[
            "test_used_for_selection"
        ]
    ):
        raise CalibrationPromotionError(
            "Experiment report states that test was used "
            "for candidate selection."
        )

    if not bool(
        selected_candidate["feasible"]
    ):
        raise CalibrationPromotionError(
            "The selected experiment candidate is not feasible."
        )

    maximum_band = str(
        selected_candidate[
            "maximum_calibrated_band"
        ]
    )

    if maximum_band not in config[
        "ordered_bands"
    ]:
        raise CalibrationPromotionError(
            f"Selected band is not declared: {maximum_band}"
        )

    selected_shrinkage = float(
        selected_candidate["shrinkage"]
    )

    if not 0.0 <= selected_shrinkage <= 1.0:
        raise CalibrationPromotionError(
            "Selected shrinkage must be between zero and one."
        )

    return (
        report,
        selected_candidate,
    )


def sha256(
    path: Path,
) -> str:
    """Calculate a file SHA-256 digest."""
    digest = hashlib.sha256()

    try:
        with path.open(
            "rb"
        ) as file:
            for chunk in iter(
                lambda: file.read(
                    1024 * 1024
                ),
                b"",
            ):
                digest.update(
                    chunk
                )

    except OSError as error:
        raise CalibrationPromotionError(
            f"Could not hash {path}: {error}"
        ) from error

    return digest.hexdigest()


def prediction_mean(
    metrics: dict[str, Any],
) -> float:
    """Calculate mean prediction from aggregate metrics."""
    row_count = int(
        metrics["row_count"]
    )

    if row_count == 0:
        return 0.0

    return (
        float(
            metrics["prediction_sum"]
        )
        / row_count
    )


def metrics_by_segment(
    frame: pl.DataFrame,
    *,
    target_column: str,
    prediction_column: str,
    high_demand_threshold: float,
) -> dict[str, Any]:
    """Calculate overall and operational-risk metrics."""
    if frame.height == 0:
        raise CalibrationPromotionError(
            "Cannot calculate metrics for an empty split."
        )

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

    if zero_demand_frame.height == 0:
        raise CalibrationPromotionError(
            "Zero-demand evaluation segment is empty."
        )

    if high_demand_frame.height == 0:
        raise CalibrationPromotionError(
            "High-demand evaluation segment is empty."
        )

    return {
        "overall": regression_metrics(
            frame,
            target_column=target_column,
            prediction_column=prediction_column,
        ),
        "zero_demand": regression_metrics(
            zero_demand_frame,
            target_column=target_column,
            prediction_column=prediction_column,
        ),
        "high_demand": regression_metrics(
            high_demand_frame,
            target_column=target_column,
            prediction_column=prediction_column,
        ),
    }


def build_split_report(
    frame: pl.DataFrame,
    *,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Compare raw, constrained, and lag-1 forecasts."""
    target_column = config[
        "target_column"
    ]

    high_demand_threshold = config[
        "high_demand_threshold"
    ]

    raw = metrics_by_segment(
        frame,
        target_column=target_column,
        prediction_column="raw_prediction",
        high_demand_threshold=high_demand_threshold,
    )

    constrained = metrics_by_segment(
        frame,
        target_column=target_column,
        prediction_column=(
            "constrained_prediction"
        ),
        high_demand_threshold=high_demand_threshold,
    )

    baseline = metrics_by_segment(
        frame,
        target_column=target_column,
        prediction_column="baseline_prediction",
        high_demand_threshold=high_demand_threshold,
    )

    raw_mae = float(
        raw["overall"]["mae"]
    )

    constrained_mae = float(
        constrained["overall"]["mae"]
    )

    baseline_mae = float(
        baseline["overall"]["mae"]
    )

    return {
        "raw": raw,
        "constrained": constrained,
        "baseline": baseline,
        "constrained_vs_raw": {
            "mae_absolute_improvement": (
                raw_mae
                - constrained_mae
            ),
            "mae_relative_improvement": (
                (
                    raw_mae
                    - constrained_mae
                )
                / raw_mae
            ),
            "constrained_improves_mae": (
                constrained_mae
                < raw_mae
            ),
        },
        "constrained_vs_baseline": {
            "mae_absolute_improvement": (
                baseline_mae
                - constrained_mae
            ),
            "mae_relative_improvement": (
                (
                    baseline_mae
                    - constrained_mae
                )
                / baseline_mae
            ),
            "constrained_beats_baseline": (
                constrained_mae
                < baseline_mae
            ),
        },
    }


def load_promotion_frame(
    config: dict[str, Any],
) -> tuple[
    pl.DataFrame,
    dict[str, int],
]:
    """Load and validate experiment predictions."""
    required_columns = {
        *config["key_columns"],
        config["split_column"],
        config["target_column"],
        config["raw_prediction_column"],
        config["baseline_prediction_column"],
        config["prediction_band_column"],
        config["correction_column"],
    }

    try:
        source = pl.read_parquet(
            config[
                "experiment_predictions"
            ]
        )

    except Exception as error:
        raise CalibrationPromotionError(
            "Could not read experiment predictions: "
            f"{error}"
        ) from error

    missing_columns = sorted(
        required_columns.difference(
            source.columns
        )
    )

    if missing_columns:
        raise CalibrationPromotionError(
            "Experiment predictions are missing columns: "
            + ", ".join(
                missing_columns
            )
        )

    selected_splits = (
        config["fit_split"],
        config["evaluation_split"],
    )

    frame = (
        source.filter(
            pl.col(
                config["split_column"]
            ).is_in(
                selected_splits
            )
        )
        .select(
            *config["key_columns"],
            config["split_column"],
            config["target_column"],
            config["raw_prediction_column"],
            config["baseline_prediction_column"],
            config["prediction_band_column"],
            config["correction_column"],
        )
        .rename(
            {
                config[
                    "raw_prediction_column"
                ]: "raw_prediction",
                config[
                    "baseline_prediction_column"
                ]: "baseline_prediction",
                config[
                    "prediction_band_column"
                ]: "prediction_band",
                config[
                    "correction_column"
                ]: "base_correction",
            }
        )
        .with_columns(
            pl.col(
                config["target_column"]
            ).cast(
                pl.Float64
            ),

            pl.col(
                "raw_prediction"
            ).cast(
                pl.Float64
            ),

            pl.col(
                "baseline_prediction"
            ).cast(
                pl.Float64
            ),

            pl.col(
                "base_correction"
            ).cast(
                pl.Float64
            ),
        )
    )

    if frame.height == 0:
        raise CalibrationPromotionError(
            "Selected promotion dataset is empty."
        )

    duplicate_key_count = (
        frame.group_by(
            *config["key_columns"]
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

    non_finite_prediction_count = (
        frame.filter(
            ~pl.col(
                "raw_prediction"
            ).is_finite()
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
        raise CalibrationPromotionError(
            "Promotion dataset contains duplicate keys."
        )

    if null_value_count:
        raise CalibrationPromotionError(
            "Promotion dataset contains null values."
        )

    if non_finite_prediction_count:
        raise CalibrationPromotionError(
            "Raw predictions contain non-finite values."
        )

    if negative_prediction_count:
        raise CalibrationPromotionError(
            "Raw predictions contain negative values."
        )

    for split in selected_splits:
        split_count = frame.filter(
            pl.col(
                config["split_column"]
            )
            == split
        ).height

        if split_count == 0:
            raise CalibrationPromotionError(
                f"Required split is empty: {split}"
            )

    return frame, {
        "row_count": frame.height,
        "duplicate_key_count": duplicate_key_count,
        "null_value_count": null_value_count,
        "non_finite_raw_prediction_count": (
            non_finite_prediction_count
        ),
        "negative_raw_prediction_count": (
            negative_prediction_count
        ),
    }


def apply_selected_candidate(
    frame: pl.DataFrame,
    *,
    config: dict[str, Any],
    selected_candidate: dict[str, Any],
) -> pl.DataFrame:
    """Apply the fixed validation-selected candidate."""
    band_mapping = pl.DataFrame(
        {
            "prediction_band": list(
                config["ordered_bands"]
            ),
            "band_order": list(
                range(
                    len(
                        config["ordered_bands"]
                    )
                )
            ),
        }
    )

    frame = frame.join(
        band_mapping,
        on="prediction_band",
        how="left",
    )

    if (
        frame.get_column(
            "band_order"
        )
        .null_count()
        > 0
    ):
        raise CalibrationPromotionError(
            "Promotion dataset contains an unknown prediction band."
        )

    maximum_band = str(
        selected_candidate[
            "maximum_calibrated_band"
        ]
    )

    maximum_band_order = config[
        "ordered_bands"
    ].index(
        maximum_band
    )

    shrinkage = float(
        selected_candidate[
            "shrinkage"
        ]
    )

    selected_for_calibration = (
        pl.col(
            "band_order"
        )
        <= maximum_band_order
    )

    effective_correction = (
        pl.when(
            selected_for_calibration
        )
        .then(
            pl.col(
                "base_correction"
            )
            * shrinkage
        )
        .otherwise(
            0.0
        )
    )

    unbounded_prediction = (
        pl.col(
            "raw_prediction"
        )
        + effective_correction
    )

    return (
        frame.with_columns(
            selected_for_calibration.alias(
                "selected_for_calibration"
            ),

            effective_correction.alias(
                "effective_correction"
            ),
        )
        .with_columns(
            pl.when(
                unbounded_prediction
                < 0
            )
            .then(
                0.0
            )
            .otherwise(
                unbounded_prediction
            )
            .alias(
                "constrained_prediction"
            ),

            pl.lit(
                shrinkage
            )
            .alias(
                "calibration_shrinkage"
            ),

            pl.lit(
                maximum_band
            )
            .alias(
                "maximum_calibrated_band"
            ),
        )
        .sort(
            *config["key_columns"]
        )
    )


def promote() -> dict[str, Any]:
    """Promote the fixed constrained-calibration candidate."""
    config = read_config(
        CONFIG_PATH
    )

    (
        experiment_report,
        selected_candidate,
    ) = read_experiment_report(
        config
    )

    frame, input_summary = (
        load_promotion_frame(
            config
        )
    )

    output = apply_selected_candidate(
        frame,
        config=config,
        selected_candidate=selected_candidate,
    )

    invalid_output_counts = {
        "negative_prediction_count": (
            output.filter(
                pl.col(
                    "constrained_prediction"
                )
                < 0
            )
            .height
        ),
        "non_finite_prediction_count": (
            output.filter(
                ~pl.col(
                    "constrained_prediction"
                ).is_finite()
            )
            .height
        ),
        "null_prediction_count": (
            output.get_column(
                "constrained_prediction"
            )
            .null_count()
        ),
    }

    if any(
        invalid_output_counts.values()
    ):
        raise CalibrationPromotionError(
            "Promoted predictions failed validation: "
            f"{invalid_output_counts}"
        )

    selected_splits = (
        config["fit_split"],
        config["evaluation_split"],
    )

    split_reports = {
        split: build_split_report(
            output.filter(
                pl.col(
                    config["split_column"]
                )
                == split
            ),
            config=config,
        )
        for split in selected_splits
    }

    prediction_artifact = (
        write_parquet_atomically(
            output,
            path=config[
                "output_predictions"
            ],
            compression=config[
                "compression"
            ],
        )
    )

    generated_at = datetime.now(
        UTC
    ).isoformat()

    maximum_band = str(
        selected_candidate[
            "maximum_calibrated_band"
        ]
    )

    maximum_band_order = config[
        "ordered_bands"
    ].index(
        maximum_band
    )

    shrinkage = float(
        selected_candidate[
            "shrinkage"
        ]
    )

    fixed_mapping = (
        output.select(
            "prediction_band",
            "band_order",
            "base_correction",
            "selected_for_calibration",
            "effective_correction",
        )
        .unique()
        .sort(
            "band_order"
        )
    )

    artifact = {
        "artifact_version": "1.0.0",
        "generated_at": generated_at,
        "dataset_name": config[
            "dataset_name"
        ],
        "status": (
            "accepted_operational_postprocessor"
        ),
        "selection_source": {
            "experiment_report": str(
                config["experiment_report"]
            ),
            "experiment_report_sha256": sha256(
                config["experiment_report"]
            ),
            "fit_split": config[
                "fit_split"
            ],
            "test_used_for_selection": False,
        },
        "selected_candidate": {
            "maximum_calibrated_band": (
                maximum_band
            ),
            "maximum_band_order": (
                maximum_band_order
            ),
            "shrinkage": shrinkage,
        },
        "mapping": fixed_mapping.to_dicts(),
        "source_predictions": str(
            config[
                "experiment_predictions"
            ]
        ),
        "output_predictions": str(
            config[
                "output_predictions"
            ]
        ),
    }

    write_json_atomically(
        config["artifact"],
        artifact,
    )

    report: dict[str, Any] = {
        "promotion_report_version": "1.0.0",
        "generated_at": generated_at,
        "selection_contract": {
            "candidate_selected_on": (
                config["fit_split"]
            ),
            "test_used_for_selection": False,
            "selection_is_not_repeated_during_promotion": True,
        },
        "selected_candidate": artifact[
            "selected_candidate"
        ],
        "input": {
            "experiment_report": {
                "path": str(
                    config["experiment_report"]
                ),
                "sha256": artifact[
                    "selection_source"
                ][
                    "experiment_report_sha256"
                ],
            },
            "experiment_predictions": {
                "path": str(
                    config[
                        "experiment_predictions"
                    ]
                ),
                "size_bytes": (
                    config[
                        "experiment_predictions"
                    ]
                    .stat()
                    .st_size
                ),
            },
            "summary": input_summary,
        },
        "splits": split_reports,
        "artifacts": {
            "predictions": (
                prediction_artifact
            ),
            "calibration": {
                "path": str(
                    config["artifact"]
                ),
                "size_bytes": (
                    config["artifact"]
                    .stat()
                    .st_size
                ),
            },
            "report": {
                "path": str(
                    config["report"]
                ),
            },
        },
        "invariants": {
            "experiment_candidate_is_feasible": bool(
                selected_candidate[
                    "feasible"
                ]
            ),
            "experiment_test_not_used_for_selection": (
                not bool(
                    experiment_report[
                        "selection_contract"
                    ][
                        "test_used_for_selection"
                    ]
                )
            ),
            "output_row_count_matches_input": (
                output.height
                == frame.height
            ),
            "duplicate_key_count": (
                input_summary[
                    "duplicate_key_count"
                ]
            ),
            "null_input_count": (
                input_summary[
                    "null_value_count"
                ]
            ),
            **invalid_output_counts,
        },
    }

    write_json_atomically(
        config["report"],
        report,
    )

    return report


def main() -> None:
    """Promote and print constrained calibration."""
    try:
        report = promote()

    except CalibrationPromotionError as error:
        raise SystemExit(
            f"Calibration promotion failed: {error}"
        ) from error

    selected = report[
        "selected_candidate"
    ]

    validation = report[
        "splits"
    ][
        "validation"
    ]

    test = report[
        "splits"
    ][
        "test"
    ]

    print(
        "Constrained calibration promotion completed"
    )

    print()

    print(
        "Maximum calibrated band:",
        selected[
            "maximum_calibrated_band"
        ],
    )

    print(
        "Shrinkage:",
        f"{selected['shrinkage']:.2f}",
    )

    print()

    print(
        "Validation MAE:",
        f"{validation['raw']['overall']['mae']:.6f}",
        "→",
        (
            f"{validation['constrained']['overall']['mae']:.6f}"
        ),
    )

    print(
        "Validation zero-demand mean:",
        (
            f"{prediction_mean(validation['raw']['zero_demand']):.4f}"
        ),
        "→",
        (
            f"{prediction_mean(validation['constrained']['zero_demand']):.4f}"
        ),
    )

    print(
        "Validation high-demand bias:",
        (
            f"{validation['raw']['high_demand']['mean_bias']:.4f}"
        ),
        "→",
        (
            f"{validation['constrained']['high_demand']['mean_bias']:.4f}"
        ),
    )

    print()

    print(
        "Test — evaluation only"
    )

    print(
        "Test MAE:",
        f"{test['raw']['overall']['mae']:.6f}",
        "→",
        (
            f"{test['constrained']['overall']['mae']:.6f}"
        ),
    )

    print(
        "Test vs raw improvement:",
        (
            f"{test['constrained_vs_raw']['mae_relative_improvement'] * 100:.3f}%"
        ),
    )

    lag_improvement = (
        test[
            "constrained_vs_baseline"
        ][
            "mae_relative_improvement"
        ]
    )

    print(
        "Test vs lag-1 improvement:",
        f"{lag_improvement * 100:.2f}%",
    )

    print(
        "Test zero-demand mean:",
        (
            f"{prediction_mean(test['raw']['zero_demand']):.4f}"
        ),
        "→",
        (
            f"{prediction_mean(test['constrained']['zero_demand']):.4f}"
        ),
    )

    print(
        "Test high-demand bias:",
        (
            f"{test['raw']['high_demand']['mean_bias']:.4f}"
        ),
        "→",
        (
            f"{test['constrained']['high_demand']['mean_bias']:.4f}"
        ),
    )

    print()

    print(
        "Predictions:",
        report[
            "artifacts"
        ][
            "predictions"
        ][
            "path"
        ],
    )

    print(
        "Artifact:",
        report[
            "artifacts"
        ][
            "calibration"
        ][
            "path"
        ],
    )

    print(
        "Report:",
        report[
            "artifacts"
        ][
            "report"
        ][
            "path"
        ],
    )


if __name__ == "__main__":
    main()