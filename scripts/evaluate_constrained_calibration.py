from __future__ import annotations

import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl
import yaml


CONFIG_PATH = Path(
    "configs/forecasting/"
    "constrained_calibration_experiment.yaml"
)


class ConstrainedCalibrationError(RuntimeError):
    """Raised when constrained calibration evaluation fails."""


def read_config(
    path: Path,
) -> dict[str, Any]:
    """Read and validate the experiment configuration."""
    if not path.exists():
        raise ConstrainedCalibrationError(
            f"Config does not exist: {path}"
        )

    try:
        config = yaml.safe_load(
            path.read_text(
                encoding="utf-8",
            )
        )
    except (
        OSError,
        yaml.YAMLError,
    ) as error:
        raise ConstrainedCalibrationError(
            f"Could not read config: {error}"
        ) from error

    if not isinstance(
        config,
        dict,
    ):
        raise ConstrainedCalibrationError(
            "Config must contain a YAML mapping."
        )

    try:
        paths = config["paths"]
        schema = config["schema"]
        experiment = config["experiment"]
        constraints = config["constraints"]

        prediction_path = Path(
            paths["predictions"]
        )

        report_path = Path(
            paths["report"]
        )

        ordered_bands = tuple(
            str(value)
            for value in experiment[
                "ordered_prediction_bands"
            ]
        )

        maximum_calibrated_bands = tuple(
            str(value)
            for value in experiment[
                "maximum_calibrated_bands"
            ]
        )

        shrinkage_candidates = tuple(
            float(value)
            for value in experiment[
                "shrinkage_candidates"
            ]
        )

        parsed = {
            "dataset_name": str(
                config["dataset"]["name"]
            ),
            "prediction_path": prediction_path,
            "report_path": report_path,
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
                schema["baseline_prediction_column"]
            ),
            "prediction_band_column": str(
                schema["prediction_band_column"]
            ),
            "correction_column": str(
                schema["correction_column"]
            ),
            "fit_split": str(
                experiment["fit_split"]
            ),
            "evaluation_split": str(
                experiment["evaluation_split"]
            ),
            "ordered_bands": ordered_bands,
            "maximum_calibrated_bands": (
                maximum_calibrated_bands
            ),
            "shrinkage_candidates": (
                shrinkage_candidates
            ),
            "high_demand_threshold": float(
                experiment[
                    "high_demand_threshold"
                ]
            ),
            "require_non_worse_mae": bool(
                constraints[
                    "require_non_worse_mae"
                ]
            ),
            "require_zero_demand_improvement": bool(
                constraints[
                    "require_zero_demand_improvement"
                ]
            ),
            "maximum_rmse_relative_degradation": float(
                constraints[
                    "maximum_rmse_relative_degradation"
                ]
            ),
            "maximum_high_demand_bias_degradation": float(
                constraints[
                    "maximum_high_demand_bias_degradation"
                ]
            ),
        }

    except (
        KeyError,
        TypeError,
        ValueError,
    ) as error:
        raise ConstrainedCalibrationError(
            f"Invalid configuration: {error}"
        ) from error

    validate_config(
        parsed
    )

    return parsed


def validate_config(
    config: dict[str, Any],
) -> None:
    """Validate the parsed configuration."""
    prediction_path = config[
        "prediction_path"
    ]

    if not prediction_path.exists():
        raise ConstrainedCalibrationError(
            "Prediction dataset does not exist: "
            f"{prediction_path}"
        )

    if prediction_path.stat().st_size == 0:
        raise ConstrainedCalibrationError(
            "Prediction dataset is empty: "
            f"{prediction_path}"
        )

    ordered_bands = config[
        "ordered_bands"
    ]

    maximum_bands = config[
        "maximum_calibrated_bands"
    ]

    if not ordered_bands:
        raise ConstrainedCalibrationError(
            "At least one ordered prediction band is required."
        )

    if len(
        ordered_bands
    ) != len(
        set(
            ordered_bands
        )
    ):
        raise ConstrainedCalibrationError(
            "Ordered prediction bands must be unique."
        )

    unknown_maximum_bands = sorted(
        set(
            maximum_bands
        ).difference(
            ordered_bands
        )
    )

    if unknown_maximum_bands:
        raise ConstrainedCalibrationError(
            "Unknown maximum calibrated bands: "
            + ", ".join(
                unknown_maximum_bands
            )
        )

    candidates = config[
        "shrinkage_candidates"
    ]

    if not candidates:
        raise ConstrainedCalibrationError(
            "At least one shrinkage candidate is required."
        )

    if any(
        not math.isfinite(
            candidate
        )
        or not 0.0 <= candidate <= 1.0
        for candidate in candidates
    ):
        raise ConstrainedCalibrationError(
            "Shrinkage candidates must be finite "
            "and between 0 and 1."
        )

    if 0.0 not in candidates:
        raise ConstrainedCalibrationError(
            "Shrinkage candidates must contain 0.0."
        )

    if (
        config[
            "maximum_rmse_relative_degradation"
        ]
        < 0
    ):
        raise ConstrainedCalibrationError(
            "Maximum RMSE degradation cannot be negative."
        )

    if (
        config[
            "maximum_high_demand_bias_degradation"
        ]
        < 0
    ):
        raise ConstrainedCalibrationError(
            "Maximum bias degradation cannot be negative."
        )


def load_prediction_frame(
    config: dict[str, Any],
) -> pl.DataFrame:
    """Load and validate calibration experiment data."""
    required_columns = {
        config["split_column"],
        config["target_column"],
        config["raw_prediction_column"],
        config["baseline_prediction_column"],
        config["prediction_band_column"],
        config["correction_column"],
    }

    try:
        frame = pl.read_parquet(
            config[
                "prediction_path"
            ]
        )
    except Exception as error:
        raise ConstrainedCalibrationError(
            f"Could not read prediction dataset: {error}"
        ) from error

    missing_columns = sorted(
        required_columns.difference(
            frame.columns
        )
    )

    if missing_columns:
        raise ConstrainedCalibrationError(
            "Prediction dataset is missing columns: "
            + ", ".join(
                missing_columns
            )
        )

    selected_splits = {
        config["fit_split"],
        config["evaluation_split"],
    }

    frame = (
        frame.filter(
            pl.col(
                config[
                    "split_column"
                ]
            ).is_in(
                list(
                    selected_splits
                )
            )
        )
        .select(
            config["split_column"],
            config["target_column"],
            config["raw_prediction_column"],
            config["baseline_prediction_column"],
            config["prediction_band_column"],
            config["correction_column"],
        )
        .with_columns(
            pl.col(
                config[
                    "target_column"
                ]
            )
            .cast(
                pl.Float64
            ),

            pl.col(
                config[
                    "raw_prediction_column"
                ]
            )
            .cast(
                pl.Float64
            ),

            pl.col(
                config[
                    "baseline_prediction_column"
                ]
            )
            .cast(
                pl.Float64
            ),

            pl.col(
                config[
                    "correction_column"
                ]
            )
            .cast(
                pl.Float64
            ),
        )
    )

    if frame.height == 0:
        raise ConstrainedCalibrationError(
            "Selected experiment dataset is empty."
        )

    null_count = sum(
        int(value)
        for value in frame.null_count().row(0)
    )

    if null_count:
        raise ConstrainedCalibrationError(
            "Experiment dataset contains null values."
        )

    band_mapping = pl.DataFrame(
        {
            config[
                "prediction_band_column"
            ]: list(
                config[
                    "ordered_bands"
                ]
            ),
            "band_order": list(
                range(
                    len(
                        config[
                            "ordered_bands"
                        ]
                    )
                )
            ),
        }
    )

    frame = frame.join(
        band_mapping,
        on=config[
            "prediction_band_column"
        ],
        how="left",
    )

    unknown_band_count = (
        frame.get_column(
            "band_order"
        )
        .null_count()
    )

    if unknown_band_count:
        raise ConstrainedCalibrationError(
            "Dataset contains prediction bands "
            "not declared in the config."
        )

    for split in selected_splits:
        split_rows = (
            frame.filter(
                pl.col(
                    config[
                        "split_column"
                    ]
                )
                == split
            )
            .height
        )

        if split_rows == 0:
            raise ConstrainedCalibrationError(
                f"Split is empty: {split}"
            )

    return frame


def regression_metrics(
    frame: pl.DataFrame,
    *,
    target_column: str,
    prediction_column: str,
) -> dict[str, Any]:
    """Calculate regression metrics for one prediction column."""
    if frame.height == 0:
        return {
            "row_count": 0,
            "actual_sum": 0.0,
            "prediction_sum": 0.0,
            "prediction_mean": None,
            "mae": None,
            "rmse": None,
            "wape": None,
            "mean_bias": None,
        }

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

            pl.col(
                prediction_column
            )
            .mean()
            .alias(
                "prediction_mean"
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
        "prediction_mean": float(
            values[
                "prediction_mean"
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
            float(
                values[
                    "absolute_error_sum"
                ]
            )
            / abs(
                actual_sum
            )
            if actual_sum != 0.0
            else None
        ),
        "mean_bias": float(
            values[
                "mean_bias"
            ]
        ),
    }


def apply_candidate(
    frame: pl.DataFrame,
    *,
    maximum_band_order: int,
    shrinkage: float,
    raw_prediction_column: str,
    correction_column: str,
) -> pl.DataFrame:
    """Apply one low-demand-only calibration candidate."""
    calibration_is_allowed = (
        pl.col(
            "band_order"
        )
        <= maximum_band_order
    )

    adjustment = (
        pl.when(
            calibration_is_allowed
        )
        .then(
            pl.col(
                correction_column
            )
            * shrinkage
        )
        .otherwise(
            0.0
        )
    )

    unbounded_prediction = (
        pl.col(
            raw_prediction_column
        )
        + adjustment
    )

    return frame.with_columns(
        adjustment.alias(
            "candidate_adjustment"
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
            "candidate_prediction"
        ),
    )


def evaluate_segments(
    frame: pl.DataFrame,
    *,
    config: dict[str, Any],
    prediction_column: str,
) -> dict[str, Any]:
    """Evaluate overall, zero-demand, and high-demand metrics."""
    target_column = config[
        "target_column"
    ]

    zero_frame = frame.filter(
        pl.col(
            target_column
        )
        == 0
    )

    high_demand_frame = frame.filter(
        pl.col(
            target_column
        )
        > config[
            "high_demand_threshold"
        ]
    )

    return {
        "overall": regression_metrics(
            frame,
            target_column=target_column,
            prediction_column=prediction_column,
        ),
        "zero_demand": regression_metrics(
            zero_frame,
            target_column=target_column,
            prediction_column=prediction_column,
        ),
        "high_demand": regression_metrics(
            high_demand_frame,
            target_column=target_column,
            prediction_column=prediction_column,
        ),
    }


def evaluate_candidate(
    validation_frame: pl.DataFrame,
    *,
    config: dict[str, Any],
    maximum_band: str,
    maximum_band_order: int,
    shrinkage: float,
    raw_metrics: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate one candidate and its feasibility constraints."""
    candidate_frame = apply_candidate(
        validation_frame,
        maximum_band_order=maximum_band_order,
        shrinkage=shrinkage,
        raw_prediction_column=(
            config[
                "raw_prediction_column"
            ]
        ),
        correction_column=(
            config[
                "correction_column"
            ]
        ),
    )

    candidate_metrics = evaluate_segments(
        candidate_frame,
        config=config,
        prediction_column="candidate_prediction",
    )

    raw_overall = raw_metrics[
        "overall"
    ]

    raw_zero = raw_metrics[
        "zero_demand"
    ]

    raw_high = raw_metrics[
        "high_demand"
    ]

    candidate_overall = candidate_metrics[
        "overall"
    ]

    candidate_zero = candidate_metrics[
        "zero_demand"
    ]

    candidate_high = candidate_metrics[
        "high_demand"
    ]

    mae_change = (
        float(
            candidate_overall[
                "mae"
            ]
        )
        - float(
            raw_overall[
                "mae"
            ]
        )
    )

    rmse_relative_degradation = (
        (
            float(
                candidate_overall[
                    "rmse"
                ]
            )
            - float(
                raw_overall[
                    "rmse"
                ]
            )
        )
        / float(
            raw_overall[
                "rmse"
            ]
        )
    )

    high_bias_change = (
        float(
            candidate_high[
                "mean_bias"
            ]
        )
        - float(
            raw_high[
                "mean_bias"
            ]
        )
    )

    high_bias_degradation = max(
        0.0,
        -high_bias_change,
    )

    raw_zero_prediction_mean = float(
        raw_zero[
            "prediction_mean"
        ]
    )

    candidate_zero_prediction_mean = float(
        candidate_zero[
            "prediction_mean"
        ]
    )

    zero_prediction_reduction = (
        raw_zero_prediction_mean
        - candidate_zero_prediction_mean
    )

    mae_constraint_passed = (
        mae_change
        <= 1e-12
        if config[
            "require_non_worse_mae"
        ]
        else True
    )

    zero_constraint_passed = (
        zero_prediction_reduction
        >= -1e-12
        if config[
            "require_zero_demand_improvement"
        ]
        else True
    )

    rmse_constraint_passed = (
        rmse_relative_degradation
        <= config[
            "maximum_rmse_relative_degradation"
        ]
        + 1e-12
    )

    high_bias_constraint_passed = (
        high_bias_degradation
        <= config[
            "maximum_high_demand_bias_degradation"
        ]
        + 1e-12
    )

    feasible = all(
        (
            mae_constraint_passed,
            zero_constraint_passed,
            rmse_constraint_passed,
            high_bias_constraint_passed,
        )
    )

    return {
        "maximum_calibrated_band": maximum_band,
        "maximum_band_order": maximum_band_order,
        "shrinkage": shrinkage,
        "feasible": feasible,
        "constraints": {
            "mae_constraint_passed": (
                mae_constraint_passed
            ),
            "zero_demand_constraint_passed": (
                zero_constraint_passed
            ),
            "rmse_constraint_passed": (
                rmse_constraint_passed
            ),
            "high_demand_bias_constraint_passed": (
                high_bias_constraint_passed
            ),
        },
        "changes": {
            "mae_change": mae_change,
            "mae_relative_improvement": (
                -mae_change
                / float(
                    raw_overall[
                        "mae"
                    ]
                )
            ),
            "rmse_relative_degradation": (
                rmse_relative_degradation
            ),
            "zero_prediction_reduction": (
                zero_prediction_reduction
            ),
            "zero_prediction_relative_reduction": (
                zero_prediction_reduction
                / raw_zero_prediction_mean
                if raw_zero_prediction_mean > 0
                else 0.0
            ),
            "high_demand_bias_change": (
                high_bias_change
            ),
            "high_demand_bias_degradation": (
                high_bias_degradation
            ),
        },
        "metrics": candidate_metrics,
    }


def select_candidate(
    candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    """Select the best feasible validation candidate."""
    feasible_candidates = [
        candidate
        for candidate in candidates
        if candidate[
            "feasible"
        ]
    ]

    if not feasible_candidates:
        raise ConstrainedCalibrationError(
            "No feasible candidate was found."
        )

    return min(
        feasible_candidates,
        key=lambda candidate: (
            float(
                candidate[
                    "metrics"
                ][
                    "overall"
                ][
                    "mae"
                ]
            ),
            -float(
                candidate[
                    "changes"
                ][
                    "zero_prediction_reduction"
                ]
            ),
            int(
                candidate[
                    "maximum_band_order"
                ]
            ),
            float(
                candidate[
                    "shrinkage"
                ]
            ),
        ),
    )


def evaluate_selected_on_test(
    test_frame: pl.DataFrame,
    *,
    config: dict[str, Any],
    selected_candidate: dict[str, Any],
) -> dict[str, Any]:
    """Apply the validation-selected candidate to the test split."""
    calibrated_test = apply_candidate(
        test_frame,
        maximum_band_order=int(
            selected_candidate[
                "maximum_band_order"
            ]
        ),
        shrinkage=float(
            selected_candidate[
                "shrinkage"
            ]
        ),
        raw_prediction_column=(
            config[
                "raw_prediction_column"
            ]
        ),
        correction_column=(
            config[
                "correction_column"
            ]
        ),
    )

    return {
        "raw": evaluate_segments(
            test_frame,
            config=config,
            prediction_column=(
                config[
                    "raw_prediction_column"
                ]
            ),
        ),
        "constrained_calibration": evaluate_segments(
            calibrated_test,
            config=config,
            prediction_column="candidate_prediction",
        ),
        "baseline": evaluate_segments(
            test_frame,
            config=config,
            prediction_column=(
                config[
                    "baseline_prediction_column"
                ]
            ),
        ),
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
    """Write a standards-compliant JSON report atomically."""
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

        raise ConstrainedCalibrationError(
            f"Could not write report: {error}"
        ) from error


def main() -> None:
    """Evaluate validation-selected constrained calibration."""
    config = read_config(
        CONFIG_PATH
    )

    frame = load_prediction_frame(
        config
    )

    split_column = config[
        "split_column"
    ]

    validation_frame = frame.filter(
        pl.col(
            split_column
        )
        == config[
            "fit_split"
        ]
    )

    test_frame = frame.filter(
        pl.col(
            split_column
        )
        == config[
            "evaluation_split"
        ]
    )

    raw_validation_metrics = evaluate_segments(
        validation_frame,
        config=config,
        prediction_column=(
            config[
                "raw_prediction_column"
            ]
        ),
    )

    band_orders = {
        band: order
        for order, band in enumerate(
            config[
                "ordered_bands"
            ]
        )
    }

    candidates: list[
        dict[str, Any]
    ] = []

    no_op_added = False

    for maximum_band in config[
        "maximum_calibrated_bands"
    ]:
        maximum_band_order = band_orders[
            maximum_band
        ]

        for shrinkage in config[
            "shrinkage_candidates"
        ]:
            if shrinkage == 0.0:
                if no_op_added:
                    continue

                no_op_added = True

            candidate = evaluate_candidate(
                validation_frame,
                config=config,
                maximum_band=maximum_band,
                maximum_band_order=(
                    maximum_band_order
                ),
                shrinkage=shrinkage,
                raw_metrics=(
                    raw_validation_metrics
                ),
            )

            candidates.append(
                candidate
            )

    selected = select_candidate(
        candidates
    )

    test_metrics = evaluate_selected_on_test(
        test_frame,
        config=config,
        selected_candidate=selected,
    )

    raw_test_mae = float(
        test_metrics[
            "raw"
        ][
            "overall"
        ][
            "mae"
        ]
    )

    calibrated_test_mae = float(
        test_metrics[
            "constrained_calibration"
        ][
            "overall"
        ][
            "mae"
        ]
    )

    baseline_test_mae = float(
        test_metrics[
            "baseline"
        ][
            "overall"
        ][
            "mae"
        ]
    )

    report = {
        "experiment_version": "0.1.0",
        "generated_at": datetime.now(
            UTC
        ).isoformat(),
        "dataset_name": config[
            "dataset_name"
        ],
        "selection_contract": {
            "fit_split": config[
                "fit_split"
            ],
            "evaluation_split": config[
                "evaluation_split"
            ],
            "test_used_for_selection": False,
            "selection_metric": "validation_mae",
        },
        "constraints": {
            "require_non_worse_mae": config[
                "require_non_worse_mae"
            ],
            "require_zero_demand_improvement": config[
                "require_zero_demand_improvement"
            ],
            "maximum_rmse_relative_degradation": config[
                "maximum_rmse_relative_degradation"
            ],
            "maximum_high_demand_bias_degradation": config[
                "maximum_high_demand_bias_degradation"
            ],
        },
        "validation_raw": raw_validation_metrics,
        "validation_candidates": candidates,
        "selected_candidate": selected,
        "test_evaluation": {
            **test_metrics,
            "constrained_vs_raw": {
                "mae_absolute_improvement": (
                    raw_test_mae
                    - calibrated_test_mae
                ),
                "mae_relative_improvement": (
                    (
                        raw_test_mae
                        - calibrated_test_mae
                    )
                    / raw_test_mae
                ),
            },
            "constrained_vs_baseline": {
                "mae_absolute_improvement": (
                    baseline_test_mae
                    - calibrated_test_mae
                ),
                "mae_relative_improvement": (
                    (
                        baseline_test_mae
                        - calibrated_test_mae
                    )
                    / baseline_test_mae
                ),
            },
        },
    }

    write_json_atomically(
        config[
            "report_path"
        ],
        report,
    )

    ranked_candidates = sorted(
        candidates,
        key=lambda candidate: (
            not candidate[
                "feasible"
            ],
            float(
                candidate[
                    "metrics"
                ][
                    "overall"
                ][
                    "mae"
                ]
            ),
        ),
    )

    print(
        "Constrained calibration experiment completed"
    )

    print()

    print(
        "Top validation candidates:"
    )

    print()

    for candidate in ranked_candidates[:10]:
        metrics = candidate[
            "metrics"
        ]

        changes = candidate[
            "changes"
        ]

        print(
            f"band<={candidate['maximum_calibrated_band']:<5} "
            f"shrinkage={candidate['shrinkage']:.2f} "
            f"MAE={metrics['overall']['mae']:.6f} "
            f"RMSE={metrics['overall']['rmse']:.6f} "
            f"zero_mean="
            f"{metrics['zero_demand']['prediction_mean']:.4f} "
            f"high_bias="
            f"{metrics['high_demand']['mean_bias']:.4f} "
            f"bias_degradation="
            f"{changes['high_demand_bias_degradation']:.4f} "
            f"feasible={candidate['feasible']}"
        )

    print()

    print(
        "Selected candidate"
    )

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

    print(
        "Validation MAE:",
        (
            f"{selected['metrics']['overall']['mae']:.6f}"
        ),
    )

    print(
        "Validation zero-demand mean prediction:",
        (
            f"{raw_validation_metrics['zero_demand']['prediction_mean']:.4f}"
        ),
        "→",
        (
            f"{selected['metrics']['zero_demand']['prediction_mean']:.4f}"
        ),
    )

    print(
        "Validation high-demand bias:",
        (
            f"{raw_validation_metrics['high_demand']['mean_bias']:.4f}"
        ),
        "→",
        (
            f"{selected['metrics']['high_demand']['mean_bias']:.4f}"
        ),
    )

    print()

    print(
        "Test — evaluation only"
    )

    print(
        "Raw MAE:",
        f"{raw_test_mae:.6f}",
    )

    print(
        "Constrained MAE:",
        f"{calibrated_test_mae:.6f}",
    )

    print(
        "Constrained vs raw improvement:",
        (
            f"{report['test_evaluation']['constrained_vs_raw']['mae_relative_improvement'] * 100:.3f}%"
        ),
    )

    print(
        "Constrained vs lag-1 improvement:",
        (
            f"{report['test_evaluation']['constrained_vs_baseline']['mae_relative_improvement'] * 100:.2f}%"
        ),
    )

    print(
        "Zero-demand mean prediction:",
        (
            f"{test_metrics['raw']['zero_demand']['prediction_mean']:.4f}"
        ),
        "→",
        (
            f"{test_metrics['constrained_calibration']['zero_demand']['prediction_mean']:.4f}"
        ),
    )

    print(
        "High-demand mean bias:",
        (
            f"{test_metrics['raw']['high_demand']['mean_bias']:.4f}"
        ),
        "→",
        (
            f"{test_metrics['constrained_calibration']['high_demand']['mean_bias']:.4f}"
        ),
    )

    print()

    print(
        "Report:",
        config[
            "report_path"
        ],
    )


if __name__ == "__main__":
    main()