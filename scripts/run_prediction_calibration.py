from __future__ import annotations

from pathlib import Path
from typing import Any

from urbanflow.forecasting.prediction_calibration import (
    PredictionCalibrationError,
    calibrate_xgboost_predictions,
)


CONFIG_PATH = Path(
    "configs/forecasting/prediction_calibration.yaml"
)


def percentage(
    value: float | None,
) -> str:
    """Format an optional ratio as a percentage."""
    if value is None:
        return "undefined"

    return f"{value * 100:.2f}%"


def mean_prediction(
    metrics: dict[str, Any],
) -> float:
    """Calculate the mean prediction from aggregate metrics."""
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


def main() -> None:
    """Run validation-only XGBoost prediction calibration."""
    try:
        report = calibrate_xgboost_predictions(
            CONFIG_PATH
        )
    except PredictionCalibrationError as error:
        raise SystemExit(
            f"Prediction calibration failed: {error}"
        ) from error

    selection = report[
        "model_selection"
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

    validation_metrics = validation[
        "metrics"
    ]

    test_metrics = test[
        "metrics"
    ]

    validation_zero = validation[
        "operational_risk"
    ][
        "zero_demand"
    ]

    validation_high = validation[
        "operational_risk"
    ][
        "high_demand"
    ]

    test_zero = test[
        "operational_risk"
    ][
        "zero_demand"
    ]

    test_high = test[
        "operational_risk"
    ][
        "high_demand"
    ]

    validation_zero_raw_mean = mean_prediction(
        validation_zero["raw"]
    )

    validation_zero_calibrated_mean = mean_prediction(
        validation_zero["calibrated"]
    )

    test_zero_raw_mean = mean_prediction(
        test_zero["raw"]
    )

    test_zero_calibrated_mean = mean_prediction(
        test_zero["calibrated"]
    )

    print(
        "XGBoost prediction calibration completed"
    )

    print()

    print(
        "Calibration fit split:",
        selection[
            "fit_split"
        ],
    )

    print(
        "Selected shrinkage:",
        f"{selection['selected_shrinkage']:.2f}",
    )

    print()

    print(
        "Validation"
    )

    print(
        "Raw MAE:",
        f"{validation_metrics['raw']['mae']:.4f}",
    )

    print(
        "Calibrated MAE:",
        f"{validation_metrics['calibrated']['mae']:.4f}",
    )

    print(
        "Calibration MAE improvement:",
        percentage(
            validation_metrics[
                "calibrated_vs_raw"
            ][
                "mae_relative_improvement"
            ]
        ),
    )

    print(
        "Calibrated WAPE:",
        percentage(
            validation_metrics[
                "calibrated"
            ][
                "wape"
            ]
        ),
    )

    print(
        "Zero-demand mean prediction:",
        f"{validation_zero_raw_mean:.4f}",
        "→",
        f"{validation_zero_calibrated_mean:.4f}",
    )

    print(
        "High-demand mean bias:",
        f"{validation_high['raw']['mean_bias']:.4f}",
        "→",
        f"{validation_high['calibrated']['mean_bias']:.4f}",
    )

    print()

    print(
        "Test — evaluation only"
    )

    print(
        "Raw MAE:",
        f"{test_metrics['raw']['mae']:.4f}",
    )

    print(
        "Calibrated MAE:",
        f"{test_metrics['calibrated']['mae']:.4f}",
    )

    print(
        "Calibration MAE improvement:",
        percentage(
            test_metrics[
                "calibrated_vs_raw"
            ][
                "mae_relative_improvement"
            ]
        ),
    )

    print(
        "Calibrated vs lag-1 improvement:",
        percentage(
            test_metrics[
                "calibrated_vs_baseline"
            ][
                "mae_relative_improvement"
            ]
        ),
    )

    print(
        "Calibrated WAPE:",
        percentage(
            test_metrics[
                "calibrated"
            ][
                "wape"
            ]
        ),
    )

    print(
        "Zero-demand mean prediction:",
        f"{test_zero_raw_mean:.4f}",
        "→",
        f"{test_zero_calibrated_mean:.4f}",
    )

    print(
        "High-demand mean bias:",
        f"{test_high['raw']['mean_bias']:.4f}",
        "→",
        f"{test_high['calibrated']['mean_bias']:.4f}",
    )

    print()

    print(
        "Calibrated predictions:",
        report[
            "effective_config"
        ][
            "paths"
        ][
            "calibrated_predictions"
        ],
    )

    print(
        "Calibration artifact:",
        report[
            "effective_config"
        ][
            "paths"
        ][
            "calibration_artifact"
        ],
    )

    print(
        "Report:",
        report[
            "effective_config"
        ][
            "paths"
        ][
            "report"
        ],
    )


if __name__ == "__main__":
    main()