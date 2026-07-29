from __future__ import annotations

from pathlib import Path

from urbanflow.evaluation.forecast_error_analysis import (
    ForecastErrorAnalysisError,
    analyze_forecast_errors,
)


CONFIG_PATH = Path(
    "configs/forecasting/champion_error_analysis.yaml"
)


def main() -> None:
    """Run segmented error analysis for the operational champion."""
    try:
        report = analyze_forecast_errors(
            CONFIG_PATH
        )
    except ForecastErrorAnalysisError as error:
        raise SystemExit(
            f"Champion error analysis failed: {error}"
        ) from error

    print(
        "Operational champion error analysis completed"
    )

    print(
        "Rows analyzed:",
        f"{report['input']['joined']['row_count']:,}",
    )

    for metrics in report[
        "overall_metrics"
    ]:
        split = metrics["split"]

        print()

        print(
            f"{split.title()} metrics"
        )

        print(
            "Champion MAE:",
            f"{metrics['model_mae']:.6f}",
        )

        print(
            "Lag-1 MAE:",
            f"{metrics['baseline_mae']:.6f}",
        )

        print(
            "Champion vs lag-1 improvement:",
            (
                f"{metrics['mae_relative_improvement'] * 100:.2f}%"
            ),
        )

        print(
            "Champion RMSE:",
            f"{metrics['model_rmse']:.6f}",
        )

        print(
            "Champion WAPE:",
            f"{metrics['model_wape'] * 100:.2f}%",
        )

        print(
            "Champion SMAPE:",
            f"{metrics['model_smape'] * 100:.2f}%",
        )

        print(
            "Champion mean bias:",
            f"{metrics['model_mean_bias']:.6f}",
        )

    print()

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

    print(
        "Tables:",
        report[
            "effective_config"
        ][
            "paths"
        ][
            "tables_directory"
        ],
    )


if __name__ == "__main__":
    main()