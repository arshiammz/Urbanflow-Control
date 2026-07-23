from __future__ import annotations

from pathlib import Path

from urbanflow.evaluation.forecast_error_analysis import (
    ForecastErrorAnalysisError,
    analyze_forecast_errors,
)


CONFIG_PATH = Path(
    "configs/forecasting/error_analysis.yaml"
)


def main() -> None:
    """Run and print the forecasting error-analysis summary."""
    try:
        report = analyze_forecast_errors(
            CONFIG_PATH
        )
    except ForecastErrorAnalysisError as error:
        raise SystemExit(
            f"Forecast error analysis failed: {error}"
        ) from error

    print(
        "Forecast error analysis completed"
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
            "Model MAE:",
            f"{metrics['model_mae']:.4f}",
        )
        print(
            "Baseline MAE:",
            f"{metrics['baseline_mae']:.4f}",
        )
        print(
            "MAE improvement:",
            (
                f"{metrics['mae_relative_improvement'] * 100:.2f}%"
            ),
        )
        print(
            "Model WAPE:",
            f"{metrics['model_wape'] * 100:.2f}%",
        )
        print(
            "Baseline WAPE:",
            f"{metrics['baseline_wape'] * 100:.2f}%",
        )

    print()
    print(
        "Report:",
        report[
            "effective_config"
        ]["paths"]["report"],
    )

    print(
        "Tables:",
        report[
            "effective_config"
        ]["paths"]["tables_directory"],
    )


if __name__ == "__main__":
    main()