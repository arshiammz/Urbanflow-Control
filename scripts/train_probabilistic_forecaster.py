"""Train and evaluate the graph/weather XGBoost quantile forecasting engine."""

from __future__ import annotations

import argparse
from pathlib import Path

from urbanflow.forecasting.probabilistic_forecaster import (
    ProbabilisticForecastError,
    train_probabilistic_forecaster,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/forecasting/probabilistic_graph_weather.yaml"),
    )
    args = parser.parse_args()
    try:
        report = train_probabilistic_forecaster(args.config)
    except ProbabilisticForecastError as error:
        raise SystemExit(f"Probabilistic forecasting failed: {error}") from error
    print("Graph/weather probabilistic forecasting completed")
    print("Zones:", report["zone_count"])
    print("Training rows:", f'{report["training_rows"]:,}')
    print("Validation rows:", f'{report["validation_rows"]:,}')
    print("Prediction rows:", f'{report["prediction_rows"]:,}')
    print("Selected estimators:", report["selected_estimators"])
    if report["test_metrics"]:
        metrics = report["test_metrics"]
        print("June P50 MAE:", f'{metrics["p50_mae"]:.4f}')
        print("June central 80% coverage:", f'{100 * metrics["central_80_coverage"]:.2f}%')
        print("June mean pinball:", f'{metrics["mean_pinball"]:.4f}')
    print("Predictions:", report["predictions_path"])
    print("Report:", args.config)


if __name__ == "__main__":
    main()
