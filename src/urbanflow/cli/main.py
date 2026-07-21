from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from urbanflow.data.aggregation import (
    DemandAggregationError,
    build_zone_demand,
)
from urbanflow.data.audit import (
    DataAuditError,
    audit_trip_data,
)
from urbanflow.data.download import (
    DownloadError,
    download_trip_data,
)
from urbanflow.data.multimonth import (
    MultiMonthPipelineError,
    build_multimonth_demand,
)
from urbanflow.data.reference import (
    download_taxi_zone_lookup,
    inspect_taxi_zone_lookup,
)
from urbanflow.data.validation import (
    ValidationError,
    validate_trip_data,
)
from urbanflow.forecasting.baselines import (
    BaselineEvaluationError,
    evaluate_forecasting_baselines,
)
from urbanflow.forecasting.features import (
    ForecastFeatureError,
    build_forecast_features,
)


app = typer.Typer(
    help="UrbanFlow Control command-line interface.",
    no_args_is_help=True,
)

console = Console()


@app.callback()
def main() -> None:
    """UrbanFlow Control data, forecasting, and simulation tools."""


@app.command("download-data")
def download_data(
    dataset: Annotated[
        str,
        typer.Option(
            help="TLC dataset identifier.",
        ),
    ] = "hvfhv",
    year: Annotated[
        int,
        typer.Option(
            min=2009,
            help="Dataset year.",
        ),
    ] = 2024,
    month: Annotated[
        int,
        typer.Option(
            min=1,
            max=12,
            help="Dataset month.",
        ),
    ] = 1,
    output_directory: Annotated[
        Path,
        typer.Option(
            "--output-dir",
            help="Directory for downloaded files.",
        ),
    ] = Path("data/raw/hvfhv"),
    force: Annotated[
        bool,
        typer.Option(
            help="Replace an existing local file.",
        ),
    ] = False,
) -> None:
    """Download one monthly TLC trip-record file."""
    try:
        destination = download_trip_data(
            dataset=dataset,
            year=year,
            month=month,
            output_directory=output_directory,
            force=force,
        )
    except (ValueError, DownloadError) as error:
        console.print(
            f"[bold red]Download failed:[/bold red] {error}"
        )
        raise typer.Exit(code=1) from error

    size_mb = (
        destination.stat().st_size
        / (1024 * 1024)
    )

    console.print(
        "[bold green]Download completed[/bold green]"
    )
    console.print(
        f"File: {destination}"
    )
    console.print(
        f"Size: {size_mb:,.2f} MB"
    )


@app.command("download-zone-lookup")
def download_zone_lookup(
    output_path: Annotated[
        Path,
        typer.Option(
            "--output",
            help="Path for the taxi-zone lookup CSV.",
        ),
    ] = Path(
        "data/reference/taxi_zone_lookup.csv"
    ),
    force: Annotated[
        bool,
        typer.Option(
            help="Replace an existing lookup file.",
        ),
    ] = False,
) -> None:
    """Download and validate the official TLC taxi-zone lookup."""
    try:
        destination = download_taxi_zone_lookup(
            output_path=output_path,
            force=force,
        )

        summary = inspect_taxi_zone_lookup(
            destination
        )

    except DownloadError as error:
        console.print(
            "[bold red]"
            f"Zone lookup download failed: {error}"
            "[/bold red]"
        )
        raise typer.Exit(code=1) from error

    console.print(
        "[bold green]"
        "Taxi-zone lookup is ready"
        "[/bold green]"
    )
    console.print(
        f"File: {destination}"
    )
    console.print(
        f"Rows: {summary['row_count']:,}"
    )
    console.print(
        "LocationID range: "
        f"{summary['minimum_location_id']}–"
        f"{summary['maximum_location_id']}"
    )
    console.print(
        "Unique LocationIDs: "
        f"{summary['unique_location_ids']:,}"
    )
    console.print(
        "Duplicate LocationIDs: "
        f"{summary['duplicate_location_ids']:,}"
    )


@app.command("audit-data")
def audit_data(
    input_path: Annotated[
        Path,
        typer.Option(
            "--input",
            help="Path to a monthly TLC Parquet file.",
        ),
    ] = Path(
        "data/raw/hvfhv/"
        "fhvhv_tripdata_2024-01.parquet"
    ),
    output_path: Annotated[
        Path,
        typer.Option(
            "--output",
            help="Path for the generated JSON audit report.",
        ),
    ] = Path(
        "reports/data_quality/"
        "2024-01-raw-audit.json"
    ),
    year: Annotated[
        int,
        typer.Option(
            min=2009,
            help="Expected dataset year.",
        ),
    ] = 2024,
    month: Annotated[
        int,
        typer.Option(
            min=1,
            max=12,
            help="Expected dataset month.",
        ),
    ] = 1,
) -> None:
    """Generate a reproducible audit report for TLC trip data."""
    console.print(
        f"Auditing [bold]{input_path}[/bold]..."
    )

    try:
        report = audit_trip_data(
            input_path=input_path,
            output_path=output_path,
            expected_year=year,
            expected_month=month,
        )
    except (
        ValueError,
        DataAuditError,
    ) as error:
        console.print(
            f"[bold red]Audit failed:[/bold red] {error}"
        )
        raise typer.Exit(code=1) from error

    console.print(
        "[bold green]Audit completed[/bold green]"
    )
    console.print(
        f"Rows: {report['row_count']:,}"
    )
    console.print(
        f"Report: {output_path}"
    )


@app.command("validate-data")
def validate_data(
    config_path: Annotated[
        Path,
        typer.Option(
            "--config",
            help="Path to the validation YAML configuration.",
        ),
    ] = Path(
        "configs/data/validation.yaml"
    ),
) -> None:
    """Validate TLC trip records and write an interim dataset."""
    console.print(
        "Running validation with "
        f"[bold]{config_path}[/bold]..."
    )

    try:
        report = validate_trip_data(
            config_path=config_path
        )
    except ValidationError as error:
        console.print(
            f"[bold red]Validation failed:[/bold red] {error}"
        )
        raise typer.Exit(code=1) from error

    status_counts = report[
        "status_counts"
    ]
    eligibility_counts = report[
        "eligibility_counts"
    ]

    console.print(
        "[bold green]Validation completed[/bold green]"
    )
    console.print(
        f"Total: {status_counts['total']:,}"
    )
    console.print(
        f"Accepted: {status_counts['accepted']:,}"
    )
    console.print(
        "Retained with flag: "
        f"{status_counts['retained_with_flag']:,}"
    )
    console.print(
        f"Rejected: {status_counts['rejected']:,}"
    )
    console.print(
        "Demand-forecast eligible: "
        f"{eligibility_counts['eligible_for_demand_forecasting']:,}"
    )
    console.print(
        f"Output: {report['output']['path']}"
    )


@app.command("build-demand")
def build_demand(
    config_path: Annotated[
        Path,
        typer.Option(
            "--config",
            help="Path to the demand aggregation YAML configuration.",
        ),
    ] = Path(
        "configs/data/demand_aggregation.yaml"
    ),
) -> None:
    """Build a dense zone-by-time demand dataset."""
    console.print(
        "Building zone demand with "
        f"[bold]{config_path}[/bold]..."
    )

    try:
        report = build_zone_demand(
            config_path=config_path
        )
    except DemandAggregationError as error:
        console.print(
            "[bold red]"
            f"Demand aggregation failed: {error}"
            "[/bold red]"
        )
        raise typer.Exit(code=1) from error

    panel = report["panel"]
    output = report["output"]

    console.print(
        "[bold green]"
        "Demand aggregation completed"
        "[/bold green]"
    )
    console.print(
        f"Rows: {output['row_count']:,}"
    )
    console.print(
        f"Intervals: {panel['interval_count']:,}"
    )
    console.print(
        f"Zones: {panel['zone_count']:,}"
    )
    console.print(
        f"Total pickups: {panel['total_pickups']:,}"
    )
    console.print(
        f"Zero-demand rows: {panel['zero_demand_rows']:,}"
    )
    console.print(
        f"Output: {output['path']}"
    )


@app.command("build-multimonth")
def build_multimonth(
    config_path: Annotated[
        Path,
        typer.Option(
            "--config",
            help="Path to the multi-month data pipeline configuration.",
        ),
    ] = Path(
        "configs/data/multimonth.yaml"
    ),
) -> None:
    """Build a resumable multi-month zone-demand dataset."""
    console.print(
        "Running multi-month data pipeline with "
        f"[bold]{config_path}[/bold]..."
    )

    try:
        report = build_multimonth_demand(
            config_path=config_path
        )
    except MultiMonthPipelineError as error:
        console.print(
            "[bold red]"
            f"Multi-month pipeline failed: {error}"
            "[/bold red]"
        )
        raise typer.Exit(code=1) from error

    execution = report["execution"]
    combined = report["combined_output"]

    console.print(
        "[bold green]"
        "Multi-month data pipeline completed"
        "[/bold green]"
    )
    console.print(
        f"Months: {execution['month_count']:,}"
    )
    console.print(
        "Combined action: "
        f"{execution['combined_action']}"
    )
    console.print(
        f"Rows: {combined['row_count']:,}"
    )
    console.print(
        f"Intervals: {combined['interval_count']:,}"
    )
    console.print(
        f"Zones: {combined['zone_count']:,}"
    )
    console.print(
        "Total pickups: "
        f"{combined['total_pickups']:,}"
    )
    console.print(
        f"Output: {combined['path']}"
    )


@app.command("build-features")
def build_features(
    config_path: Annotated[
        Path,
        typer.Option(
            "--config",
            help="Path to the forecasting feature YAML configuration.",
        ),
    ] = Path(
        "configs/forecasting/features.yaml"
    ),
) -> None:
    """Build leakage-safe forecasting features."""
    console.print(
        "Building forecasting features with "
        f"[bold]{config_path}[/bold]..."
    )

    try:
        report = build_forecast_features(
            config_path=config_path
        )
    except ForecastFeatureError as error:
        console.print(
            "[bold red]"
            f"Feature generation failed: {error}"
            "[/bold red]"
        )
        raise typer.Exit(code=1) from error

    output = report["output"]
    splits = report["splits"]
    feature_schema = report[
        "feature_schema"
    ]

    console.print(
        "[bold green]"
        "Forecasting features completed"
        "[/bold green]"
    )
    console.print(
        f"Rows: {output['row_count']:,}"
    )
    console.print(
        f"Intervals: {output['interval_count']:,}"
    )
    console.print(
        f"Zones: {output['zone_count']:,}"
    )
    console.print(
        f"Train rows: {splits['train_rows']:,}"
    )
    console.print(
        "Validation rows: "
        f"{splits['validation_rows']:,}"
    )
    console.print(
        f"Test rows: {splits['test_rows']:,}"
    )
    console.print(
        "Model features: "
        f"{feature_schema['feature_count']:,}"
    )
    console.print(
        f"Output: {output['path']}"
    )


@app.command("evaluate-baselines")
def evaluate_baselines(
    config_path: Annotated[
        Path,
        typer.Option(
            "--config",
            help="Path to the forecasting baseline YAML configuration.",
        ),
    ] = Path(
        "configs/forecasting/baselines.yaml"
    ),
) -> None:
    """Evaluate forecasting baseline models."""
    console.print(
        "Evaluating forecasting baselines with "
        f"[bold]{config_path}[/bold]..."
    )

    try:
        report = evaluate_forecasting_baselines(
            config_path=config_path
        )
    except BaselineEvaluationError as error:
        console.print(
            "[bold red]"
            f"Baseline evaluation failed: {error}"
            "[/bold red]"
        )
        raise typer.Exit(code=1) from error

    validation_best = report[
        "best_baseline_by_mae"
    ]["validation"]

    test_best = report[
        "best_baseline_by_mae"
    ]["test"]

    validation_metrics = report[
        "metrics"
    ]["validation"][validation_best]

    test_metrics = report[
        "metrics"
    ]["test"][test_best]

    console.print(
        "[bold green]"
        "Baseline evaluation completed"
        "[/bold green]"
    )
    console.print(
        f"Validation best: {validation_best}"
    )
    console.print(
        "Validation MAE: "
        f"{validation_metrics['mae']:,.4f}"
    )
    console.print(
        f"Test best: {test_best}"
    )
    console.print(
        f"Test MAE: {test_metrics['mae']:,.4f}"
    )
    console.print(
        "Predictions: "
        f"{report['predictions']['path']}"
    )


if __name__ == "__main__":
    app()