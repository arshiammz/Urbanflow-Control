from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from urbanflow.data.audit import (
    DataAuditError,
    audit_trip_data,
)
from urbanflow.data.download import (
    DownloadError,
    download_trip_data,
)
from urbanflow.data.reference import (
    download_taxi_zone_lookup,
    inspect_taxi_zone_lookup,
)
from urbanflow.data.validation import (
    ValidationError,
    validate_trip_data,
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

    size_mb = destination.stat().st_size / (1024 * 1024)

    console.print(
        "[bold green]Download completed[/bold green]"
    )
    console.print(f"File: {destination}")
    console.print(f"Size: {size_mb:,.2f} MB")


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
    console.print(f"File: {destination}")
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

    status_counts = report["status_counts"]
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


if __name__ == "__main__":
    app()