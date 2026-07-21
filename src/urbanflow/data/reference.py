from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import polars as pl

from urbanflow.data.download import (
    DownloadError,
    calculate_sha256,
)


TAXI_ZONE_LOOKUP_URL = (
    "https://d37ci6vzurychx.cloudfront.net/"
    "misc/taxi_zone_lookup.csv"
)

REQUIRED_ZONE_COLUMNS = {
    "LocationID",
    "Borough",
    "Zone",
    "service_zone",
}


def download_taxi_zone_lookup(
    output_path: Path,
    *,
    force: bool = False,
) -> Path:
    """Download the official NYC TLC taxi-zone lookup table."""
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = output_path.with_suffix(
        output_path.suffix + ".part"
    )
    metadata_path = output_path.with_suffix(
        ".metadata.json"
    )

    if output_path.exists() and not force:
        if output_path.stat().st_size == 0:
            raise DownloadError(
                f"Existing zone lookup is empty: {output_path}"
            )

        return output_path

    temporary_path.unlink(missing_ok=True)

    timeout = httpx.Timeout(
        connect=30.0,
        read=120.0,
        write=30.0,
        pool=30.0,
    )

    headers = {
        "User-Agent": "UrbanFlow-Control/0.1",
    }

    try:
        with httpx.Client(
            timeout=timeout,
            follow_redirects=True,
            headers=headers,
        ) as client:
            with client.stream(
                "GET",
                TAXI_ZONE_LOOKUP_URL,
            ) as response:
                response.raise_for_status()

                content_type = response.headers.get(
                    "content-type",
                    "",
                )

                with temporary_path.open("wb") as file:
                    for chunk in response.iter_bytes(
                        chunk_size=1024 * 1024
                    ):
                        file.write(chunk)

        if not temporary_path.exists():
            raise DownloadError(
                "Temporary taxi-zone lookup was not created."
            )

        if temporary_path.stat().st_size == 0:
            raise DownloadError(
                "Downloaded taxi-zone lookup is empty."
            )

        temporary_path.replace(output_path)

        metadata = {
            "dataset": "NYC TLC Taxi Zone Lookup",
            "source": "NYC Taxi and Limousine Commission",
            "source_url": TAXI_ZONE_LOOKUP_URL,
            "downloaded_at": datetime.now(UTC).isoformat(),
            "filename": output_path.name,
            "size_bytes": output_path.stat().st_size,
            "sha256": calculate_sha256(output_path),
            "content_type": content_type,
        }

        metadata_path.write_text(
            json.dumps(
                metadata,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    except (
        httpx.HTTPError,
        OSError,
    ) as error:
        temporary_path.unlink(missing_ok=True)

        raise DownloadError(
            f"Could not download taxi-zone lookup: {error}"
        ) from error

    return output_path


def inspect_taxi_zone_lookup(
    path: Path,
) -> dict[str, Any]:
    """Validate and summarize the taxi-zone lookup table."""
    if not path.exists():
        raise DownloadError(
            f"Taxi-zone lookup does not exist: {path}"
        )

    if path.stat().st_size == 0:
        raise DownloadError(
            f"Taxi-zone lookup is empty: {path}"
        )

    frame = pl.read_csv(path)

    if frame.height == 0:
        raise DownloadError(
            "Taxi-zone lookup contains no rows."
        )

    missing_columns = sorted(
        REQUIRED_ZONE_COLUMNS.difference(
            frame.columns
        )
    )

    if missing_columns:
        missing_text = ", ".join(missing_columns)

        raise DownloadError(
            f"Taxi-zone lookup is missing columns: {missing_text}"
        )

    null_counts = {
        column: int(
            frame[column].null_count()
        )
        for column in sorted(REQUIRED_ZONE_COLUMNS)
    }

    duplicate_location_ids = int(
        frame.select(
            pl.col("LocationID")
            .is_duplicated()
            .sum()
            .alias("duplicate_count")
        ).item()
    )

    minimum_location_id = frame[
        "LocationID"
    ].min()

    maximum_location_id = frame[
        "LocationID"
    ].max()

    if (
        minimum_location_id is None
        or maximum_location_id is None
    ):
        raise DownloadError(
            "Taxi-zone lookup has invalid LocationID values."
        )

    borough_counts = (
        frame.group_by("Borough")
        .agg(
            pl.len().alias("zone_count")
        )
        .sort("Borough")
        .to_dicts()
    )

    return {
        "path": str(path),
        "row_count": frame.height,
        "column_count": frame.width,
        "columns": frame.columns,
        "minimum_location_id": int(
            minimum_location_id
        ),
        "maximum_location_id": int(
            maximum_location_id
        ),
        "unique_location_ids": int(
            frame["LocationID"].n_unique()
        ),
        "duplicate_location_ids": (
            duplicate_location_ids
        ),
        "null_counts": null_counts,
        "borough_counts": borough_counts,
    }