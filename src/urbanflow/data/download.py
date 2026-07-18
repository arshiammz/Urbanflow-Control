from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import httpx


TLC_TRIP_DATA_BASE_URL = (
    "https://d37ci6vzurychx.cloudfront.net/trip-data"
)

DATASET_PREFIXES = {
    "hvfhv": "fhvhv_tripdata",
}


class DownloadError(RuntimeError):
    """Raised when a TLC dataset cannot be downloaded safely."""


def build_trip_data_url(
    dataset: str,
    year: int,
    month: int,
) -> str:
    """Construct the official NYC TLC monthly Parquet URL."""
    if dataset not in DATASET_PREFIXES:
        supported = ", ".join(sorted(DATASET_PREFIXES))
        raise ValueError(
            f"Unsupported dataset {dataset!r}. Supported datasets: {supported}"
        )

    if not 2009 <= year <= datetime.now(UTC).year:
        raise ValueError(f"Invalid year: {year}")

    if not 1 <= month <= 12:
        raise ValueError(f"Invalid month: {month}")

    prefix = DATASET_PREFIXES[dataset]
    filename = f"{prefix}_{year}-{month:02d}.parquet"

    return f"{TLC_TRIP_DATA_BASE_URL}/{filename}"


def calculate_sha256(path: Path) -> str:
    """Calculate the SHA-256 checksum of a local file."""
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def download_trip_data(
    dataset: str,
    year: int,
    month: int,
    output_directory: Path,
    *,
    force: bool = False,
) -> Path:
    """Download one monthly NYC TLC Parquet file atomically."""
    url = build_trip_data_url(
        dataset=dataset,
        year=year,
        month=month,
    )

    filename = f"{DATASET_PREFIXES[dataset]}_{year}-{month:02d}.parquet"

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    destination = output_directory / filename
    temporary_file = destination.with_suffix(
        destination.suffix + ".part"
    )
    metadata_file = destination.with_suffix(
        ".metadata.json"
    )

    if destination.exists() and not force:
        if destination.stat().st_size == 0:
            raise DownloadError(
                f"Existing file is empty: {destination}"
            )

        return destination

    temporary_file.unlink(missing_ok=True)

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
            with client.stream("GET", url) as response:
                response.raise_for_status()

                content_type = response.headers.get(
                    "content-type",
                    "",
                )

                with temporary_file.open("wb") as file:
                    for chunk in response.iter_bytes(
                        chunk_size=1024 * 1024
                    ):
                        file.write(chunk)

        if not temporary_file.exists():
            raise DownloadError(
                "Temporary download file was not created."
            )

        file_size = temporary_file.stat().st_size

        if file_size == 0:
            raise DownloadError(
                "Downloaded file is empty."
            )

        temporary_file.replace(destination)

        metadata = {
            "dataset": dataset,
            "year": year,
            "month": month,
            "source": "NYC Taxi and Limousine Commission",
            "source_url": url,
            "downloaded_at": datetime.now(UTC).isoformat(),
            "filename": destination.name,
            "size_bytes": destination.stat().st_size,
            "sha256": calculate_sha256(destination),
            "content_type": content_type,
        }

        metadata_file.write_text(
            json.dumps(
                metadata,
                indent=2,
            ),
            encoding="utf-8",
        )

    except (
        httpx.HTTPError,
        OSError,
    ) as error:
        temporary_file.unlink(missing_ok=True)

        raise DownloadError(
            f"Could not download {url}: {error}"
        ) from error

    return destination