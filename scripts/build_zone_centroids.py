"""Download the official TLC zone shapefile and build WGS84 centroids."""

from __future__ import annotations

import csv
import zipfile
from pathlib import Path

import httpx
import shapefile
from pyproj import CRS, Transformer


SHAPEFILE_URL = "https://s3.amazonaws.com/nyc-tlc/misc/taxi_zones.zip"
ZIP_PATH = Path("data/reference/taxi_zones.zip")
EXTRACT_DIRECTORY = Path("data/reference/taxi_zones")
LOOKUP_PATH = Path("data/reference/taxi_zone_lookup.csv")
OUTPUT_PATH = Path("data/reference/taxi_zone_centroids.csv")


def ring_centroid(points: list[tuple[float, float]]) -> tuple[float, float, float]:
    """Return signed area and centroid for one polygon ring."""
    if len(points) < 3:
        return 0.0, 0.0, 0.0
    area_twice = 0.0
    centroid_x = 0.0
    centroid_y = 0.0
    for index, point in enumerate(points):
        next_point = points[(index + 1) % len(points)]
        cross = point[0] * next_point[1] - next_point[0] * point[1]
        area_twice += cross
        centroid_x += (point[0] + next_point[0]) * cross
        centroid_y += (point[1] + next_point[1]) * cross
    if abs(area_twice) < 1e-9:
        return 0.0, 0.0, 0.0
    return (
        area_twice / 2.0,
        centroid_x / (3.0 * area_twice),
        centroid_y / (3.0 * area_twice),
    )


def shape_centroid(shape: shapefile.Shape) -> tuple[float, float]:
    """Compute an area-weighted centroid for multipart polygon geometry."""
    starts = list(shape.parts) + [len(shape.points)]
    total_area = 0.0
    weighted_x = 0.0
    weighted_y = 0.0
    for start, end in zip(starts[:-1], starts[1:], strict=True):
        ring = [(float(x), float(y)) for x, y in shape.points[start:end]]
        area, x_value, y_value = ring_centroid(ring)
        total_area += area
        weighted_x += area * x_value
        weighted_y += area * y_value
    if abs(total_area) < 1e-9:
        points = [(float(x), float(y)) for x, y in shape.points]
        if not points:
            raise ValueError("Taxi-zone shape has no coordinate points.")
        return (
            sum(point[0] for point in points) / len(points),
            sum(point[1] for point in points) / len(points),
        )
    return weighted_x / total_area, weighted_y / total_area


def download_shapefile() -> Path:
    """Download and safely extract the TLC taxi-zone shapefile."""
    ZIP_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not ZIP_PATH.exists():
        with httpx.Client(timeout=120.0, follow_redirects=True) as client:
            response = client.get(SHAPEFILE_URL)
            response.raise_for_status()
            ZIP_PATH.write_bytes(response.content)
    EXTRACT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ZIP_PATH) as archive:
        root = EXTRACT_DIRECTORY.resolve()
        for member in archive.infolist():
            destination = (EXTRACT_DIRECTORY / member.filename).resolve()
            if root not in destination.parents and destination != root:
                raise ValueError(f"Unsafe path in shapefile archive: {member.filename}")
        archive.extractall(EXTRACT_DIRECTORY)
    shp_files = sorted(EXTRACT_DIRECTORY.rglob("*.shp"))
    if len(shp_files) != 1:
        raise RuntimeError(f"Expected one .shp file, found {len(shp_files)}.")
    return shp_files[0]


def main() -> None:
    if not LOOKUP_PATH.exists():
        raise SystemExit(f"TLC zone lookup is missing: {LOOKUP_PATH}")
    shp_path = download_shapefile()
    prj_path = shp_path.with_suffix(".prj")
    if prj_path.exists():
        source_crs = CRS.from_wkt(prj_path.read_text(encoding="utf-8"))
    else:
        source_crs = CRS.from_epsg(2263)
    transformer = Transformer.from_crs(source_crs, CRS.from_epsg(4326), always_xy=True)

    with LOOKUP_PATH.open("r", encoding="utf-8-sig", newline="") as file:
        lookup_rows = list(csv.DictReader(file))
    lookup = {
        int(row["LocationID"]): {
            "borough": row["Borough"],
            "zone_name": row["Zone"],
            "service_zone": row["service_zone"],
        }
        for row in lookup_rows
    }

    reader = shapefile.Reader(str(shp_path), encoding="utf-8")
    field_names = [field[0] for field in reader.fields[1:]]
    location_field = next(
        (name for name in field_names if name.lower() in {"locationid", "location_id"}),
        None,
    )
    if location_field is None:
        raise RuntimeError(f"No LocationID field in shapefile: {field_names}")

    output_rows: list[dict[str, object]] = []
    for shape_record in reader.iterShapeRecords():
        record = dict(zip(field_names, shape_record.record, strict=True))
        zone_id = int(record[location_field])
        if zone_id not in lookup:
            continue
        x_value, y_value = shape_centroid(shape_record.shape)
        longitude, latitude = transformer.transform(x_value, y_value)
        output_rows.append(
            {
                "zone_id": zone_id,
                "borough": lookup[zone_id]["borough"],
                "zone_name": lookup[zone_id]["zone_name"],
                "service_zone": lookup[zone_id]["service_zone"],
                "latitude": latitude,
                "longitude": longitude,
            }
        )

    if len({int(row["zone_id"]) for row in output_rows}) != len(output_rows):
        raise RuntimeError("Duplicate zone IDs found in centroid output.")
    if len(output_rows) < 250:
        raise RuntimeError(f"Unexpectedly small taxi-zone coverage: {len(output_rows)}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT_PATH.with_name(OUTPUT_PATH.name + ".part")
    with temporary.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(sorted(output_rows, key=lambda row: int(row["zone_id"])))
    temporary.replace(OUTPUT_PATH)
    print("TLC zone centroids completed")
    print("Zones:", len(output_rows))
    print("Shapefile:", shp_path)
    print("Output:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
