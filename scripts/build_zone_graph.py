"""Build a geometry-only k-nearest-neighbour graph for TLC zones."""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

from urbanflow.geo.zone_graph import ZoneGraph


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--centroids",
        type=Path,
        default=Path("data/reference/taxi_zone_centroids.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/spatial/nyc_taxi_zone_graph_all_eligible.json"),
    )
    parser.add_argument("--neighbors", type=int, default=8)
    parser.add_argument(
        "--zone-index",
        type=Path,
        default=Path(
            "data/processed/simulation/nyc_2024-01_2024-06_all_zones/zone_index.parquet"
        ),
    )
    args = parser.parse_args()
    try:
        if not args.zone_index.exists():
            raise FileNotFoundError(
                "All-zone simulation index is missing; build all-zone simulation inputs first: "
                f"{args.zone_index}"
            )
        frame = pl.read_csv(args.centroids).select(
            pl.col("zone_id").cast(pl.Int64),
            pl.col("latitude").cast(pl.Float64),
            pl.col("longitude").cast(pl.Float64),
        )
        selected = pl.read_parquet(args.zone_index).select(
            pl.col("zone_id").cast(pl.Int64)
        )
        frame = frame.join(selected, on="zone_id", how="inner").sort("zone_id")
        if frame.height != selected.height:
            raise ValueError(
                "Some simulation zones are missing centroid coordinates: "
                f"expected={selected.height}, observed={frame.height}."
            )
        graph = ZoneGraph.from_centroids(
            zone_ids=frame.get_column("zone_id").to_list(),
            latitudes=frame.get_column("latitude").to_list(),
            longitudes=frame.get_column("longitude").to_list(),
            k_neighbors=args.neighbors,
        )
        graph.save(args.output)
    except (OSError, ValueError) as error:
        raise SystemExit(f"Zone graph build failed: {error}") from error
    print("Zone graph completed")
    print("Zones:", len(graph.zone_ids))
    print("K neighbours:", graph.k_neighbors)
    print("Edges:", int((graph.adjacency > 0).sum() // 2))
    print("Output:", args.output)


if __name__ == "__main__":
    main()
