"""Geometry-derived taxi-zone graph and graph-lag feature engineering.

The graph is built from taxi-zone centroids, not future trip outcomes.  Each
zone is connected to its k nearest zones using a symmetric, distance-decayed
adjacency matrix.  Row-normalized adjacency is then used to aggregate only
historically available lag features from neighboring zones.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from numpy.typing import NDArray


class ZoneGraphError(ValueError):
    """Raised when zone graph inputs or invariants are invalid."""


FloatMatrix = NDArray[np.float64]
FloatVector = NDArray[np.float64]


def haversine_km(
    latitude_a: FloatVector,
    longitude_a: FloatVector,
    latitude_b: FloatVector,
    longitude_b: FloatVector,
) -> FloatMatrix:
    """Return pairwise great-circle distances in kilometres."""
    lat_a = np.radians(np.asarray(latitude_a, dtype=np.float64))[:, None]
    lon_a = np.radians(np.asarray(longitude_a, dtype=np.float64))[:, None]
    lat_b = np.radians(np.asarray(latitude_b, dtype=np.float64))[None, :]
    lon_b = np.radians(np.asarray(longitude_b, dtype=np.float64))[None, :]

    delta_lat = lat_b - lat_a
    delta_lon = lon_b - lon_a
    hav = (
        np.sin(delta_lat / 2.0) ** 2
        + np.cos(lat_a) * np.cos(lat_b) * np.sin(delta_lon / 2.0) ** 2
    )
    return 6371.0088 * 2.0 * np.arcsin(np.sqrt(np.clip(hav, 0.0, 1.0)))


@dataclass(frozen=True)
class ZoneGraph:
    """Immutable weighted zone graph in deterministic zone-id order."""

    zone_ids: tuple[int, ...]
    latitudes: FloatVector
    longitudes: FloatVector
    adjacency: FloatMatrix
    k_neighbors: int
    version: str = "centroid_knn_symmetric_v1"

    def __post_init__(self) -> None:
        count = len(self.zone_ids)
        if count < 2 or len(set(self.zone_ids)) != count:
            raise ZoneGraphError("zone_ids must contain at least two unique zones.")
        if self.latitudes.shape != (count,) or self.longitudes.shape != (count,):
            raise ZoneGraphError("Centroid arrays do not match zone_ids.")
        if self.adjacency.shape != (count, count):
            raise ZoneGraphError("Adjacency matrix has an invalid shape.")
        if not np.isfinite(self.latitudes).all() or not np.isfinite(self.longitudes).all():
            raise ZoneGraphError("Zone coordinates must be finite.")
        if not np.isfinite(self.adjacency).all() or np.any(self.adjacency < 0):
            raise ZoneGraphError("Adjacency weights must be finite and non-negative.")
        if not np.allclose(self.adjacency, self.adjacency.T, atol=1e-10):
            raise ZoneGraphError("Adjacency matrix must be symmetric.")
        if not np.allclose(np.diag(self.adjacency), 0.0):
            raise ZoneGraphError("Self-loops are not allowed in the zone graph.")
        row_sums = self.adjacency.sum(axis=1)
        if np.any(row_sums <= 0):
            raise ZoneGraphError("Every zone must have at least one neighbour.")
        self.latitudes.setflags(write=False)
        self.longitudes.setflags(write=False)
        self.adjacency.setflags(write=False)

    @classmethod
    def from_centroids(
        cls,
        *,
        zone_ids: Iterable[int],
        latitudes: Iterable[float],
        longitudes: Iterable[float],
        k_neighbors: int = 8,
        distance_scale_km: float | None = None,
    ) -> ZoneGraph:
        """Build a symmetric k-nearest-neighbour graph from WGS84 centroids."""
        raw_ids = tuple(int(value) for value in zone_ids)
        raw_lat = np.asarray(tuple(latitudes), dtype=np.float64)
        raw_lon = np.asarray(tuple(longitudes), dtype=np.float64)
        if raw_lat.shape != (len(raw_ids),) or raw_lon.shape != (len(raw_ids),):
            raise ZoneGraphError("Each zone must have one latitude and longitude.")
        order = np.argsort(np.asarray(raw_ids, dtype=np.int64))
        ids = tuple(raw_ids[index] for index in order)
        lat = raw_lat[order]
        lon = raw_lon[order]
        if len(ids) < 2 or len(set(ids)) != len(ids):
            raise ZoneGraphError("At least two unique zone IDs are required.")
        if not 1 <= k_neighbors < len(ids):
            raise ZoneGraphError("k_neighbors must be between 1 and zone_count - 1.")
        if np.any(~np.isfinite(lat)) or np.any(~np.isfinite(lon)):
            raise ZoneGraphError("Centroid coordinates must be finite.")
        if np.any((lat < -90.0) | (lat > 90.0)):
            raise ZoneGraphError("Latitude must be between -90 and 90 degrees.")
        if np.any((lon < -180.0) | (lon > 180.0)):
            raise ZoneGraphError("Longitude must be between -180 and 180 degrees.")

        distance = haversine_km(lat, lon, lat, lon)
        np.fill_diagonal(distance, np.inf)
        nearest = np.argsort(distance, axis=1)[:, :k_neighbors]
        binary = np.zeros((len(ids), len(ids)), dtype=np.float64)
        binary[np.arange(len(ids))[:, None], nearest] = 1.0
        binary = np.maximum(binary, binary.T)

        positive_distances = distance[np.isfinite(distance) & (distance > 0)]
        if distance_scale_km is None:
            distance_scale_km = float(np.median(positive_distances))
        if not math.isfinite(distance_scale_km) or distance_scale_km <= 0:
            raise ZoneGraphError("distance_scale_km must be finite and positive.")

        weights = np.exp(-distance / distance_scale_km)
        weights[~np.isfinite(weights)] = 0.0
        weights *= binary
        np.fill_diagonal(weights, 0.0)
        row_sums = weights.sum(axis=1)
        if np.any(row_sums <= 0):
            raise ZoneGraphError("At least one zone has no graph neighbours.")
        # Keep the stored undirected graph symmetric.  Row normalization is
        # applied only when a neighbour aggregation is requested.
        symmetric = (weights + weights.T) / 2.0
        max_sum = float(symmetric.sum(axis=1).max())
        if max_sum <= 0:
            raise ZoneGraphError("Graph has no edges.")
        symmetric = symmetric / max_sum
        # Symmetric edge weights are converted to row-stochastic weights only
        # at aggregation time; this keeps the graph representation symmetric.
        return cls(
            zone_ids=ids,
            latitudes=lat,
            longitudes=lon,
            adjacency=symmetric,
            k_neighbors=k_neighbors,
        )

    @property
    def neighbor_counts(self) -> NDArray[np.int64]:
        return np.count_nonzero(self.adjacency, axis=1).astype(np.int64)

    @property
    def normalized_adjacency(self) -> FloatMatrix:
        row_sums = self.adjacency.sum(axis=1, keepdims=True)
        return np.divide(
            self.adjacency,
            row_sums,
            out=np.zeros_like(self.adjacency),
            where=row_sums > 0,
        )

    def neighbor_aggregate(self, values: NDArray[np.float64]) -> NDArray[np.float64]:
        """Aggregate the final zone dimension using graph neighbours."""
        array = np.asarray(values, dtype=np.float64)
        if array.shape[-1] != len(self.zone_ids):
            raise ZoneGraphError("Input final dimension must match graph zone count.")
        if np.any(~np.isfinite(array)):
            raise ZoneGraphError("Graph aggregation input must be finite.")
        return array @ self.normalized_adjacency.T

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible graph artifact."""
        return {
            "version": self.version,
            "zone_ids": list(self.zone_ids),
            "latitudes": self.latitudes.tolist(),
            "longitudes": self.longitudes.tolist(),
            "adjacency": self.adjacency.tolist(),
            "k_neighbors": self.k_neighbors,
            "invariants": {
                "symmetric": True,
                "no_self_loops": True,
                "all_zones_have_neighbors": bool(np.all(self.neighbor_counts > 0)),
                "geometry_only": True,
            },
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ZoneGraph:
        """Restore a graph artifact from JSON data."""
        return cls(
            zone_ids=tuple(int(value) for value in payload["zone_ids"]),
            latitudes=np.asarray(payload["latitudes"], dtype=np.float64),
            longitudes=np.asarray(payload["longitudes"], dtype=np.float64),
            adjacency=np.asarray(payload["adjacency"], dtype=np.float64),
            k_neighbors=int(payload["k_neighbors"]),
            version=str(payload.get("version", "centroid_knn_symmetric_v1")),
        )

    def save(self, path: Path) -> None:
        """Write graph JSON atomically."""
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".part")
        temporary.write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)

    @classmethod
    def load(cls, path: Path) -> ZoneGraph:
        """Read a graph JSON artifact."""
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ZoneGraphError("Graph artifact must contain a JSON object.")
        return cls.from_dict(payload)


def add_graph_features(
    frame: Any,
    *,
    graph: ZoneGraph,
    timestamp_column: str = "interval_start",
    zone_column: str = "zone_id",
    source_columns: tuple[str, ...] = (
        "pickup_count_lag_1",
        "pickup_count_lag_4",
        "pickup_count_rolling_mean_4",
    ),
) -> Any:
    """Add graph-lag features to a dense Polars zone-time panel.

    Rows must contain exactly one observation for every (timestamp, zone) pair.
    The dense-panel contract prevents silent zone misalignment during matrix
    multiplication and makes the operation deterministic.
    """
    import polars as pl

    required = {timestamp_column, zone_column, *source_columns}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ZoneGraphError("Feature frame is missing columns: " + ", ".join(missing))

    zone_ids = tuple(
        int(value)
        for value in frame.get_column(zone_column).unique().sort().to_list()
    )
    if zone_ids != tuple(sorted(graph.zone_ids)):
        raise ZoneGraphError("Feature-frame zones do not match graph zone IDs.")
    if len(zone_ids) != len(graph.zone_ids):
        raise ZoneGraphError("Feature-frame zone count does not match graph.")

    ordered = frame.sort([timestamp_column, zone_column])
    timestamp_count = ordered.get_column(timestamp_column).n_unique()
    expected_rows = timestamp_count * len(zone_ids)
    if ordered.height != expected_rows:
        raise ZoneGraphError(
            f"Feature panel is not dense: expected={expected_rows}, observed={ordered.height}."
        )
    duplicate_count = (
        ordered.group_by(timestamp_column, zone_column)
        .len()
        .filter(pl.col("len") != 1)
        .height
    )
    if duplicate_count:
        raise ZoneGraphError("Feature panel contains duplicate timestamp-zone keys.")

    additions: list[pl.Series] = []
    for source_column in source_columns:
        values = ordered.get_column(source_column).cast(pl.Float64).to_numpy()
        if np.any(~np.isfinite(values)):
            raise ZoneGraphError(f"Source feature contains non-finite values: {source_column}")
        matrix = values.reshape(timestamp_count, len(zone_ids))
        aggregated = graph.neighbor_aggregate(matrix).reshape(-1)
        suffix = source_column.removeprefix("pickup_count_")
        additions.append(pl.Series(f"graph_neighbor_{suffix}", aggregated))

    degree = pl.Series(
        "graph_neighbor_count",
        graph.neighbor_counts.astype(np.int64),
    )
    coordinate_frame = pl.DataFrame(
        {
            zone_column: list(graph.zone_ids),
            "zone_latitude": graph.latitudes,
            "zone_longitude": graph.longitudes,
        }
    )
    ordered = ordered.with_columns(additions).join(
        coordinate_frame,
        on=zone_column,
        how="left",
    )
    degree_frame = pl.DataFrame(
        {zone_column: list(graph.zone_ids), "graph_neighbor_count": degree}
    )
    ordered = ordered.join(degree_frame, on=zone_column, how="left")
    return ordered.sort([timestamp_column, zone_column])
