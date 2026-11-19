from __future__ import annotations

import numpy as np
import pytest

from urbanflow.geo.zone_graph import ZoneGraph, ZoneGraphError


def build_graph() -> ZoneGraph:
    return ZoneGraph.from_centroids(
        zone_ids=(3, 1, 2, 4),
        latitudes=(40.72, 40.70, 40.71, 40.73),
        longitudes=(-73.98, -74.00, -73.99, -73.97),
        k_neighbors=2,
    )


def test_graph_is_sorted_symmetric_and_has_no_self_loops() -> None:
    graph = build_graph()
    assert graph.zone_ids == (1, 2, 3, 4)
    assert np.allclose(graph.adjacency, graph.adjacency.T)
    assert np.allclose(np.diag(graph.adjacency), 0.0)
    assert (graph.neighbor_counts > 0).all()
    assert np.allclose(graph.normalized_adjacency.sum(axis=1), 1.0)


def test_neighbor_aggregate_is_weighted_and_deterministic() -> None:
    graph = build_graph()
    values = np.asarray([[10.0, 20.0, 30.0, 40.0]])
    first = graph.neighbor_aggregate(values)
    second = graph.neighbor_aggregate(values)
    assert first.shape == (1, 4)
    assert np.isfinite(first).all()
    assert np.array_equal(first, second)
    assert not np.array_equal(first, values)


def test_graph_rejects_invalid_k_and_non_finite_coordinates() -> None:
    with pytest.raises(ZoneGraphError, match="k_neighbors"):
        ZoneGraph.from_centroids(
            zone_ids=(1, 2), latitudes=(40.0, 41.0), longitudes=(-73.0, -74.0), k_neighbors=2
        )
    with pytest.raises(ZoneGraphError, match="finite"):
        ZoneGraph.from_centroids(
            zone_ids=(1, 2),
            latitudes=(40.0, float("nan")),
            longitudes=(-73.0, -74.0),
            k_neighbors=1,
        )


def test_graph_round_trip_preserves_adjacency() -> None:
    graph = build_graph()
    restored = ZoneGraph.from_dict(graph.to_dict())
    assert restored.zone_ids == graph.zone_ids
    assert np.array_equal(restored.adjacency, graph.adjacency)
