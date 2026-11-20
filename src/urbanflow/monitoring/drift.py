from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np


class MonitoringError(ValueError):
    """Monitoring inputs or reference statistics are invalid."""


def population_stability_index(
    reference_values: Any,
    current_values: Any,
    *,
    bins: int = 10,
    epsilon: float = 1e-6,
) -> float:
    """Calculate PSI using quantile bins fitted only on the reference sample."""
    reference = np.asarray(reference_values, dtype=np.float64)
    current = np.asarray(current_values, dtype=np.float64)
    if reference.ndim != 1 or current.ndim != 1 or not reference.size or not current.size:
        raise MonitoringError("PSI samples must be non-empty one-dimensional arrays.")
    if not np.isfinite(reference).all() or not np.isfinite(current).all():
        raise MonitoringError("PSI samples must contain only finite values.")
    if bins < 2 or not 0 < epsilon < 0.1:
        raise MonitoringError("PSI bin count or epsilon is invalid.")
    quantiles = np.linspace(0.0, 1.0, bins + 1)
    edges = np.unique(np.quantile(reference, quantiles))
    if edges.size == 1:
        edges = np.asarray([-np.inf, edges[0] + 0.5, np.inf], dtype=np.float64)
    else:
        edges[0] = -np.inf
        edges[-1] = np.inf
    ref_counts = np.histogram(reference, bins=edges)[0].astype(np.float64)
    cur_counts = np.histogram(current, bins=edges)[0].astype(np.float64)
    ref_share = np.maximum(ref_counts / ref_counts.sum(), epsilon)
    cur_share = np.maximum(cur_counts / cur_counts.sum(), epsilon)
    ref_share /= ref_share.sum()
    cur_share /= cur_share.sum()
    return float(np.sum((cur_share - ref_share) * np.log(cur_share / ref_share)))


def distribution_summary(values: Any) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not array.size or not np.isfinite(array).all():
        raise MonitoringError("Distribution values must be finite and non-empty.")
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "std": float(array.std()),
        "zero_rate": float(np.mean(array == 0.0)),
        "p50": float(np.quantile(array, 0.50)),
        "p90": float(np.quantile(array, 0.90)),
        "p99": float(np.quantile(array, 0.99)),
    }


def build_reference(values: Any, *, split: str, source_path: str) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    summary = distribution_summary(array)
    quantiles = np.linspace(0.0, 1.0, 11)
    edges = np.unique(np.quantile(array, quantiles))
    if edges.size == 1:
        edges = np.asarray([-np.inf, edges[0] + 0.5, np.inf], dtype=np.float64)
    else:
        edges[0] = -np.inf
        edges[-1] = np.inf
    counts = np.histogram(array, bins=edges)[0]
    return {
        "version": "urbanflow_demand_monitoring_reference_v1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "source_path": source_path,
        "reference_split": split,
        "summary": summary,
        "histogram_edges": [
            None if not np.isfinite(value) else float(value) for value in edges
        ],
        "histogram_counts": [int(value) for value in counts],
    }


def psi_from_reference(reference: dict[str, Any], current_values: Any, *, epsilon: float = 1e-6) -> float:
    current = np.asarray(current_values, dtype=np.float64)
    raw_edges = reference["histogram_edges"]
    edges = np.asarray(
        [
            -np.inf if index == 0 and value is None
            else np.inf if index == len(raw_edges) - 1 and value is None
            else float(value)
            for index, value in enumerate(raw_edges)
        ],
        dtype=np.float64,
    )
    counts = np.asarray(reference["histogram_counts"], dtype=np.float64)
    if current.ndim != 1 or not current.size or not np.isfinite(current).all():
        raise MonitoringError("Current monitoring values must be finite and non-empty.")
    current_counts = np.histogram(current, bins=edges)[0].astype(np.float64)
    expected = np.maximum(counts / counts.sum(), epsilon)
    observed = np.maximum(current_counts / current_counts.sum(), epsilon)
    expected /= expected.sum()
    observed /= observed.sum()
    return float(np.sum((observed - expected) * np.log(observed / expected)))


def write_monitoring_report(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
