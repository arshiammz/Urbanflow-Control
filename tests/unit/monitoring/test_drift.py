from __future__ import annotations

import numpy as np
import pytest

from urbanflow.monitoring.drift import (
    MonitoringError,
    build_reference,
    distribution_summary,
    population_stability_index,
    psi_from_reference,
)


def test_psi_is_zero_for_identical_samples() -> None:
    values = np.arange(100, dtype=np.float64)
    assert population_stability_index(values, values) == pytest.approx(0.0, abs=1e-12)


def test_psi_detects_distribution_shift() -> None:
    reference = np.zeros(500, dtype=np.float64)
    current = np.full(500, 20.0, dtype=np.float64)
    assert population_stability_index(reference, current) > 0.5


def test_reference_round_trip_and_summary() -> None:
    values = np.asarray([0, 0, 1, 2, 3, 4, 5], dtype=np.float64)
    reference = build_reference(values, split="train", source_path="train.parquet")
    assert psi_from_reference(reference, values) == pytest.approx(0.0, abs=1e-12)
    assert distribution_summary(values)["zero_rate"] == pytest.approx(2 / 7)


def test_psi_rejects_non_finite_values() -> None:
    with pytest.raises(MonitoringError):
        population_stability_index([1.0, 2.0], [1.0, float("nan")])
