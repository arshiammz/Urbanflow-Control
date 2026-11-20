from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pytest

from urbanflow.rl.episodes import Episode, RLEpisodeError


def example() -> Episode:
    start = datetime(2024, 5, 15, 0, 0, tzinfo=ZoneInfo("America/New_York"))
    times = tuple(start + timedelta(minutes=15 * i) for i in range(3))
    q = np.ones((3, 2, 2), dtype=float)
    return Episode(
        timestamps=times,
        demand=np.ones((3, 2), dtype=np.int64),
        p10=q * 0.8, p50=q, p90=q * 1.2,
        warmup_steps=1, split="validation",
        forecast_provenance="unit_fixture",
    )


def test_dense_episode_contract() -> None:
    example().validate(horizon=2, zones=2, interval_minutes=15)


def test_non_monotonic_quantiles_fail_closed() -> None:
    from dataclasses import replace
    sample = example()
    with pytest.raises(RLEpisodeError):
        replace(sample, p10=sample.p90).validate(horizon=2, zones=2, interval_minutes=15)


def test_spring_dst_uses_utc_elapsed_time_not_wall_time() -> None:
    zone = ZoneInfo("America/New_York")
    times = (
        datetime(2024, 3, 10, 1, 45, tzinfo=zone),
        datetime(2024, 3, 10, 3, 0, tzinfo=zone),
    )
    q = np.ones((2, 2, 2))
    episode = Episode(
        timestamps=times, demand=np.ones((2, 2), dtype=np.int64),
        p10=q * .5, p50=q, p90=q * 1.5, warmup_steps=1,
        split="train", forecast_provenance="unit_fixture",
    )
    episode.validate(horizon=2, zones=2, interval_minutes=15)
