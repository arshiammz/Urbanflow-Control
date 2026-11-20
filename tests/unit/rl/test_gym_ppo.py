"""Tiny synthetic end-to-end tests; the real NYC Parquet files are unnecessary."""
from __future__ import annotations

from datetime import datetime, timedelta
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import numpy as np
import pytest

gymnasium = pytest.importorskip("gymnasium")
pytest.importorskip("polars")
torch = pytest.importorskip("torch")

from urbanflow.rl.action_projection import RelocationConstraints
from urbanflow.rl.environment import FleetRebalancingEnv
from urbanflow.rl.episodes import Episode, EpisodeBundle
from urbanflow.rl.ppo import GraphActorCritic, PPOSettings, train_one_seed
from urbanflow.simulation.real_episode_baseline import NetworkInputs


def env_factory(*, future_demand: int = 1, split: str = "train") -> FleetRebalancingEnv:
    now = datetime(2024, 4, 15, 11, 0, tzinfo=ZoneInfo("America/New_York"))
    ts = tuple(now + timedelta(minutes=15 * i) for i in range(3))
    q = np.ones((3, 3, 2), dtype=np.float64) * 2.0
    demand = np.asarray([[0, 0], [future_demand, 2], [1, 1]], dtype=np.int64)
    network = NetworkInputs(
        zone_ids=(1, 2), fit_pickup_shares=np.asarray([.5, .5]),
        destination_probabilities=np.eye(2),
        passenger_travel_steps=np.ones((2, 2), dtype=np.int64),
        relocation_travel_steps=np.asarray([[0, 1], [1, 0]], dtype=np.int64),
    )
    episode = Episode(
        timestamps=ts, demand=demand, p10=q * .8, p50=q, p90=q * 1.2,
        warmup_steps=1, split=split, forecast_provenance="unit_proxy",
    )
    bundle = EpisodeBundle(
        network=network, initial_vehicles=np.asarray([5, 5], dtype=np.int64),
        allowed_edges=np.asarray([[False, True], [True, False]]),
        region_index=np.asarray([0, 1], dtype=np.int64),
        region_names=("A", "B"),
        baseline=SimpleNamespace(
            interval_minutes=15, unmet_demand_penalty=10.0,
            relocation_cost_per_vehicle_step=.5, idle_vehicle_cost=.01,
        ),
        episodes=(episode,), forecast_horizon_steps=3, source=split,
    )
    return FleetRebalancingEnv(
        bundle=bundle, constraints=RelocationConstraints(
            maximum_relocation_steps=1,
            maximum_relocation_fraction=0.2,
            minimum_zone_reserve=0,
            demand_buffer_multiplier=1.0,
            in_transit_supply_credit=0.25,
        ),
    )


def test_current_observation_does_not_reveal_current_realized_demand() -> None:
    left, right = env_factory(future_demand=1), env_factory(future_demand=100)
    first_obs, _ = left.reset(seed=101)
    second_obs, _ = right.reset(seed=101)
    np.testing.assert_array_equal(first_obs, second_obs)
    action = np.ones(left.action_space.shape, dtype=np.float32)
    _, _, _, _, a = left.step(action)
    _, _, _, _, b = right.step(action)
    assert a["demand"] != b["demand"]
    assert a["fleet_conservation_passed"] and b["fleet_conservation_passed"]
    assert a["action_relocated"] == b["action_relocated"]


def test_actor_actions_are_bounded_and_logprob_finite() -> None:
    env = env_factory()
    obs, _ = env.reset(seed=1)
    actor = GraphActorCritic(env)
    with torch.no_grad():
        action, log_prob, value = actor.act(torch.tensor(obs[None]))
        score, entropy, compared_value = actor.evaluate_actions(torch.tensor(obs[None]), action)
    assert action.shape == (1, env.action_space.shape[0])
    assert torch.all(action >= -1) and torch.all(action <= 1)
    assert torch.isfinite(log_prob).all() and torch.isfinite(score).all()
    assert torch.isfinite(value).all() and torch.isfinite(entropy).all()
    assert torch.allclose(value, compared_value)


def test_small_ppo_training_uses_train_only() -> None:
    training_env = env_factory(split="train")
    settings = PPOSettings(updates=1, rollout_steps=8, epochs=1, minibatch_size=4)
    model, history = train_one_seed(training_env, seed=5, settings=settings)
    assert isinstance(model, GraphActorCritic)
    assert len(history) == 1
    assert np.isfinite(history[0]["mean_training_reward"])
    with pytest.raises(RuntimeError, match="train-only"):
        train_one_seed(env_factory(split="validation"), seed=5, settings=settings)


def test_checkpoint_roundtrip_and_graph_signature(tmp_path) -> None:
    from urbanflow.rl.workflow import (
        _save_checkpoint, graph_contract_signature, load_checkpoint, RLWorkflowError,
    )
    environment = env_factory()
    network = GraphActorCritic(environment)
    metadata = {
        "graph_contract_signature": graph_contract_signature(environment),
        "action_size": environment.action_space.shape[0],
        "observation_size": environment.observation_space.shape[0],
        "constraints": {
            "maximum_relocation_steps": 1,
            "maximum_relocation_fraction": 0.2,
            "minimum_zone_reserve": 0,
            "demand_buffer_multiplier": 1.0,
            "in_transit_supply_credit": 0.25,
        },
    }
    checkpoint = tmp_path / "policy.pt"
    _save_checkpoint(checkpoint, network, metadata=metadata)
    restored, restored_metadata = load_checkpoint(checkpoint, environment)
    assert restored_metadata == metadata
    for key, value in network.state_dict().items():
        assert torch.equal(value, restored.state_dict()[key])

    incompatible = env_factory()
    incompatible.bundle.allowed_edges[0, 1] = False
    with pytest.raises(RLWorkflowError, match="signature"):
        load_checkpoint(checkpoint, incompatible)
