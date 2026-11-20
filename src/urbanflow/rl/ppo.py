"""From-scratch PyTorch PPO with bounded continuous edge proposals.

PPO acts on tanh-squashed Gaussian edge/budget proposals. The environment
projects every proposal into a graph-feasible integer relocation matrix.
Fairness uses a nonnegative primal-dual penalty, whereas supply/graph/time
constraints are enforced hard by projection and by FleetSimulator.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal

from urbanflow.rl.environment import FleetRebalancingEnv


@dataclass(frozen=True)
class PPOSettings:
    updates: int = 24
    rollout_steps: int = 512
    epochs: int = 4
    minibatch_size: int = 128
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.20
    entropy_coefficient: float = 0.001
    value_coefficient: float = 0.5
    max_gradient_norm: float = 0.5
    dual_learning_rate: float = 0.10
    dual_maximum: float = 5.0

    def __post_init__(self) -> None:
        if any(v <= 0 for v in (self.updates, self.rollout_steps, self.epochs,
                                  self.minibatch_size)):
            raise ValueError("PPO iteration counts must be positive.")
        for name, value in vars(self).items():
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(f"PPO {name} must be finite.")
        if not (0 < self.learning_rate and 0 < self.gamma <= 1
                and 0 < self.gae_lambda <= 1 and 0 < self.clip_epsilon < 1):
            raise ValueError("Invalid PPO learning rate/discount/clip settings.")
        if min(self.entropy_coefficient, self.value_coefficient,
               self.max_gradient_norm, self.dual_learning_rate,
               self.dual_maximum) < 0:
            raise ValueError("PPO regularization/dual controls must be nonnegative.")


class GraphActorCritic(nn.Module):
    """Parameter-shared zone/edge encoder with zone-order-specific graph buffers."""

    def __init__(self, env: FleetRebalancingEnv) -> None:
        super().__init__()
        self.zone_count = env.zone_count
        self.features_per_zone = env.features_per_zone
        self.global_features = env.global_features
        p = env.projector
        self.action_size = p.action_size
        self.register_buffer("origin", torch.as_tensor(p.origins, dtype=torch.long))
        self.register_buffer("destination", torch.as_tensor(p.destinations, dtype=torch.long))
        distances = p.travel_steps[p.origins, p.destinations].astype(np.float32)
        distances /= max(1, p.constraints.maximum_relocation_steps)
        self.register_buffer("distance", torch.from_numpy(distances))
        self.encoder = nn.Sequential(
            nn.Linear(self.features_per_zone, 64), nn.Tanh(),
            nn.Linear(64, 32), nn.Tanh(),
        )
        self.edge_actor = nn.Sequential(
            nn.Linear(65, 64), nn.Tanh(), nn.Linear(64, 1),
        )
        self.budget_actor = nn.Sequential(
            nn.Linear(32 + self.global_features, 32), nn.Tanh(),
            nn.Linear(32, 1),
        )
        self.critic = nn.Sequential(
            nn.Linear(32 + self.global_features, 64), nn.Tanh(),
            nn.Linear(64, 1),
        )
        self.edge_log_std = nn.Parameter(torch.tensor(-0.4))
        self.budget_log_std = nn.Parameter(torch.tensor(-0.6))

    def forward(self, observation: torch.Tensor) -> tuple[Normal, torch.Tensor]:
        if observation.ndim != 2 or observation.shape[1] != (
            self.zone_count * self.features_per_zone + self.global_features
        ):
            raise ValueError("ActorCritic received an observation of wrong shape.")
        n = self.zone_count
        zones = observation[:, :n * self.features_per_zone].reshape(
            -1, n, self.features_per_zone
        )
        global_obs = observation[:, n * self.features_per_zone:]
        embedded = self.encoder(zones)
        pooled = embedded.mean(dim=1)
        context = torch.cat((pooled, global_obs), dim=-1)
        if len(self.origin):
            origin = embedded[:, self.origin, :]
            destination = embedded[:, self.destination, :]
            distance = self.distance.view(1, -1, 1).expand(len(observation), -1, -1)
            edge_mean = self.edge_actor(torch.cat((origin, destination, distance), dim=-1))
            edge_mean = edge_mean.squeeze(-1)
        else:
            edge_mean = observation.new_zeros((len(observation), 0))
        budget_mean = self.budget_actor(context)
        latent_mean = torch.cat((edge_mean, budget_mean), dim=-1)
        latent_log_std = torch.cat((
            self.edge_log_std.clamp(-4, 1).expand(len(self.origin)),
            self.budget_log_std.clamp(-4, 1).view(1),
        ))
        distribution = Normal(latent_mean, latent_log_std.exp())
        value = self.critic(context).squeeze(-1)
        return distribution, value

    @staticmethod
    def _log_probability(distribution: Normal, latent: torch.Tensor) -> torch.Tensor:
        action = latent.tanh()
        return (distribution.log_prob(latent)
                - torch.log((1.0 - action.square()).clamp_min(1e-6))).sum(dim=-1)

    def act(
        self, observation: torch.Tensor, *, deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        distribution, value = self.forward(observation)
        latent = distribution.mean if deterministic else distribution.rsample()
        return latent.tanh(), self._log_probability(distribution, latent), value

    def evaluate_actions(
        self, observation: torch.Tensor, actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        distribution, value = self.forward(observation)
        clipped = actions.clamp(-0.999999, 0.999999)
        latent = torch.atanh(clipped)
        log_prob = self._log_probability(distribution, latent)
        # Entropy of underlying Normal is a differentiable exploration proxy.
        entropy_proxy = distribution.entropy().sum(dim=-1)
        return log_prob, entropy_proxy, value


@dataclass
class Rollout:
    observations: np.ndarray
    actions: np.ndarray
    old_log_probs: np.ndarray
    advantages: np.ndarray
    returns: np.ndarray
    rewards: np.ndarray
    fairness_violations: np.ndarray
    fairness_constraint_signals: np.ndarray


def gather_rollout(
    model: GraphActorCritic, env: FleetRebalancingEnv,
    observation: np.ndarray, settings: PPOSettings,
) -> tuple[Rollout, np.ndarray]:
    obs_rows: list[np.ndarray] = []
    act_rows: list[np.ndarray] = []
    logprob_rows: list[float] = []
    value_rows: list[float] = []
    reward_rows: list[float] = []
    done_rows: list[bool] = []
    violation_rows: list[float] = []
    constraint_rows: list[float] = []
    model.eval()
    for _ in range(settings.rollout_steps):
        with torch.no_grad():
            action, logprob, value = model.act(torch.as_tensor(observation[None], dtype=torch.float32))
        raw = action[0].cpu().numpy().astype(np.float32)
        next_obs, reward, terminated, truncated, info = env.step(raw)
        obs_rows.append(observation.copy())
        act_rows.append(raw)
        logprob_rows.append(float(logprob.item()))
        value_rows.append(float(value.item()))
        reward_rows.append(float(reward))
        done_rows.append(bool(terminated or truncated))
        violation_rows.append(float(info["fairness_violation"]))
        constraint_rows.append(float(info["fairness_constraint_signal"]))
        if terminated or truncated:
            observation, _ = env.reset()
        else:
            observation = next_obs

    with torch.no_grad():
        bootstrap = float(model.forward(torch.as_tensor(observation[None], dtype=torch.float32))[1].item())
    advantages = np.zeros(settings.rollout_steps, dtype=np.float32)
    last_gae = 0.0
    for t in range(settings.rollout_steps - 1, -1, -1):
        nonterminal = 1.0 - float(done_rows[t])
        next_value = bootstrap if t == settings.rollout_steps - 1 else value_rows[t + 1]
        delta = reward_rows[t] + settings.gamma * next_value * nonterminal - value_rows[t]
        last_gae = delta + settings.gamma * settings.gae_lambda * nonterminal * last_gae
        advantages[t] = last_gae
    values = np.asarray(value_rows, dtype=np.float32)
    rollout = Rollout(
        observations=np.stack(obs_rows), actions=np.stack(act_rows),
        old_log_probs=np.asarray(logprob_rows, dtype=np.float32),
        advantages=advantages,
        returns=advantages + values,
        rewards=np.asarray(reward_rows, dtype=np.float32),
        fairness_violations=np.asarray(violation_rows, dtype=np.float32),
        fairness_constraint_signals=np.asarray(constraint_rows, dtype=np.float32),
    )
    return rollout, observation


def update_ppo(
    model: GraphActorCritic, optimizer: torch.optim.Optimizer,
    rollout: Rollout, settings: PPOSettings,
    *, rng: np.random.Generator,
) -> dict[str, float]:
    """Clipped PPO objective with GAE, value loss and entropy regularization."""
    model.train()
    observation = torch.as_tensor(rollout.observations, dtype=torch.float32)
    actions = torch.as_tensor(rollout.actions, dtype=torch.float32)
    old_log_prob = torch.as_tensor(rollout.old_log_probs, dtype=torch.float32)
    returns = torch.as_tensor(rollout.returns, dtype=torch.float32)
    advantage = torch.as_tensor(rollout.advantages, dtype=torch.float32)
    advantage = (advantage - advantage.mean()) / (advantage.std(unbiased=False) + 1e-8)
    batch_size = len(rollout.rewards)
    aggregate = {"policy_loss": [], "value_loss": [], "entropy": [], "approx_kl": []}
    for _ in range(settings.epochs):
        permutation = rng.permutation(batch_size)
        for start in range(0, batch_size, settings.minibatch_size):
            index = permutation[start:start + settings.minibatch_size]
            if len(index) == 0:
                continue
            new_log_prob, entropy, value = model.evaluate_actions(observation[index], actions[index])
            log_ratio = new_log_prob - old_log_prob[index]
            ratio = log_ratio.exp()
            surrogate = torch.minimum(
                ratio * advantage[index],
                ratio.clamp(1 - settings.clip_epsilon, 1 + settings.clip_epsilon) * advantage[index],
            )
            policy_loss = -surrogate.mean()
            value_loss = 0.5 * (returns[index] - value).square().mean()
            total_loss = (policy_loss + settings.value_coefficient * value_loss
                          - settings.entropy_coefficient * entropy.mean())
            if not torch.isfinite(total_loss):
                raise RuntimeError("Non-finite PPO training loss.")
            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), settings.max_gradient_norm)
            optimizer.step()
            aggregate["policy_loss"].append(float(policy_loss.item()))
            aggregate["value_loss"].append(float(value_loss.item()))
            aggregate["entropy"].append(float(entropy.mean().item()))
            aggregate["approx_kl"].append(float(((ratio - 1) - log_ratio).mean().item()))
    return {key: float(np.mean(values)) for key, values in aggregate.items()}


def train_one_seed(
    env: FleetRebalancingEnv, *, seed: int, settings: PPOSettings,
) -> tuple[GraphActorCritic, list[dict[str, Any]]]:
    """Train on TRAIN episodes only. Never rank seeds with May observations."""
    if env.bundle.source != "train" or any(e.split != "train" for e in env.bundle.episodes):
        raise RuntimeError("PPO training requires train-only replay episodes.")
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    rng = np.random.default_rng(seed)
    model = GraphActorCritic(env)
    optimizer = torch.optim.Adam(model.parameters(), lr=settings.learning_rate, eps=1e-5)
    observation, _ = env.reset(seed=seed)
    progress: list[dict[str, Any]] = []
    for update in range(settings.updates):
        rollout, observation = gather_rollout(model, env, observation, settings)
        stats = update_ppo(model, optimizer, rollout, settings, rng=rng)
        fairness_violation = float(rollout.fairness_violations.mean())
        mean_constraint = float(rollout.fairness_constraint_signals.mean())
        env.fairness_multiplier = float(np.clip(
            env.fairness_multiplier + settings.dual_learning_rate * mean_constraint,
            0.0, settings.dual_maximum,
        ))
        progress.append({
            "update": update + 1, "seed": seed,
            "mean_training_reward": float(rollout.rewards.mean()),
            "mean_training_fairness_violation": fairness_violation,
            "mean_fairness_constraint_signal": mean_constraint,
            "fairness_dual_multiplier": env.fairness_multiplier,
            **stats,
        })
    return model, progress
