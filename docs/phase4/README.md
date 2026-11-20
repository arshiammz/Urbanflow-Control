# Phase 4 — Constrained Deep Reinforcement Learning (Gymnasium + PyTorch PPO)

Phase 4 extends the exact Phase 3 simulation/forecast/network contract. The
Phase 3 four-policy optimization benchmark is left intact. This phase adds
**a fifth, train-only learned policy**, then evaluates it on **the identical
May validation day** used by Phase 3. It does not claim that RL is better than
Min-Cost Flow or MPC unless the resulting report supports that conclusion.

**State:** the code and synthetic tests are implemented. Large-data NYC PPO
training and May comparison are **not yet executed**. No trained neural model
or measured performance is bundled in the source ZIP.

## 1. What is trained, what is held out?

| Stage | Date/source | Information boundary |
|---|---|---|
| Offline model selection | Phase 1/2 train/April | Inherited from prior phase |
| PPO training | 2024-02-01 through 2024-04-30 (train only) | Episodes use only historical train demand and a causal lag-1 seasonal **training proxy** |
| PPO seed selection | Last 5 training rollout rewards | Never uses May or June metrics |
| PPO evaluation | 2024-05-15 (May validation) | Same precomputed, decision-time-safe Phase 2 P10/P50/P90 tensor as Phase 3 |
| Phase 3 comparison | Same May day/warmup/fleet/OD/costs/graph | No retraining or PPO selection on May |
| Final development test | June | Not touched by Phase 4 scripts |
| Final untouched holdout | July | Reserved for Phase 5 |

### Important distribution shift

Phase 2 quantile forecasts are only available on the May/June prediction splits.
There are no honest historical out-of-fold quantile forecasts available for
January–April in the supplied Phase 3 snapshot. Consequently training uses a
**training-only causal lag-1 forecast** with a train-fitted weekly seasonal
profile fitted on **training days before February 1**, and an **uncalibrated
+/-20% uncertainty augmentation**, all clearly
marked in the artifact metadata. It is **not** claimed to be calibrated P10/P90.
The May inference path switches to the actual Phase 2 archived P10/P50/P90.
This distribution shift is a limitation and must be reported. A later upgrade
can use rolling-origin out-of-fold training forecasts before changing the
training regimen; never use May realized labels to fake training forecasts.

## 2. The environment

`src/urbanflow/rl/environment.py` exposes `FleetRebalancingEnv(gym.Env)`.

- One training episode is a service date, with the preceding date as a
  **shared no-rebalancing warmup**.
- Observation: `N x 10` zone features plus 6 global time/fleet features.
  Includes available and in-transit vehicles, P50/P90 forecasts, uncertainty,
  previous observed demand, historical regional service rate and graph degree.
  **The current realized demand is not in the policy observation.**
- Continuous actor proposal: `E` directed legal-edge scores plus one budget
  fraction (squashed Gaussian in `[-1,1]`).
- Hard projection maps proposals to integer, graph-feasible OD relocations.
- The simulator advances using **actual realized historical demand**, releases
  scheduled arrivals at the correct step and checks fleet conservation.
- The simulator itself is not modified or bypassed by the neural policy.

The environment is deterministic conditional on its episode, action and
initial fleet; randomness in PPO comes from policy sampling and episode
selection.

## 3. Hard constraints and fairness

Hard invariants in `GraphActionProjector`:

- no relocations outside the fitted Phase 2 graph;
- no relocation exceeding configured max travel steps;
- no moving more vehicles from a zone than those available;
- no exceeding the global relocation cap;
- no self relocation, negatives, or fractional vehicle counts;
- per-zone donor reserve based on *as-of* current forecast;
- target deficits based on *as-of* multi-step forecasts and credited in-transit vehicles.

Time-zero scheduled arrivals are included in decision-time availability,
identically to the Phase 3 benchmark.

**Fairness is a soft constrained objective**, not an impossible promise that
all boroughs meet a minimum service rate under every fleet/demand situation.
The training reward uses service rate, normalized operational cost, idle fleet
and a learned nonnegative penalty for violation of the target regional
service-rate gap. The primal-dual coefficient rises with observed violations.
Region service-rate history contains only realized data up to the current
step, and the forecast model does not receive future realized demand.

## 4. Policy architecture and PPO

The actor has a shared zone encoder and an edge scorer using encoded origin,
encoded destination and graph travel time. A separate budget head controls
relocation intensity. The value network uses pooled zone embeddings and
global features. This architecture is more tractable for hundreds of zones
than a fully-connected `N x N` dense action head.

The implementation uses:

- tanh-squashed Gaussian actor with log-probability Jacobian correction;
- clipped PPO policy objective;
- generalized advantage estimation (GAE);
- value loss and latent Gaussian entropy regularization;
- per-epoch minibatches, gradient clipping and Adam;
- 3 independently seeded training runs;
- training-only seed selection and a frozen selected checkpoint;
- CPU-first reproducible execution (MPS can be added after determinism checks).

## 5. Install, train and evaluate

```bash
# From Urbanflow-Control repository root
python -m pip install -e '.[dev,rl]'

# Fast unit tests (no TLC Parquet data needed)
pytest -q tests/unit/rl

ruff check src/urbanflow/rl scripts/train_constrained_ppo.py \
  scripts/evaluate_constrained_ppo.py tests/unit/rl

# These build artifacts must exist from Phase 1–3:
# - all-zone historical_demand + zone_index + od_profile Parquets
# - full zone graph and TLC zone lookup
# - Phase 2 decision-time P10/P50/P90 horizon Parquet

# Trains THREE seeds on Feb–Apr TRAIN episodes, not May:
python scripts/train_constrained_ppo.py \
  --config configs/rl/constrained_ppo.yaml

# The exact Phase 3 four-policy baseline must be available first:
python scripts/run_optimization_benchmark.py \
  --config configs/simulation/optimization_benchmark.yaml

# Evaluate selected RL checkpoint on the exact same May validation date:
python scripts/evaluate_constrained_ppo.py \
  --config configs/rl/constrained_ppo.yaml
```

Both scripts error out if their required data, graph, benchmark report or
checkpoint is missing. They never invent a result.

## 6. Produced artifacts

| Path | Description |
|---|---|
| `artifacts/models/constrained_ppo/ppo_seed_{seed}.pt` | Frozen PyTorch state + graph-contract metadata per seed |
| `artifacts/models/constrained_ppo/champion_train_selected.pt` | Champion selected using training rollouts only |
| `reports/rl/constrained_ppo_training.json` | All 3 seed trajectories + training-only selection |
| `reports/rl/constrained_ppo_vs_phase3_may_validation.json` | RL metrics, comparison against 4 Phase 3 policies, invariants |
| `reports/rl/constrained_ppo_may_step_metrics.parquet` | Every real evaluation transition |
| `reports/rl/constrained_ppo_may_zone_metrics.parquet` | Zone service, unmet demand and fairness |
| `reports/rl/constrained_ppo_may_region_metrics.parquet` | Borough-level service/fairness |
| `reports/rl/constrained_ppo_may_actions.parquet` | Nonzero OD relocation decisions for audit and reproducibility |

Large binary checkpoints/Parquet files remain local artifacts and are ignored
by `.gitignore`. Commit the source, configs, tests and documentation.

## 7. What does comparison mean?

The evaluation report includes No Rebalancing, Greedy, Min-Cost Flow, MPC and
Constrained PPO under the same actual May demand, initial fleet, graph,
OD/travel times, warmup and operational cost coefficients.

`strongest_phase3_optimizer` is the lower-MAY-validation-operating-cost of
Min-Cost Flow and MPC, not a claim of test-set model selection. The report
records RL deltas against **each** Phase 3 reference as well as this strongest
optimizer. Lower mean cost by itself does not prove superior operational
reliability: compare worst-region service, relocations and decision latency.

The completion of this phase means a **runnable framework** for a trained RL
policy and fair comparison exists. Training and reported performance remain
unverified until the full local commands above have executed successfully.
