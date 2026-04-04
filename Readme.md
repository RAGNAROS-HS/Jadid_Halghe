# Jadid Halghe

**Teach AI agents to dominate an agar.io clone through reinforcement learning.**

Built in Python + PyTorch with a hard focus on simulation speed: every design decision optimizes for maximum training throughput while preserving faithful agar.io gameplay. Human play is supported alongside trained agents.

---

## Project Phases

### Phase 1 — Game Engine (headless, vectorized) ✅

Goal: a fully correct, blazing-fast agar.io simulation that can run thousands of steps per second without a renderer.

**1.1 — World & entity model**
- [x] Define world bounds (walled, configurable size)
- [x] `Cell` — player-controlled blob: position, radius, mass, velocity, owner ID
- [x] `Food` — static pellets spawned randomly across the map
- [x] `Virus` — stationary hazard that splits large cells on contact
- [x] `EjectedMass` — projectile fired by player; becomes food after settling
- [x] All entities stored as contiguous NumPy arrays (no Python objects in the hot path)
- [x] Pre-allocated buffers for entity slots; dead slots recycled via free-list

**1.2 — Physics & movement**
- [x] Continuous movement: velocity vector per cell, speed inversely proportional to mass (`base / mass^0.439`)
- [x] Direction input → instant velocity update (vectorized across all cells of a player)
- [x] Split momentum (`split_vel`) decays each tick via `split_decay` multiplier
- [x] Cell merging: split cells belonging to the same player merge after a configurable tick threshold
- [x] Mass decay: cells above `mass_decay_threshold` lose a fraction of mass each tick

**1.3 — Splitting & ejection**
- [x] Split action: cell divides into two, minimum mass enforced, max-split-count per player
- [x] Ejection action: fires a small mass pellet in movement direction
- [x] Both actions handled as batch operations (all eligible cells processed in one call)

**1.4 — Collision & eating**
- [x] Vectorized O(n²) NumPy broadcasting over alive-cell subsets
- [x] Eating rule: `mass_A > 1.21 × mass_B` and center of B inside radius of A
- [x] Virus interaction: large cell (`mass > virus_mass`) hits virus → split into fragments
- [x] Player elimination when last cell is eaten
- [x] Ejected mass absorbed by enemy cells immediately, own cells after settle timer

**1.5 — Food & spawning**
- [x] Food pellets maintain target density; batch-spawn each tick when count drops below threshold
- [x] Virus spawning: background maintenance + split triggered by ejected-mass hits
- [x] Player spawn at random position within inner 80% of world

**1.6 — Game loop & tick API**
- [x] Fixed-timestep tick: `world.step(actions) → (rewards, dones, info)`; `world.get_state()` on demand
- [x] `GameState` is a plain dataclass of arrays (zero-copy friendly)
- [x] Deterministic seeding: `world.reset(seed=N)` reproduces identical runs
- [x] Throughput benchmark script (`benchmark.py`) — see numbers below

**1.7 — Tests & benchmarks**
- [x] Unit tests for each mechanic (eating rules, split physics, merge timer, boundary clamping)
- [x] Mass-conservation tests, no-NaN checks, determinism test
- [ ] Property-based tests with Hypothesis
- [ ] `pytest-benchmark` baseline committed to repo

---

### Phase 2 — Human-Playable UI ✅

Goal: render the simulation with Pygame so a human can play against (or alongside) agents.

**2.1 — Renderer**
- [x] Pygame window; configurable resolution
- [x] Draw food pellets, viruses, cells (color-coded by owner)
- [x] Cell labels (player name / mass)
- [x] Smooth camera follow: viewport centered on player's mass centroid
- [x] Zoom scales with player size

**2.2 — Input handling**
- [x] Mouse position → movement direction (relative to viewport center)
- [x] `Space` → split, `W` → eject mass
- [x] Pause / speed multiplier keys for debugging

**2.3 — HUD**
- [x] Live leaderboard (top N players by mass)
- [x] Minimap showing all cells and food
- [x] FPS and tick-rate counter

**2.4 — Mixed human + agent sessions**
- [x] Human player occupies one agent slot; RL agents fill the rest
- [x] UI runs at display FPS; game ticks decoupled (render every K ticks)
- [x] `main.py --agents <N> --seed <N> --width <N> --height <N>` CLI entry point

---

### Phase 3 — RL Environment Wrapper ✅

Goal: wrap the game engine in a Gymnasium-compatible interface ready for standard RL libraries.

**3.1 — Observation space**
- [x] Ego-centric local view: K-nearest-neighbor list centered on agent centroid
- [x] Channels: own cells, food, viruses, **threat enemies** (larger than self), **prey enemies** (smaller than self) — relative position + mass features
- [x] Enemy mass encoded as `delta_log_mass = log(enemy_mass / own_total_mass + 1e-6) / 5`; positive = threat, negative = prey — no subtraction needed by the network
- [x] Global features: own total log-mass, cell-count fraction
- [x] Observation normalized and clipped to `[-10, 10]`; dtype `float32`; total dim = 170
- [ ] Configurable K per channel (currently fixed: own=16, food=20, virus=10, threat=10, prey=10)

**3.2 — Action space**
- [x] Continuous: `Box([-1,-1,-1,-1], [1,1,1,1])` — `(dx, dy)` direction + split/eject logits (thresholded at 0)
- [ ] Discrete variant (for DQN baselines): 8 movement directions × {none, split, eject}

**3.3 — Reward function**
- [x] Primary: `Δ(own_mass) / start_mass` per tick
- [x] Death penalty: `-player_mass / reward_scale` on elimination — proportional to current mass so dying is always a net loss regardless of size
- [x] Survival bonus: `+survival_bonus` per tick alive (default `0.01 = food_mass / start_mass`); configurable via `env.survival_bonus` in YAML
- [x] No NaN/Inf guaranteed (tested)

**3.4 — Multi-agent interface**
- [x] `PettingZoo`-compatible `AgarParallelEnv` (all agents step simultaneously; dead agents removed from `self.agents`)
- [x] Gymnasium `AgarEnv` single-agent wrapper (agent 0 is RL; remaining slots are random-direction bots with auto-respawn)
- [x] `env.reset(seed=...)` resets world and re-seeds RNG

**3.5 — Vectorized / batched environments**
- [x] `VecAgarEnv`: N independent worlds stepped synchronously; auto-reset on episode end; terminal obs in `info["final_observation"]`
- [ ] Shared-memory / multiprocessing transport (sequential is sufficient for current throughput target)
- [ ] ≥ 1M agent-steps/sec (current: ~10 k steps/sec with 3 envs; Numba path available if needed)

---

### Phase 4 — Agent Architecture ✅

Goal: neural network policies that can efficiently process the agar.io observation.

**4.1 — Baseline MLP policy**
- [x] 3-layer MLP (256→128) over flat 170-dim observation
- [x] Separate actor head (mean) + shared `log_std` parameter + critic head

**4.2 — Attention-based policy (primary)**
- [x] Per-group projections to `embed_dim=64` + learned type embeddings (5 types: own, food, virus, threat, prey)
- [x] Pre-LN TransformerEncoder (2 layers, 4 heads) over 66 entity tokens
- [x] Zero-pad masking (real vs. padded slots detected via feature norm); safe mean pool
- [x] Actor + critic heads on (pooled entities ‖ scalars)

**4.3 — Recurrent option**
- [x] `RecurrentPolicy`: GRU-wrapped MLP; `initial_state()` → hidden, passed through `act()`
- [x] Documented that the provided `Runner` is non-recurrent; custom loop required

**4.4 — Model utilities**
- [x] Orthogonal init on all Linear layers (gain √2; output heads: 0.01 / 1.0)
- [x] Gradient clipping in PPO (`max_grad_norm`)
- [x] `policy.save(path, step=N)` / `Policy.load(path)` with type + config + optimizer state
- [ ] ONNX export (deferred)

---

### Phase 5 — Training Infrastructure ✅

Goal: stable, reproducible RL training with clean logging and checkpointing.

**5.1 — PPO implementation**
- [x] Clipped surrogate objective (`L_clip`), entropy bonus, MSE value loss
- [x] GAE (λ-weighted) in `RolloutBuffer.compute_returns_and_advantages()`
- [x] Mini-batch updates; `n_epochs` passes per rollout; advantage normalisation per rollout
- [x] All hyperparameters in `configs/default.yaml`

**5.2 — Rollout collection**
- [x] `Runner`: stateful synchronous collect-then-learn; `tanh(z)` sent to env, `z` stored in buffer
- [x] `RolloutBuffer`: pre-allocated T×N tensors; no per-step allocation
- [ ] Mixed-precision (deferred)

**5.3 — Logging & metrics**
- [x] TensorBoard via `torch.utils.tensorboard.SummaryWriter`
- [x] Logs per rollout: `train/reward`, `train/policy_loss`, `train/value_loss`, `train/entropy`, `train/approx_kl`, `train/clip_fraction`
- [ ] W&B integration, action histograms (deferred)

**5.4 — Checkpointing & resuming**
- [x] Checkpoint every `save_interval` rollouts: policy weights + optimizer state + step count
- [x] `train.py --resume <path>` restores policy and optimizer state seamlessly
- [ ] Best-model tracking by eval win rate (deferred to Phase 6)

**5.5 — Curriculum & self-play**
- [ ] Self-play pool with ELO tracker (deferred)

---

### Phase 6 — Evaluation & Analysis ✅

Goal: understand what the agent learned and how well it generalizes.

**6.1 — Evaluation harness**
- [x] Run N evaluation episodes with agent in `.eval()` mode; log aggregate stats
- [x] Opponents: random bots, greedy bots, past checkpoints
- [x] `eval.py --checkpoint <path> --opponents <type> --episodes <N>`

**6.2 — Visualization**
- [x] Replay system: save episode trajectories to disk; replay via Pygame UI
- [x] Attention weight overlay: `AttentionPolicy.attention_maps(obs)` returns per-head weights
- [x] Mass-over-time plot per episode (matplotlib)

**6.3 — Ablations & baselines**
- [x] `RandomPolicy` baseline (uniform random actions)
- [x] `GreedyPolicy` heuristic (chase edible enemies → food → flee threats)
- [ ] DQN baseline (using discrete action space)
- [ ] Comparison table in docs

---

## Architecture

```
Jadid_Halghe/
  game/              # Engine — Phase 1 ✅
    config.py        #   WorldConfig frozen dataclass (all sim constants)
    entities.py      #   CellArrays, FoodArrays, VirusArrays, EjectedArrays
    physics.py       #   update_cells(), update_ejected()
    collision.py     #   resolve_*() eating / merging functions
    spawner.py       #   spawn_food/viruses, add_player, handle_split/eject
    world.py         #   World class + GameState; step() / reset() / get_state()
  rl/                # RL layer — Phase 3–5 ✅
    env.py           #   AgarEnv (Gymnasium) — single-agent; random-bot opponents
    multi_env.py     #   AgarParallelEnv (PettingZoo) — all-agent RL
    vec_env.py       #   VecAgarEnv — synchronous N-env wrapper, auto-reset
    ma_vec_env.py    #   VecAgarMAEnv — N worlds × M RL agents, shared policy
    agent.py         #   MLPPolicy, AttentionPolicy, RecurrentPolicy; build/load helpers
    buffer.py        #   RolloutBuffer — pre-allocated, GAE, minibatch iterator
    ppo.py           #   PPO — clipped surrogate + value + entropy
    runner.py        #   Runner — stateful rollout collector
    video.py         #   render_episode_to_video(), record_video() — headless GIF/MP4
  ui/                # Pygame renderer — Phase 2 ✅
    renderer.py      #   draw food/viruses/cells/ejected with culling
    camera.py        #   viewport follow, zoom, world↔screen transforms
    hud.py           #   leaderboard, FPS counter, minimap
    input.py         #   mouse direction, Space/W/P keys
  eval/              # Evaluation & replay — Phase 6 ✅
    harness.py       #   Harness — run N eval episodes, compute EvalResult
    replay.py        #   ReplayEpisode, save/load, Pygame playback, mass plot
    baselines.py     #   RandomPolicy, GreedyPolicy
  tests/
    game/
      test_mechanics.py   # 34 tests — all passing
    rl/
      test_env.py         # 23 tests — all passing
      test_agent.py       # 24 tests — all passing
      test_training.py    # 12 tests — all passing
    eval/
      test_eval.py        # 34 tests — all passing
  configs/
    default.yaml     #   Default PPO hyperparameters (attention policy, 3 envs, 10M steps)
  train.py           # PPO training entry point ✅ (`python train.py --config configs/default.yaml`)
  eval.py            # Eval entry point ✅ (`python eval.py --checkpoint <path>`)
  main.py            # Human-play entry point ✅ (`python main.py --agents N`)
```

---

## Tech Stack

| Concern | Choice |
|---|---|
| Language | Python 3.12+ |
| Array ops | NumPy (game engine) |
| RL / neural nets | PyTorch 2.x |
| RL env interface | Gymnasium + PettingZoo |
| Renderer | Pygame-CE |
| Logging | TensorBoard + optional W&B |
| Testing | pytest + Hypothesis + pytest-benchmark |
| Lint/format | Ruff (line length 88) |

---

## Simulation Constants

Key values baked into `WorldConfig` defaults (all tunable):

| Constant | Value | Notes |
|---|---|---|
| World size | 14 142 × 14 142 | Matches real agar.io map |
| Start mass | 2 500 | → radius = √2500 = 50 units |
| Food mass | 25 | → radius = 5 units |
| Virus mass | 2 500 | → radius = 50 units |
| Speed formula | `20 000 / mass^0.439` | units/sec; × dt(1/25) per tick |
| Min split mass | 2 500 | = start mass; can split immediately |
| Merge time | `100 + mass/50` ticks | ≈ 4 s base at 25 TPS |
| Mass decay | disabled | `mass_decay_rate = 0`; enable via `world: {mass_decay_rate: 0.002}` in YAML |
| Survival bonus | 0.01 / tick | `= food_mass / start_mass`; configurable via `env.survival_bonus` |
| eat_ratio | 1.1 | Need 1.21× mass to eat |

---

## Throughput

`python benchmark.py` produces (measured on a modern laptop):

| Scenario | TPS | Worlds for 10k |
|---|---|---|
| 2 players / 500 food | ~4 300 | 3 |
| 4 players / 1 000 food | ~4 000 | 3 |
| 8 players / 2 000 food | ~3 500 | 3 |

**Single-world limit:** ~25 NumPy calls per tick × ~3 µs Python dispatch = ~75 µs floor. Pure-NumPy single-world throughput caps at ~3–4k TPS regardless of further NumPy micro-optimisations.

**Path to 10k TPS:**
- **VecEnv (Phase 3.5):** 3 parallel worlds × 3 500 TPS = ~10 500 agent-steps/sec. This is the planned solution.
- **Numba (optional):** `@njit` on the physics/collision hot path would yield 20–50k single-world TPS. Can be added as an optional dependency later.

---

## Quick Start

```bash
pip install -r requirements.txt

# Run all tests
pytest

# Throughput benchmark
python benchmark.py

# Lint + format check
ruff check . && ruff format .

# Human play
python main.py                    # 4 random bots + human
python main.py --agents 8         # 8 random bots + human
python main.py --agents 0         # solo
python main.py --no-human         # spectate bots only

# Play against a trained agent
python main.py --checkpoint checkpoints/run_default/ckpt_000100.pt
python main.py --checkpoint ckpt.pt --agents 7
python main.py --checkpoint ckpt.pt --no-human   # spectate trained agents

# Train — all 8 agents learn simultaneously (multi-agent, default config)
python train.py --config configs/default.yaml

# Resume from checkpoint
python train.py --config configs/default.yaml --resume checkpoints/run_default/ckpt_000100.pt

# Train on GPU
python train.py --config configs/default.yaml --device cuda

# Evaluate a checkpoint
python eval.py --checkpoint checkpoints/run_default/ckpt_000100.pt --episodes 20
python eval.py --checkpoint ckpt.pt --opponents greedy --save-replay replays/ep.pkl --plot

# Replay a saved episode
python eval.py --replay replays/ep.pkl
```
