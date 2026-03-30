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

### Phase 3 — RL Environment Wrapper

Goal: wrap the game engine in a Gymnasium-compatible interface ready for standard RL libraries.

**3.1 — Observation space**
- [ ] Ego-centric local view: fixed-size grid or K-nearest-neighbor list centered on agent
- [ ] Channels: food, friendly cells, enemy cells, viruses — each with (dx, dy, radius) features
- [ ] Global features: own total mass, number of splits remaining, merge cooldown
- [ ] Observation normalized to `[0, 1]` or `[-1, 1]`; dtype `float32`
- [ ] Configurable view radius and max entity count per channel

**3.2 — Action space**
- [ ] Continuous: `Box([-1,-1, 0, 0], [1, 1, 1, 1])` — (dx, dy, split, eject) where split/eject are Bernoulli thresholded
- [ ] Discrete variant (for DQN baselines): 8 movement directions × {none, split, eject}
- [ ] Both wrappers provided; flag-selectable

**3.3 — Reward function**
- [ ] Primary: `Δ(own_mass)` per tick — growing is rewarded, shrinking penalized
- [ ] Elimination bonus: large reward for eating the last cell of an opponent
- [ ] Death penalty: large negative reward on elimination
- [ ] Survival bonus: small per-tick reward for staying alive (tunable weight)
- [ ] No NaN/Inf guaranteed; reward clipping with assertion checks

**3.4 — Multi-agent interface**
- [ ] `PettingZoo`-compatible `ParallelEnv` wrapper (all agents step simultaneously)
- [ ] Gymnasium single-agent wrapper available (one agent, rest are bots/frozen)
- [ ] `env.reset(seed=...)` resets world and re-seeds all RNG sources

**3.5 — Vectorized / batched environments**
- [ ] `VecEnv` wrapper: N independent worlds stepped in parallel (NumPy, then optional multiprocessing)
- [ ] Shared-memory transport between worker processes (avoid pickle overhead)
- [ ] Step throughput target: ≥ 1 M agent-steps/sec across 16 parallel envs

---

### Phase 4 — Agent Architecture

Goal: neural network policies that can efficiently process the agar.io observation.

**4.1 — Baseline MLP policy**
- [ ] Simple 3-layer MLP over flattened observation vector
- [ ] Separate value head (actor-critic)
- [ ] Serves as a fast-to-train sanity-check baseline

**4.2 — Attention-based policy (primary)**
- [ ] Each nearby entity encoded as a feature vector (type embedding + position + radius)
- [ ] Transformer encoder over entity set (permutation-invariant)
- [ ] Pooled representation fed to actor and critic heads
- [ ] Handles variable-length entity lists without padding waste

**4.3 — Recurrent option**
- [ ] Optional GRU wrapper around the policy for partial observability experiments
- [ ] Hidden state managed per-agent across episodes

**4.4 — Model utilities**
- [ ] Orthogonal weight initialization
- [ ] Gradient clipping configurable per module
- [ ] `model.save(path)` / `model.load(path)` with metadata (hyperparams, step count)
- [ ] ONNX export for potential browser/non-Python deployment

---

### Phase 5 — Training Infrastructure

Goal: stable, reproducible RL training with clean logging and checkpointing.

**5.1 — PPO implementation**
- [ ] Clipped surrogate objective, entropy bonus, value loss coefficient
- [ ] GAE (Generalized Advantage Estimation) with configurable λ
- [ ] Mini-batch updates over rollout buffer; configurable epochs per rollout
- [ ] All hyperparameters in a single config file (YAML / dataclass)

**5.2 — Rollout collection**
- [ ] Parallel rollout workers feeding a central learner (or synchronous collect-then-learn)
- [ ] Rollout buffer pre-allocated; no per-step allocation
- [ ] Mixed-precision (fp16 for inference, fp32 for gradient accumulation) optional

**5.3 — Logging & metrics**
- [ ] Weights & Biases integration (optional, off by default)
- [ ] TensorBoard fallback always available
- [ ] Key metrics logged every N steps: `train/reward`, `train/loss_policy`, `train/loss_value`, `train/entropy`, `train/kl`, `eval/mean_mass`, `eval/survival_time`, `eval/win_rate`
- [ ] Histogram of action distributions logged periodically

**5.4 — Checkpointing & resuming**
- [ ] Save checkpoint every K steps: model weights, optimizer state, step count, config
- [ ] `train.py --resume <checkpoint>` loads and continues seamlessly
- [ ] Best-model tracking by eval win rate

**5.5 — Curriculum & self-play**
- [ ] Start with fewer/weaker opponents; ramp up difficulty as agent improves
- [ ] Self-play pool: maintain a frozen snapshot of past policies; sample opponents from pool
- [ ] ELO tracker for pool members

---

### Phase 6 — Evaluation & Analysis

Goal: understand what the agent learned and how well it generalizes.

**6.1 — Evaluation harness**
- [ ] Run N evaluation episodes with agent in `.eval()` mode; log aggregate stats
- [ ] Opponents: random bots, greedy bots, past checkpoints, human (via UI)
- [ ] `eval.py --checkpoint <path> --opponents <type> --episodes <N>`

**6.2 — Visualization**
- [ ] Replay system: save episode trajectories to disk; replay via UI
- [ ] Attention weight overlay: highlight which entities the agent is "watching"
- [ ] Mass-over-time plot per episode

**6.3 — Ablations & baselines**
- [ ] Random policy baseline
- [ ] Greedy heuristic baseline (always move toward nearest smaller cell)
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
  rl/                # RL layer — Phase 3–5 (not yet implemented)
    env.py           #   Gymnasium single-agent wrapper
    multi_env.py     #   PettingZoo parallel wrapper
    vec_env.py       #   Vectorized env
    agent.py         #   Neural network policies
    ppo.py           #   PPO algorithm
    buffer.py        #   Rollout buffer
    runner.py        #   Rollout collection
  ui/                # Pygame renderer — Phase 2 ✅
    renderer.py      #   draw food/viruses/cells/ejected with culling
    camera.py        #   viewport follow, zoom, world↔screen transforms
    hud.py           #   leaderboard, FPS counter, minimap
    input.py         #   mouse direction, Space/W/P keys
  eval/              # Evaluation & replay — Phase 6 (not yet implemented)
    harness.py
    replay.py
    baselines.py
  tests/
    game/
      test_mechanics.py   # 34 tests — all passing
  configs/           # YAML training configs (not yet implemented)
  train.py           # Training entry point (not yet implemented)
  eval.py            # Eval entry point (not yet implemented)
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
| Mass decay | disabled | `mass_decay_rate = 0` |
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

# Human play (Phase 2 complete)
python main.py                    # 4 bots + human
python main.py --agents 8         # 8 bots + human
python main.py --agents 0         # solo
python main.py --no-human         # spectate bots only

# (Coming in Phase 5) Train from scratch
python train.py --config configs/default.yaml
```
