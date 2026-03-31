# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [Unreleased]

---

## [0.6.0] — Phase 6: Evaluation & Analysis

### Added
- `eval/baselines.py`: `RandomPolicy` — seeded uniform-random baseline; `GreedyPolicy` — heuristic policy that chases edible enemies, moves to nearest food, and flees nearby threats.  Neither emits split or eject actions.
- `eval/harness.py`: `Harness` — runs N evaluation episodes using `World` directly (not `AgarEnv`).  Supports trained policies (`MLPPolicy` / `AttentionPolicy`), baseline policies, or checkpoint paths as both eval agent and opponent.  Returns `EvalResult` (aggregate stats) + optional per-tick `GameState` replay frames.  `EpisodeResult` stores per-episode reward, length, final mass, survival flag, and rank.
- `eval/replay.py`: `ReplayEpisode` dataclass (config + frames + metadata); `save_replay()` / `load_replay()` via pickle; `replay_with_ui()` — Pygame playback with arrow-key scrubbing; `plot_mass_over_time()` — matplotlib mass curves over episodes.
- `eval/__init__.py`: package-level re-exports for all public eval symbols.
- `eval.py`: CLI entry point.  Eval mode: `--checkpoint`, `--opponents`, `--episodes`, `--n-bots`, `--max-ticks`, `--seed`, `--device`, `--save-replay`, `--plot`, `--output`.  Replay mode: `--replay`, `--replay-fps`, `--replay-speed`.
- `rl/agent.py`: `AttentionPolicy.attention_maps(obs)` — manually steps through Pre-LN transformer layers with `need_weights=True` to return per-head attention weights `(n_layers × (1, n_heads, N, N))` without modifying the existing forward pass.
- `tests/eval/test_eval.py`: 34 tests covering `RandomPolicy`, `GreedyPolicy`, `Harness` (shapes, no-NaN, determinism, trained policies), replay save/load roundtrip, `plot_mass_over_time`, and `attention_maps` (shape, softmax sum, no-NaN).
- `requirements.txt`: added `matplotlib>=3.8` for mass-over-time plots.

---

## [0.5.0] — Phase 4+5: Agent Architecture & Training Infrastructure

### Added
- `rl/agent.py`: `MLPPolicy` — 3-layer MLP actor-critic (256→128 hidden, Tanh activations). Shared `log_std` parameter; orthogonal init with output-head gains 0.01 (actor) and 1.0 (critic).
- `rl/agent.py`: `AttentionPolicy` — entity-encoder actor-critic. Per-group linear projections to `embed_dim=64` + learned type embeddings for 4 entity types. Pre-LN transformer (2 layers, 4 heads) over all 66 entity tokens; zero-pad masking via feature norm; safe mean pool; scalars appended post-pooling.
- `rl/agent.py`: `RecurrentPolicy` — GRU-wrapped MLP (hidden_dim=256, gru_hidden=128). Explicit hidden state threading via `initial_state()` / `act(obs, hidden_state)`. Documented as requiring a custom training loop (the provided `Runner` is non-recurrent).
- `rl/agent.py`: `build_policy(type, **kwargs)` factory and `load_policy(path)` dispatcher. All three policies share a consistent `save(path, step)` / `ClassName.load(path)` checkpoint interface storing type, config, and step count.
- `rl/agent.py`: Numerically stable tanh-squashed log-probability correction: `log(1−tanh²(z)) = 2·(log2 − z − softplus(−2z))`. Buffer stores pre-tanh `z`; no `atanh` inversion needed during PPO updates.
- `rl/buffer.py`: `RolloutBuffer` — pre-allocated T×N tensors on a specified device. Stores `(obs, z_actions, log_probs, rewards, values, dones)`. `compute_returns_and_advantages()` runs GAE backward pass. `get_batches()` shuffles and normalises advantages in-place before yielding minibatches.
- `rl/ppo.py`: `PPO` — clipped surrogate loss + MSE value loss + entropy bonus. Logs `approx_kl` and `clip_fraction` diagnostics per update. Gradient norm clipping via `max_grad_norm`.
- `rl/runner.py`: `Runner` — stateful rollout collector. Sends `tanh(z)` to the vectorised env; stores pre-tanh `z` in the buffer. Tracks per-episode reward and length; returns aggregated stats alongside the filled buffer.
- `configs/default.yaml`: Default hyperparameters — attention policy, `n_envs=3`, `n_steps=512`, `n_epochs=4`, `batch_size=256`, `lr=3e-4`, `gamma=0.99`, `gae_lambda=0.95`, `total_steps=10M`.
- `train.py`: CLI training entry point (`--config`, `--resume`, `--device`). Seeds `random`, `numpy`, and `torch`. TensorBoard logging every rollout. Checkpoints policy + optimizer state every `save_interval` rollouts. `--resume` restores both policy weights and optimizer state.
- `tests/rl/test_agent.py`: 24 tests — forward pass shapes, deterministic/stochastic act, no-NaN on zero/non-zero obs, evaluate log-prob consistency, save/load roundtrip, `build_policy` factory.
- `tests/rl/test_training.py`: 12 tests — `RolloutBuffer` fill/GAE/batch coverage, `PPO` loss keys/finite values/weight change, `Runner` buffer shape/no-NaN/contiguous collects/attention policy compatibility.
- `requirements.txt`: added `pyyaml>=6.0` for YAML config loading.

---

## [0.3.0] — Phase 3: RL Environment

### Added
- `rl/env.py`: `AgarEnv(gym.Env)` — single-agent Gymnasium wrapper. Agent 0 is RL-controlled; remaining slots run a random-direction bot policy with auto-respawn on death.
- `rl/multi_env.py`: `AgarParallelEnv(pettingzoo.ParallelEnv)` — all-agent RL via PettingZoo parallel API. Dead agents are removed from `self.agents`; truncation clears all remaining agents at `max_ticks`.
- `rl/vec_env.py`: `VecAgarEnv` — synchronous N-env vectorised wrapper. Auto-resets on episode end; terminal observation stored in `info["final_observation"]` (standard PPO convention). Pre-allocated output buffers avoid per-step heap allocation.
- `rl/env.py`: `build_observation()` module-level helper shared between `AgarEnv` and `AgarParallelEnv`. Builds a 170-dim flat `float32` observation: own cells (K=16), nearest food (K=20), nearest viruses (K=10), nearest enemy cells (K=20), plus 2 global scalars. All positions are centroid-relative, normalised by `max(width, height) / 2`, and clipped to `[-10, 10]`.
- Action space: `Box([-1,-1,-1,-1], [1,1,1,1])` — `(dx, dy)` direction projected to world-space target `centroid + dir * world_size`; `action[2] > 0` → split; `action[3] > 0` → eject.
- Reward: `Δmass / start_mass` per tick; death penalty `-1.0`. Guaranteed no NaN/Inf.
- `tests/rl/test_env.py`: 23 tests covering obs shape, bounds, no-NaN/Inf, truncation, auto-reset, reproducible seeding, and edge cases (zero bots, n_envs=0).

---

## [0.2.1] — UI Fixes & Config Tuning

### Changed
- `mass_decay_rate` set to `0.0` (passive mass decay disabled). Decay was removing mass even at low cell sizes, making it hard to grow in a sparse world.
- Bot AI in `main.py` updated: bots chase the nearest eatable enemy cell when one exists within range; otherwise wander with a slowly-rotating direction.
- `main.py` action latching fixed: split/eject flags are set on `KEYDOWN` and cleared only after a simulation tick consumes them — key presses are no longer dropped.

### Fixed
- Camera follow jitter: position now updates instantly to the mass-weighted centroid each frame; only zoom lerps smoothly. Previously lerping position against a 25 TPS target caused visible grid shimmer at 60 FPS.
- `input.py` mouse direction: returns raw screen coordinates; `main.py` converts via `camera.screen_to_world()`. Previously the raw coords were used as world positions directly.

---

## [0.2.0] — Phase 2: Pygame UI

### Added
- `ui/camera.py`: `Camera` class — mass-weighted centroid follow, zoom scales with player size, `world_to_screen_arr()`, `screen_to_world()`, `visible_mask()`.
- `ui/input.py`: `handle_events()` returning `(action[4], quit, paused)` — mouse direction, Space=split, W=eject, P=pause, Escape=quit.
- `ui/renderer.py`: `Renderer.draw()` — grid, food pellets, spiky viruses, ejected mass, cells (sorted by mass for correct layering, labels at radius ≥ 14 px), frustum culling via `camera.visible_mask()`.
- `ui/hud.py`: `HUD.draw()` — semi-transparent leaderboard panel, FPS counter, minimap with viewport rectangle, food dots, and per-player cell circles.
- `main.py`: argparse CLI (`--agents N --seed N --width N --height N --fps N --tps N --no-human`), fixed-timestep accumulator loop decoupled from render FPS, sub-tick position interpolation (`_interp_state`) for smooth 60 FPS visuals from a 25 TPS sim, player respawn with 50-tick cooldown.
- `benchmark.py`: headless throughput benchmark (no Pygame); reports TPS across several player/food configurations.

---

## [0.1.0] — Phase 1: Game Engine

### Added
- `game/config.py`: `WorldConfig` — frozen dataclass holding all simulation constants. Key calibration: `radius = sqrt(mass)` with no separate scale factor; `start_mass=2500 → radius=50`, `food_mass=25 → radius=5`.
- `game/entities.py`: `CellArrays`, `FoodArrays`, `VirusArrays`, `EjectedArrays` — pre-allocated contiguous NumPy buffers with free-list allocators. `FoodArrays` uses compact (no-holes) layout with swap-with-last free for O(1) `alive_indices`.
- `game/physics.py`: `update_cells()` — vectorised velocity from steering direction and speed formula; split-velocity decay; merge-timer countdown; mass decay. `update_ejected()` — friction deceleration, wall clamping.
- `game/collision.py`: `resolve_food_eating()`, `resolve_cell_eating()`, `resolve_ejected_eating()`, `resolve_virus_collision()`, `resolve_merging()` — all use O(n²) NumPy broadcasting over alive-cell subsets. `np.add.at` for correct mass accumulation when one predator eats multiple prey in the same tick.
- `game/spawner.py`: `spawn_food()`, `spawn_viruses()`, `add_player()`, `handle_split()`, `handle_eject()`, `resolve_virus_feeding()`, `apply_virus_splits()`.
- `game/world.py`: `World` class with `reset()`, `step()`, `add_player()`, `remove_player()`, `get_state()`. `step()` returns `(rewards, dones, info)` — `get_state()` is kept off the hot path to save ~15 % tick time.
- `tests/game/test_mechanics.py`: 34 unit tests — eating rules, mass conservation, split physics, merge timer, virus splitting, boundary clamping, determinism, no-NaN.
