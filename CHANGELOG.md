# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

add graphing/data collection. Average survival time of blobs, average size, number of kills, biggest blob per generation etc

## [Unreleased]

### Fixed
- `game/world.py` docstring: example in `World` class docstring unpacked 4 values from `world.step()` which returns 3 (`rewards, dones, info`). Would raise `ValueError` if copied verbatim.
- `eval/elo.py`: progress counter formula `i * (n - 1 - i // 2) + (j - i)` produced incorrect pair numbers (e.g. pair 5 shown twice for n=4). Corrected to `i*(n-1) - i*(i-1)//2 + (j-i)`.

### Performance
- `game/world.py`: `get_state()` replaced per-player `player_indices()` calls (O(n_players × capacity) scan) with a single vectorised pass over the alive cell index array — same `c_idx`/`owners`/`valid` arrays already available in the function. Eliminates N linear scans per `get_state()` call; called every tick during training and evaluation.
- `rl/env.py`: `_world_actions` and `_bot_obs_buf` pre-allocated in `__init__`; `step()` zeroes them in-place with `[:] = 0.0` instead of allocating a new array each call.
- `rl/env.py`: bot observations in `step()` now built with `build_observation_batch()` in a single call instead of per-bot `build_observation()` inside a Python loop. Eliminates per-bot `np.zeros` + `np.concatenate` allocations when a `bot_policy` is active.
- `rl/env.py`: random bot movement angles now sampled with a single `rng.uniform(..., size=n)` batch call instead of one `rng.uniform()` call per bot in a loop.
- `eval/harness.py`: `world_actions` array moved outside the episode and tick loops; zeroed in-place each tick. Eliminates one `np.zeros((max_players, 4))` allocation per simulation tick during evaluation.
- `rl/buffer.py`: `compute_returns_and_advantages()` GAE loop now copies rollout tensors to CPU before the sequential backward pass. Avoids launching ~`n_steps` (up to 2048) tiny CUDA kernels when training on GPU; loop runs on CPU then results are copied back via `copy_()`.
- `rl/selfplay.py`: `sample_policy()` now caches the `obs → action` closure in the ring slot alongside the policy at load time. Subsequent calls to the same slot return the cached callable directly instead of constructing a new closure object each rollout.

### Changed
- `configs/default.yaml`: training world shrunk from 4000×4000 to **2000×2000**; `target_food_count` 600→200, `target_virus_count` 4→2. Smaller arena means ~4× more agent encounters per tick, producing denser reward signal.
- `configs/default.yaml`: switched from 4 parallel worlds to **1 world × 32 agents** (`n_envs: 1`, `n_agents: 32`, `world.max_players: 32`). All 32 RL agents now compete in one visible world; CPU game-sim overhead drops proportionally.
- `configs/default.yaml`: episode length increased from 1000 to **1500 ticks** (~60 s at 25 TPS) to allow full mass-growth arcs before world reset.
- `configs/default.yaml`: PPO minibatch size raised from 256 to **2048**, `n_steps` 512→2048, `n_epochs` 4→8. Total transitions per rollout: 65 536. Larger minibatches improve GPU utilisation during the PPO update phase.

### Performance
- `rl/env.py`: `build_observation_batch()` — new vectorised helper builds observations for all agents in a world in a single call. Shared arrays (`cell_owner`, `cell_pos`, `cell_mass`, `food_pos`, `virus_pos`) are accessed once and results are written directly into a pre-allocated `(B, OBS_DIM)` output buffer, eliminating per-agent `np.zeros` + `np.concatenate` allocations. Produces bit-identical output to the scalar loop.
- `rl/ma_vec_env.py`: `_fill_obs()`, the step output loop, and the truncation final-obs block all replaced with `build_observation_batch()` calls. Added a pre-allocated `_obs_scratch` buffer so the terminal-obs path avoids heap allocation. Reduces per-step Python overhead from O(n_agents) function calls to 1–2 NumPy calls.

### Added (prior unreleased)
- `game/world.py`: `_prev_mass` snapshot vectorised — replaced a Python `for` loop calling `cells.player_indices()` once per active player (O(capacity) scan each) with a single `np.add.at` mass-accumulation pass that reuses the `c_idx`/`owners`/`valid` arrays already computed in the same tick for death detection. Eliminates N linear scans per tick (N = active players).
- `rl/ma_vec_env.py`: `_world_actions` pre-allocated in `__init__` — the `np.zeros((max_players, 4))` array previously re-allocated inside the `step()` loop on every call. Now cleared with `[:] = 0.0` in-place each iteration.
- `rl/ma_vec_env.py`: removed `.copy()` from `step()` return values — output buffers (`_obs_buf`, `_rew_buf`, `_term_buf`, `_trunc_buf`) are now returned directly. Safe because the `Runner` consumes them via `torch.from_numpy().to(device)` and `buffer.add()` (which calls `copy_()`) before the next `step()` overwrites the buffers.
- `rl/env.py`: `build_observation` enemy distance computation unified — squared distances to all enemy cells are now computed once, then boolean-indexed per threat/prey category for `argpartition`. Previously `_k_nearest_indices` was called twice, recomputing distances independently for each subgroup. Also removes the intermediate `t_pos`/`p_pos`/`t_mass`/`p_mass` array allocations.

### Added (Phases 7–9)
- `rl/selfplay.py`: `OpponentPool` — ring buffer (configurable size) of past policy snapshots written to disk. `maybe_update(policy, rollout_idx, step)` saves a snapshot every `update_interval` rollouts. `sample_policy()` returns a `obs → action` callable drawn uniformly from the pool, or `None` (random walk) with probability `1 − selfplay_prob`. Policies are lazy-loaded on first use and evicted when the ring slot is overwritten. `save_state()` / `load_state()` persist pool metadata to `pool_state.json` for resume support.
- `rl/env.py`: `AgarEnv` gains a `bot_policy: Callable | None` constructor parameter and a `set_bot_policy()` method. When set, each bot calls `build_observation()` + the policy instead of sampling a random angle; a single `world.get_state()` snapshot is taken before the bot action loop. Default (`None`) is identical to prior behaviour.
- `rl/vec_env.py`: `VecAgarEnv.set_bot_policy()` propagates a new bot policy to all underlying `AgarEnv` instances.
- `train.py`: reads optional `selfplay:` YAML section (`pool_size`, `update_interval`, `selfplay_prob`). When present and `n_agents == 0`, instantiates `OpponentPool` and calls `pool.maybe_update()` + `venv.set_bot_policy()` before each rollout. Logs a warning and disables the pool if `n_agents > 0`.
- `configs/selfplay.yaml`: dedicated self-play training config (`n_agents=0`, `n_bots=7`, selfplay pool enabled).
- `configs/default.yaml`: `selfplay` section added as commented-out template.
- `eval/elo.py`: `EloRating` class — standard Elo with configurable K-factor (default 32) and initial rating (default 1000). `record_result(label_a, label_b, score_a)` updates both ratings after one game. `table_str()` returns a formatted ranked table. `to_dict()` / `from_dict()` for JSON serialisation. `run_tournament(checkpoint_paths, ...)` — round-robin across all checkpoint pairs; each unordered pair plays in both directions to cancel home-field advantage; score per episode is rank-based (rank 1 → 1.0, tie → 0.5, last → 0.0); policies loaded and deleted pair-by-pair to bound memory use.
- `eval.py`: `--elo` mode — `--checkpoint-dir <dir>` globs checkpoints matching `--ckpt-glob` (default `ckpt_[0-9]*.pt`), runs `run_tournament()`, prints ranked table, saves `elo_results.json`. Additional flags: `--elo-output`, `--elo-bots`, `--k-factor`.
- `ui/renderer.py`: `AttentionWeights` dataclass — `food_weight: ndarray` (length = food count) and `cell_weight: ndarray` (length = cell count) for per-entity attention scores. `Renderer.draw()` accepts an optional `attention: AttentionWeights` parameter. `_draw_food()` and `_draw_cells()` draw a semi-transparent glow halo before each entity fill when `attention` is set; halo radius scales as `r × (1 + scale × weight)`. `_draw_glow()` helper renders a `SRCALPHA` circle onto the display surface.
- `main.py`: `--attention-viz` flag enables attention overlay (requires `--checkpoint` with an `AttentionPolicy`; silently disabled otherwise). `--viz-player N` selects which bot's perspective to visualise (default: first bot). `_build_attn_weights()` helper — takes last-layer attention weights from `attention_maps()`, averages over heads, column-sums to get "attention received" per token, normalises to [0, 1], then maps food/threat/prey token slots back to world entity indices via the same k-nearest logic used in `build_observation()`. Computed once per simulation tick (25 Hz) and cached as `attn_cache` for all render frames until the next tick.

---

### Fixed (prior unreleased)
- `rl/buffer.py`: GAE off-by-one in `compute_returns_and_advantages()` — inner steps used `~dones[t+1]` as the `not_done` mask instead of `~dones[t]`. When an agent died at step t and respawned before the step returned, `dones[t+1]` was always False, so the bootstrap was never zeroed and the GAE accumulation never reset at the episode boundary. Now uses `~dones[t]`: δ_t = r_t − V(s_t) on death, and advantage carry-over is correctly cut at every episode boundary.
- `rl/video.py`: fixed `extra_args` passed via `anim.save()` string shorthand — `extra_args` is only valid on `FFMpegWriter` instances, not the string dispatcher. Writer is now constructed explicitly as `FFMpegWriter(fps, extra_args=[...])`.
- `rl/video.py`: video render no longer raises when ffmpeg is not installed — falls back to Pillow and saves as `.gif` instead of `.mp4`. `render_episode_to_video()` now returns the actual path written so callers can log the correct filename.

### Changed
- `train.py`: training video is now also recorded after the very first rollout (`rollout_idx == 1`), in addition to every `video_interval` rollouts thereafter. Log line updated to show the actual saved path (`.gif` vs `.mp4` depending on ffmpeg availability).

### Added
- `rl/env.py`: `survival_bonus` parameter on `AgarEnv` — added to reward each tick the agent is alive.  Default `0.0`; set to `0.01` (`food_mass / start_mass`) in `configs/default.yaml` to give a dense gradient toward staying alive before death ever occurs.
- `rl/ma_vec_env.py`: `survival_bonus` parameter on `VecAgarMAEnv` — same behaviour as `AgarEnv`.  Threaded through `train.py` from `env.survival_bonus` in YAML.
- `configs/default.yaml`: `env.survival_bonus: 0.01`.

### Changed
- `rl/env.py`: enemy observation encoding replaced.  Former `K_ENEMY=20` single group (absolute `log_mass_norm`) split into `K_THREAT=10` (enemies larger than self) + `K_PREY=10` (enemies smaller than self).  Mass feature changed to `delta_log_mass = log(enemy_mass / own_total_mass + 1e-6) / 5` — positive for threats, negative for prey — so the network never needs to subtract two distant features to classify an enemy.  `OBS_DIM` remains 170.
- `rl/agent.py` (`AttentionPolicy`): `enemy_proj` replaced with `threat_proj` + `prey_proj`; `type_emb` expanded from 4 to 5 types (own, food, virus, threat, prey).  Token count stays 66.  **Existing checkpoints are incompatible — retrain from scratch.**
- `eval/baselines.py` (`GreedyPolicy`): updated to use threat/prey slot indices; removed redundant own-mass subtraction (slot assignment encodes threat vs. prey directly).
- `game/world.py`: death penalty changed from fixed `-start_mass` to `-prev_mass[pid]` (the player's actual mass at time of death).  A mass-tracking update loop now snapshots each surviving player's mass at the end of every tick.  This makes dying proportionally costly regardless of how much the agent has grown.
- `configs/default.yaml`: `video_interval` halved from 100 to 50 rollouts (2× more frequent training videos); training videos now saved as `.mp4` instead of `.gif`.
- `configs/default.yaml`: `n_agents` increased from 8 to 16; `n_envs` increased from 3 to 4.

### Added (prior unreleased)
- `rl/ma_vec_env.py`: `VecAgarMAEnv` — vectorised multi-agent env with N worlds × M RL-controlled agents all sharing a single policy.  Each `(world, agent)` pair is an independent stream; Runner/buffer/PPO require no changes.  Agents auto-respawn on death; worlds reset on `max_ticks`.
- `rl/video.py`: `render_episode_to_video()` — headless matplotlib renderer that converts a list of `GameState` frames to a GIF or MP4.  `record_video()` — runs a short episode with the current policy and saves it; called automatically during training.
- `train.py`: multi-agent training mode — when `env.n_agents > 0`, uses `VecAgarMAEnv` instead of `VecAgarEnv`; all agents learn simultaneously via shared-weight PPO.  Saves a training video every `video_interval` rollouts.
- `main.py`: `--checkpoint PATH` flag — bots use a trained policy loaded from a checkpoint instead of the random heuristic.  Bot names on the leaderboard change from "Bot N" to "Agent N" when a checkpoint is active.

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
