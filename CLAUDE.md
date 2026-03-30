# Jadid_Halghe

> Teach AI agents to play an agar.io clone via reinforcement learning, implemented in Python + PyTorch.

## Phases

- **Phase 1** ✅ — Game engine: fast, local, accurate agar.io clone. Human-playable alongside trained agents.
- **Phase 2** — Pygame UI: renderer, camera, HUD, human + agent mixed sessions.
- **Phase 3** — RL environment: Gymnasium + PettingZoo wrappers, observation/action spaces, reward function, VecEnv.
- **Phase 4** — Agent architecture: MLP baseline, attention-based primary policy, recurrent option.
- **Phase 5** — Training: PPO, rollout collection, logging, checkpointing, self-play curriculum.
- **Phase 6** — Evaluation: harness, replay, baselines, ablations.

## Commands

| Command | Description |
|---------|-------------|
| `pip install -r requirements.txt` | Install dependencies |
| `pytest` | Run all tests (34 tests, ~0.4 s) |
| `ruff check .` | Lint |
| `ruff format .` | Format |
| `python main.py` | Run the game — Phase 2 (not yet implemented) |
| `python train.py --config configs/default.yaml` | Start RL training — Phase 5 (not yet implemented) |
| `python eval.py --checkpoint <path>` | Evaluate a checkpoint — Phase 6 (not yet implemented) |

**Python version note:** Two Python installs exist on this machine. Use `C:/Users/Hugo/AppData/Local/Programs/Python/Python312/python.exe` (3.12) — packages are installed there. Python 3.13 at the default `python` path is missing most deps.

## Architecture

```
Jadid_Halghe/
  game/              # Game engine ✅
    config.py        #   WorldConfig — all simulation constants (frozen dataclass)
    entities.py      #   CellArrays, FoodArrays, VirusArrays, EjectedArrays
    physics.py       #   update_cells(), update_ejected()
    collision.py     #   resolve_food_eating(), resolve_cell_eating(),
                     #   resolve_ejected_eating(), resolve_virus_collision(),
                     #   resolve_merging()
    spawner.py       #   spawn_food/viruses, add_player, handle_split/eject,
                     #   resolve_virus_feeding(), apply_virus_splits()
    world.py         #   World class + GameState dataclass
  rl/                # RL layer (Phase 3–5, not yet implemented)
    env.py           #   Gymnasium single-agent wrapper
    multi_env.py     #   PettingZoo parallel wrapper
    vec_env.py       #   Vectorized env
    agent.py         #   Neural network policies
    ppo.py           #   PPO algorithm
    buffer.py        #   Rollout buffer
    runner.py        #   Rollout collection
  ui/                # Pygame renderer (Phase 2, not yet implemented)
    renderer.py
    camera.py
    hud.py
    input.py
  eval/              # Evaluation & replay (Phase 6, not yet implemented)
    harness.py
    replay.py
    baselines.py
  tests/
    game/
      test_mechanics.py   # 34 unit tests — all passing
  configs/           # YAML training configs (Phase 5)
  train.py           # Training entry point (Phase 5)
  eval.py            # Eval entry point (Phase 6)
  main.py            # Human-play entry point (Phase 2)
```

## Code Style

- **Language**: Python 3.12+
- **Type hints**: Required on all public functions and class attributes. Use `from __future__ import annotations` at the top of files.
- **Formatter/linter**: Ruff. Line length 88. Run `ruff check . && ruff format .` before committing.
- **Docstrings**: NumPy style on all public functions and classes.
- **Imports**: stdlib → third-party → local, separated by blank lines.
- **No `Any`** in type hints without a comment explaining why.
- **Naming**: `snake_case` for functions/variables, `PascalCase` for classes, `UPPER_SNAKE` for module-level constants.

## Performance Rules

Performance is a first-class requirement. The game loop must support RL training at high speed (many steps/second).

- **No Python loops over entities in the game tick.** Use NumPy or PyTorch tensor operations instead.
- **Avoid object allocation in the hot path.** Pre-allocate buffers.
- **Batch everything possible.** Entity positions, radii, collisions — all should work on arrays, not individual objects.
- **Measure before optimizing.** Use `cProfile` or `torch.profiler`. Don't optimize speculatively.
- When adding a feature, always ask: "does this break vectorization?"

## RL Conventions

- Environment must implement the [Gymnasium](https://gymnasium.farama.org/) interface (`reset`, `step`, `observation_space`, `action_space`).
- Always seed everything for reproducibility: `numpy`, `random`, `torch`, environment.
- Separate training and inference clearly — model in `.train()` during training, `.eval()` during inference.
- Reward function must never return `NaN` or `Inf`. Add assertions or clipping.
- Log metrics with a consistent key format: `train/reward`, `train/loss`, `eval/win_rate`.

## Git Workflow

### Branch Naming
```
feat/<short-description>      # New feature or game mechanic
fix/<short-description>       # Bug fix
perf/<short-description>      # Performance improvement
chore/<short-description>     # Deps, config, tooling
docs/<short-description>      # Documentation only
refactor/<short-description>  # Restructure, no behavior change
test/<short-description>      # Tests only
```

### Commit Format
```
<emoji> <type>(<scope>): <subject>

[optional body: explain the why, not the what]
```

Scopes: `game`, `rl`, `ui`, `env`, `agent`, `train`, `tests`, `deps`

Use `/commit` for an assisted commit with message preview.

### PR Process
Use `/pr` to auto-generate a PR with summary, test plan, and performance notes.

## Claude Instructions

**When uncertain about scope, ask.** Don't implement more than what's asked.

**Don't add speculative abstractions.** If one environment is needed, don't build a plugin system for N environments.

**Performance is always in scope.** Even when fixing a bug, note if your fix could hurt training speed.

**Before adding new files**, check if the functionality belongs in an existing module.

**Read before editing.** Never modify a file without reading it first.

**No backward-compat shims.** This is a new project — break things cleanly when refactoring.

## Security

- Never commit model checkpoints >50MB (use Git LFS or external storage).
- Never commit `.env` files, API keys, or tokens.
- `checkpoints/`, `runs/`, `wandb/`, and `data/` are in `.gitignore` — keep it that way.

## Known Gotchas

- **`radius = sqrt(mass)` — no separate scale factor.** Mass values are chosen so radii fall in the right range: `start_mass=2500 → radius=50`, `food_mass=25 → radius=5`. Don't add a radius scale factor; adjust mass constants instead.
- **`split_vel` is separate from `vel`.** `vel` is overwritten every tick by the steering direction. `split_vel` holds split momentum and decays by `split_decay` each tick. Both are summed for position update.
- **Free-list uses a Python `deque`.** Allocation/deallocation is not on the hot path (only happens at discrete events). The hot path only reads/writes contiguous NumPy arrays via alive-index slices.
- **`WorldConfig` is frozen (`frozen=True`).** Do not try to mutate it at runtime. Create a new instance if you need a different config.
- **`np.where(cond, a/b, 0)` still evaluates `a/b` eagerly.** Use `safe_denom = np.where(cond, denom, 1.0)` before dividing to avoid `RuntimeWarning: invalid value encountered in divide` when `denom` contains zeros.
- **Eating condition uses mass ratio, not radius ratio directly.** `mass_A > eat_mass_ratio * mass_B` where `eat_mass_ratio = eat_ratio**2 = 1.21`. This avoids two `sqrt` calls per pair.
- **`np.add.at` is used for mass accumulation after eating.** It handles the case where one predator eats multiple prey in the same tick correctly (unbuffered add). It is slower than `+=` but correct for repeated indices.
