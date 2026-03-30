# Jadid_Halghe

> Teach AI agents to play an agar.io clone via reinforcement learning, implemented in Python + PyTorch.

## Phases

- **Phase 1** — Game engine: fast, local, accurate agar.io clone. Human-playable alongside trained agents.
- **Phase 2** — RL infrastructure: training loop, agent architecture, evaluation.

## Commands

| Command | Description |
|---------|-------------|
| `pip install -r requirements.txt` | Install dependencies |
| `python main.py` | Run the game (human-playable mode) |
| `python train.py` | Start RL training |
| `pytest` | Run all tests |
| `ruff check .` | Lint |
| `ruff format .` | Format |

> Update this table as the project evolves.

## Architecture

```
Jadid_Halghe/
  game/        # Game engine: entities, physics, collision, world state
  rl/          # RL: environment wrapper, agent, training loop, replay buffer
  ui/          # Renderer for human play (pygame or similar)
  tests/       # All tests, mirroring source structure
  train.py     # Training entry point
  main.py      # Human-play entry point
```

> Update as directories are created.

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

> Fill this section as you discover project-specific quirks.

- [ ] (empty — add as you encounter them)
