"""Phase 1 throughput benchmark.

Run with:
    python benchmark.py

Prints ticks/sec for several entity-count scenarios and shows how many
parallel worlds are needed to reach the 10 000 agent-steps/sec training
target.
"""

from __future__ import annotations

import time

import numpy as np

from game.config import WorldConfig
from game.world import World


def measure(cfg: WorldConfig, n_players: int, n_ticks: int = 10_000) -> float:
    """Return ticks/sec for a warmed-up world."""
    world = World(cfg)
    world.reset(seed=0)
    for i in range(n_players):
        world.add_player(i)

    actions = np.zeros((cfg.max_players, 4), dtype=np.float32)
    actions[:n_players, 0] = 1.0  # all players move right

    # Warmup
    for _ in range(500):
        world.step(actions)

    t0 = time.perf_counter()
    for _ in range(n_ticks):
        world.step(actions)
    elapsed = time.perf_counter() - t0

    return n_ticks / elapsed


SCENARIOS = [
    # (label, WorldConfig kwargs, n_players)
    ("2 players / 500 food",   {"max_players": 2,  "target_food_count": 500},  2),
    ("4 players / 1000 food",  {"max_players": 4,  "target_food_count": 1000}, 4),
    ("8 players / 2000 food",  {"max_players": 8,  "target_food_count": 2000}, 8),
    ("16 players / 4000 food", {"max_players": 16, "target_food_count": 4000, "max_cells": 1024, "max_food": 8192}, 16),
]

TARGET_TPS = 10_000

print(f"{'Scenario':<28} {'TPS':>8}  {'Worlds for 10k':>15}")
print("-" * 56)
for label, kwargs, n_players in SCENARIOS:
    cfg = WorldConfig(**kwargs)
    tps = measure(cfg, n_players)
    worlds_needed = int(np.ceil(TARGET_TPS / tps))
    print(f"{label:<28} {tps:>8,.0f}  {worlds_needed:>15}")

print()
print("Note: 'Worlds for 10k' = parallel VecEnv instances needed to")
print("      reach 10 000 total agent-steps/sec (Phase 3.5 target).")
print()
print("Single-world throughput is bounded by NumPy per-call Python")
print("overhead (~2-3 us/call x ~25 calls/tick ~= 70 us minimum).")
print("Numba JIT on the hot path would achieve 20-50k TPS single-world.")
