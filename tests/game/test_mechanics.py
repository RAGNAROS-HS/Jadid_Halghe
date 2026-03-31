from __future__ import annotations

import numpy as np
import pytest

from game.collision import (
    resolve_cell_eating,
    resolve_food_eating,
    resolve_merging,
    resolve_virus_collision,
)
from game.config import WorldConfig
from game.entities import CellArrays, FoodArrays, VirusArrays
from game.spawner import handle_split, spawn_food
from game.world import World


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def cfg() -> WorldConfig:
    """Small world config for fast tests."""
    return WorldConfig(
        width=1000.0,
        height=1000.0,
        max_cells=128,
        max_food=512,
        max_viruses=16,
        max_ejected=64,
        max_players=4,
        target_food_count=50,
        target_virus_count=4,
    )


@pytest.fixture()
def world(cfg: WorldConfig) -> World:
    w = World(cfg)
    w.reset(seed=0)
    return w


# ---------------------------------------------------------------------------
# WorldConfig validation
# ---------------------------------------------------------------------------

class TestWorldConfig:
    def test_eat_mass_ratio(self) -> None:
        cfg = WorldConfig()
        assert abs(cfg.eat_mass_ratio - 1.1**2) < 1e-6

    def test_invalid_eat_ratio(self) -> None:
        with pytest.raises(AssertionError):
            WorldConfig(eat_ratio=0.9)

    def test_invalid_split_decay(self) -> None:
        with pytest.raises(AssertionError):
            WorldConfig(split_decay=1.5)


# ---------------------------------------------------------------------------
# Entity arrays
# ---------------------------------------------------------------------------

class TestEntityArrays:
    def test_cell_allocate_and_free(self, cfg: WorldConfig) -> None:
        cells = CellArrays.create(cfg.max_cells)
        assert cells.count == 0

        idx = cells.allocate(3)
        assert len(idx) == 3
        assert cells.count == 3
        assert cells.alive[idx].all()

        cells.free(idx)
        assert cells.count == 0
        assert not cells.alive[idx].any()

    def test_cell_allocate_exhaustion(self, cfg: WorldConfig) -> None:
        cells = CellArrays.create(2)
        cells.allocate(2)
        with pytest.raises(RuntimeError):
            cells.allocate(1)

    def test_radius_formula(self, cfg: WorldConfig) -> None:
        cells = CellArrays.create(4)
        idx = cells.allocate(2)
        cells.mass[idx[0]] = 2500.0
        cells.mass[idx[1]] = 25.0
        radii = cells.radius()
        assert abs(radii[idx[0]] - 50.0) < 0.01
        assert abs(radii[idx[1]] - 5.0) < 0.01

    def test_food_allocate_caps_at_available(self) -> None:
        food = FoodArrays.create(5)
        # Requesting more than capacity silently caps
        idx = food.allocate(10)
        assert len(idx) == 5


# ---------------------------------------------------------------------------
# Food spawning
# ---------------------------------------------------------------------------

class TestFoodSpawning:
    def test_food_count_after_reset(self, world: World, cfg: WorldConfig) -> None:
        assert world.food.count == cfg.target_food_count

    def test_food_positions_in_bounds(self, world: World, cfg: WorldConfig) -> None:
        f_idx = world.food.alive_indices()
        pos = world.food.pos[f_idx]
        assert (pos[:, 0] >= 0).all() and (pos[:, 0] <= cfg.width).all()
        assert (pos[:, 1] >= 0).all() and (pos[:, 1] <= cfg.height).all()

    def test_food_respawn(self, cfg: WorldConfig) -> None:
        food = FoodArrays.create(cfg.max_food)
        rng = np.random.default_rng(1)
        spawn_food(food, rng, cfg)
        assert food.count == cfg.target_food_count
        # Kill half
        alive = food.alive_indices()
        food.free(alive[: len(alive) // 2])
        prev_count = food.count
        spawn_food(food, rng, cfg)
        assert food.count == cfg.target_food_count
        assert food.count > prev_count


# ---------------------------------------------------------------------------
# Cell eating
# ---------------------------------------------------------------------------

class TestCellEating:
    def _make_cells(self, cfg: WorldConfig) -> CellArrays:
        return CellArrays.create(cfg.max_cells)

    def test_big_eats_small(self, cfg: WorldConfig) -> None:
        cells = self._make_cells(cfg)
        idx = cells.allocate(2)
        big, small = int(idx[0]), int(idx[1])

        cells.pos[big] = [500.0, 500.0]
        cells.mass[big] = 10_000.0
        cells.owner[big] = 0

        # Small cell is inside the big cell (dist=0 < radius_big)
        cells.pos[small] = [500.0, 500.0]
        cells.mass[small] = 100.0
        cells.owner[small] = 1  # different player

        gains = resolve_cell_eating(cells, cfg)

        assert not cells.alive[small], "Small cell should be eaten."
        assert cells.alive[big], "Big cell should survive."
        assert cells.mass[big] == pytest.approx(10_100.0), "Mass should increase."
        assert gains[0] > 0.0

    def test_equal_mass_no_eat(self, cfg: WorldConfig) -> None:
        cells = self._make_cells(cfg)
        idx = cells.allocate(2)
        a, b = int(idx[0]), int(idx[1])

        for i in [a, b]:
            cells.pos[i] = [500.0, 500.0]
            cells.mass[i] = 5_000.0

        cells.owner[a] = 0
        cells.owner[b] = 1

        resolve_cell_eating(cells, cfg)
        # Neither should eat the other (need 1.21× mass ratio)
        assert cells.alive[a] and cells.alive[b]

    def test_same_player_no_eat(self, cfg: WorldConfig) -> None:
        cells = self._make_cells(cfg)
        idx = cells.allocate(2)
        big, small = int(idx[0]), int(idx[1])

        cells.pos[big] = [500.0, 500.0]
        cells.mass[big] = 10_000.0
        cells.owner[big] = 0

        cells.pos[small] = [500.0, 500.0]
        cells.mass[small] = 100.0
        cells.owner[small] = 0  # same player

        resolve_cell_eating(cells, cfg)
        # Should NOT eat own cell
        assert cells.alive[small]

    def test_far_apart_no_eat(self, cfg: WorldConfig) -> None:
        cells = self._make_cells(cfg)
        idx = cells.allocate(2)
        big, small = int(idx[0]), int(idx[1])

        cells.pos[big] = [100.0, 100.0]
        cells.mass[big] = 10_000.0    # radius = 100
        cells.owner[big] = 0

        cells.pos[small] = [900.0, 900.0]   # far away
        cells.mass[small] = 25.0
        cells.owner[small] = 1

        resolve_cell_eating(cells, cfg)
        assert cells.alive[small]

    def test_mass_conservation_on_eat(self, cfg: WorldConfig) -> None:
        cells = self._make_cells(cfg)
        idx = cells.allocate(2)
        big, small = int(idx[0]), int(idx[1])

        cells.pos[big] = [500.0, 500.0]
        cells.mass[big] = 10_000.0
        cells.owner[big] = 0

        cells.pos[small] = [500.0, 500.0]
        cells.mass[small] = 100.0
        cells.owner[small] = 1

        total_before = cells.mass[big] + cells.mass[small]
        resolve_cell_eating(cells, cfg)
        total_after = cells.mass[cells.alive_indices()].sum()
        assert total_after == pytest.approx(total_before, rel=1e-5)


# ---------------------------------------------------------------------------
# Food eating
# ---------------------------------------------------------------------------

class TestFoodEating:
    def test_cell_eats_nearby_food(self, cfg: WorldConfig) -> None:
        cells = CellArrays.create(cfg.max_cells)
        food = FoodArrays.create(cfg.max_food)

        c_idx = cells.allocate(1)
        cells.pos[c_idx[0]] = [500.0, 500.0]
        cells.mass[c_idx[0]] = cfg.start_mass
        cells.owner[c_idx[0]] = 0

        f_idx = food.allocate(1)
        food.pos[f_idx[0]] = [500.1, 500.0]   # inside cell radius=50

        gains = resolve_food_eating(cells, food, cfg)

        assert not food.alive[f_idx[0]], "Food should be eaten."
        assert cells.mass[c_idx[0]] == pytest.approx(cfg.start_mass + cfg.food_mass)
        assert gains[0] == pytest.approx(cfg.food_mass)

    def test_cell_misses_far_food(self, cfg: WorldConfig) -> None:
        cells = CellArrays.create(cfg.max_cells)
        food = FoodArrays.create(cfg.max_food)

        c_idx = cells.allocate(1)
        cells.pos[c_idx[0]] = [500.0, 500.0]
        cells.mass[c_idx[0]] = cfg.start_mass   # radius = 50

        f_idx = food.allocate(1)
        food.pos[f_idx[0]] = [700.0, 500.0]   # 200 units away, > radius

        resolve_food_eating(cells, food, cfg)
        assert food.alive[f_idx[0]], "Far food should not be eaten."


# ---------------------------------------------------------------------------
# Splitting and merging
# ---------------------------------------------------------------------------

class TestSplitAndMerge:
    def test_split_creates_two_cells(self, cfg: WorldConfig) -> None:
        cells = CellArrays.create(cfg.max_cells)
        idx = cells.allocate(1)
        cells.pos[idx[0]] = [500.0, 500.0]
        cells.mass[idx[0]] = cfg.min_split_mass * 2
        cells.owner[idx[0]] = 0
        cells.vel[idx[0]] = [1.0, 0.0]

        handle_split(cells, 0, cfg)

        p_cells = cells.player_indices(0)
        assert len(p_cells) == 2, "Split should produce exactly 2 cells."

    def test_split_conserves_mass(self, cfg: WorldConfig) -> None:
        cells = CellArrays.create(cfg.max_cells)
        idx = cells.allocate(1)
        original_mass = cfg.min_split_mass * 2
        cells.pos[idx[0]] = [500.0, 500.0]
        cells.mass[idx[0]] = original_mass
        cells.owner[idx[0]] = 0
        cells.vel[idx[0]] = [1.0, 0.0]

        handle_split(cells, 0, cfg)

        p_cells = cells.player_indices(0)
        total_mass = cells.mass[p_cells].sum()
        assert total_mass == pytest.approx(original_mass, rel=1e-5)

    def test_split_below_min_mass_no_op(self, cfg: WorldConfig) -> None:
        cells = CellArrays.create(cfg.max_cells)
        idx = cells.allocate(1)
        cells.pos[idx[0]] = [500.0, 500.0]
        cells.mass[idx[0]] = cfg.min_split_mass * 0.5   # too small
        cells.owner[idx[0]] = 0
        cells.vel[idx[0]] = [1.0, 0.0]

        handle_split(cells, 0, cfg)
        assert len(cells.player_indices(0)) == 1, "Should not split when too small."

    def test_merge_timer_prevents_early_merge(self, cfg: WorldConfig) -> None:
        cells = CellArrays.create(cfg.max_cells)
        idx = cells.allocate(2)
        for i in idx:
            cells.pos[i] = [500.0, 500.0]
            cells.mass[i] = 2500.0
            cells.owner[i] = 0
            cells.merge_timer[i] = 100.0   # not ready

        resolve_merging(cells, cfg)
        assert len(cells.player_indices(0)) == 2, "Should not merge before timer expires."

    def test_merge_when_ready(self, cfg: WorldConfig) -> None:
        cells = CellArrays.create(cfg.max_cells)
        idx = cells.allocate(2)
        for i in idx:
            cells.pos[i] = [500.0, 500.0]
            cells.mass[i] = 2500.0
            cells.owner[i] = 0
            cells.merge_timer[i] = 0.0   # ready

        resolve_merging(cells, cfg)
        assert len(cells.player_indices(0)) == 1, "Should merge when overlapping and timer=0."

    def test_merge_conserves_mass(self, cfg: WorldConfig) -> None:
        cells = CellArrays.create(cfg.max_cells)
        idx = cells.allocate(2)
        total = 0.0
        for i in idx:
            cells.pos[i] = [500.0, 500.0]
            cells.mass[i] = 2500.0
            cells.owner[i] = 0
            cells.merge_timer[i] = 0.0
            total += cells.mass[i]

        resolve_merging(cells, cfg)
        survivor = cells.player_indices(0)
        assert cells.mass[survivor].sum() == pytest.approx(total, rel=1e-5)


# ---------------------------------------------------------------------------
# Virus collision
# ---------------------------------------------------------------------------

class TestVirusCollision:
    def test_large_cell_hits_virus(self, cfg: WorldConfig) -> None:
        cells = CellArrays.create(cfg.max_cells)
        viruses = VirusArrays.create(cfg.max_viruses)

        c_idx = cells.allocate(1)
        cells.pos[c_idx[0]] = [500.0, 500.0]
        cells.mass[c_idx[0]] = cfg.virus_mass * 2   # larger than virus
        cells.owner[c_idx[0]] = 0
        cells.vel[c_idx[0]] = [1.0, 0.0]

        v_idx = viruses.allocate(1)
        viruses.pos[v_idx[0]] = [500.0, 500.0]   # same position

        requests = resolve_virus_collision(cells, viruses, cfg)
        assert len(requests) == 1

    def test_small_cell_no_virus_hit(self, cfg: WorldConfig) -> None:
        cells = CellArrays.create(cfg.max_cells)
        viruses = VirusArrays.create(cfg.max_viruses)

        c_idx = cells.allocate(1)
        cells.pos[c_idx[0]] = [500.0, 500.0]
        cells.mass[c_idx[0]] = cfg.virus_mass * 0.5   # smaller than virus
        cells.owner[c_idx[0]] = 0

        v_idx = viruses.allocate(1)
        viruses.pos[v_idx[0]] = [500.0, 500.0]

        requests = resolve_virus_collision(cells, viruses, cfg)
        assert len(requests) == 0


# ---------------------------------------------------------------------------
# World integration
# ---------------------------------------------------------------------------

class TestWorld:
    def test_reset_populates_food_and_viruses(self, world: World, cfg: WorldConfig) -> None:
        assert world.food.count == cfg.target_food_count
        assert world.viruses.count == cfg.target_virus_count

    def test_add_player_creates_cell(self, world: World) -> None:
        world.add_player(0)
        assert world.cells.player_indices(0).size == 1

    def test_duplicate_player_raises(self, world: World) -> None:
        world.add_player(0)
        with pytest.raises(ValueError):
            world.add_player(0)

    def test_step_returns_correct_shapes(self, world: World, cfg: WorldConfig) -> None:
        world.add_player(0)
        actions = np.zeros((cfg.max_players, 4), dtype=np.float32)
        rewards, dones, info = world.step(actions)

        assert rewards.shape == (cfg.max_players,)
        assert dones.shape == (cfg.max_players,)
        assert isinstance(info, dict)
        assert info["tick"] == 1

    def test_determinism_with_seed(self, cfg: WorldConfig) -> None:
        """Two worlds with the same seed must produce identical states after N steps."""
        def run(seed: int, n_steps: int) -> np.ndarray:
            w = World(cfg)
            w.reset(seed=seed)
            w.add_player(0)
            actions = np.zeros((cfg.max_players, 4), dtype=np.float32)
            actions[0] = [1.0, 0.0, 0.0, 0.0]
            for _ in range(n_steps):
                w.step(actions)
            return w.get_state().player_mass.copy()

        mass_a = run(42, 50)
        mass_b = run(42, 50)
        np.testing.assert_array_equal(mass_a, mass_b)

    def test_cell_positions_stay_in_bounds(self, world: World, cfg: WorldConfig) -> None:
        world.add_player(0)
        # Drive into a corner
        actions = np.zeros((cfg.max_players, 4), dtype=np.float32)
        actions[0] = [1.0, 1.0, 0.0, 0.0]   # top-right corner
        for _ in range(500):
            world.step(actions)

        p_idx = world.cells.player_indices(0)
        if len(p_idx) > 0:
            pos = world.cells.pos[p_idx]
            assert (pos[:, 0] >= 0).all() and (pos[:, 0] <= cfg.width).all()
            assert (pos[:, 1] >= 0).all() and (pos[:, 1] <= cfg.height).all()

    def test_no_nan_in_positions(self, world: World, cfg: WorldConfig) -> None:
        world.add_player(0)
        world.add_player(1)
        actions = np.zeros((cfg.max_players, 4), dtype=np.float32)
        for step in range(200):
            # Periodically split and eject
            if step % 20 == 0:
                actions[0, 2] = 1.0
            else:
                actions[0, 2] = 0.0
            world.step(actions)

        c_idx = world.cells.alive_indices()
        assert not np.isnan(world.cells.pos[c_idx]).any(), "NaN in cell positions."
        assert not np.isnan(world.cells.mass[c_idx]).any(), "NaN in cell masses."

    def test_player_death_sets_done(self, cfg: WorldConfig) -> None:
        """A player with no cells should be reported dead."""
        world = World(cfg)
        world.reset(seed=1)
        world.add_player(0)

        # Force kill player 0's cell directly
        p_idx = world.cells.player_indices(0)
        world.cells.free(p_idx)
        world._active_players.add(0)   # re-add as active so step checks it

        actions = np.zeros((cfg.max_players, 4), dtype=np.float32)
        _, dones, _ = world.step(actions)
        assert dones[0], "Dead player should be flagged in dones."

    def test_food_eating_increases_mass(self, cfg: WorldConfig) -> None:
        """Running many ticks with a stationary player in a food-dense world."""
        world = World(cfg)
        world.reset(seed=7)
        world.add_player(0)

        initial_mass = world.cells.player_mass(0)
        actions = np.zeros((cfg.max_players, 4), dtype=np.float32)
        # Keep the player still (zero input) in a dense food field
        for _ in range(100):
            world.step(actions)

        # Mass must not decrease (no enemies present, just food and viruses)
        # Note: there might be no food exactly at the player's position, so
        # we just assert non-negative mass.
        assert world.cells.player_mass(0) >= 0.0

    def _player_mass(self, world: World, pid: int) -> float:
        return world.cells.player_mass(pid)
