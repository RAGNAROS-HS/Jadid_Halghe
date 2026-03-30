from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from game.collision import (
    resolve_cell_eating,
    resolve_ejected_eating,
    resolve_food_eating,
    resolve_merging,
    resolve_virus_collision,
)
from game.config import WorldConfig
from game.entities import CellArrays, EjectedArrays, FoodArrays, VirusArrays
from game.physics import update_cells, update_ejected
from game.spawner import (
    add_player,
    apply_virus_splits,
    handle_eject,
    handle_split,
    resolve_virus_feeding,
    spawn_food,
    spawn_viruses,
)


# ---------------------------------------------------------------------------
# GameState — snapshot of observable world state
# ---------------------------------------------------------------------------

@dataclass
class GameState:
    """Immutable snapshot of the world at a given tick.

    All arrays contain only *live* entities.

    Parameters
    ----------
    tick : int
        Current simulation tick.
    cell_pos : ndarray, shape (n_cells, 2), float32
        Positions of all live cell fragments.
    cell_mass : ndarray, shape (n_cells,), float32
        Mass of each live cell fragment.
    cell_owner : ndarray, shape (n_cells,), int32
        Player ID of each live cell fragment.
    food_pos : ndarray, shape (n_food, 2), float32
        Positions of all live food pellets.
    virus_pos : ndarray, shape (n_viruses, 2), float32
        Positions of all live viruses.
    ejected_pos : ndarray, shape (n_ejected, 2), float32
        Positions of all live ejected-mass pellets.
    player_alive : ndarray, shape (max_players,), bool
        Whether each player slot has at least one live cell.
    player_mass : ndarray, shape (max_players,), float32
        Total mass of all live cells per player (0 if dead).
    """

    tick: int
    cell_pos: np.ndarray
    cell_mass: np.ndarray
    cell_owner: np.ndarray
    food_pos: np.ndarray
    virus_pos: np.ndarray
    ejected_pos: np.ndarray
    player_alive: np.ndarray
    player_mass: np.ndarray


# ---------------------------------------------------------------------------
# World
# ---------------------------------------------------------------------------

class World:
    """Headless agar.io simulation.

    Typical usage
    -------------
    >>> cfg = WorldConfig()
    >>> world = World(cfg)
    >>> world.reset(seed=42)
    >>> world.add_player(0)
    >>> world.add_player(1)
    >>> actions = np.zeros((cfg.max_players, 4), dtype=np.float32)
    >>> actions[0] = [1.0, 0.0, 0.0, 0.0]   # player 0 moves right
    >>> state, rewards, dones, info = world.step(actions)

    Parameters
    ----------
    config : WorldConfig
        Simulation parameters.  Defaults to ``WorldConfig()``.
    """

    def __init__(self, config: WorldConfig | None = None) -> None:
        self.config = config or WorldConfig()
        cfg = self.config

        self.cells = CellArrays.create(cfg.max_cells)
        self.food = FoodArrays.create(cfg.max_food)
        self.viruses = VirusArrays.create(cfg.max_viruses)
        self.ejected = EjectedArrays.create(cfg.max_ejected)

        self._rng = np.random.default_rng(0)
        self._tick = 0
        self._active_players: set[int] = set()
        # Track previous mass per player for reward calculation
        self._prev_mass = np.zeros(cfg.max_players, dtype=np.float32)

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self, seed: int | None = None) -> GameState:
        """Reset the world to a clean state and populate initial entities.

        Parameters
        ----------
        seed : int, optional
            If provided, re-seeds the internal RNG for full reproducibility.

        Returns
        -------
        GameState
            Initial observation.
        """
        cfg = self.config

        # Re-initialise entity buffers
        self.cells = CellArrays.create(cfg.max_cells)
        self.food = FoodArrays.create(cfg.max_food)
        self.viruses = VirusArrays.create(cfg.max_viruses)
        self.ejected = EjectedArrays.create(cfg.max_ejected)

        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._tick = 0
        self._active_players = set()
        self._prev_mass = np.zeros(cfg.max_players, dtype=np.float32)

        spawn_food(self.food, self._rng, cfg)
        spawn_viruses(self.viruses, self._rng, cfg)

        return self.get_state()

    # ------------------------------------------------------------------
    # Player management
    # ------------------------------------------------------------------

    def add_player(self, player_id: int) -> None:
        """Add a player to the simulation (spawns their starting cell).

        Parameters
        ----------
        player_id : int
            Must be in ``[0, config.max_players)``.

        Raises
        ------
        ValueError
            If *player_id* is out of range or already active.
        """
        if not (0 <= player_id < self.config.max_players):
            raise ValueError(
                f"player_id {player_id} out of range [0, {self.config.max_players})"
            )
        if player_id in self._active_players:
            raise ValueError(f"Player {player_id} is already active.")
        add_player(self.cells, player_id, self._rng, self.config)
        self._active_players.add(player_id)
        self._prev_mass[player_id] = self.config.start_mass

    def remove_player(self, player_id: int) -> None:
        """Remove a player and free all their cell slots.

        Parameters
        ----------
        player_id : int
        """
        p_idx = self.cells.player_indices(player_id)
        if len(p_idx) > 0:
            self.cells.free(p_idx)
        self._active_players.discard(player_id)
        self._prev_mass[player_id] = 0.0

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(
        self,
        actions: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, dict]:
        """Advance the simulation by one tick.

        This method is optimised for headless RL throughput.  It does **not**
        call :meth:`get_state` — callers that need a :class:`GameState`
        snapshot should call it explicitly after ``step()``.

        Parameters
        ----------
        actions : ndarray, shape (max_players, 4), float32
            One row per player slot: ``[dx, dy, split_flag, eject_flag]``.
            For inactive players the row is ignored.
            ``dx``, ``dy`` set the desired movement direction (need not be
            normalised; zero vector means no steering).
            ``split_flag >= 0.5`` triggers a split; ``eject_flag >= 0.5``
            fires ejected mass.

        Returns
        -------
        rewards : ndarray, shape (max_players,), float32
            Per-player reward:

            * ``+Δmass`` from eating food or cells.
            * ``−start_mass`` on death (eliminated this tick).
        dones : ndarray, shape (max_players,), bool
            True for players that died this tick.
        info : dict
            Auxiliary diagnostics (tick number, live cell counts, etc.).
        """
        cfg = self.config
        actions = np.asarray(actions, dtype=np.float32)
        assert actions.shape == (cfg.max_players, 4), (
            f"Expected actions shape ({cfg.max_players}, 4), got {actions.shape}"
        )

        rewards = np.zeros(cfg.max_players, dtype=np.float32)
        dones = np.zeros(cfg.max_players, dtype=bool)

        # ── 1. Handle discrete actions (split / eject) ─────────────────
        for pid in self._active_players:
            if actions[pid, 2] >= 0.5:
                handle_split(self.cells, pid, cfg)
            if actions[pid, 3] >= 0.5:
                handle_eject(self.cells, self.ejected, pid, cfg)

        # ── 2. Physics ─────────────────────────────────────────────────
        update_cells(self.cells, actions, cfg)
        update_ejected(self.ejected, cfg)

        # ── 3. Collision resolution ─────────────────────────────────────
        food_gains = resolve_food_eating(self.cells, self.food, cfg)
        rewards += food_gains

        cell_gains = resolve_cell_eating(self.cells, cfg)
        rewards += cell_gains

        resolve_ejected_eating(self.cells, self.ejected, cfg)

        # Ejected mass hitting viruses
        resolve_virus_feeding(self.ejected, self.viruses, self._rng, cfg)

        # Virus collisions → split requests
        split_requests = resolve_virus_collision(self.cells, self.viruses, cfg)
        apply_virus_splits(self.cells, split_requests, cfg)

        # Cell–cell merging
        resolve_merging(self.cells, cfg)

        # ── 4. Detect deaths ──────────────────────────────────────────
        # Compute per-player cell count in one pass over alive cells to avoid
        # calling player_indices() once per player (each call scans all slots).
        p_cell_count = np.zeros(cfg.max_players, dtype=np.int32)
        c_idx = self.cells.alive_indices()
        if len(c_idx):
            owners = self.cells.owner[c_idx]
            valid = (owners >= 0) & (owners < cfg.max_players)
            np.add.at(p_cell_count, owners[valid], 1)

        dead_players: list[int] = []
        for pid in list(self._active_players):
            if p_cell_count[pid] == 0:
                dones[pid] = True
                rewards[pid] -= cfg.start_mass  # death penalty
                dead_players.append(pid)

        for pid in dead_players:
            self._active_players.discard(pid)

        # ── 5. Spawning maintenance ────────────────────────────────────
        spawn_food(self.food, self._rng, cfg)
        spawn_viruses(self.viruses, self._rng, cfg)

        self._tick += 1

        # ── 6. Diagnostics (no get_state() call — callers request it explicitly)
        info = {
            "tick": self._tick,
            "live_cells": self.cells.count,
            "live_food": self.food.count,
            "live_viruses": self.viruses.count,
            "live_ejected": self.ejected.count,
            "active_players": len(self._active_players),
        }

        return rewards, dones, info

    # ------------------------------------------------------------------
    # State accessor
    # ------------------------------------------------------------------

    def get_state(self) -> GameState:
        """Build and return a :class:`GameState` snapshot of the current world.

        Returns
        -------
        GameState
        """
        cfg = self.config

        c_idx = self.cells.alive_indices()
        f_idx = self.food.alive_indices()
        v_idx = self.viruses.alive_indices()
        e_idx = self.ejected.alive_indices()

        player_alive = np.zeros(cfg.max_players, dtype=bool)
        player_mass = np.zeros(cfg.max_players, dtype=np.float32)
        for pid in self._active_players:
            p_idx = self.cells.player_indices(pid)
            if len(p_idx) > 0:
                player_alive[pid] = True
                player_mass[pid] = float(self.cells.mass[p_idx].sum())

        return GameState(
            tick=self._tick,
            cell_pos=self.cells.pos[c_idx].copy(),
            cell_mass=self.cells.mass[c_idx].copy(),
            cell_owner=self.cells.owner[c_idx].copy(),
            food_pos=self.food.pos[f_idx].copy() if len(f_idx) > 0 else np.empty((0, 2), dtype=np.float32),
            virus_pos=self.viruses.pos[v_idx].copy() if len(v_idx) > 0 else np.empty((0, 2), dtype=np.float32),
            ejected_pos=self.ejected.pos[e_idx].copy() if len(e_idx) > 0 else np.empty((0, 2), dtype=np.float32),
            player_alive=player_alive,
            player_mass=player_mass,
        )
