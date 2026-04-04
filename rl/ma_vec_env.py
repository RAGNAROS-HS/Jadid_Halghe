from __future__ import annotations

import numpy as np
from gymnasium import spaces

from game.config import WorldConfig
from game.world import GameState, World
from rl.agent import ACT_DIM
from rl.env import OBS_DIM, build_observation


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _centroid(state: GameState, player_id: int, cfg: WorldConfig) -> np.ndarray:
    """Mass-weighted centroid of *player_id*'s cells, falling back to world centre."""
    mask = state.cell_owner == player_id
    if not mask.any():
        return np.array([cfg.width / 2.0, cfg.height / 2.0], dtype=np.float32)
    pos = state.cell_pos[mask]
    mass = state.cell_mass[mask]
    total = float(mass.sum())
    if total <= 0:
        return pos.mean(axis=0)
    return (pos * mass[:, None]).sum(axis=0) / total


# ---------------------------------------------------------------------------
# Multi-agent vectorised environment
# ---------------------------------------------------------------------------

class VecAgarMAEnv:
    """Vectorised multi-agent env: N worlds × M RL-controlled agents.

    All agents share a single policy (parameter sharing).  The Runner and
    :class:`~rl.buffer.RolloutBuffer` see ``n_envs * n_agents`` independent
    streams and require no modification.

    Each agent auto-respawns immediately on death so that every slot always
    produces a valid observation.  The death reward (``-1.0``) is already
    included in the step reward from :class:`~game.world.World`.  A world
    resets after *max_ticks* ticks (truncation).

    Parameters
    ----------
    n_envs : int
        Number of parallel world instances.
    n_agents : int
        Number of RL-controlled agents per world.
    config : WorldConfig, optional
    max_ticks : int
        Ticks per episode before the world resets.
    """

    def __init__(
        self,
        n_envs: int,
        n_agents: int,
        config: WorldConfig | None = None,
        max_ticks: int = 2000,
        reward_scale: float | None = None,
        survival_bonus: float = 0.0,
    ) -> None:
        if n_envs < 1:
            raise ValueError(f"n_envs must be >= 1, got {n_envs}")
        if n_agents < 1:
            raise ValueError(f"n_agents must be >= 1, got {n_agents}")

        self.n_envs = n_envs
        self.n_agents = n_agents
        self.num_envs = n_envs * n_agents   # exposed to Runner / buffer
        self.config = config or WorldConfig()
        self.max_ticks = max_ticks

        cfg = self.config
        self._worlds: list[World] = [World(cfg) for _ in range(n_envs)]
        self._ticks: list[int] = [0] * n_envs
        self._seeds: list[int] = list(range(n_envs))
        self._states: list[GameState | None] = [None] * n_envs
        self._agent_ids: list[int] = list(range(n_agents))

        self._pos_scale = float(max(cfg.width, cfg.height)) / 2.0
        self._large_scale = float(max(cfg.width, cfg.height))
        self._reward_scale = reward_scale if reward_scale is not None else float(cfg.start_mass)
        self._survival_bonus = survival_bonus

        self.single_observation_space: spaces.Box = spaces.Box(
            -10.0, 10.0, (OBS_DIM,), dtype=np.float32
        )
        self.single_action_space: spaces.Box = spaces.Box(
            -1.0, 1.0, (ACT_DIM,), dtype=np.float32
        )

        self._obs_buf = np.zeros((self.num_envs, OBS_DIM), dtype=np.float32)
        self._rew_buf = np.zeros(self.num_envs, dtype=np.float32)
        self._term_buf = np.zeros(self.num_envs, dtype=bool)
        self._trunc_buf = np.zeros(self.num_envs, dtype=bool)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _reset_world(self, env_idx: int, seed: int | None = None) -> None:
        world = self._worlds[env_idx]
        s = self._seeds[env_idx] if seed is None else seed
        world.reset(seed=s)
        self._seeds[env_idx] = s + self.n_envs   # different seed next time
        for aid in self._agent_ids:
            world.add_player(aid)
        self._ticks[env_idx] = 0
        self._states[env_idx] = world.get_state()

    def _fill_obs(self, env_idx: int) -> None:
        state = self._states[env_idx]
        cfg = self.config
        for agent_idx, aid in enumerate(self._agent_ids):
            flat_idx = env_idx * self.n_agents + agent_idx
            self._obs_buf[flat_idx] = build_observation(
                state, aid, cfg, self._pos_scale
            )

    # ------------------------------------------------------------------
    # VecEnv API
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: int | None = None,
    ) -> tuple[np.ndarray, list[dict]]:
        """Reset all worlds.

        Parameters
        ----------
        seed : int, optional
            World *i* is seeded with ``seed + i``.

        Returns
        -------
        obs : ndarray, shape (num_envs, obs_dim)
        infos : list[dict], length num_envs
        """
        for i in range(self.n_envs):
            if seed is not None:
                self._seeds[i] = seed + i
            self._reset_world(i)
            self._fill_obs(i)
        return self._obs_buf.copy(), [{} for _ in range(self.num_envs)]

    def step(
        self,
        actions: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict]]:
        """Step all worlds.

        Parameters
        ----------
        actions : ndarray, shape (num_envs, act_dim)
            Tanh-squashed ``(dx, dy, split, eject)`` actions from the policy,
            one row per ``(env, agent)`` slot.

        Returns
        -------
        obs : ndarray, shape (num_envs, obs_dim)
        rewards : ndarray, shape (num_envs,)
        terminated : ndarray, shape (num_envs,)
            True for each agent that died this tick.
        truncated : ndarray, shape (num_envs,)
            True for all agents in a world when *max_ticks* is reached.
        infos : list[dict], length num_envs
        """
        self._rew_buf[:] = 0.0
        self._term_buf[:] = False
        self._trunc_buf[:] = False
        infos: list[dict] = [{} for _ in range(self.num_envs)]

        cfg = self.config

        for env_idx in range(self.n_envs):
            world = self._worlds[env_idx]
            state = self._states[env_idx]

            # ── Build world-space actions ──────────────────────────────
            world_actions = np.zeros((cfg.max_players, 4), dtype=np.float32)
            for agent_idx, aid in enumerate(self._agent_ids):
                flat_idx = env_idx * self.n_agents + agent_idx
                act = actions[flat_idx]
                if aid in world._active_players:
                    c = _centroid(state, aid, cfg)
                    world_actions[aid, 0] = c[0] + act[0] * self._large_scale
                    world_actions[aid, 1] = c[1] + act[1] * self._large_scale
                    world_actions[aid, 2] = 1.0 if act[2] > 0.0 else 0.0
                    world_actions[aid, 3] = 1.0 if act[3] > 0.0 else 0.0

            rewards, dones, _ = world.step(world_actions)
            self._ticks[env_idx] += 1
            truncated = self._ticks[env_idx] >= self.max_ticks

            # ── Respawn dead agents (keeps them active next tick) ──────
            if not truncated:
                for aid in self._agent_ids:
                    if dones[aid]:
                        world.add_player(aid)

            new_state = world.get_state()

            if truncated:
                for agent_idx, aid in enumerate(self._agent_ids):
                    flat_idx = env_idx * self.n_agents + agent_idx
                    infos[flat_idx]["final_observation"] = build_observation(
                        new_state, aid, cfg, self._pos_scale
                    )
                self._reset_world(env_idx)
                new_state = self._states[env_idx]
            else:
                self._states[env_idx] = new_state

            # ── Fill output buffers ────────────────────────────────────
            for agent_idx, aid in enumerate(self._agent_ids):
                flat_idx = env_idx * self.n_agents + agent_idx
                self._obs_buf[flat_idx] = build_observation(
                    new_state, aid, cfg, self._pos_scale
                )
                raw_rew = float(rewards[aid]) / self._reward_scale
                if not dones[aid]:
                    raw_rew += self._survival_bonus
                self._rew_buf[flat_idx] = raw_rew
                self._term_buf[flat_idx] = bool(dones[aid])
                self._trunc_buf[flat_idx] = truncated

        return (
            self._obs_buf.copy(),
            self._rew_buf.copy(),
            self._term_buf.copy(),
            self._trunc_buf.copy(),
            infos,
        )

    def close(self) -> None:
        """No-op (no external resources)."""
