from __future__ import annotations

import functools
from typing import Any

import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv

from game.config import WorldConfig
from game.world import World
from rl.env import OBS_DIM, build_observation


# ---------------------------------------------------------------------------
# Multi-agent parallel environment (PettingZoo)
# ---------------------------------------------------------------------------

class AgarParallelEnv(ParallelEnv):
    """PettingZoo :class:`ParallelEnv` wrapper for the agar.io simulation.

    All ``n_agents`` player slots are RL-controlled.  When an agent dies
    it is removed from :attr:`agents`; no automatic respawn is performed
    (terminated agents must be handled by the training loop).

    Observation and action spaces are identical to :class:`~rl.env.AgarEnv`.
    See :func:`~rl.env.build_observation` for the observation layout.

    Parameters
    ----------
    config : WorldConfig, optional
        Simulation config.
    n_agents : int
        Number of concurrent RL agents (capped at ``max_players``).
    max_ticks : int
        Episode truncation length.
    reward_scale : float, optional
        Divides raw rewards.  Defaults to ``config.start_mass``.
    """

    metadata: dict[str, Any] = {"name": "agar_v0", "render_modes": []}

    def __init__(
        self,
        config: WorldConfig | None = None,
        n_agents: int = 8,
        max_ticks: int = 2000,
        reward_scale: float | None = None,
    ) -> None:
        self.config = config or WorldConfig()
        cfg = self.config
        self._n_agents = min(n_agents, cfg.max_players)
        self.possible_agents: list[str] = [
            f"agent_{i}" for i in range(self._n_agents)
        ]
        self.max_ticks = max_ticks
        self._reward_scale = (
            reward_scale if reward_scale is not None else float(cfg.start_mass)
        )
        self._world = World(cfg)
        self._tick: int = 0
        self._agent_to_id: dict[str, int] = {
            a: i for i, a in enumerate(self.possible_agents)
        }
        self._pos_scale: float = max(cfg.width, cfg.height) / 2.0

        obs_bound = np.full(OBS_DIM, 10.0, dtype=np.float32)
        self._obs_space = spaces.Box(-obs_bound, obs_bound, dtype=np.float32)
        self._act_space = spaces.Box(
            low=np.full(4, -1.0, dtype=np.float32),
            high=np.full(4, 1.0, dtype=np.float32),
            dtype=np.float32,
        )

        # PettingZoo requires this to be initialised before reset
        self.agents: list[str] = []

    # ------------------------------------------------------------------
    # PettingZoo space accessors (cached per PettingZoo convention)
    # ------------------------------------------------------------------

    @functools.cache
    def observation_space(self, agent: str) -> spaces.Box:  # noqa: ARG002
        """Return observation space (identical for all agents)."""
        return self._obs_space

    @functools.cache
    def action_space(self, agent: str) -> spaces.Box:  # noqa: ARG002
        """Return action space (identical for all agents)."""
        return self._act_space

    # ------------------------------------------------------------------
    # PettingZoo API
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, dict]]:
        """Reset the world and return initial observations for all agents.

        Parameters
        ----------
        seed : int, optional
        options : dict, optional
            Unused; present for API compatibility.

        Returns
        -------
        obs : dict[agent_name → ndarray shape (OBS_DIM,)]
        infos : dict[agent_name → dict]
        """
        self._world.reset(seed=seed)
        self._tick = 0
        self.agents = list(self.possible_agents)
        for agent in self.agents:
            self._world.add_player(self._agent_to_id[agent])

        state = self._world.get_state()
        obs = {
            agent: build_observation(
                state, self._agent_to_id[agent], self.config, self._pos_scale
            )
            for agent in self.agents
        }
        infos: dict[str, dict] = {agent: {} for agent in self.agents}
        return obs, infos

    def step(
        self,
        actions: dict[str, np.ndarray],
    ) -> tuple[
        dict[str, np.ndarray],
        dict[str, float],
        dict[str, bool],
        dict[str, bool],
        dict[str, dict],
    ]:
        """Advance one simulation tick for all live agents.

        Parameters
        ----------
        actions : dict[agent_name → ndarray shape (4,)]
            Only live agents (in :attr:`agents`) need be present.

        Returns
        -------
        obs, rewards, terminations, truncations, infos
            Each is a dict keyed by agent name, covering all agents that
            were alive at the *start* of this step.
        """
        cfg = self.config
        world_actions = np.zeros((cfg.max_players, 4), dtype=np.float32)
        large_scale = float(max(cfg.width, cfg.height))
        step_agents = list(self.agents)  # snapshot before modifications

        for agent in step_agents:
            act = actions.get(agent)
            if act is None:
                continue
            pid = self._agent_to_id[agent]
            centroid = self._centroid(pid)
            world_actions[pid, 0] = centroid[0] + float(act[0]) * large_scale
            world_actions[pid, 1] = centroid[1] + float(act[1]) * large_scale
            world_actions[pid, 2] = 1.0 if act[2] > 0.0 else 0.0
            world_actions[pid, 3] = 1.0 if act[3] > 0.0 else 0.0

        raw_rewards, raw_dones, info = self._world.step(world_actions)
        self._tick += 1
        truncated_all = self._tick >= self.max_ticks

        state = self._world.get_state()
        obs: dict[str, np.ndarray] = {}
        rewards: dict[str, float] = {}
        terminations: dict[str, bool] = {}
        truncations: dict[str, bool] = {}
        infos: dict[str, dict] = {}

        for agent in step_agents:
            pid = self._agent_to_id[agent]
            obs[agent] = build_observation(
                state, pid, self.config, self._pos_scale
            )
            rewards[agent] = float(raw_rewards[pid]) / self._reward_scale
            terminations[agent] = bool(raw_dones[pid])
            truncations[agent] = truncated_all
            infos[agent] = {"tick": self._tick, **info}

        # Remove dead/truncated agents from the live list
        self.agents = [
            a for a in step_agents
            if not terminations[a] and not truncations[a]
        ]

        return obs, rewards, terminations, truncations, infos

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _centroid(self, player_id: int) -> np.ndarray:
        """World-space centroid of *player_id*'s live cells."""
        idx = self._world.cells.player_indices(player_id)
        if len(idx) == 0:
            cfg = self.config
            return np.array([cfg.width / 2.0, cfg.height / 2.0], dtype=np.float32)
        return self._world.cells.pos[idx].mean(axis=0)
