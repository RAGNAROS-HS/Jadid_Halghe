from __future__ import annotations

from typing import Callable

import numpy as np
from gymnasium import spaces

from game.config import WorldConfig
from rl.env import AgarEnv


# ---------------------------------------------------------------------------
# Synchronous vectorised environment
# ---------------------------------------------------------------------------

class VecAgarEnv:
    """Synchronous vectorised wrapper over *n_envs* independent :class:`AgarEnv` instances.

    Environments are stepped in sequence (no multiprocessing).  Each
    environment auto-resets when an episode ends, storing the terminal
    observation in ``info["final_observation"]`` (standard PPO convention).

    Pre-allocated NumPy buffers avoid per-step heap allocation.

    Parameters
    ----------
    n_envs : int
        Number of parallel environment instances.
    **env_kwargs
        Forwarded verbatim to :class:`AgarEnv`.

    Attributes
    ----------
    num_envs : int
    single_observation_space : spaces.Box
    single_action_space : spaces.Box
    """

    def __init__(self, n_envs: int, **env_kwargs) -> None:
        if n_envs < 1:
            raise ValueError(f"n_envs must be >= 1, got {n_envs}")
        self.num_envs = n_envs
        self._envs: list[AgarEnv] = [AgarEnv(**env_kwargs) for _ in range(n_envs)]
        self.single_observation_space: spaces.Box = self._envs[0].observation_space
        self.single_action_space: spaces.Box = self._envs[0].action_space

        obs_dim: int = int(self.single_observation_space.shape[0])
        act_dim: int = int(self.single_action_space.shape[0])

        # Pre-allocated output buffers
        self._obs_buf = np.zeros((n_envs, obs_dim), dtype=np.float32)
        self._rew_buf = np.zeros(n_envs, dtype=np.float32)
        self._term_buf = np.zeros(n_envs, dtype=bool)
        self._trunc_buf = np.zeros(n_envs, dtype=bool)
        self._act_buf = np.zeros((n_envs, act_dim), dtype=np.float32)

    # ------------------------------------------------------------------
    # Batch spaces (tiled single spaces)
    # ------------------------------------------------------------------

    @property
    def observation_space(self) -> spaces.Box:
        """Batched observation space, shape ``(n_envs, obs_dim)``."""
        low = np.tile(self.single_observation_space.low, (self.num_envs, 1))
        high = np.tile(self.single_observation_space.high, (self.num_envs, 1))
        return spaces.Box(low, high, dtype=np.float32)

    @property
    def action_space(self) -> spaces.Box:
        """Batched action space, shape ``(n_envs, act_dim)``."""
        low = np.tile(self.single_action_space.low, (self.num_envs, 1))
        high = np.tile(self.single_action_space.high, (self.num_envs, 1))
        return spaces.Box(low, high, dtype=np.float32)

    # ------------------------------------------------------------------
    # VecEnv API
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: int | None = None,
    ) -> tuple[np.ndarray, list[dict]]:
        """Reset all environments.

        Parameters
        ----------
        seed : int, optional
            If provided, env *i* is seeded with ``seed + i``.

        Returns
        -------
        obs : ndarray, shape (n_envs, obs_dim)
        infos : list[dict], length n_envs
        """
        infos: list[dict] = []
        for i, env in enumerate(self._envs):
            env_seed = None if seed is None else seed + i
            obs, info = env.reset(seed=env_seed)
            self._obs_buf[i] = obs
            infos.append(info)
        return self._obs_buf.copy(), infos

    def step(
        self,
        actions: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict]]:
        """Step all environments with the given batch of actions.

        Episodes that end (terminated or truncated) are automatically reset.
        The observation returned for a done environment is the *first*
        observation of the new episode; the terminal observation is stored
        in ``info["final_observation"]``.

        Parameters
        ----------
        actions : ndarray, shape (n_envs, act_dim)

        Returns
        -------
        obs : ndarray, shape (n_envs, obs_dim)
        rewards : ndarray, shape (n_envs,), float32
        terminated : ndarray, shape (n_envs,), bool
        truncated : ndarray, shape (n_envs,), bool
        infos : list[dict], length n_envs
        """
        infos: list[dict] = []
        for i, env in enumerate(self._envs):
            obs, reward, terminated, truncated, info = env.step(actions[i])
            if terminated or truncated:
                info["final_observation"] = obs.copy()
                obs, _ = env.reset()
            self._obs_buf[i] = obs
            self._rew_buf[i] = reward
            self._term_buf[i] = terminated
            self._trunc_buf[i] = truncated
            infos.append(info)
        return (
            self._obs_buf.copy(),
            self._rew_buf.copy(),
            self._term_buf.copy(),
            self._trunc_buf.copy(),
            infos,
        )

    def set_bot_policy(
        self,
        policy: Callable[[np.ndarray], np.ndarray] | None,
    ) -> None:
        """Propagate a new bot policy to all underlying :class:`AgarEnv` instances.

        Parameters
        ----------
        policy : Callable[[np.ndarray], np.ndarray] | None
            New bot policy, or ``None`` to revert to the random-walk baseline.
        """
        for env in self._envs:
            env.set_bot_policy(policy)

    def close(self) -> None:
        """No-op (no external resources to release)."""
