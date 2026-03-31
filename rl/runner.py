from __future__ import annotations

from typing import Union

import numpy as np
import torch
from torch import Tensor

from rl.agent import AttentionPolicy, MLPPolicy
from rl.buffer import RolloutBuffer
from rl.vec_env import VecAgarEnv


_Policy = Union[MLPPolicy, AttentionPolicy]


class Runner:
    """Collects fixed-length rollouts from a vectorised environment.

    Each call to :meth:`collect` steps the environment for ``n_steps``
    ticks across all ``n_envs`` parallel instances and returns a filled
    :class:`~rl.buffer.RolloutBuffer` ready for a PPO update.

    The runner is stateful: it carries the current observation between
    calls so that rollouts are contiguous across PPO updates.

    Parameters
    ----------
    venv : VecAgarEnv
        Vectorised environment.
    policy : MLPPolicy or AttentionPolicy
        Policy to query for actions and values.
    n_steps : int
        Steps per environment per rollout.
    device : torch.device
        Device used for tensor operations.
    gamma : float
        Discount factor (passed to GAE).
    gae_lambda : float
        GAE smoothing parameter (passed to GAE).
    """

    def __init__(
        self,
        venv: VecAgarEnv,
        policy: _Policy,
        n_steps: int,
        device: torch.device,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ) -> None:
        self.venv = venv
        self.policy = policy
        self.n_steps = n_steps
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda

        obs_dim = venv.single_observation_space.shape[0]
        act_dim = venv.single_action_space.shape[0]
        self.buffer = RolloutBuffer(
            n_steps=n_steps,
            n_envs=venv.num_envs,
            obs_dim=obs_dim,
            act_dim=act_dim,
            device=device,
        )

        # Warm-start: reset all envs and store initial obs
        obs_np, _ = venv.reset()
        self._obs: Tensor = torch.from_numpy(obs_np).float().to(device)
        self._dones: Tensor = torch.zeros(venv.num_envs, dtype=torch.bool, device=device)

    def collect(self) -> tuple[RolloutBuffer, dict[str, float]]:
        """Collect ``n_steps`` steps and compute GAE.

        Returns
        -------
        buffer : RolloutBuffer
            Filled buffer with ``returns`` and ``advantages`` computed.
        info : dict[str, float]
            Episode statistics aggregated over this rollout:
            ``mean_reward``, ``mean_episode_length``, ``n_episodes``.
        """
        self.buffer.reset()
        self.policy.eval()

        ep_rewards: list[float] = []
        ep_lengths: list[int] = []
        # Running episode accumulators per env
        running_rew = np.zeros(self.venv.num_envs, dtype=np.float32)
        running_len = np.zeros(self.venv.num_envs, dtype=np.int32)

        with torch.no_grad():
            for _ in range(self.n_steps):
                z, log_prob, value = self.policy.act(self._obs)

                # Convert pre-tanh sample to squashed action for the env
                env_action = torch.tanh(z).cpu().numpy().astype(np.float32)

                next_obs_np, rewards_np, terminated_np, truncated_np, infos = self.venv.step(env_action)

                dones_np = terminated_np | truncated_np
                rewards_t = torch.from_numpy(rewards_np).float().to(self.device)
                dones_t = torch.from_numpy(dones_np).to(self.device)

                self.buffer.add(self._obs, z, log_prob, rewards_t, value, dones_t)

                running_rew += rewards_np
                running_len += 1
                for i in range(self.venv.num_envs):
                    if dones_np[i]:
                        ep_rewards.append(float(running_rew[i]))
                        ep_lengths.append(int(running_len[i]))
                        running_rew[i] = 0.0
                        running_len[i] = 0

                self._obs = torch.from_numpy(next_obs_np).float().to(self.device)
                self._dones = dones_t

            # Bootstrap value for the last step
            _, _, last_value = self.policy.act(self._obs)

        self.buffer.compute_returns_and_advantages(
            last_values=last_value,
            last_dones=self._dones,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
        )

        n_ep = len(ep_rewards)
        rollout_info: dict[str, float] = {
            "mean_reward": float(np.mean(ep_rewards)) if ep_rewards else float("nan"),
            "mean_episode_length": float(np.mean(ep_lengths)) if ep_lengths else float("nan"),
            "n_episodes": float(n_ep),
        }
        return self.buffer, rollout_info
