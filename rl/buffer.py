from __future__ import annotations

import torch
from torch import Tensor


class RolloutBuffer:
    """Fixed-length rollout buffer for on-policy algorithms.

    Stores ``n_steps × n_envs`` transitions collected by a vectorised
    environment.  After collection, :meth:`compute_returns_and_advantages`
    fills the ``returns`` and ``advantages`` arrays using Generalised
    Advantage Estimation (GAE).

    Actions are stored as **pre-tanh samples** ``z`` (not squashed).  The
    caller must pass ``tanh(z)`` to the environment and keep ``z`` here so
    that :meth:`~rl.ppo.PPO` can recompute log-probabilities without an
    expensive ``atanh`` inversion.

    Parameters
    ----------
    n_steps : int
        Rollout length per environment.
    n_envs : int
        Number of parallel environments.
    obs_dim : int
        Flat observation dimension.
    act_dim : int
        Action dimension.
    device : torch.device
        All tensors are allocated on this device.
    """

    def __init__(
        self,
        n_steps: int,
        n_envs: int,
        obs_dim: int,
        act_dim: int,
        device: torch.device,
    ) -> None:
        self.n_steps = n_steps
        self.n_envs = n_envs
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.device = device
        self._ptr: int = 0
        self._full: bool = False

        # Core storage — pre-allocated, never re-allocated during training
        self.obs = torch.zeros(n_steps, n_envs, obs_dim, device=device)
        self.actions = torch.zeros(n_steps, n_envs, act_dim, device=device)  # pre-tanh z
        self.log_probs = torch.zeros(n_steps, n_envs, device=device)
        self.rewards = torch.zeros(n_steps, n_envs, device=device)
        self.values = torch.zeros(n_steps, n_envs, device=device)
        self.dones = torch.zeros(n_steps, n_envs, dtype=torch.bool, device=device)

        # Filled by compute_returns_and_advantages
        self.returns = torch.zeros(n_steps, n_envs, device=device)
        self.advantages = torch.zeros(n_steps, n_envs, device=device)

    @property
    def full(self) -> bool:
        """True once ``n_steps`` transitions have been added since the last reset."""
        return self._full

    def reset(self) -> None:
        """Clear the buffer (pointer back to 0, not-full)."""
        self._ptr = 0
        self._full = False

    def add(
        self,
        obs: Tensor,
        actions: Tensor,
        log_probs: Tensor,
        rewards: Tensor,
        values: Tensor,
        dones: Tensor,
    ) -> None:
        """Store one step of experience from all environments.

        Parameters
        ----------
        obs : Tensor, shape (n_envs, obs_dim)
        actions : Tensor, shape (n_envs, act_dim)
            Pre-tanh samples ``z`` (not squashed actions).
        log_probs : Tensor, shape (n_envs,)
        rewards : Tensor, shape (n_envs,)
        values : Tensor, shape (n_envs,)
        dones : Tensor, shape (n_envs,), bool
            ``terminated | truncated`` flags.
        """
        t = self._ptr
        self.obs[t].copy_(obs)
        self.actions[t].copy_(actions)
        self.log_probs[t].copy_(log_probs)
        self.rewards[t].copy_(rewards)
        self.values[t].copy_(values)
        self.dones[t].copy_(dones)
        self._ptr += 1
        if self._ptr >= self.n_steps:
            self._full = True

    def compute_returns_and_advantages(
        self,
        last_values: Tensor,
        last_dones: Tensor,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ) -> None:
        """Fill ``returns`` and ``advantages`` using GAE.

        Parameters
        ----------
        last_values : Tensor, shape (n_envs,)
            Critic estimate for the observation *after* the last stored step.
        last_dones : Tensor, shape (n_envs,), bool
            Done flags for the step *after* the last stored step.
        gamma : float
            Discount factor.
        gae_lambda : float
            GAE smoothing parameter (λ).

        Notes
        -----
        ``done`` is treated as ``terminated | truncated``.  For truncated
        episodes the bootstrap is slightly underestimated (the reset obs
        value is used instead of the true terminal obs value), but in
        practice with long episodes (≥ 2 000 ticks) this bias is negligible.
        """
        T = self.n_steps
        last_gae = torch.zeros(self.n_envs, device=self.device)
        not_done_last = (~last_dones).float()

        for t in reversed(range(T)):
            if t == T - 1:
                next_value = last_values
                not_done = not_done_last
            else:
                next_value = self.values[t + 1]
                not_done = (~self.dones[t + 1]).float()

            delta = self.rewards[t] + gamma * not_done * next_value - self.values[t]
            last_gae = delta + gamma * gae_lambda * not_done * last_gae
            self.advantages[t] = last_gae

        self.returns = self.advantages + self.values

    def get_batches(
        self,
        batch_size: int,
        shuffle: bool = True,
    ):
        """Yield minibatches of flattened (T × N) transitions.

        Each minibatch is a tuple of Tensors on ``self.device``.

        Parameters
        ----------
        batch_size : int
            Number of transitions per minibatch.
        shuffle : bool
            If True, permute the data before batching.

        Yields
        ------
        tuple of Tensors
            ``(obs, actions, old_log_probs, returns, advantages)``,
            each shape ``(batch_size, ...)``.
        """
        total = self.n_steps * self.n_envs
        obs_flat = self.obs.view(total, self.obs_dim)
        act_flat = self.actions.view(total, self.act_dim)
        lp_flat = self.log_probs.view(total)
        ret_flat = self.returns.view(total)
        adv_flat = self.advantages.view(total)

        # Normalise advantages in-place over this rollout
        adv_flat = (adv_flat - adv_flat.mean()) / (adv_flat.std() + 1e-8)

        indices = torch.randperm(total, device=self.device) if shuffle else torch.arange(total, device=self.device)
        for start in range(0, total, batch_size):
            idx = indices[start: start + batch_size]
            yield (
                obs_flat[idx],
                act_flat[idx],
                lp_flat[idx],
                ret_flat[idx],
                adv_flat[idx],
            )
