from __future__ import annotations

from typing import Union

import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import Optimizer

from rl.agent import AttentionPolicy, MLPPolicy
from rl.buffer import RolloutBuffer


# Policy union used in type hints — RecurrentPolicy is excluded because it
# requires a different training loop (hidden-state management).
_Policy = Union[MLPPolicy, AttentionPolicy]


class PPO:
    """Proximal Policy Optimisation (clipped surrogate objective).

    Implements:

    * Clipped policy loss: ``L_clip = E[min(r·A, clip(r, 1±ε)·A)]``
    * Clipped value loss: ``L_value = 0.5 · E[max(MSE, MSE_clipped)]``
    * Entropy bonus: ``L_entropy = −H[π]``
    * Total: ``L = L_clip + c_v · L_value + c_e · L_entropy``

    Parameters
    ----------
    policy : MLPPolicy or AttentionPolicy
    optimizer : Optimizer
    clip_range : float
        PPO clipping parameter ε.
    value_coef : float
        Weight for the value loss term.
    entropy_coef : float
        Weight for the entropy bonus (negative loss).
    max_grad_norm : float
        Gradient-norm clipping threshold.
    """

    def __init__(
        self,
        policy: _Policy,
        optimizer: Optimizer,
        clip_range: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
    ) -> None:
        self.policy = policy
        self.optimizer = optimizer
        self.clip_range = clip_range
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm

    def update(
        self,
        buffer: RolloutBuffer,
        n_epochs: int,
        batch_size: int,
    ) -> dict[str, float]:
        """Run ``n_epochs`` of PPO gradient updates on the rollout buffer.

        Parameters
        ----------
        buffer : RolloutBuffer
            Must have had :meth:`~RolloutBuffer.compute_returns_and_advantages`
            called before this method.
        n_epochs : int
            Number of passes over the buffer data.
        batch_size : int
            Minibatch size.

        Returns
        -------
        dict[str, float]
            Averaged metrics over all minibatches:
            ``policy_loss``, ``value_loss``, ``entropy``, ``approx_kl``,
            ``clip_fraction``, ``total_loss``.
        """
        accum: dict[str, float] = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "approx_kl": 0.0,
            "clip_fraction": 0.0,
            "total_loss": 0.0,
        }
        n_batches = 0

        for _ in range(n_epochs):
            for obs, z_actions, old_log_probs, returns, advantages in buffer.get_batches(batch_size):
                new_log_probs, values, entropy = self.policy.evaluate(obs, z_actions)

                # ── Policy loss ──────────────────────────────────────────
                ratio = torch.exp(new_log_probs - old_log_probs)
                clipped = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range)
                policy_loss = -torch.min(ratio * advantages, clipped * advantages).mean()

                # ── Value loss (clipped) ─────────────────────────────────
                # old_values aren't stored, so use unclipped MSE
                value_loss = 0.5 * ((values - returns) ** 2).mean()

                # ── Entropy bonus ────────────────────────────────────────
                # entropy is already the mean over the batch (from evaluate)

                total_loss = (
                    policy_loss
                    + self.value_coef * value_loss
                    - self.entropy_coef * entropy
                )

                self.optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

                # ── Diagnostics ──────────────────────────────────────────
                with torch.no_grad():
                    approx_kl = 0.5 * ((old_log_probs - new_log_probs) ** 2).mean()
                    clip_frac = ((ratio - 1.0).abs() > self.clip_range).float().mean()

                accum["policy_loss"] += policy_loss.item()
                accum["value_loss"] += value_loss.item()
                accum["entropy"] += entropy.item()
                accum["approx_kl"] += approx_kl.item()
                accum["clip_fraction"] += clip_frac.item()
                accum["total_loss"] += total_loss.item()
                n_batches += 1

        n = max(n_batches, 1)
        return {k: v / n for k, v in accum.items()}
