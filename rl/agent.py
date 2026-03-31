from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import Normal

from rl.env import K_OWN, K_FOOD, K_VIRUS, K_ENEMY, OBS_DIM


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ACT_DIM: int = 4

# Obs slice offsets — must stay in sync with rl/env.build_observation
_OWN_S: int = 0
_OWN_E: int = K_OWN * 3           # 48
_FOOD_S: int = _OWN_E              # 48
_FOOD_E: int = _FOOD_S + K_FOOD * 2   # 88
_VIR_S: int = _FOOD_E              # 88
_VIR_E: int = _VIR_S + K_VIRUS * 2   # 108
_ENM_S: int = _VIR_E               # 108
_ENM_E: int = _ENM_S + K_ENEMY * 3   # 168
_SCL_S: int = _ENM_E               # 168
_SCL_E: int = OBS_DIM              # 170

LOG_STD_MIN: float = -4.0
LOG_STD_MAX: float = 1.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _init_weights(module: nn.Module) -> None:
    """Apply orthogonal initialisation to all Linear layers."""
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain=math.sqrt(2))
        nn.init.zeros_(module.bias)


def _tanh_log_prob(z: Tensor, dist: Normal) -> Tensor:
    """Log-probability of a tanh-squashed action with Jacobian correction.

    Uses the numerically stable identity
    ``log(1 − tanh²(z)) = 2·(log 2 − z − softplus(−2z))``.

    Parameters
    ----------
    z : Tensor, shape (B, act_dim)
        Pre-tanh sample.
    dist : Normal
        Distribution from which *z* was sampled.

    Returns
    -------
    Tensor, shape (B,)
        ``Σᵢ [log p(zᵢ) − log(1 − tanh²(zᵢ))]``.
    """
    log_prob = dist.log_prob(z).sum(-1)
    correction = 2.0 * (math.log(2.0) - z - F.softplus(-2.0 * z))
    return log_prob - correction.sum(-1)


def _to_tensor(obs: np.ndarray | Tensor, device: torch.device) -> Tensor:
    """Convert obs to float32 tensor on *device*."""
    if isinstance(obs, np.ndarray):
        return torch.from_numpy(obs).float().to(device)
    return obs.float().to(device)


# ---------------------------------------------------------------------------
# MLP policy
# ---------------------------------------------------------------------------

class MLPPolicy(nn.Module):
    """Three-layer MLP actor-critic with tanh-squashed Gaussian actions.

    The encoder maps the flat observation to a shared feature vector.
    A separate actor head outputs the action mean; a scalar ``log_std``
    parameter (shared across actions, *not* input-dependent) controls
    exploration. The critic head outputs a scalar state value.

    Parameters
    ----------
    obs_dim : int
        Flat observation dimension (default: ``OBS_DIM`` = 170).
    act_dim : int
        Action dimension (default: ``ACT_DIM`` = 4).
    hidden_dim : int
        Width of each hidden layer.
    """

    def __init__(
        self,
        obs_dim: int = OBS_DIM,
        act_dim: int = ACT_DIM,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden_dim = hidden_dim

        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
        )
        self.actor_mean = nn.Linear(hidden_dim // 2, act_dim)
        self.log_std = nn.Parameter(torch.zeros(act_dim))
        self.critic = nn.Linear(hidden_dim // 2, 1)

        self.apply(_init_weights)
        # Smaller gain for output layers (standard PPO practice)
        nn.init.orthogonal_(self.actor_mean.weight, gain=0.01)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)

    @property
    def device(self) -> torch.device:
        """Device this module lives on."""
        return next(self.parameters()).device

    def _features(self, obs: Tensor) -> Tensor:
        return self.net(obs)

    def act(
        self,
        obs: np.ndarray | Tensor,
        deterministic: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Sample an action and return it with log-probability and value.

        Parameters
        ----------
        obs : ndarray or Tensor, shape (B, obs_dim) or (obs_dim,)
            Current observations.
        deterministic : bool
            If True, return the mean of the distribution (no noise).

        Returns
        -------
        z : Tensor, shape (B, act_dim)
            Pre-tanh action sample (store this in the rollout buffer).
        log_prob : Tensor, shape (B,)
            Log-probability of the tanh-squashed action.
        value : Tensor, shape (B,)
            Critic state-value estimate.
        """
        obs_t = _to_tensor(obs, self.device)
        if obs_t.ndim == 1:
            obs_t = obs_t.unsqueeze(0)

        feat = self._features(obs_t)
        mean = self.actor_mean(feat)
        log_std = self.log_std.clamp(LOG_STD_MIN, LOG_STD_MAX)
        std = log_std.exp()
        dist = Normal(mean, std)

        z = mean if deterministic else dist.rsample()
        log_prob = _tanh_log_prob(z, dist)
        value = self.critic(feat).squeeze(-1)
        return z, log_prob, value

    def evaluate(
        self,
        obs: Tensor,
        z_actions: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Evaluate log-probabilities and value for stored rollout data.

        Parameters
        ----------
        obs : Tensor, shape (B, obs_dim)
        z_actions : Tensor, shape (B, act_dim)
            Pre-tanh actions stored in the rollout buffer.

        Returns
        -------
        log_prob : Tensor, shape (B,)
        value : Tensor, shape (B,)
        entropy : Tensor, scalar
            Mean entropy of the action distribution.
        """
        feat = self._features(obs)
        mean = self.actor_mean(feat)
        log_std = self.log_std.clamp(LOG_STD_MIN, LOG_STD_MAX)
        std = log_std.exp()
        dist = Normal(mean, std)

        log_prob = _tanh_log_prob(z_actions, dist)
        value = self.critic(feat).squeeze(-1)
        entropy = dist.entropy().sum(-1).mean()
        return log_prob, value, entropy

    def save(self, path: str | Path, step: int = 0, **meta: Any) -> None:
        """Save policy weights and config to *path*.

        Parameters
        ----------
        path : str or Path
        step : int
            Training step to store for resuming.
        **meta
            Additional key-value pairs stored alongside the checkpoint.
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "type": "MLPPolicy",
                "config": {
                    "obs_dim": self.obs_dim,
                    "act_dim": self.act_dim,
                    "hidden_dim": self.hidden_dim,
                },
                "state_dict": self.state_dict(),
                "step": step,
                **meta,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> tuple[MLPPolicy, dict[str, Any]]:
        """Load a policy from a checkpoint.

        Parameters
        ----------
        path : str or Path

        Returns
        -------
        policy : MLPPolicy
        meta : dict
            All non-config checkpoint keys (``step``, etc.).
        """
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        policy = cls(**ckpt["config"])
        policy.load_state_dict(ckpt["state_dict"])
        meta = {k: v for k, v in ckpt.items() if k not in ("type", "config", "state_dict")}
        return policy, meta


# ---------------------------------------------------------------------------
# Attention policy
# ---------------------------------------------------------------------------

class AttentionPolicy(nn.Module):
    """Transformer-based entity-encoder actor-critic.

    Each entity type (own cells, food, viruses, enemy cells) is projected
    to a common embedding dimension and enriched with a learned type
    embedding.  A transformer encoder then attends over all 66 entity
    slots; zero-padded slots are masked out.  The attended representation
    is mean-pooled (over real tokens only) and concatenated with the two
    global scalar features before being fed to the actor and critic heads.

    Parameters
    ----------
    obs_dim : int
        Flat observation dimension.
    act_dim : int
        Action dimension.
    embed_dim : int
        Entity embedding dimension (transformer model width).
    n_heads : int
        Number of attention heads (must divide *embed_dim*).
    n_layers : int
        Number of transformer encoder layers.
    """

    def __init__(
        self,
        obs_dim: int = OBS_DIM,
        act_dim: int = ACT_DIM,
        embed_dim: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.n_layers = n_layers

        # Per-group input projections
        self.own_proj = nn.Linear(3, embed_dim)
        self.food_proj = nn.Linear(2, embed_dim)
        self.virus_proj = nn.Linear(2, embed_dim)
        self.enemy_proj = nn.Linear(3, embed_dim)

        # Learned type embeddings: 4 types × embed_dim
        self.type_emb = nn.Parameter(torch.zeros(4, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.0,
            batch_first=True,
            norm_first=True,  # Pre-LN: more stable training
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers, enable_nested_tensor=False
        )

        feat_dim = embed_dim + 2  # pooled entities + scalars
        self.actor_mean = nn.Linear(feat_dim, act_dim)
        self.log_std = nn.Parameter(torch.zeros(act_dim))
        self.critic = nn.Linear(feat_dim, 1)

        self.apply(_init_weights)
        nn.init.orthogonal_(self.actor_mean.weight, gain=0.01)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)
        nn.init.zeros_(self.type_emb)

    @property
    def device(self) -> torch.device:
        """Device this module lives on."""
        return next(self.parameters()).device

    def _features(self, obs: Tensor) -> Tensor:
        """Extract entity-pooled features from a batch of flat observations.

        Parameters
        ----------
        obs : Tensor, shape (B, obs_dim)

        Returns
        -------
        Tensor, shape (B, embed_dim + 2)
        """
        B = obs.shape[0]

        own = obs[:, _OWN_S:_OWN_E].view(B, K_OWN, 3)
        food = obs[:, _FOOD_S:_FOOD_E].view(B, K_FOOD, 2)
        virus = obs[:, _VIR_S:_VIR_E].view(B, K_VIRUS, 2)
        enemy = obs[:, _ENM_S:_ENM_E].view(B, K_ENEMY, 3)
        scalars = obs[:, _SCL_S:_SCL_E]  # (B, 2)

        # Project + add type embedding (broadcasts over entity dimension)
        own_emb = self.own_proj(own) + self.type_emb[0]    # (B, K_OWN, embed_dim)
        food_emb = self.food_proj(food) + self.type_emb[1]
        virus_emb = self.virus_proj(virus) + self.type_emb[2]
        enemy_emb = self.enemy_proj(enemy) + self.type_emb[3]

        tokens = torch.cat([own_emb, food_emb, virus_emb, enemy_emb], dim=1)  # (B, 66, embed_dim)

        # Padding mask: True = ignore slot (all features == 0 → padded)
        own_real = own.abs().sum(-1) > 1e-6     # (B, K_OWN)
        food_real = food.abs().sum(-1) > 1e-6
        virus_real = virus.abs().sum(-1) > 1e-6
        enemy_real = enemy.abs().sum(-1) > 1e-6
        is_real = torch.cat([own_real, food_real, virus_real, enemy_real], dim=1)  # (B, 66)
        pad_mask = ~is_real  # (B, 66) — True = ignore

        out = self.transformer(tokens, src_key_padding_mask=pad_mask)  # (B, 66, embed_dim)

        # Safe mean pool over real tokens only
        n_real = is_real.float().sum(-1, keepdim=True).clamp(min=1.0)
        pooled = (out * is_real.unsqueeze(-1).float()).sum(1) / n_real  # (B, embed_dim)

        return torch.cat([pooled, scalars], dim=-1)  # (B, embed_dim + 2)

    def act(
        self,
        obs: np.ndarray | Tensor,
        deterministic: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Sample an action. See :meth:`MLPPolicy.act` for full documentation."""
        obs_t = _to_tensor(obs, self.device)
        if obs_t.ndim == 1:
            obs_t = obs_t.unsqueeze(0)

        feat = self._features(obs_t)
        mean = self.actor_mean(feat)
        log_std = self.log_std.clamp(LOG_STD_MIN, LOG_STD_MAX)
        std = log_std.exp()
        dist = Normal(mean, std)

        z = mean if deterministic else dist.rsample()
        log_prob = _tanh_log_prob(z, dist)
        value = self.critic(feat).squeeze(-1)
        return z, log_prob, value

    def evaluate(
        self,
        obs: Tensor,
        z_actions: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Evaluate stored rollout data. See :meth:`MLPPolicy.evaluate`."""
        feat = self._features(obs)
        mean = self.actor_mean(feat)
        log_std = self.log_std.clamp(LOG_STD_MIN, LOG_STD_MAX)
        std = log_std.exp()
        dist = Normal(mean, std)

        log_prob = _tanh_log_prob(z_actions, dist)
        value = self.critic(feat).squeeze(-1)
        entropy = dist.entropy().sum(-1).mean()
        return log_prob, value, entropy

    def save(self, path: str | Path, step: int = 0, **meta: Any) -> None:
        """Save policy to *path*. See :meth:`MLPPolicy.save`."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "type": "AttentionPolicy",
                "config": {
                    "obs_dim": self.obs_dim,
                    "act_dim": self.act_dim,
                    "embed_dim": self.embed_dim,
                    "n_heads": self.n_heads,
                    "n_layers": self.n_layers,
                },
                "state_dict": self.state_dict(),
                "step": step,
                **meta,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> tuple[AttentionPolicy, dict[str, Any]]:
        """Load an AttentionPolicy from a checkpoint. See :meth:`MLPPolicy.load`."""
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        policy = cls(**ckpt["config"])
        policy.load_state_dict(ckpt["state_dict"])
        meta = {k: v for k, v in ckpt.items() if k not in ("type", "config", "state_dict")}
        return policy, meta


# ---------------------------------------------------------------------------
# Recurrent policy
# ---------------------------------------------------------------------------

class RecurrentPolicy(nn.Module):
    """GRU-wrapped MLP actor-critic for partial observability experiments.

    The MLP encodes the flat observation; the GRU maintains temporal
    context across steps.  The hidden state must be managed by the
    caller: pass it into :meth:`act` at each step and store the returned
    ``new_hidden`` for the next step.

    .. note::

        The provided :class:`~rl.runner.Runner` does **not** manage
        hidden states.  Using this policy requires a custom rollout loop
        that threads ``hidden_state`` through each step.

    Parameters
    ----------
    obs_dim : int
    act_dim : int
    hidden_dim : int
        MLP hidden layer width.
    gru_hidden : int
        GRU hidden state size.
    """

    def __init__(
        self,
        obs_dim: int = OBS_DIM,
        act_dim: int = ACT_DIM,
        hidden_dim: int = 256,
        gru_hidden: int = 128,
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden_dim = hidden_dim
        self.gru_hidden = gru_hidden

        self.feature_net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.gru = nn.GRU(hidden_dim, gru_hidden, batch_first=True)
        self.actor_mean = nn.Linear(gru_hidden, act_dim)
        self.log_std = nn.Parameter(torch.zeros(act_dim))
        self.critic = nn.Linear(gru_hidden, 1)

        self.feature_net.apply(_init_weights)
        nn.init.orthogonal_(self.actor_mean.weight, gain=0.01)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)

    @property
    def device(self) -> torch.device:
        """Device this module lives on."""
        return next(self.parameters()).device

    def initial_state(self, n_envs: int) -> Tensor:
        """Return a zero initial hidden state.

        Parameters
        ----------
        n_envs : int

        Returns
        -------
        Tensor, shape (1, n_envs, gru_hidden)
        """
        return torch.zeros(1, n_envs, self.gru_hidden, device=self.device)

    def act(
        self,
        obs: np.ndarray | Tensor,
        hidden_state: Tensor,
        deterministic: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Sample an action given the current hidden state.

        Parameters
        ----------
        obs : ndarray or Tensor, shape (B, obs_dim) or (obs_dim,)
        hidden_state : Tensor, shape (1, B, gru_hidden)
        deterministic : bool

        Returns
        -------
        z : Tensor, shape (B, act_dim)
            Pre-tanh action sample.
        log_prob : Tensor, shape (B,)
        value : Tensor, shape (B,)
        new_hidden : Tensor, shape (1, B, gru_hidden)
        """
        obs_t = _to_tensor(obs, self.device)
        if obs_t.ndim == 1:
            obs_t = obs_t.unsqueeze(0)

        feat = self.feature_net(obs_t)               # (B, hidden_dim)
        feat_seq = feat.unsqueeze(1)                 # (B, 1, hidden_dim)
        gru_out, new_hidden = self.gru(feat_seq, hidden_state)
        out = gru_out.squeeze(1)                     # (B, gru_hidden)

        mean = self.actor_mean(out)
        log_std = self.log_std.clamp(LOG_STD_MIN, LOG_STD_MAX)
        dist = Normal(mean, log_std.exp())

        z = mean if deterministic else dist.rsample()
        log_prob = _tanh_log_prob(z, dist)
        value = self.critic(out).squeeze(-1)
        return z, log_prob, value, new_hidden

    def evaluate(
        self,
        obs: Tensor,
        z_actions: Tensor,
        hidden_states: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Evaluate a sequence of stored rollout data.

        Parameters
        ----------
        obs : Tensor, shape (T, B, obs_dim) or (B, obs_dim)
        z_actions : Tensor, shape (T, B, act_dim) or (B, act_dim)
        hidden_states : Tensor, shape (1, B, gru_hidden)
            Initial hidden state for the sequence.

        Returns
        -------
        log_prob : Tensor, shape (T*B,) or (B,)
        value : Tensor, shape (T*B,) or (B,)
        entropy : Tensor, scalar
        """
        if obs.ndim == 2:
            obs = obs.unsqueeze(0)
            z_actions = z_actions.unsqueeze(0)

        T, B = obs.shape[:2]
        feat = self.feature_net(obs.reshape(T * B, -1)).view(T, B, -1)
        gru_out, _ = self.gru(feat.transpose(0, 1), hidden_states)  # (B, T, gru_hidden)
        out = gru_out.transpose(0, 1).reshape(T * B, -1)            # (T*B, gru_hidden)

        mean = self.actor_mean(out)
        log_std = self.log_std.clamp(LOG_STD_MIN, LOG_STD_MAX)
        dist = Normal(mean, log_std.exp())

        z_flat = z_actions.reshape(T * B, -1)
        log_prob = _tanh_log_prob(z_flat, dist)
        value = self.critic(out).squeeze(-1)
        entropy = dist.entropy().sum(-1).mean()
        return log_prob, value, entropy

    def save(self, path: str | Path, step: int = 0, **meta: Any) -> None:
        """Save policy to *path*."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "type": "RecurrentPolicy",
                "config": {
                    "obs_dim": self.obs_dim,
                    "act_dim": self.act_dim,
                    "hidden_dim": self.hidden_dim,
                    "gru_hidden": self.gru_hidden,
                },
                "state_dict": self.state_dict(),
                "step": step,
                **meta,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> tuple[RecurrentPolicy, dict[str, Any]]:
        """Load a RecurrentPolicy from a checkpoint."""
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        policy = cls(**ckpt["config"])
        policy.load_state_dict(ckpt["state_dict"])
        meta = {k: v for k, v in ckpt.items() if k not in ("type", "config", "state_dict")}
        return policy, meta


# ---------------------------------------------------------------------------
# Policy factory
# ---------------------------------------------------------------------------

def build_policy(
    policy_type: str,
    obs_dim: int = OBS_DIM,
    act_dim: int = ACT_DIM,
    **kwargs: Any,  # forwarded to the specific policy class
) -> MLPPolicy | AttentionPolicy | RecurrentPolicy:
    """Construct a policy by name.

    Parameters
    ----------
    policy_type : str
        One of ``"mlp"``, ``"attention"``, ``"recurrent"``.
    obs_dim, act_dim : int
    **kwargs
        Passed to the policy constructor (e.g. ``hidden_dim``, ``embed_dim``).

    Returns
    -------
    MLPPolicy | AttentionPolicy | RecurrentPolicy
    """
    mapping: dict[str, type] = {
        "mlp": MLPPolicy,
        "attention": AttentionPolicy,
        "recurrent": RecurrentPolicy,
    }
    if policy_type not in mapping:
        raise ValueError(f"Unknown policy type {policy_type!r}. Choose from {list(mapping)}")
    return mapping[policy_type](obs_dim=obs_dim, act_dim=act_dim, **kwargs)


def load_policy(path: str | Path) -> tuple[MLPPolicy | AttentionPolicy | RecurrentPolicy, dict[str, Any]]:
    """Load any policy type from a checkpoint.

    Parameters
    ----------
    path : str or Path

    Returns
    -------
    policy, meta : (policy instance, dict of extra checkpoint keys)
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    loader_map: dict[str, type] = {
        "MLPPolicy": MLPPolicy,
        "AttentionPolicy": AttentionPolicy,
        "RecurrentPolicy": RecurrentPolicy,
    }
    cls = loader_map[ckpt["type"]]
    return cls.load(path)
