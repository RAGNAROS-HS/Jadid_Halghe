"""Self-play opponent pool for training against past policy snapshots.

The :class:`OpponentPool` maintains a ring buffer of checkpoint paths on disk.
At each ``update_interval`` rollouts it saves a snapshot of the current policy;
between rollouts it returns a callable that can be injected as the bot policy in
:class:`~rl.env.AgarEnv` / :class:`~rl.vec_env.VecAgarEnv`.

Usage in ``train.py``::

    pool = OpponentPool(
        pool_size=20,
        checkpoint_dir=Path("checkpoints/run_selfplay"),
        update_interval=50,
        selfplay_prob=0.8,
        device=device,
    )

    # Inside training loop, before runner.collect():
    pool.maybe_update(policy, rollout_idx=rollout_idx, step=global_step)
    bot_fn = pool.sample_policy()
    venv.set_bot_policy(bot_fn)   # None → fall back to random walk
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import numpy as np
import torch


class OpponentPool:
    """Ring-buffer pool of past policy snapshots used as training opponents.

    The pool saves a copy of the current policy every ``update_interval``
    rollouts.  When the ring is full the oldest entry is overwritten.
    :meth:`sample_policy` returns a callable ``obs → action`` for use as a
    bot policy, drawn uniformly at random from the pool.  With probability
    ``1 − selfplay_prob`` it returns ``None`` instead, which causes the env to
    fall back to the random-walk baseline.

    Loaded policies are cached in memory after first use; the cache entry is
    cleared when a slot is overwritten so the old weights can be garbage
    collected.

    Parameters
    ----------
    pool_size : int
        Maximum number of snapshots kept simultaneously.
    checkpoint_dir : Path
        Directory where snapshots are written (created if absent).
    update_interval : int
        Number of rollouts between consecutive snapshot writes.
    selfplay_prob : float
        Probability of returning a pool policy vs. ``None`` (random walk).
    device : torch.device
        Device used for opponent inference.
    """

    def __init__(
        self,
        pool_size: int,
        checkpoint_dir: Path,
        update_interval: int,
        selfplay_prob: float,
        device: torch.device,
    ) -> None:
        self._pool_size = pool_size
        self._dir = Path(checkpoint_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._update_interval = update_interval
        self._selfplay_prob = selfplay_prob
        self._device = device
        # Ring: list of [path, loaded_policy | None]
        self._ring: list[list[Path | object]] = []
        self._write_idx: int = 0
        self._rng = np.random.default_rng()

    # ------------------------------------------------------------------
    # Pool management
    # ------------------------------------------------------------------

    def maybe_update(self, policy: object, rollout_idx: int, step: int) -> None:
        """Conditionally snapshot *policy* into the ring buffer.

        Called every rollout; only writes when ``rollout_idx`` is a multiple
        of ``update_interval``.

        Parameters
        ----------
        policy : MLPPolicy | AttentionPolicy
            Current training policy (must have a ``.save(path, step=…)`` method).
        rollout_idx : int
            Current rollout counter (1-indexed).
        step : int
            Global environment step count (stored in the checkpoint).
        """
        if rollout_idx % self._update_interval != 0:
            return

        snap_path = self._dir / f"ckpt_selfplay_{rollout_idx:06d}.pt"
        policy.save(snap_path, step=step)  # type: ignore[union-attr]

        if len(self._ring) < self._pool_size:
            self._ring.append([snap_path, None])
        else:
            self._ring[self._write_idx] = [snap_path, None]
            self._write_idx = (self._write_idx + 1) % self._pool_size

        self.save_state()

    def sample_policy(self) -> Callable[[np.ndarray], np.ndarray] | None:
        """Return a callable opponent policy, or ``None`` (random walk).

        With probability ``1 − selfplay_prob``, or when the pool is empty,
        returns ``None``.  Otherwise samples a random slot from the ring,
        lazy-loads the policy if needed, and returns a
        ``torch.no_grad()``-wrapped callable.

        Returns
        -------
        Callable[[np.ndarray], np.ndarray] | None
            ``obs → action`` callable (tanh-squashed), or ``None``.
        """
        if not self._ring or self._rng.random() > self._selfplay_prob:
            return None

        idx = int(self._rng.integers(len(self._ring)))
        path, loaded = self._ring[idx]
        if loaded is None:
            from rl.agent import load_policy
            pol, _ = load_policy(path)  # type: ignore[arg-type]
            pol = pol.to(self._device).eval()
            self._ring[idx][1] = pol
            loaded = pol

        policy = loaded
        device = self._device

        def _fn(obs: np.ndarray) -> np.ndarray:
            with torch.no_grad():
                z, _, _ = policy.act(obs, deterministic=True)  # type: ignore[union-attr]
                return torch.tanh(z).squeeze(0).cpu().numpy().astype(np.float32)

        return _fn

    # ------------------------------------------------------------------
    # Persistence (resume support)
    # ------------------------------------------------------------------

    def save_state(self) -> None:
        """Write pool metadata to ``pool_state.json`` in the checkpoint dir."""
        state = {
            "pool_size": self._pool_size,
            "write_idx": self._write_idx,
            "paths": [str(entry[0]) for entry in self._ring],
        }
        (self._dir / "pool_state.json").write_text(
            json.dumps(state, indent=2), encoding="utf-8"
        )

    def load_state(self) -> None:
        """Restore pool metadata from ``pool_state.json`` (called on resume).

        Missing snapshot files are silently skipped so partial runs do not
        crash on resume.
        """
        state_path = self._dir / "pool_state.json"
        if not state_path.exists():
            return
        state = json.loads(state_path.read_text(encoding="utf-8"))
        self._write_idx = state.get("write_idx", 0)
        self._ring = []
        for p in state.get("paths", []):
            path = Path(p)
            if path.exists():
                self._ring.append([path, None])
