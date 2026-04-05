from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Union

import numpy as np
import torch

from eval.baselines import GreedyPolicy, RandomPolicy
from game.config import WorldConfig
from game.world import GameState, World
from rl.agent import AttentionPolicy, MLPPolicy, RecurrentPolicy, load_policy
from rl.env import OBS_DIM, build_observation

_AnyPolicy = Union[MLPPolicy, AttentionPolicy, RecurrentPolicy, RandomPolicy, GreedyPolicy]


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class EpisodeResult:
    """Statistics collected from one evaluation episode.

    Parameters
    ----------
    total_reward : float
        Sum of raw (un-scaled) rewards received by the eval agent.
    length : int
        Number of ticks the episode lasted.
    final_mass : float
        Eval agent's total mass at the end of the episode (0 if dead).
    survived : bool
        True if the episode ended by truncation (agent reached ``max_ticks``).
    rank : int
        Agent's rank by mass at episode end (1 = largest).  Computed among
        all players that were alive at the final tick.
    """

    total_reward: float
    length: int
    final_mass: float
    survived: bool
    rank: int


@dataclass
class EvalResult:
    """Aggregate statistics over *n_episodes* evaluation episodes.

    All ``mean_*`` and ``std_*`` fields are computed over episodes.
    """

    n_episodes: int
    mean_reward: float
    std_reward: float
    mean_length: float
    survival_rate: float   # fraction of episodes where agent survived max_ticks
    mean_final_mass: float
    mean_rank: float
    episodes: list[EpisodeResult] = field(default_factory=list, repr=False)

    def summary(self) -> str:
        """Return a human-readable one-line summary."""
        return (
            f"episodes={self.n_episodes} "
            f"reward={self.mean_reward:+.3f}±{self.std_reward:.3f} "
            f"survival={self.survival_rate:.1%} "
            f"mass={self.mean_final_mass:.0f} "
            f"rank={self.mean_rank:.1f}"
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _centroid_from_state(state: GameState, player_id: int, cfg: WorldConfig) -> np.ndarray:
    """Mass-weighted centroid of *player_id*'s cells, falling back to world centre."""
    mask = state.cell_owner == player_id
    if not mask.any():
        return np.array([cfg.width / 2.0, cfg.height / 2.0], dtype=np.float32)
    pos = state.cell_pos[mask]
    mass = state.cell_mass[mask]
    total = mass.sum()
    if total <= 0:
        return pos.mean(axis=0)
    return (pos * mass[:, None]).sum(axis=0) / total


def _wrap_policy(
    policy: _AnyPolicy,
    device: torch.device,
) -> Callable[[np.ndarray], np.ndarray]:
    """Return a unified ``obs → action`` callable for any policy type.

    Trained policies (MLPPolicy / AttentionPolicy) use deterministic
    ``act()`` and tanh-squash the pre-tanh sample.  Baseline policies
    (RandomPolicy / GreedyPolicy) are called directly.
    """
    if isinstance(policy, (MLPPolicy, AttentionPolicy)):
        policy.eval()
        def fn(obs: np.ndarray) -> np.ndarray:
            with torch.no_grad():
                z, _, _ = policy.act(obs, deterministic=True)
                return torch.tanh(z).squeeze(0).cpu().numpy().astype(np.float32)
        return fn
    elif isinstance(policy, RecurrentPolicy):
        raise ValueError(
            "RecurrentPolicy requires a custom evaluation loop with hidden-state "
            "threading.  Use MLPPolicy or AttentionPolicy with the Harness."
        )
    else:
        # RandomPolicy / GreedyPolicy
        return policy.act  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------

class Harness:
    """Runs evaluation episodes for a trained policy against configurable opponents.

    Parameters
    ----------
    policy : trained policy or path
        The agent being evaluated.  Pass a :class:`~rl.agent.MLPPolicy` /
        :class:`~rl.agent.AttentionPolicy` instance, or a ``str``/``Path``
        checkpoint path (auto-loaded via :func:`~rl.agent.load_policy`).
    opponent : str or policy
        Opponent bot behaviour.  One of:

        * ``"random"`` — uniform-random actions (default).
        * ``"greedy"`` — heuristic chase/flee policy.
        * :class:`~rl.agent.MLPPolicy` / :class:`~rl.agent.AttentionPolicy`
          — any trained policy instance.
        * ``str``/``Path`` — checkpoint path, auto-loaded.

    config : WorldConfig, optional
        World configuration.  Defaults to ``WorldConfig()``.
    n_bots : int
        Number of opponent bot slots.
    max_ticks : int
        Episode truncation length.
    device : torch.device
        Device used for trained-policy inference.
    """

    def __init__(
        self,
        policy: _AnyPolicy | str,
        opponent: _AnyPolicy | str = "random",
        config: WorldConfig | None = None,
        n_bots: int = 7,
        max_ticks: int = 2000,
        device: torch.device | None = None,
    ) -> None:
        self.config = config or WorldConfig()
        self.n_bots = min(n_bots, self.config.max_players - 1)
        self.max_ticks = max_ticks
        self.device = device or torch.device("cpu")

        # Load eval policy
        if isinstance(policy, str):
            _pol, _ = load_policy(policy)
            policy = _pol
        self._eval_fn = _wrap_policy(policy, self.device)

        # Load opponent policy
        if isinstance(opponent, str) and opponent not in ("random", "greedy"):
            # Treat as checkpoint path
            _opp, _ = load_policy(opponent)
            opponent = _opp
        if isinstance(opponent, str):
            if opponent == "random":
                opponent = RandomPolicy()
            elif opponent == "greedy":
                opponent = GreedyPolicy()
        self._opp_fn = _wrap_policy(opponent, self.device)

        self._eval_pid = 0
        self._bot_pids = list(range(1, 1 + self.n_bots))
        self._pos_scale = float(max(self.config.width, self.config.height)) / 2.0
        self._large_scale = float(max(self.config.width, self.config.height))

    def run(
        self,
        n_episodes: int,
        seed: int = 0,
        record: bool = False,
    ) -> tuple[EvalResult, list[list[GameState]]]:
        """Run *n_episodes* evaluation episodes.

        Parameters
        ----------
        n_episodes : int
        seed : int
            Episode *i* is seeded with ``seed + i``.
        record : bool
            If True, collect per-tick :class:`~game.world.GameState`
            snapshots for each episode (for replay / plotting).  This
            incurs extra memory and slows evaluation slightly.

        Returns
        -------
        result : EvalResult
        replays : list of list[GameState]
            One inner list per episode (empty lists when ``record=False``).
        """
        cfg = self.config
        world = World(cfg)
        ep_results: list[EpisodeResult] = []
        replays: list[list[GameState]] = []
        world_actions = np.zeros((cfg.max_players, 4), dtype=np.float32)

        for ep in range(n_episodes):
            world.reset(seed=seed + ep)
            world.add_player(self._eval_pid)
            for pid in self._bot_pids:
                world.add_player(pid)

            ep_reward = 0.0
            frames: list[GameState] = []
            survived = False
            last_state: GameState | None = None

            for tick in range(self.max_ticks):
                state = world.get_state()
                last_state = state

                if record:
                    frames.append(state)

                world_actions[:] = 0.0

                # Eval agent
                if self._eval_pid in world._active_players:
                    obs = build_observation(
                        state, self._eval_pid, cfg, self._pos_scale
                    )
                    act = self._eval_fn(obs)
                    c = _centroid_from_state(state, self._eval_pid, cfg)
                    world_actions[self._eval_pid, 0] = c[0] + act[0] * self._large_scale
                    world_actions[self._eval_pid, 1] = c[1] + act[1] * self._large_scale
                    world_actions[self._eval_pid, 2] = 1.0 if act[2] > 0.0 else 0.0
                    world_actions[self._eval_pid, 3] = 1.0 if act[3] > 0.0 else 0.0

                # Opponent bots
                for pid in self._bot_pids:
                    if pid in world._active_players:
                        obs_b = build_observation(state, pid, cfg, self._pos_scale)
                        act_b = self._opp_fn(obs_b)
                        c_b = _centroid_from_state(state, pid, cfg)
                        world_actions[pid, 0] = c_b[0] + act_b[0] * self._large_scale
                        world_actions[pid, 1] = c_b[1] + act_b[1] * self._large_scale

                rewards, dones, _ = world.step(world_actions)
                ep_reward += float(rewards[self._eval_pid])

                # Respawn dead bots
                for pid in self._bot_pids:
                    if dones[pid]:
                        world.add_player(pid)

                if dones[self._eval_pid]:
                    survived = False
                    break
            else:
                survived = True

            # Rank: count how many alive players have more mass than us
            assert last_state is not None
            final_mass = float(last_state.player_mass[self._eval_pid])
            rank = 1 + int(
                (last_state.player_mass > final_mass).sum()
            )

            ep_results.append(
                EpisodeResult(
                    total_reward=ep_reward,
                    length=tick + 1,
                    final_mass=final_mass,
                    survived=survived,
                    rank=rank,
                )
            )
            replays.append(frames)

        rewards_arr = np.array([r.total_reward for r in ep_results])
        result = EvalResult(
            n_episodes=n_episodes,
            mean_reward=float(rewards_arr.mean()),
            std_reward=float(rewards_arr.std()),
            mean_length=float(np.mean([r.length for r in ep_results])),
            survival_rate=float(np.mean([r.survived for r in ep_results])),
            mean_final_mass=float(np.mean([r.final_mass for r in ep_results])),
            mean_rank=float(np.mean([r.rank for r in ep_results])),
            episodes=ep_results,
        )
        return result, replays
