"""Elo rating system for ranking trained policies via round-robin tournament.

Usage example::

    from pathlib import Path
    from eval.elo import run_tournament

    paths = sorted(Path("checkpoints/run_default").glob("ckpt_[0-9]*.pt"))
    elo = run_tournament(paths, episodes_per_pair=20)
    print(elo.table_str())
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Elo rating
# ---------------------------------------------------------------------------

class EloRating:
    """Standard Elo rating system for multiple policies.

    Parameters
    ----------
    k_factor : float
        Controls how quickly ratings change after each game (default 32).
    initial_rating : float
        Starting Elo score for each newly-added player (default 1000).
    """

    def __init__(self, k_factor: float = 32.0, initial_rating: float = 1000.0) -> None:
        self._ratings: dict[str, float] = {}
        self._k = k_factor
        self._initial = initial_rating

    def add_player(self, label: str) -> None:
        """Register *label* with the initial rating if not already present.

        Parameters
        ----------
        label : str
            Unique player identifier (e.g. checkpoint filename).
        """
        if label not in self._ratings:
            self._ratings[label] = self._initial

    def record_result(self, label_a: str, label_b: str, score_a: float) -> None:
        """Update ratings after one game.

        Parameters
        ----------
        label_a, label_b : str
            Player labels; both must have been added via :meth:`add_player`.
        score_a : float
            Score for player A: ``1.0`` = win, ``0.5`` = draw, ``0.0`` = loss.
        """
        ra = self._ratings[label_a]
        rb = self._ratings[label_b]
        ea = 1.0 / (1.0 + 10.0 ** ((rb - ra) / 400.0))
        self._ratings[label_a] = ra + self._k * (score_a - ea)
        self._ratings[label_b] = rb + self._k * ((1.0 - score_a) - (1.0 - ea))

    def ratings(self) -> dict[str, float]:
        """Return a copy of ratings sorted by score (descending).

        Returns
        -------
        dict[str, float]
        """
        return dict(sorted(self._ratings.items(), key=lambda kv: -kv[1]))

    def table_str(self) -> str:
        """Return a formatted ranking table string.

        Returns
        -------
        str
        """
        rows = list(self.ratings().items())
        if not rows:
            return "(no players)"
        lines = [f"{'Rank':>4}  {'Label':<44}  {'Elo':>7}"]
        lines.append("-" * 62)
        for i, (label, score) in enumerate(rows, 1):
            lines.append(f"{i:>4}  {label:<44}  {score:>7.1f}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-compatible dict.

        Returns
        -------
        dict
        """
        return {
            "k_factor": self._k,
            "initial_rating": self._initial,
            "ratings": dict(self._ratings),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EloRating:
        """Deserialise from a dict produced by :meth:`to_dict`.

        Parameters
        ----------
        data : dict
        """
        obj = cls(k_factor=data["k_factor"], initial_rating=data["initial_rating"])
        obj._ratings = dict(data["ratings"])
        return obj


# ---------------------------------------------------------------------------
# Tournament runner
# ---------------------------------------------------------------------------

def run_tournament(
    checkpoint_paths: list[Path],
    episodes_per_pair: int = 20,
    n_bots: int = 0,
    max_ticks: int = 2000,
    device: torch.device | None = None,
    k_factor: float = 32.0,
    seed: int = 0,
) -> EloRating:
    """Run a round-robin Elo tournament between checkpoint policies.

    Each unordered pair ``(A, B)`` plays ``episodes_per_pair`` games in each
    direction (A vs B and B vs A) to cancel home-field advantage.  Within a
    game, A is the eval agent and B is the sole opponent bot (plus any extra
    random ``n_bots``).  Win/loss/draw is determined per episode by final rank:

    * rank 1 → score ``1.0`` (win)
    * rank 2 → score ``0.0`` (loss) in a pure 1v1, or interpolated for more players
    * exact tie in final mass → score ``0.5`` (draw)

    Memory: policies are loaded pair-by-pair and deleted immediately after
    their games to avoid holding all checkpoints in memory simultaneously.

    Parameters
    ----------
    checkpoint_paths : list[Path]
        Paths to ``.pt`` checkpoint files.  Labels are taken from
        ``Path.name``.
    episodes_per_pair : int
        Episodes played per ordered direction (default 20).
    n_bots : int
        Extra random-bot bystanders in each game (0 = pure 1v1).
    max_ticks : int
        Episode truncation length.
    device : torch.device, optional
        Device for policy inference (default: CPU).
    k_factor : float
        Elo K-factor.
    seed : int
        Base random seed; each matchup uses a deterministic offset.

    Returns
    -------
    EloRating
        Populated with final scores after the full round-robin.
    """
    from eval.harness import Harness

    _device = device or torch.device("cpu")
    paths = list(checkpoint_paths)
    labels = [p.name for p in paths]
    elo = EloRating(k_factor=k_factor)
    for label in labels:
        elo.add_player(label)

    n = len(paths)
    total_pairs = n * (n - 1) // 2

    for i in range(n):
        for j in range(i + 1, n):
            path_a, label_a = paths[i], labels[i]
            path_b, label_b = paths[j], labels[j]
            print(f"  [{i * (n - 1) - i * (i - 1) // 2 + (j - i):>3}/{total_pairs}]"
                  f"  {label_a}  vs  {label_b}")

            for direction, (ep, op, el, ol) in enumerate([
                (path_a, path_b, label_a, label_b),
                (path_b, path_a, label_b, label_a),
            ]):
                harness = Harness(
                    policy=str(ep),
                    opponent=str(op),
                    n_bots=n_bots,
                    max_ticks=max_ticks,
                    device=_device,
                )
                pair_seed = seed + i * n * 2 + j * 2 + direction
                result, _ = harness.run(n_episodes=episodes_per_pair, seed=pair_seed)

                # Score per episode: rank-1 = 1.0, tie = 0.5, else 0.0
                # With n_bots extra bystanders, max_rank = 2 + n_bots.
                max_rank = 2 + n_bots
                ep_scores = []
                for ep_result in result.episodes:
                    if ep_result.rank == 1:
                        ep_scores.append(1.0)
                    elif ep_result.rank >= max_rank:
                        ep_scores.append(0.0)
                    else:
                        ep_scores.append(0.5)
                mean_score = float(np.mean(ep_scores)) if ep_scores else 0.5

                elo.record_result(el, ol, mean_score)
                del harness  # free policy memory

    return elo
