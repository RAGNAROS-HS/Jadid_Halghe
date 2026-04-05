from __future__ import annotations

from typing import Any, Callable

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from game.config import WorldConfig
from game.world import GameState, World


# ---------------------------------------------------------------------------
# Observation layout
# ---------------------------------------------------------------------------

#: Observation feature-group sizes.
K_OWN: int = 16     # own cell slots              (rel_x, rel_y, log_mass_norm)
K_FOOD: int = 20    # nearest food                (rel_x, rel_y)
K_VIRUS: int = 10   # nearest virus               (rel_x, rel_y)
K_THREAT: int = 10  # nearest larger enemies      (rel_x, rel_y, delta_log_mass)
K_PREY: int = 10    # nearest smaller enemies     (rel_x, rel_y, delta_log_mass)

#: Total flat observation dimension.
OBS_DIM: int = K_OWN * 3 + K_FOOD * 2 + K_VIRUS * 2 + K_THREAT * 3 + K_PREY * 3 + 2  # 170


# ---------------------------------------------------------------------------
# Observation helpers
# ---------------------------------------------------------------------------

def _k_nearest_indices(query: np.ndarray, pool: np.ndarray, k: int) -> np.ndarray:
    """Return indices of the *k* nearest rows of *pool* to *query*.

    Parameters
    ----------
    query : ndarray, shape (2,)
        Reference point.
    pool : ndarray, shape (n, 2)
        Candidate points.
    k : int
        Number of neighbours to return.

    Returns
    -------
    ndarray, shape (min(k, n),), int32
    """
    n = min(k, len(pool))
    dists = np.sum((pool - query) ** 2, axis=1)
    if n == len(pool):
        return np.arange(n, dtype=np.int32)
    return np.argpartition(dists, n - 1)[:n].astype(np.int32)


def build_observation(
    state: GameState,
    player_id: int,
    cfg: WorldConfig,
    pos_scale: float,
) -> np.ndarray:
    """Construct a flat observation vector for *player_id* from *state*.

    Layout (all float32, shape ``(OBS_DIM,)``):

    * **Own cells** ``[K_OWN × 3]`` — sorted by mass (largest first);
      each entry is ``(rel_x, rel_y, log_mass_norm)`` relative to centroid.
    * **Food** ``[K_FOOD × 2]`` — *K* nearest food pellets:
      ``(rel_x, rel_y)`` relative to centroid.
    * **Viruses** ``[K_VIRUS × 2]`` — *K* nearest viruses.
    * **Threats** ``[K_THREAT × 3]`` — *K* nearest enemies larger than self:
      ``(rel_x, rel_y, delta_log_mass)`` where
      ``delta_log_mass = log(enemy_mass / own_total_mass + 1e-6) / 5 > 0``.
    * **Prey** ``[K_PREY × 3]`` — *K* nearest enemies smaller than or equal to self:
      ``(rel_x, rel_y, delta_log_mass)`` where ``delta_log_mass <= 0``.
    * **Scalars** ``[2]`` — ``(log_total_mass_norm, n_cells / max_cells_per_player)``.

    Positions are divided by *pos_scale* and clipped to ``[-10, 10]``.
    Log-mass features use ``log(mass / start_mass + 1e-6) / 5`` and are
    clipped to ``[-10, 10]``.  Unused slots are zero-padded.

    Parameters
    ----------
    state : GameState
        Current world snapshot.
    player_id : int
        Which player's perspective to encode.
    cfg : WorldConfig
        Simulation configuration (needed for normalisation constants).
    pos_scale : float
        Position normalisation denominator, typically ``max(width, height) / 2``.

    Returns
    -------
    ndarray, shape (OBS_DIM,), float32
    """
    own_mask = state.cell_owner == player_id
    own_pos = state.cell_pos[own_mask]    # (n_own, 2)
    own_mass = state.cell_mass[own_mask]  # (n_own,)

    centroid: np.ndarray
    if len(own_pos) > 0:
        centroid = own_pos.mean(axis=0)
    else:
        centroid = np.array([cfg.width / 2.0, cfg.height / 2.0], dtype=np.float32)

    # ── Own cells ───────────────────────────────────────────────────────────
    own_feat = np.zeros((K_OWN, 3), dtype=np.float32)
    if len(own_pos) > 0:
        order = np.argsort(own_mass)[::-1][:K_OWN]
        n = len(order)
        rel = np.clip((own_pos[order] - centroid) / pos_scale, -10.0, 10.0)
        lm = np.clip(
            np.log(own_mass[order] / cfg.start_mass + 1e-6) / 5.0, -10.0, 10.0
        )
        own_feat[:n, :2] = rel
        own_feat[:n, 2] = lm

    # ── Nearest food ────────────────────────────────────────────────────────
    food_feat = np.zeros((K_FOOD, 2), dtype=np.float32)
    if len(state.food_pos) > 0:
        nn = _k_nearest_indices(centroid, state.food_pos, K_FOOD)
        food_feat[: len(nn)] = np.clip(
            (state.food_pos[nn] - centroid) / pos_scale, -10.0, 10.0
        )

    # ── Nearest viruses ──────────────────────────────────────────────────────
    virus_feat = np.zeros((K_VIRUS, 2), dtype=np.float32)
    if len(state.virus_pos) > 0:
        nn = _k_nearest_indices(centroid, state.virus_pos, K_VIRUS)
        virus_feat[: len(nn)] = np.clip(
            (state.virus_pos[nn] - centroid) / pos_scale, -10.0, 10.0
        )

    # ── Own total mass (needed for delta encoding below) ────────────────────
    total_mass = float(state.player_mass[player_id])
    safe_own_mass = max(total_mass, 1.0)

    # ── Nearest enemies — split into threats (larger) and prey (smaller) ────
    enemy_mask = (state.cell_owner != player_id) & (state.cell_owner >= 0)
    enemy_pos = state.cell_pos[enemy_mask]
    enemy_mass = state.cell_mass[enemy_mask]

    threat_feat = np.zeros((K_THREAT, 3), dtype=np.float32)
    prey_feat = np.zeros((K_PREY, 3), dtype=np.float32)
    if len(enemy_pos) > 0:
        # Compute distances once; reuse for both threat and prey k-nearest searches.
        dists_sq = np.sum((enemy_pos - centroid) ** 2, axis=1)
        delta_lm_all = np.log(enemy_mass / safe_own_mass + 1e-6) / 5.0
        is_threat = delta_lm_all > 0.0

        for feat, mask, k in (
            (threat_feat, is_threat, K_THREAT),
            (prey_feat, ~is_threat, K_PREY),
        ):
            ep = enemy_pos[mask]
            em = enemy_mass[mask]
            ed = dists_sq[mask]
            if len(ep) == 0:
                continue
            n = min(k, len(ep))
            nn = (
                np.argpartition(ed, n - 1)[:n].astype(np.int32)
                if n < len(ep)
                else np.arange(len(ep), dtype=np.int32)
            )
            feat[:n, :2] = np.clip((ep[nn] - centroid) / pos_scale, -10.0, 10.0)
            feat[:n, 2] = np.clip(
                np.log(em[nn] / safe_own_mass + 1e-6) / 5.0, -10.0, 10.0
            )

    # ── Scalars ─────────────────────────────────────────────────────────────
    log_mass = float(
        np.clip(np.log(total_mass / cfg.start_mass + 1e-6) / 5.0, -10.0, 10.0)
    )
    cells_frac = float(own_mask.sum()) / cfg.max_cells_per_player

    return np.concatenate([
        own_feat.ravel(),
        food_feat.ravel(),
        virus_feat.ravel(),
        threat_feat.ravel(),
        prey_feat.ravel(),
        np.array([log_mass, cells_frac], dtype=np.float32),
    ])


def build_observation_batch(
    state: GameState,
    player_ids: list[int],
    cfg: WorldConfig,
    pos_scale: float,
    out: np.ndarray | None = None,
) -> np.ndarray:
    """Build observation vectors for *B* agents from a single :class:`GameState`.

    More efficient than calling :func:`build_observation` in a loop: shared
    arrays (food positions, virus positions, cell arrays) are accessed once and
    the output is written directly into a pre-allocated buffer, avoiding per-
    agent ``np.zeros`` + ``np.concatenate`` allocations.

    Parameters
    ----------
    state : GameState
        Current world snapshot.
    player_ids : list[int]
        Agent IDs to build observations for, in order.
    cfg : WorldConfig
        Simulation configuration.
    pos_scale : float
        Position normalisation denominator (``max(width, height) / 2``).
    out : ndarray, shape (B, OBS_DIM), float32, optional
        Pre-allocated output buffer.  Allocated internally if not provided.

    Returns
    -------
    ndarray, shape (len(player_ids), OBS_DIM), float32
    """
    B = len(player_ids)
    if out is None:
        out = np.zeros((B, OBS_DIM), dtype=np.float32)
    else:
        out[:] = 0.0

    # Hoist shared arrays — accessed once, reused across all agents.
    cell_owner = state.cell_owner
    cell_pos = state.cell_pos
    cell_mass = state.cell_mass
    food_pos = state.food_pos
    virus_pos = state.virus_pos
    player_mass = state.player_mass
    n_food = len(food_pos)
    n_virus = len(virus_pos)

    # Flat-vector segment boundaries (must match OBS_DIM layout above).
    s_food = K_OWN * 3
    s_virus = s_food + K_FOOD * 2
    s_threat = s_virus + K_VIRUS * 2
    s_prey = s_threat + K_THREAT * 3
    s_scalar = s_prey + K_PREY * 3

    for i, pid in enumerate(player_ids):
        own_mask = cell_owner == pid
        own_pos = cell_pos[own_mask]
        own_mass_arr = cell_mass[own_mask]

        if len(own_pos) > 0:
            centroid = own_pos.mean(axis=0)
        else:
            centroid = np.array([cfg.width / 2.0, cfg.height / 2.0], dtype=np.float32)

        # Own cells — write directly into pre-allocated output slice.
        if len(own_pos) > 0:
            order = np.argsort(own_mass_arr)[::-1][:K_OWN]
            n = len(order)
            rel = np.clip((own_pos[order] - centroid) / pos_scale, -10.0, 10.0)
            lm = np.clip(
                np.log(own_mass_arr[order] / cfg.start_mass + 1e-6) / 5.0, -10.0, 10.0
            )
            own_block = out[i, :s_food].reshape(K_OWN, 3)
            own_block[:n, :2] = rel
            own_block[:n, 2] = lm

        # Food
        if n_food > 0:
            nn = _k_nearest_indices(centroid, food_pos, K_FOOD)
            food_block = out[i, s_food:s_virus].reshape(K_FOOD, 2)
            food_block[:len(nn)] = np.clip(
                (food_pos[nn] - centroid) / pos_scale, -10.0, 10.0
            )

        # Viruses
        if n_virus > 0:
            nn = _k_nearest_indices(centroid, virus_pos, K_VIRUS)
            virus_block = out[i, s_virus:s_threat].reshape(K_VIRUS, 2)
            virus_block[:len(nn)] = np.clip(
                (virus_pos[nn] - centroid) / pos_scale, -10.0, 10.0
            )

        # Threats & prey
        total_mass = float(player_mass[pid])
        safe_own_mass = max(total_mass, 1.0)
        enemy_mask = (cell_owner != pid) & (cell_owner >= 0)
        enemy_pos = cell_pos[enemy_mask]
        enemy_mass_arr = cell_mass[enemy_mask]

        if len(enemy_pos) > 0:
            dists_sq = np.sum((enemy_pos - centroid) ** 2, axis=1)
            delta_lm_all = np.log(enemy_mass_arr / safe_own_mass + 1e-6) / 5.0
            is_threat = delta_lm_all > 0.0

            for s_start, s_end, mask, k in (
                (s_threat, s_prey, is_threat, K_THREAT),
                (s_prey, s_scalar, ~is_threat, K_PREY),
            ):
                ep = enemy_pos[mask]
                em = enemy_mass_arr[mask]
                ed = dists_sq[mask]
                if len(ep) == 0:
                    continue
                n = min(k, len(ep))
                nn = (
                    np.argpartition(ed, n - 1)[:n].astype(np.int32)
                    if n < len(ep)
                    else np.arange(len(ep), dtype=np.int32)
                )
                block = out[i, s_start:s_end].reshape(k, 3)
                block[:n, :2] = np.clip((ep[nn] - centroid) / pos_scale, -10.0, 10.0)
                block[:n, 2] = np.clip(
                    np.log(em[nn] / safe_own_mass + 1e-6) / 5.0, -10.0, 10.0
                )

        # Scalars
        log_mass = float(
            np.clip(np.log(total_mass / cfg.start_mass + 1e-6) / 5.0, -10.0, 10.0)
        )
        out[i, s_scalar] = log_mass
        out[i, s_scalar + 1] = float(own_mask.sum()) / cfg.max_cells_per_player

    return out


# ---------------------------------------------------------------------------
# Single-agent Gymnasium environment
# ---------------------------------------------------------------------------

class AgarEnv(gym.Env):
    """Single-agent Gymnasium wrapper for the agar.io simulation.

    Agent slot 0 is the RL-controlled player.  The remaining ``n_bots``
    player slots follow a random-direction policy and are automatically
    respawned on death.

    Observation space
    -----------------
    ``Box([-10]*OBS_DIM, [10]*OBS_DIM, float32)`` — see :func:`build_observation`.

    Action space
    ------------
    ``Box([-1,-1,-1,-1], [1,1,1,1], float32)``:

    * ``action[0:2]`` — desired movement direction ``(dx, dy)``.
      Converted to a world-space target position:
      ``target = centroid + direction * world_diagonal``.
    * ``action[2] > 0`` — trigger split.
    * ``action[3] > 0`` — fire ejected mass.

    Reward
    ------
    Raw world reward (mass delta from eating) divided by ``reward_scale``
    (defaults to ``start_mass``), plus ``survival_bonus`` each tick the
    agent is alive.  Death penalty is proportional to the agent's current
    mass at time of death.

    Parameters
    ----------
    config : WorldConfig, optional
        Simulation config.
    n_bots : int
        Number of random-bot opponents (capped at ``max_players - 1``).
    max_ticks : int
        Episode length before truncation.
    reward_scale : float, optional
        Divides raw rewards.  Defaults to ``config.start_mass``.
    survival_bonus : float
        Added to reward each tick the agent survives.  Calibrate to
        ``food_mass / start_mass`` (default 0.01) to match food-eating scale.
    bot_policy : Callable[[np.ndarray], np.ndarray] | None, optional
        If provided, each bot's action is produced by calling
        ``bot_policy(obs)`` instead of the random-walk heuristic.
        Swap it between rollouts via :meth:`set_bot_policy` without
        reconstructing the environment.
    """

    metadata: dict[str, Any] = {"render_modes": []}

    def __init__(
        self,
        config: WorldConfig | None = None,
        n_bots: int = 7,
        max_ticks: int = 2000,
        reward_scale: float | None = None,
        survival_bonus: float = 0.0,
        bot_policy: Callable[[np.ndarray], np.ndarray] | None = None,
    ) -> None:
        super().__init__()
        self.config = config or WorldConfig()
        cfg = self.config
        self.n_bots = min(n_bots, cfg.max_players - 1)
        self.max_ticks = max_ticks
        self._reward_scale = (
            reward_scale if reward_scale is not None else float(cfg.start_mass)
        )
        self._survival_bonus = survival_bonus
        self._world = World(cfg)
        self._agent_id = 0
        self._bot_ids: list[int] = list(range(1, 1 + self.n_bots))
        self._tick: int = 0
        self._pos_scale: float = max(cfg.width, cfg.height) / 2.0
        self._bot_policy: Callable[[np.ndarray], np.ndarray] | None = bot_policy

        obs_bound = np.full(OBS_DIM, 10.0, dtype=np.float32)
        self.observation_space = spaces.Box(-obs_bound, obs_bound, dtype=np.float32)
        self.action_space = spaces.Box(
            low=np.full(4, -1.0, dtype=np.float32),
            high=np.full(4, 1.0, dtype=np.float32),
            dtype=np.float32,
        )

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        """Reset the world and return the initial observation.

        Parameters
        ----------
        seed : int, optional
            Passed to the underlying RNG for reproducibility.
        options : dict, optional
            Unused; present for API compatibility.

        Returns
        -------
        obs : ndarray, shape (OBS_DIM,)
        info : dict
        """
        super().reset(seed=seed)
        self._world.reset(seed=seed)
        self._tick = 0
        self._world.add_player(self._agent_id)
        for bid in self._bot_ids:
            self._world.add_player(bid)
        state = self._world.get_state()
        obs = build_observation(state, self._agent_id, self.config, self._pos_scale)
        return obs, {}

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Advance one simulation tick.

        Parameters
        ----------
        action : ndarray, shape (4,)
            RL agent's action: ``[dx, dy, split_logit, eject_logit]``.

        Returns
        -------
        obs : ndarray, shape (OBS_DIM,)
        reward : float
        terminated : bool
        truncated : bool
        info : dict
        """
        cfg = self.config
        world_actions = np.zeros((cfg.max_players, 4), dtype=np.float32)
        large_scale = float(max(cfg.width, cfg.height))

        # Agent: convert direction to world-space target
        ac = self._centroid(self._agent_id)
        world_actions[self._agent_id, 0] = ac[0] + float(action[0]) * large_scale
        world_actions[self._agent_id, 1] = ac[1] + float(action[1]) * large_scale
        world_actions[self._agent_id, 2] = 1.0 if action[2] > 0.0 else 0.0
        world_actions[self._agent_id, 3] = 1.0 if action[3] > 0.0 else 0.0

        # Bots: policy-driven or random direction each tick
        if self._bot_policy is not None:
            # Fetch state once so all bots can build observations from the same snapshot.
            bot_state = self._world.get_state()
            for bid in self._bot_ids:
                if bid in self._world._active_players:
                    obs_b = build_observation(bot_state, bid, self.config, self._pos_scale)
                    act_b = self._bot_policy(obs_b)
                    bc = self._centroid(bid)
                    world_actions[bid, 0] = bc[0] + float(act_b[0]) * large_scale
                    world_actions[bid, 1] = bc[1] + float(act_b[1]) * large_scale
                    world_actions[bid, 2] = 1.0 if act_b[2] > 0.0 else 0.0
                    world_actions[bid, 3] = 1.0 if act_b[3] > 0.0 else 0.0
        else:
            rng = self._world._rng
            for bid in self._bot_ids:
                if bid in self._world._active_players:
                    angle = float(rng.uniform(0.0, 2.0 * np.pi))
                    bc = self._centroid(bid)
                    world_actions[bid, 0] = bc[0] + np.cos(angle) * large_scale
                    world_actions[bid, 1] = bc[1] + np.sin(angle) * large_scale

        raw_rewards, raw_dones, info = self._world.step(world_actions)

        # Respawn dead bots immediately
        for bid in self._bot_ids:
            if raw_dones[bid]:
                self._world.add_player(bid)

        self._tick += 1
        reward = float(raw_rewards[self._agent_id]) / self._reward_scale
        terminated = bool(raw_dones[self._agent_id])
        truncated = self._tick >= self.max_ticks
        if not terminated:
            reward += self._survival_bonus

        state = self._world.get_state()
        obs = build_observation(state, self._agent_id, self.config, self._pos_scale)
        return obs, reward, terminated, truncated, info

    def set_bot_policy(
        self,
        policy: Callable[[np.ndarray], np.ndarray] | None,
    ) -> None:
        """Replace the bot policy without reconstructing the environment.

        Parameters
        ----------
        policy : Callable[[np.ndarray], np.ndarray] | None
            New bot policy, or ``None`` to revert to the random-walk baseline.
        """
        self._bot_policy = policy

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
