from __future__ import annotations

import numpy as np

from rl.env import K_OWN, K_FOOD, K_VIRUS, K_THREAT, K_PREY, OBS_DIM

# Obs slice offsets — must match rl/env.build_observation
_OWN_S = 0
_OWN_E = K_OWN * 3                # 48
_FOOD_S = _OWN_E                  # 48
_FOOD_E = _FOOD_S + K_FOOD * 2    # 88
_VIR_S = _FOOD_E                  # 88
_VIR_E = _VIR_S + K_VIRUS * 2    # 108
_THR_S = _VIR_E                   # 108
_THR_E = _THR_S + K_THREAT * 3    # 138
_PRY_S = _THR_E                   # 138
_PRY_E = _PRY_S + K_PREY * 3      # 168
_SCL_S = _PRY_E                   # 168


class RandomPolicy:
    """Baseline that samples uniformly from the action space.

    Actions are drawn from ``U([-1, 1]^4)`` regardless of the observation.
    Serves as a lower-bound reference in evaluation comparisons.

    Parameters
    ----------
    seed : int, optional
        Seed for the internal RNG.
    """

    def __init__(self, seed: int | None = None) -> None:
        self._rng = np.random.default_rng(seed)

    def act(self, obs: np.ndarray) -> np.ndarray:
        """Return a random action.

        Parameters
        ----------
        obs : ndarray, shape (obs_dim,)
            Ignored.

        Returns
        -------
        ndarray, shape (4,), float32
        """
        return self._rng.uniform(-1.0, 1.0, size=4).astype(np.float32)


class GreedyPolicy:
    """Heuristic baseline: chase food/edible enemies, flee threats.

    Strategy (evaluated from the centroid-relative observation each tick):

    1. **Chase** the nearest edible prey cell.  Prey slots already contain
       only enemies smaller than self; ``delta_log_mass < -0.038`` confirms
       the eating threshold ``mass_self > 1.21 × mass_enemy`` is met.
    2. **Eat food** — move toward the nearest food pellet if no edible
       prey is visible.
    3. **Flee** — if any threat cell (larger than self) is within 30% of the
       normalised view radius, reverse direction away from the nearest one.
    4. **Wander** randomly when nothing else applies.

    No split or eject actions are ever emitted (``action[2:] = 0``).
    """

    def act(self, obs: np.ndarray) -> np.ndarray:
        """Choose a greedy action from the observation.

        Parameters
        ----------
        obs : ndarray, shape (obs_dim,)

        Returns
        -------
        ndarray, shape (4,), float32
        """
        # Threat slots: delta_log_mass > 0 (enemy larger than self)
        threat = obs[_THR_S:_THR_E].reshape(K_THREAT, 3)
        # Prey slots: delta_log_mass <= 0 (enemy smaller than or equal to self)
        prey = obs[_PRY_S:_PRY_E].reshape(K_PREY, 3)
        food = obs[_FOOD_S:_FOOD_E].reshape(K_FOOD, 2)

        action = np.zeros(4, dtype=np.float32)
        direction = np.zeros(2, dtype=np.float32)

        # ── 1. Chase nearest edible prey ────────────────────────────────
        # delta_log_mass < -0.038 ↔ enemy_mass < own_mass / 1.21 (eating threshold)
        _EAT_THRESH = -0.038
        is_real_prey = np.any(prey != 0.0, axis=1)
        edible = is_real_prey & (prey[:, 2] < _EAT_THRESH)
        if edible.any():
            edible_pos = prey[edible, :2]
            nearest = edible_pos[np.argmin(np.linalg.norm(edible_pos, axis=1))]
            direction = nearest.copy()

        # ── 2. Move toward nearest food if no edible prey ───────────────
        if not edible.any():
            is_real_food = np.any(food != 0.0, axis=1)
            if is_real_food.any():
                real_food = food[is_real_food]
                nearest_food = real_food[np.argmin(np.linalg.norm(real_food, axis=1))]
                direction = nearest_food.copy()
            else:
                # Wander: fixed angle from obs hash for determinism-in-context
                angle = float(np.sum(obs[:4])) % (2.0 * np.pi)
                direction = np.array([np.cos(angle), np.sin(angle)], dtype=np.float32)

        # ── 3. Flee nearest threat if close ─────────────────────────────
        _FLEE_DIST = 0.3  # within 30% of normalised view radius → flee
        is_real_threat = np.any(threat != 0.0, axis=1)
        if is_real_threat.any():
            threat_pos = threat[is_real_threat, :2]
            dists = np.linalg.norm(threat_pos, axis=1)
            nearest_threat = threat_pos[np.argmin(dists)]
            if dists.min() < _FLEE_DIST:
                norm = np.linalg.norm(nearest_threat) + 1e-6
                direction = -(nearest_threat / norm)

        # ── Normalise and clip to [-1, 1] ───────────────────────────────
        norm = np.linalg.norm(direction)
        if norm > 1e-6:
            direction = direction / norm
        action[:2] = np.clip(direction, -1.0, 1.0)
        return action
