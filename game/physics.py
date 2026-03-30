from __future__ import annotations

import numpy as np

from game.config import WorldConfig
from game.entities import CellArrays, EjectedArrays


# ---------------------------------------------------------------------------
# Cell physics
# ---------------------------------------------------------------------------

def update_cells(
    cells: CellArrays,
    actions: np.ndarray,
    config: WorldConfig,
) -> None:
    """Apply one physics tick to all live cells.

    Order of operations
    -------------------
    1. Compute per-cell target velocity from the owning player's action.
    2. Decay split_vel toward zero.
    3. Advance position by (vel + split_vel) × dt.
    4. Clamp positions to world bounds.
    5. Decrement merge timers.
    6. Apply mass decay for cells above the decay threshold.

    Parameters
    ----------
    cells : CellArrays
        Mutable cell state.  Modified in-place.
    actions : ndarray, shape (max_players, 4), float32
        One row per player slot: ``[dx, dy, split_flag, eject_flag]``.
        dx/dy are the desired movement direction (need not be normalised).
        split_flag and eject_flag are read by the spawner, not here.
    config : WorldConfig
        World configuration constants.

    Notes
    -----
    *No Python loops over individual cells* — all operations are vectorised
    over the alive-cell subset via NumPy array indexing.
    """
    idx = cells.alive_indices()
    if len(idx) == 0:
        return

    owners = cells.owner[idx]  # (n,)
    masses = cells.mass[idx]   # (n,)

    # ------------------------------------------------------------------
    # 1. Target velocity from player input
    # ------------------------------------------------------------------
    # Clamp owner ids to valid player range so we can index `actions` safely.
    valid_owner = (owners >= 0) & (owners < config.max_players)

    raw_dx = np.zeros(len(idx), dtype=np.float32)
    raw_dy = np.zeros(len(idx), dtype=np.float32)
    raw_dx[valid_owner] = actions[owners[valid_owner], 0]
    raw_dy[valid_owner] = actions[owners[valid_owner], 1]

    # Normalise direction; zero-length input → cell drifts with split_vel only
    mag = np.hypot(raw_dx, raw_dy)
    nonzero = mag > 1e-8
    safe_mag = np.where(nonzero, mag, 1.0)   # avoid divide-by-zero; result masked below
    dx = np.where(nonzero, raw_dx / safe_mag, 0.0).astype(np.float32)
    dy = np.where(nonzero, raw_dy / safe_mag, 0.0).astype(np.float32)

    # Speed = base_speed / mass^speed_exp, capped at base_speed
    speed = (config.base_speed / np.power(masses, config.speed_exp)).astype(np.float32)
    speed = np.minimum(speed, config.base_speed, out=speed)

    cells.vel[idx, 0] = dx * speed
    cells.vel[idx, 1] = dy * speed

    # ------------------------------------------------------------------
    # 2. Decay split velocity
    # ------------------------------------------------------------------
    cells.split_vel[idx] *= config.split_decay

    # ------------------------------------------------------------------
    # 3. Advance positions
    # ------------------------------------------------------------------
    total_vel = cells.vel[idx] + cells.split_vel[idx]  # (n, 2)
    cells.pos[idx] += total_vel * config.dt

    # ------------------------------------------------------------------
    # 4. Clamp to world bounds
    # ------------------------------------------------------------------
    cells.pos[idx, 0] = np.clip(cells.pos[idx, 0], 0.0, config.width)
    cells.pos[idx, 1] = np.clip(cells.pos[idx, 1], 0.0, config.height)

    # ------------------------------------------------------------------
    # 5. Merge timers
    # ------------------------------------------------------------------
    mt = cells.merge_timer[idx]
    cells.merge_timer[idx] = np.where(mt > 0, mt - 1.0, 0.0)

    # ------------------------------------------------------------------
    # 6. Mass decay for large cells
    # ------------------------------------------------------------------
    decay_mask = masses > config.mass_decay_threshold
    if decay_mask.any():
        decay = masses * config.mass_decay_rate
        new_mass = np.where(
            decay_mask,
            np.maximum(masses - decay, config.mass_decay_threshold),
            masses,
        )
        cells.mass[idx] = new_mass.astype(np.float32)


# ---------------------------------------------------------------------------
# Ejected-mass physics
# ---------------------------------------------------------------------------

def update_ejected(
    ejected: EjectedArrays,
    config: WorldConfig,
) -> None:
    """Advance ejected-mass pellets and count down their settle timers.

    Ejected mass decelerates the same way a split cell does — via the same
    ``split_decay`` multiplier.  Once velocity is negligible the pellet sits
    still and is absorbed when a cell overlaps it.

    Parameters
    ----------
    ejected : EjectedArrays
        Mutable ejected-mass state.  Modified in-place.
    config : WorldConfig
        World configuration constants.
    """
    idx = ejected.alive_indices()
    if len(idx) == 0:
        return

    # Decelerate
    ejected.vel[idx] *= config.split_decay

    # Advance position
    ejected.pos[idx] += ejected.vel[idx] * config.dt

    # Clamp to world bounds
    ejected.pos[idx, 0] = np.clip(ejected.pos[idx, 0], 0.0, config.width)
    ejected.pos[idx, 1] = np.clip(ejected.pos[idx, 1], 0.0, config.height)

    # Settle timer
    st = ejected.settle_timer[idx]
    ejected.settle_timer[idx] = np.where(st > 0, st - 1, 0)
