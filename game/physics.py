from __future__ import annotations

import numpy as np

from game.config import WorldConfig
from game.entities import CellArrays, EjectedArrays

# Pre-computed dt constant used as a float32 scalar to keep all arithmetic in float32.
_F32 = np.float32


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
    over the alive-cell subset via NumPy array indexing.  All intermediate
    arrays are kept as float32 to avoid dtype-promotion overhead.
    """
    idx = cells.alive_indices()
    if len(idx) == 0:
        return

    owners = cells.owner[idx]   # (n,) int32
    masses = cells.mass[idx]    # (n,) float32

    # ------------------------------------------------------------------
    # 1. Target velocity from player input
    #
    # actions[:, :2] holds the cursor's world-space position (target_x,
    # target_y).  Each cell steers independently toward that point so that
    # split cells naturally converge when the cursor is between them.
    # ------------------------------------------------------------------
    valid_owner = (owners >= 0) & (owners < config.max_players)

    direction = np.zeros((len(idx), 2), dtype=np.float32)
    if valid_owner.any():
        cursor = actions[owners[valid_owner], :2]           # (n_valid, 2)
        delta = cursor - cells.pos[idx[valid_owner]]        # per-cell vector
        direction[valid_owner] = delta.astype(np.float32)

    # Normalise in-place; zero-length delta leaves direction as zero
    mag_sq = (direction * direction).sum(axis=1)            # (n,) float32
    nonzero = mag_sq > _F32(1e-8)
    if nonzero.any():
        mag = np.sqrt(mag_sq[nonzero], dtype=np.float32)
        direction[nonzero] /= mag[:, None]
    direction[~nonzero] = _F32(0.0)

    # Speed: base / mass^exp, all float32
    # masses is float32; float32 ** float64-scalar promotes to float64, so cast exp.
    speed = np.power(masses, _F32(config.speed_exp))           # (n,) float32
    np.divide(config.base_speed, speed, out=speed)             # in-place
    np.minimum(speed, config.base_speed, out=speed)            # cap

    # Write target velocity directly into cells array
    cells.vel[idx] = direction * speed[:, None]                # (n, 2)

    # ------------------------------------------------------------------
    # 2. Decay split velocity in-place
    # ------------------------------------------------------------------
    cells.split_vel[idx] *= _F32(config.split_decay)

    # ------------------------------------------------------------------
    # 3 & 4. Advance positions and clamp
    # ------------------------------------------------------------------
    # Compute total_vel directly into pos update to avoid allocating a temp array
    pos = cells.pos[idx]                                       # view (n, 2)
    pos += (cells.vel[idx] + cells.split_vel[idx]) * _F32(config.dt)
    np.clip(pos[:, 0], _F32(0.0), _F32(config.width),  out=pos[:, 0])
    np.clip(pos[:, 1], _F32(0.0), _F32(config.height), out=pos[:, 1])
    cells.pos[idx] = pos

    # ------------------------------------------------------------------
    # 5. Merge timers — subtract 1 where positive
    # ------------------------------------------------------------------
    mt = cells.merge_timer[idx]                                # (n,) float32
    positive = mt > _F32(0.0)
    if positive.any():
        mt[positive] -= _F32(1.0)
        cells.merge_timer[idx] = mt

    # ------------------------------------------------------------------
    # 6. Mass decay for large cells
    # ------------------------------------------------------------------
    decay_mask = masses > _F32(config.mass_decay_threshold)
    if decay_mask.any():
        masses[decay_mask] *= _F32(1.0 - config.mass_decay_rate)
        np.maximum(
            masses,
            _F32(config.mass_decay_threshold),
            out=masses,
            where=decay_mask,
        )
        cells.mass[idx] = masses


# ---------------------------------------------------------------------------
# Ejected-mass physics
# ---------------------------------------------------------------------------

def update_ejected(
    ejected: EjectedArrays,
    config: WorldConfig,
) -> None:
    """Advance ejected-mass pellets and count down their settle timers.

    Ejected mass decelerates via the same ``split_decay`` multiplier as split
    cells.  Once velocity is negligible the pellet sits still until absorbed.

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

    ejected.vel[idx] *= _F32(config.split_decay)
    ejected.pos[idx] += ejected.vel[idx] * _F32(config.dt)

    pos = ejected.pos[idx]
    np.clip(pos[:, 0], _F32(0.0), _F32(config.width),  out=pos[:, 0])
    np.clip(pos[:, 1], _F32(0.0), _F32(config.height), out=pos[:, 1])
    ejected.pos[idx] = pos

    st = ejected.settle_timer[idx]
    positive = st > 0
    if positive.any():
        st[positive] -= 1
        ejected.settle_timer[idx] = st
