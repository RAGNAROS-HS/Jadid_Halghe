from __future__ import annotations

import numpy as np

from game.config import WorldConfig
from game.entities import CellArrays, EjectedArrays, FoodArrays, VirusArrays


# ---------------------------------------------------------------------------
# Food spawning
# ---------------------------------------------------------------------------

def spawn_food(
    food: FoodArrays,
    rng: np.random.Generator,
    config: WorldConfig,
) -> None:
    """Spawn food pellets until the live count reaches ``config.target_food_count``.

    Pellets are placed uniformly at random within the world bounds.

    Parameters
    ----------
    food : FoodArrays
        Modified in-place.
    rng : numpy.random.Generator
        Seeded RNG for reproducibility.
    config : WorldConfig
    """
    deficit = config.target_food_count - food.count
    if deficit <= 0:
        return

    # allocate() returns contiguous indices [old_count, old_count + n)
    indices = food.allocate(deficit)
    n = len(indices)
    if n == 0:
        return

    food.pos[indices, 0] = rng.uniform(0.0, config.width,  n).astype(np.float32)
    food.pos[indices, 1] = rng.uniform(0.0, config.height, n).astype(np.float32)


# ---------------------------------------------------------------------------
# Virus spawning
# ---------------------------------------------------------------------------

def spawn_viruses(
    viruses: VirusArrays,
    rng: np.random.Generator,
    config: WorldConfig,
) -> None:
    """Spawn viruses until the live count reaches ``config.target_virus_count``.

    Parameters
    ----------
    viruses : VirusArrays
        Modified in-place.
    rng : numpy.random.Generator
    config : WorldConfig
    """
    deficit = config.target_virus_count - viruses.count
    if deficit <= 0:
        return

    indices = viruses.allocate(deficit)
    if len(indices) == 0:
        return

    viruses.pos[indices, 0] = rng.uniform(0.0, config.width, len(indices)).astype(np.float32)
    viruses.pos[indices, 1] = rng.uniform(0.0, config.height, len(indices)).astype(np.float32)
    viruses.feed_count[indices] = 0


# ---------------------------------------------------------------------------
# Player spawning
# ---------------------------------------------------------------------------

def add_player(
    cells: CellArrays,
    player_id: int,
    rng: np.random.Generator,
    config: WorldConfig,
) -> int:
    """Spawn a fresh player cell for *player_id* at a random position.

    Parameters
    ----------
    cells : CellArrays
        Modified in-place.
    player_id : int
        Player slot index (0 … max_players-1).
    rng : numpy.random.Generator
    config : WorldConfig

    Returns
    -------
    int
        Global slot index of the newly created cell.

    Raises
    ------
    RuntimeError
        If no cell slots are available.
    """
    (slot,) = cells.allocate(1)
    slot = int(slot)

    cells.pos[slot, 0] = float(rng.uniform(config.width * 0.1, config.width * 0.9))
    cells.pos[slot, 1] = float(rng.uniform(config.height * 0.1, config.height * 0.9))
    cells.vel[slot] = 0.0
    cells.split_vel[slot] = 0.0
    cells.mass[slot] = config.start_mass
    cells.owner[slot] = player_id
    cells.merge_timer[slot] = 0.0

    return slot


# ---------------------------------------------------------------------------
# Split action
# ---------------------------------------------------------------------------

def handle_split(
    cells: CellArrays,
    player_id: int,
    config: WorldConfig,
) -> None:
    """Split all eligible cells belonging to *player_id*.

    Rules
    -----
    * The player must have fewer than ``config.max_cells_per_player`` live cells.
    * Only cells with ``mass >= config.min_split_mass`` can split.
    * A split halves the cell's mass between parent and new child.
    * The new child launches in the direction of the parent's current
      velocity at ``config.split_speed``.
    * Both cells receive a fresh ``merge_timer``.

    Parameters
    ----------
    cells : CellArrays
        Modified in-place.
    player_id : int
    config : WorldConfig
    """
    p_idx = cells.player_indices(player_id)
    current_count = len(p_idx)

    if current_count == 0 or current_count >= config.max_cells_per_player:
        return

    can_split_mask = cells.mass[p_idx] >= config.min_split_mass
    split_candidates = p_idx[can_split_mask]

    if len(split_candidates) == 0:
        return

    # How many new cells can we create?
    slots_available = min(
        config.max_cells_per_player - current_count,
        cells.free_count(),
        len(split_candidates),
    )
    split_candidates = split_candidates[:slots_available]

    new_slots = cells.allocate(len(split_candidates))

    # For each split: child starts at parent's position with half the mass
    cells.pos[new_slots] = cells.pos[split_candidates]
    cells.mass[new_slots] = cells.mass[split_candidates] / 2.0
    cells.mass[split_candidates] /= 2.0
    cells.owner[new_slots] = player_id

    # Merge timer based on mass (larger cells take longer to merge)
    new_timers = (
        config.merge_time_base
        + cells.mass[split_candidates] / config.merge_time_mass_factor
    ).astype(np.float32)
    cells.merge_timer[new_slots] = new_timers
    cells.merge_timer[split_candidates] = new_timers

    # Launch direction: normalised current velocity, fall back to +x
    vel = cells.vel[split_candidates]                              # (k, 2)
    mag = np.linalg.norm(vel, axis=1, keepdims=True)              # (k, 1)
    safe_mag = np.where(mag > 1e-8, mag, np.float32(1.0))
    dir_vec = np.where(mag > 1e-8, vel / safe_mag, np.array([[1.0, 0.0]])).astype(np.float32)

    cells.split_vel[new_slots] = dir_vec * config.split_speed
    cells.split_vel[split_candidates] = -dir_vec * (config.split_speed * 0.2)
    cells.vel[new_slots] = cells.vel[split_candidates]


# ---------------------------------------------------------------------------
# Eject action
# ---------------------------------------------------------------------------

def handle_eject(
    cells: CellArrays,
    ejected: EjectedArrays,
    player_id: int,
    config: WorldConfig,
) -> None:
    """Fire ejected-mass pellets from all eligible cells of *player_id*.

    A cell can eject when its mass exceeds ``eject_loss`` (so it stays
    viable after ejection).  The cell loses ``eject_loss`` mass; a pellet
    of mass ``eject_mass`` is created; the difference is discarded.

    Parameters
    ----------
    cells : CellArrays
        Modified in-place (mass reduced).
    ejected : EjectedArrays
        Modified in-place (new pellets added).
    player_id : int
    config : WorldConfig
    """
    p_idx = cells.player_indices(player_id)
    if len(p_idx) == 0:
        return

    # Only cells with enough mass can eject
    # Keep a small buffer above eject_loss so the cell doesn't become tiny
    min_mass_to_eject = config.eject_loss * 2.0
    can_eject_mask = cells.mass[p_idx] >= min_mass_to_eject
    eject_candidates = p_idx[can_eject_mask]

    if len(eject_candidates) == 0:
        return

    # Limit by available ejected slots
    n = min(len(eject_candidates), ejected.free_count())
    if n == 0:
        return
    eject_candidates = eject_candidates[:n]
    new_slots = ejected.allocate(n)

    # Deduct mass from ejecting cells
    cells.mass[eject_candidates] -= config.eject_loss

    # Launch direction: same as the cell's current movement direction
    vel = cells.vel[eject_candidates]                              # (n, 2)
    mag = np.linalg.norm(vel, axis=1, keepdims=True)              # (n, 1)
    safe_mag = np.where(mag > 1e-8, mag, np.float32(1.0))
    dir_vec = np.where(mag > 1e-8, vel / safe_mag, np.array([[1.0, 0.0]])).astype(np.float32)

    ejected.pos[new_slots] = cells.pos[eject_candidates] + dir_vec * (
        np.sqrt(cells.mass[eject_candidates])[:, None] + 1.0
    )
    ejected.vel[new_slots] = dir_vec * config.eject_speed
    ejected.owner[new_slots] = player_id
    ejected.settle_timer[new_slots] = config.eject_settle_ticks

    # Clamp spawn position to world bounds
    ejected.pos[new_slots, 0] = np.clip(ejected.pos[new_slots, 0], 0.0, config.width)
    ejected.pos[new_slots, 1] = np.clip(ejected.pos[new_slots, 1], 0.0, config.height)


# ---------------------------------------------------------------------------
# Virus feeding (ejected mass hitting a virus)
# ---------------------------------------------------------------------------

def resolve_virus_feeding(
    ejected: EjectedArrays,
    viruses: VirusArrays,
    rng: np.random.Generator,
    config: WorldConfig,
) -> None:
    """Check whether ejected mass hits and eventually splits a virus.

    When ``virus_feed_count`` pellets have hit a virus it ejects a new virus
    in a random direction and its feed_count resets to 0.

    Parameters
    ----------
    ejected : EjectedArrays
        Modified in-place (absorbed pellets removed).
    viruses : VirusArrays
        Modified in-place (feed_count incremented; new virus spawned).
    rng : numpy.random.Generator
    config : WorldConfig
    """
    ej_idx = ejected.alive_indices()
    vir_idx = viruses.alive_indices()

    if len(ej_idx) == 0 or len(vir_idx) == 0:
        return

    ej_pos = ejected.pos[ej_idx]              # (ne, 2)
    vir_pos = viruses.pos[vir_idx]            # (nv, 2)
    vir_rad = float(np.sqrt(config.virus_mass))

    # Distance from each ejected pellet to each virus
    diff = ej_pos[:, None, :] - vir_pos[None, :, :]   # (ne, nv, 2)
    dist_sq = (diff * diff).sum(axis=2)               # (ne, nv)
    hit = dist_sq < (vir_rad ** 2)                    # (ne, nv)

    if not hit.any():
        return

    # Each ejected pellet can only hit one virus (the closest)
    # For each ejected pellet, find which virus it hit (if any)
    hit_ej_local, hit_vir_local = np.where(hit)

    # Process each ejected → virus hit
    absorbed_ej: list[int] = []
    for ej_loc, vir_loc in zip(hit_ej_local.tolist(), hit_vir_local.tolist()):
        ej_glob = int(ej_idx[ej_loc])
        vir_glob = int(vir_idx[vir_loc])

        if not ejected.alive[ej_glob] or not viruses.alive[vir_glob]:
            continue

        absorbed_ej.append(ej_glob)
        viruses.feed_count[vir_glob] += 1

        if viruses.feed_count[vir_glob] >= config.virus_feed_count:
            viruses.feed_count[vir_glob] = 0
            # Spawn a new virus nearby if capacity allows
            new_v = viruses.allocate(1)
            if len(new_v) > 0:
                angle = float(rng.uniform(0, 2 * np.pi))
                offset = vir_rad * 2.5
                vx = viruses.pos[vir_glob, 0] + offset * np.cos(angle)
                vy = viruses.pos[vir_glob, 1] + offset * np.sin(angle)
                viruses.pos[new_v[0], 0] = float(np.clip(vx, 0, config.width))
                viruses.pos[new_v[0], 1] = float(np.clip(vy, 0, config.height))

    if absorbed_ej:
        ejected.free(np.array(absorbed_ej, dtype=np.int32))


# ---------------------------------------------------------------------------
# Virus-triggered cell split
# ---------------------------------------------------------------------------

def apply_virus_splits(
    cells: CellArrays,
    split_requests: list[tuple[int, int, np.ndarray]],
    config: WorldConfig,
) -> None:
    """Split cells that collided with viruses (from ``resolve_virus_collision``).

    Each hit cell is split into as many pieces as possible (up to
    ``max_cells_per_player``) with each fragment having half the prior mass,
    until the cell is smaller than the virus.

    Parameters
    ----------
    cells : CellArrays
        Modified in-place.
    split_requests : list of (cell_idx, virus_idx, direction)
        Output of ``collision.resolve_virus_collision``.
    config : WorldConfig
    """
    for cell_glob, _vir_glob, direction in split_requests:
        if not cells.alive[cell_glob]:
            continue

        player_id = int(cells.owner[cell_glob])
        # Current player cell count
        p_count = len(cells.player_indices(player_id))

        # Split up to fill max_cells_per_player, or until mass drops below virus mass
        while (
            cells.mass[cell_glob] > config.virus_mass
            and p_count < config.max_cells_per_player
            and cells.free_count() > 0
        ):
            (new_slot,) = cells.allocate(1)
            new_slot = int(new_slot)

            cells.mass[cell_glob] /= 2.0
            cells.mass[new_slot] = cells.mass[cell_glob]
            cells.pos[new_slot] = cells.pos[cell_glob]
            cells.owner[new_slot] = player_id

            timer = float(
                config.merge_time_base + cells.mass[cell_glob] / config.merge_time_mass_factor
            )
            cells.merge_timer[new_slot] = timer
            cells.merge_timer[cell_glob] = timer

            cells.split_vel[new_slot] = direction * config.split_speed
            cells.vel[new_slot] = cells.vel[cell_glob]

            p_count += 1
