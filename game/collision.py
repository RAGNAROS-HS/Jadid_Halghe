from __future__ import annotations

import numpy as np

from game.config import WorldConfig
from game.entities import CellArrays, EjectedArrays, FoodArrays, VirusArrays


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _pairwise_dist_sq(pos_a: np.ndarray, pos_b: np.ndarray) -> np.ndarray:
    """Return squared pairwise distances using the dot-product identity.

    ``‖a − b‖² = ‖a‖² + ‖b‖² − 2 aᵀb``

    This avoids allocating the ``(na, nb, 2)`` intermediate array that the
    naive ``(a[:,None,:] − b[None,:,:]) ** 2).sum(axis=2)`` approach
    requires, which dominates memory bandwidth at large food counts.

    Parameters
    ----------
    pos_a : ndarray, shape (na, 2), float32
    pos_b : ndarray, shape (nb, 2), float32

    Returns
    -------
    ndarray, shape (na, nb), float32
    """
    sq_a = (pos_a * pos_a).sum(axis=1)            # (na,)
    sq_b = (pos_b * pos_b).sum(axis=1)            # (nb,)
    cross = pos_a @ pos_b.T                        # (na, nb) — BLAS SGEMM
    dist_sq = sq_a[:, None] + sq_b[None, :] - 2.0 * cross
    # Clamp negatives from floating-point round-off
    np.maximum(dist_sq, 0.0, out=dist_sq)
    return dist_sq


# ---------------------------------------------------------------------------
# Cell-eats-food
# ---------------------------------------------------------------------------

def resolve_food_eating(
    cells: CellArrays,
    food: FoodArrays,
    config: WorldConfig,
) -> np.ndarray:
    """Award food mass to cells whose centre overlaps a food pellet.

    A cell at position ``p`` with radius ``r`` eats a food pellet at
    position ``q`` when ``‖p − q‖ < r``.

    Parameters
    ----------
    cells : CellArrays
        Modified in-place (mass increased for eating cells).
    food : FoodArrays
        Modified in-place (eaten food marked dead).
    config : WorldConfig

    Returns
    -------
    ndarray, shape (max_players,), float32
        Mass gained from food per player this tick (used for reward shaping).

    Notes
    -----
    Uses O(nc × nf) broadcasting where nc = live cells, nf = live food.
    With typical counts (nc ≤ 200, nf ≤ 2 000) this is fast enough for
    10 000 TPS; add a spatial grid if counts grow significantly larger.
    """
    cell_idx = cells.alive_indices()
    food_idx = food.alive_indices()
    food_gains = np.zeros(config.max_players, dtype=np.float32)

    if len(cell_idx) == 0 or len(food_idx) == 0:
        return food_gains

    cell_pos = cells.pos[cell_idx]          # (nc, 2)
    cell_rad = np.sqrt(cells.mass[cell_idx])  # (nc,)  radius = sqrt(mass)
    food_pos = food.pos[food_idx]           # (nf, 2)

    dist_sq = _pairwise_dist_sq(cell_pos, food_pos)   # (nc, nf)
    rad_sq = (cell_rad ** 2)[:, None]                  # (nc, 1)

    # cell i eats food j when dist_sq[i,j] < cell_rad[i]^2
    eats = dist_sq < rad_sq   # (nc, nf) bool

    # For each food pellet, pick the first (lowest-index) eater.
    # Multiple cells can overlap the same food; we give it to the first.
    # np.argmax on bool array finds first True along axis 0.
    any_eater = eats.any(axis=0)                      # (nf,)
    eaten_local = np.where(any_eater)[0]              # local food indices
    if len(eaten_local) == 0:
        return food_gains

    eater_local = np.argmax(eats[:, eaten_local], axis=0)  # (n_eaten,)

    eaten_global = food_idx[eaten_local]
    eater_global = cell_idx[eater_local]

    # Award mass
    np.add.at(cells.mass, eater_global, config.food_mass)

    # Accumulate per-player gains
    owners = cells.owner[eater_global]
    valid = (owners >= 0) & (owners < config.max_players)
    np.add.at(food_gains, owners[valid], config.food_mass)

    # Kill eaten food
    food.free(eaten_global)

    return food_gains


# ---------------------------------------------------------------------------
# Cell-eats-cell
# ---------------------------------------------------------------------------

def resolve_cell_eating(
    cells: CellArrays,
    config: WorldConfig,
) -> np.ndarray:
    """Resolve predator–prey eating between player cells.

    Eating rules
    ------------
    Cell A eats cell B when ALL of the following hold:

    * ``mass_A > eat_mass_ratio × mass_B``  (A is large enough)
    * ``‖pos_A − pos_B‖ < radius_A``        (B's centre is inside A)
    * ``owner_A ≠ owner_B``                 (different players)

    If multiple cells could eat the same prey, the one with the highest
    mass wins.

    Parameters
    ----------
    cells : CellArrays
        Modified in-place.
    config : WorldConfig

    Returns
    -------
    ndarray, shape (max_players,), float32
        Mass gained from eating enemy cells per player this tick.
    """
    idx = cells.alive_indices()
    gains = np.zeros(config.max_players, dtype=np.float32)

    if len(idx) < 2:
        return gains

    pos = cells.pos[idx]       # (n, 2)
    mass = cells.mass[idx]     # (n,)
    owner = cells.owner[idx]   # (n,)
    radius = np.sqrt(mass)     # (n,)

    dist_sq = _pairwise_dist_sq(pos, pos)                       # (n, n)
    rad_sq_pred = (radius ** 2)[:, None]                        # (n, 1)

    # Eating conditions (i is predator, j is prey)
    mass_ok = mass[:, None] > config.eat_mass_ratio * mass[None, :]   # (n, n)
    dist_ok = dist_sq < rad_sq_pred                                    # (n, n)
    diff_owner = owner[:, None] != owner[None, :]                      # (n, n)
    np.fill_diagonal(mass_ok, False)

    can_eat = mass_ok & dist_ok & diff_owner   # (n, n)

    if not can_eat.any():
        return gains

    # For each prey j, choose the heaviest predator
    mass_if_pred = np.where(can_eat, mass[:, None], 0.0)   # (n, n)
    best_pred_local = np.argmax(mass_if_pred, axis=0)       # (n,)
    has_pred = mass_if_pred[best_pred_local, np.arange(len(idx))] > 0  # (n,)

    prey_local = np.where(has_pred)[0]
    pred_local = best_pred_local[prey_local]

    # Avoid self-eat (should be impossible, but guard anyway)
    valid = pred_local != prey_local
    prey_local = prey_local[valid]
    pred_local = pred_local[valid]

    if len(prey_local) == 0:
        return gains

    prey_global = idx[prey_local]
    pred_global = idx[pred_local]

    # Award mass to predators
    np.add.at(cells.mass, pred_global, cells.mass[prey_global])

    # Accumulate per-player gains
    pred_owners = cells.owner[pred_global]
    prey_mass = cells.mass[prey_global].copy()
    valid_own = (pred_owners >= 0) & (pred_owners < config.max_players)
    np.add.at(gains, pred_owners[valid_own], prey_mass[valid_own])

    # Kill eaten cells
    cells.free(prey_global)

    return gains


# ---------------------------------------------------------------------------
# Cell absorbs ejected mass
# ---------------------------------------------------------------------------

def resolve_ejected_eating(
    cells: CellArrays,
    ejected: EjectedArrays,
    config: WorldConfig,
) -> None:
    """Allow cells to absorb settled ejected-mass pellets.

    Rules
    -----
    * Any cell can absorb ejected mass from *enemy* players immediately.
    * Own-cell absorption is blocked until ``settle_timer == 0``.
    * Eating condition: ``‖cell_pos − ejected_pos‖ < cell_radius``.

    Parameters
    ----------
    cells : CellArrays
        Modified in-place (mass increased).
    ejected : EjectedArrays
        Modified in-place (absorbed pellets marked dead).
    config : WorldConfig
    """
    cell_idx = cells.alive_indices()
    ej_idx = ejected.alive_indices()

    if len(cell_idx) == 0 or len(ej_idx) == 0:
        return

    cell_pos = cells.pos[cell_idx]             # (nc, 2)
    cell_rad = np.sqrt(cells.mass[cell_idx])   # (nc,)
    cell_own = cells.owner[cell_idx]           # (nc,)
    ej_pos = ejected.pos[ej_idx]               # (ne, 2)
    ej_own = ejected.owner[ej_idx]             # (ne,)
    ej_settled = ejected.settle_timer[ej_idx] == 0  # (ne,)

    dist_sq = _pairwise_dist_sq(cell_pos, ej_pos)   # (nc, ne)
    rad_sq = (cell_rad ** 2)[:, None]               # (nc, 1)
    dist_ok = dist_sq < rad_sq                      # (nc, ne)

    # Own cells can only absorb after settle; enemy cells can absorb always
    same_owner = cell_own[:, None] == ej_own[None, :]          # (nc, ne)
    own_allowed = ~same_owner | ej_settled[None, :]            # (nc, ne)
    can_absorb = dist_ok & own_allowed                         # (nc, ne)

    if not can_absorb.any():
        return

    any_absorber = can_absorb.any(axis=0)                       # (ne,)
    absorbed_local = np.where(any_absorber)[0]
    absorber_local = np.argmax(can_absorb[:, absorbed_local], axis=0)

    absorbed_global = ej_idx[absorbed_local]
    absorber_global = cell_idx[absorber_local]

    np.add.at(cells.mass, absorber_global, config.eject_mass)
    ejected.free(absorbed_global)


# ---------------------------------------------------------------------------
# Cell hits virus
# ---------------------------------------------------------------------------

def resolve_virus_collision(
    cells: CellArrays,
    viruses: VirusArrays,
    config: WorldConfig,
) -> list[tuple[int, int, np.ndarray]]:
    """Detect cells colliding with viruses and return split requests.

    A cell collides with a virus when:

    * ``‖cell_pos − virus_pos‖ < cell_radius + virus_radius``  (circles overlap)
    * ``cell_mass > virus_mass``  (cell must be larger than the virus)

    Rather than performing the split here (which requires spawner logic),
    this function returns a list of ``(cell_global_idx, virus_global_idx,
    cell_direction)`` tuples for the spawner to process.

    Parameters
    ----------
    cells : CellArrays
    viruses : VirusArrays
    config : WorldConfig

    Returns
    -------
    list of (int, int, ndarray shape (2,))
        Each tuple: (cell_idx, virus_idx, movement_direction).
    """
    cell_idx = cells.alive_indices()
    vir_idx = viruses.alive_indices()
    splits: list[tuple[int, int, np.ndarray]] = []

    if len(cell_idx) == 0 or len(vir_idx) == 0:
        return splits

    cell_pos = cells.pos[cell_idx]           # (nc, 2)
    cell_mass = cells.mass[cell_idx]         # (nc,)
    cell_vel = cells.vel[cell_idx]           # (nc, 2) — split direction
    cell_rad = np.sqrt(cell_mass)            # (nc,)
    vir_pos = viruses.pos[vir_idx]           # (nv, 2)
    vir_rad = np.sqrt(config.virus_mass)     # scalar

    dist_sq = _pairwise_dist_sq(cell_pos, vir_pos)                 # (nc, nv)
    hit_rad = (cell_rad + vir_rad)[:, None]                        # (nc, 1)
    collides = dist_sq < (hit_rad ** 2)                            # (nc, nv)
    big_enough = (cell_mass > config.virus_mass)[:, None]          # (nc, 1)
    hits = collides & big_enough                                   # (nc, nv)

    if not hits.any():
        return splits

    hit_cells_local, hit_virs_local = np.where(hits)

    # Each (cell, virus) pair triggers a split; deduplicate by cell (one split per cell)
    seen_cells: set[int] = set()
    for c_loc, v_loc in zip(hit_cells_local.tolist(), hit_virs_local.tolist()):
        if c_loc in seen_cells:
            continue
        seen_cells.add(c_loc)
        c_glob = int(cell_idx[c_loc])
        v_glob = int(vir_idx[v_loc])
        vel = cells.vel[c_glob]
        mag = float(np.linalg.norm(vel))
        direction = (vel / mag).astype(np.float32) if mag > 1e-8 else np.array([1.0, 0.0], dtype=np.float32)
        splits.append((c_glob, v_glob, direction))

    return splits


# ---------------------------------------------------------------------------
# Cell-cell merging (same player)
# ---------------------------------------------------------------------------

def resolve_merging(
    cells: CellArrays,
    config: WorldConfig,
) -> None:
    """Merge sibling cells of the same player when they overlap and are ready.

    Two cells of the same player merge when:

    * Both have ``merge_timer == 0``.
    * Their circles overlap: ``‖pos_i − pos_j‖ < radius_i + radius_j``.

    The smaller cell is absorbed into the larger one.  Processing continues
    until no more mergeable pairs remain (handles chain merges in one call).

    Parameters
    ----------
    cells : CellArrays
        Modified in-place.
    config : WorldConfig
    """
    # Iterate until stable (typically 0–1 passes)
    while True:
        idx = cells.alive_indices()
        if len(idx) < 2:
            return

        pos = cells.pos[idx]          # (n, 2)
        mass = cells.mass[idx]        # (n,)
        owner = cells.owner[idx]      # (n,)
        timer = cells.merge_timer[idx]  # (n,)
        radius = np.sqrt(mass)         # (n,)

        # Same owner, both ready
        same_owner = owner[:, None] == owner[None, :]       # (n, n)
        both_ready = (timer == 0)[:, None] & (timer == 0)[None, :]  # (n, n)

        dist_sq = _pairwise_dist_sq(pos, pos)               # (n, n)
        merge_rad = (radius[:, None] + radius[None, :])     # (n, n)
        overlaps = dist_sq < (merge_rad ** 2)               # (n, n)

        can_merge = same_owner & both_ready & overlaps      # (n, n)
        np.fill_diagonal(can_merge, False)

        # Only look at upper triangle to avoid processing each pair twice
        upper = np.triu(can_merge)
        if not upper.any():
            return

        # Merge pairs: smaller into larger
        rows, cols = np.where(upper)
        # Sort pairs so larger is always row (predator convention)
        for r, c in zip(rows.tolist(), cols.tolist()):
            gi = int(idx[r])
            gj = int(idx[c])
            # Guard: both still alive (earlier iteration may have freed one)
            if not cells.alive[gi] or not cells.alive[gj]:
                continue
            # Larger absorbs smaller
            if cells.mass[gi] >= cells.mass[gj]:
                cells.mass[gi] += cells.mass[gj]
                cells.free(np.array([gj], dtype=np.int32))
            else:
                cells.mass[gj] += cells.mass[gi]
                cells.free(np.array([gi], dtype=np.int32))
