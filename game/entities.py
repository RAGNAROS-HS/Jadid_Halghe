from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_free(capacity: int) -> deque[int]:
    return deque(range(capacity))


# ---------------------------------------------------------------------------
# Cell arrays
# ---------------------------------------------------------------------------

@dataclass
class CellArrays:
    """Pre-allocated buffers for all player-controlled cell fragments.

    All fields are NumPy arrays indexed by slot number.  Only slots where
    ``alive[i]`` is True contain valid data.  Slots are recycled via a
    free-list; the free-list is managed in Python (allocation is not on the
    hot path).

    Radius is derived on demand: ``radius = sqrt(mass)``.

    Parameters
    ----------
    capacity : int
        Total number of pre-allocated slots.
    pos : ndarray, shape (capacity, 2), float32
        World-space position (x, y).
    vel : ndarray, shape (capacity, 2), float32
        Input-driven velocity for this tick.  Recomputed each tick from the
        owning player's action direction and cell speed.
    split_vel : ndarray, shape (capacity, 2), float32
        Extra momentum injected at split time; decays each tick by
        ``config.split_decay``.
    mass : ndarray, shape (capacity,), float32
        Cell mass.  ``radius = sqrt(mass)``.
    owner : ndarray, shape (capacity,), int32
        Player ID of the owning player.  -1 means the slot is unused.
    merge_timer : ndarray, shape (capacity,), float32
        Remaining ticks before this cell may merge with a sibling.
        0 means merge is allowed.
    alive : ndarray, shape (capacity,), bool
        Validity mask.  Dead slots must not be read by the simulation.
    """

    capacity: int
    pos: np.ndarray
    vel: np.ndarray
    split_vel: np.ndarray
    mass: np.ndarray
    owner: np.ndarray
    merge_timer: np.ndarray
    alive: np.ndarray

    _free: deque[int] = field(default_factory=deque, init=False, repr=False, compare=False)
    _count: int = field(default=0, init=False, repr=False, compare=False)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def create(cls, capacity: int) -> CellArrays:
        """Allocate zero-initialised buffers for *capacity* cell slots."""
        obj = cls(
            capacity=capacity,
            pos=np.zeros((capacity, 2), dtype=np.float32),
            vel=np.zeros((capacity, 2), dtype=np.float32),
            split_vel=np.zeros((capacity, 2), dtype=np.float32),
            mass=np.zeros(capacity, dtype=np.float32),
            owner=np.full(capacity, -1, dtype=np.int32),
            merge_timer=np.zeros(capacity, dtype=np.float32),
            alive=np.zeros(capacity, dtype=bool),
        )
        obj._free = _make_free(capacity)
        obj._count = 0
        return obj

    # ------------------------------------------------------------------
    # Slot management (not on hot path)
    # ------------------------------------------------------------------

    def allocate(self, n: int = 1) -> np.ndarray:
        """Return *n* unused slot indices and mark them alive.

        Raises
        ------
        RuntimeError
            If fewer than *n* free slots remain.
        """
        if len(self._free) < n:
            raise RuntimeError(
                f"CellArrays capacity exhausted: need {n}, have {len(self._free)}"
            )
        indices = np.array([self._free.popleft() for _ in range(n)], dtype=np.int32)
        self.alive[indices] = True
        self._count += n
        return indices

    def free(self, indices: np.ndarray) -> None:
        """Return *indices* to the free pool and reset their alive flag."""
        indices = np.asarray(indices, dtype=np.int32).ravel()
        self.alive[indices] = False
        self.owner[indices] = -1
        self._free.extend(indices.tolist())
        self._count -= len(indices)

    def free_count(self) -> int:
        """Number of available (unused) slots."""
        return len(self._free)

    # ------------------------------------------------------------------
    # Derived quantities
    # ------------------------------------------------------------------

    @property
    def count(self) -> int:
        """Number of live cells (O(1) — maintained by allocate/free)."""
        return self._count

    def radius(self) -> np.ndarray:
        """Return radius array (sqrt of mass) for all slots.

        Returns
        -------
        ndarray, shape (capacity,), float32
            ``sqrt(max(mass, 0))`` for every slot.
        """
        return np.sqrt(np.maximum(self.mass, 0.0, out=np.empty_like(self.mass)))

    def alive_indices(self) -> np.ndarray:
        """Indices of all live cells as an int32 array."""
        return np.where(self.alive)[0].astype(np.int32)

    def player_indices(self, player_id: int) -> np.ndarray:
        """Indices of all live cells owned by *player_id*."""
        mask = self.alive & (self.owner == player_id)
        return np.where(mask)[0].astype(np.int32)

    def player_mass(self, player_id: int) -> float:
        """Total mass of all live cells owned by *player_id*."""
        idx = self.player_indices(player_id)
        return float(self.mass[idx].sum()) if len(idx) > 0 else 0.0


# ---------------------------------------------------------------------------
# Food arrays
# ---------------------------------------------------------------------------

@dataclass
class FoodArrays:
    """Pre-allocated buffers for food pellets.

    Parameters
    ----------
    capacity : int
        Total number of pre-allocated slots.
    pos : ndarray, shape (capacity, 2), float32
        World-space position (x, y).
    alive : ndarray, shape (capacity,), bool
        Validity mask.
    """

    capacity: int
    pos: np.ndarray
    alive: np.ndarray

    _free: deque[int] = field(default_factory=deque, init=False, repr=False, compare=False)
    _count: int = field(default=0, init=False, repr=False, compare=False)

    @classmethod
    def create(cls, capacity: int) -> FoodArrays:
        """Allocate zero-initialised buffers for *capacity* food slots."""
        obj = cls(
            capacity=capacity,
            pos=np.zeros((capacity, 2), dtype=np.float32),
            alive=np.zeros(capacity, dtype=bool),
        )
        obj._free = _make_free(capacity)
        obj._count = 0
        return obj

    def allocate(self, n: int) -> np.ndarray:
        """Return up to *n* free slot indices (silently caps at available count)."""
        n = min(n, len(self._free))
        if n == 0:
            return np.empty(0, dtype=np.int32)
        indices = np.array([self._free.popleft() for _ in range(n)], dtype=np.int32)
        self.alive[indices] = True
        self._count += n
        return indices

    def free(self, indices: np.ndarray) -> None:
        """Return *indices* to the free pool."""
        indices = np.asarray(indices, dtype=np.int32).ravel()
        self.alive[indices] = False
        self._free.extend(indices.tolist())
        self._count -= len(indices)

    def free_count(self) -> int:
        return len(self._free)

    @property
    def count(self) -> int:
        """Number of live food pellets (O(1))."""
        return self._count

    def alive_indices(self) -> np.ndarray:
        return np.where(self.alive)[0].astype(np.int32)


# ---------------------------------------------------------------------------
# Virus arrays
# ---------------------------------------------------------------------------

@dataclass
class VirusArrays:
    """Pre-allocated buffers for viruses.

    Parameters
    ----------
    capacity : int
        Total number of pre-allocated slots.
    pos : ndarray, shape (capacity, 2), float32
        World-space position (x, y).
    feed_count : ndarray, shape (capacity,), int32
        Number of ejected-mass hits absorbed so far.  When this reaches
        ``config.virus_feed_count`` the virus splits.
    alive : ndarray, shape (capacity,), bool
        Validity mask.
    """

    capacity: int
    pos: np.ndarray
    feed_count: np.ndarray
    alive: np.ndarray

    _free: deque[int] = field(default_factory=deque, init=False, repr=False, compare=False)
    _count: int = field(default=0, init=False, repr=False, compare=False)

    @classmethod
    def create(cls, capacity: int) -> VirusArrays:
        obj = cls(
            capacity=capacity,
            pos=np.zeros((capacity, 2), dtype=np.float32),
            feed_count=np.zeros(capacity, dtype=np.int32),
            alive=np.zeros(capacity, dtype=bool),
        )
        obj._free = _make_free(capacity)
        obj._count = 0
        return obj

    def allocate(self, n: int = 1) -> np.ndarray:
        if len(self._free) < n:
            n = len(self._free)
        if n == 0:
            return np.empty(0, dtype=np.int32)
        indices = np.array([self._free.popleft() for _ in range(n)], dtype=np.int32)
        self.alive[indices] = True
        self.feed_count[indices] = 0
        self._count += len(indices)
        return indices

    def free(self, indices: np.ndarray) -> None:
        indices = np.asarray(indices, dtype=np.int32).ravel()
        self.alive[indices] = False
        self._free.extend(indices.tolist())
        self._count -= len(indices)

    def free_count(self) -> int:
        return len(self._free)

    @property
    def count(self) -> int:
        """Number of live viruses (O(1))."""
        return self._count

    def alive_indices(self) -> np.ndarray:
        return np.where(self.alive)[0].astype(np.int32)


# ---------------------------------------------------------------------------
# Ejected-mass arrays
# ---------------------------------------------------------------------------

@dataclass
class EjectedArrays:
    """Pre-allocated buffers for ejected-mass pellets.

    Ejected mass is fired by a player cell and travels until it slows down.
    After ``settle_timer`` reaches 0 it can be absorbed by the ejector's own
    cells; it can always be absorbed by enemy cells.

    Parameters
    ----------
    capacity : int
        Total number of pre-allocated slots.
    pos : ndarray, shape (capacity, 2), float32
        World-space position (x, y).
    vel : ndarray, shape (capacity, 2), float32
        Current velocity.
    owner : ndarray, shape (capacity,), int32
        Player ID of the cell that fired this pellet.
    settle_timer : ndarray, shape (capacity,), int32
        Ticks remaining before own-cell absorption is permitted.
    alive : ndarray, shape (capacity,), bool
        Validity mask.
    """

    capacity: int
    pos: np.ndarray
    vel: np.ndarray
    owner: np.ndarray
    settle_timer: np.ndarray
    alive: np.ndarray

    _free: deque[int] = field(default_factory=deque, init=False, repr=False, compare=False)
    _count: int = field(default=0, init=False, repr=False, compare=False)

    @classmethod
    def create(cls, capacity: int) -> EjectedArrays:
        obj = cls(
            capacity=capacity,
            pos=np.zeros((capacity, 2), dtype=np.float32),
            vel=np.zeros((capacity, 2), dtype=np.float32),
            owner=np.full(capacity, -1, dtype=np.int32),
            settle_timer=np.zeros(capacity, dtype=np.int32),
            alive=np.zeros(capacity, dtype=bool),
        )
        obj._free = _make_free(capacity)
        obj._count = 0
        return obj

    def allocate(self, n: int = 1) -> np.ndarray:
        if len(self._free) < n:
            n = len(self._free)
        if n == 0:
            return np.empty(0, dtype=np.int32)
        indices = np.array([self._free.popleft() for _ in range(n)], dtype=np.int32)
        self.alive[indices] = True
        self._count += len(indices)
        return indices

    def free(self, indices: np.ndarray) -> None:
        indices = np.asarray(indices, dtype=np.int32).ravel()
        self.alive[indices] = False
        self.owner[indices] = -1
        self._free.extend(indices.tolist())
        self._count -= len(indices)

    def free_count(self) -> int:
        return len(self._free)

    @property
    def count(self) -> int:
        """Number of live ejected pellets (O(1))."""
        return self._count

    def alive_indices(self) -> np.ndarray:
        return np.where(self.alive)[0].astype(np.int32)
