from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WorldConfig:
    """Immutable configuration for one agar.io world simulation.

    All numeric constants are calibrated so that:
      - ``radius = sqrt(mass)``  (no separate scale factor)
      - A fresh player cell has mass 2500 → radius 50 world-units
      - Food has mass 25 → radius 5 world-units
      - The world is 14 142 × 14 142 units (matching real agar.io map)

    Parameters
    ----------
    width, height : float
        World dimensions in world-units.
    dt : float
        Seconds per tick.  Default is 1/25 (25 TPS).
    max_cells : int
        Total pre-allocated cell slots (all players, all splits combined).
    max_food : int
        Total pre-allocated food slots.
    max_viruses : int
        Total pre-allocated virus slots.
    max_ejected : int
        Total pre-allocated ejected-mass slots.
    max_players : int
        Maximum concurrent players (human + agent).
    start_mass : float
        Mass of a freshly spawned player cell.
    food_mass : float
        Mass of a food pellet.
    virus_mass : float
        Mass (and squared radius) of a virus.
    min_split_mass : float
        Minimum cell mass required to perform a split action.
    max_cells_per_player : int
        Maximum number of live cell fragments per player.
    merge_time_base : float
        Ticks a split cell must wait before it can merge back.
        Full formula (matching agar.io): merge_time = merge_time_base + mass / merge_time_mass_factor.
    merge_time_mass_factor : float
        Divisor for the mass-dependent part of merge time.
    mass_decay_rate : float
        Fraction of mass lost per tick for cells above mass_decay_threshold.
    mass_decay_threshold : float
        Only cells with mass above this value lose mass each tick.
    base_speed : float
        Speed constant in ``speed = base_speed / mass ** speed_exp``.
    speed_exp : float
        Mass exponent for speed formula.
    split_speed : float
        Initial speed of a newly split fragment (world-units / tick).
    split_decay : float
        Per-tick multiplicative decay applied to split_vel.
    eat_ratio : float
        Predator radius must exceed ``eat_ratio * prey_radius`` to eat.
        Equivalently predator mass > eat_ratio**2 * prey mass.
    target_food_count : int
        Desired number of live food pellets; spawner maintains this level.
    target_virus_count : int
        Desired number of live viruses; spawner maintains this level.
    virus_feed_count : int
        Number of ejected-mass hits needed to split a virus.
    eject_mass : float
        Mass of the fired ejected-mass pellet (awarded to whoever absorbs it).
    eject_loss : float
        Mass removed from the ejecting cell (> eject_mass; the rest is "wasted").
    eject_speed : float
        Launch speed of ejected mass (world-units / tick).
    eject_settle_ticks : int
        Ticks after ejection before the pellet can be absorbed by the ejector's
        own cells.
    """

    # ── World ──────────────────────────────────────────────────────────────
    width: float = 14_142.0
    height: float = 14_142.0

    # ── Tick ───────────────────────────────────────────────────────────────
    dt: float = 1.0 / 25.0  # 25 ticks per second

    # ── Buffer capacities ──────────────────────────────────────────────────
    max_cells: int = 1024
    max_food: int = 4096
    max_viruses: int = 64
    max_ejected: int = 512
    max_players: int = 16

    # ── Cell / mass ────────────────────────────────────────────────────────
    start_mass: float = 2_500.0   # → radius ≈ 50 units
    food_mass: float = 25.0       # → radius  = 5 units
    virus_mass: float = 2_500.0   # → radius ≈ 50 units
    min_split_mass: float = 2_500.0
    max_cells_per_player: int = 16
    merge_time_base: float = 100.0         # ticks (= 4 s at 25 TPS)
    merge_time_mass_factor: float = 50.0   # adds mass/50 ticks to merge time
    mass_decay_rate: float = 0.0           # disabled — set to e.g. 0.002 to enable
    # (0.002 ≈ 0.2 %/tick = ~5 %/s at 25 TPS for cells above mass_decay_threshold)
    # Creates pressure to keep eating to maintain size; enable via configs/default.yaml
    # by adding: world: { mass_decay_rate: 0.002 }
    mass_decay_threshold: float = 10_000.0

    # ── Physics ────────────────────────────────────────────────────────────
    base_speed: float = 20000.0
    speed_exp: float = 0.439
    split_speed: float = 3000.0
    split_decay: float = 0.9   # per tick; after 25 ticks ≈ 7 % remains

    # ── Eating ─────────────────────────────────────────────────────────────
    eat_ratio: float = 1.1   # → mass threshold = 1.21 ×

    # ── Food ───────────────────────────────────────────────────────────────
    target_food_count: int = 2_000

    # ── Viruses ────────────────────────────────────────────────────────────
    target_virus_count: int = 12
    virus_feed_count: int = 7

    # ── Ejected mass ───────────────────────────────────────────────────────
    eject_mass: float = 13.0
    eject_loss: float = 43.0
    eject_speed: float = 780.0
    eject_settle_ticks: int = 25

    def __post_init__(self) -> None:
        assert self.width > 0 and self.height > 0, "World dimensions must be positive."
        assert 0 < self.dt <= 1.0, "dt must be in (0, 1]."
        assert self.max_cells >= self.max_players, "max_cells must be >= max_players."
        assert self.eat_ratio > 1.0, "eat_ratio must be > 1."
        assert self.split_decay < 1.0, "split_decay must be < 1 (it is a per-tick decay)."
        assert self.eject_loss >= self.eject_mass, "eject_loss must be >= eject_mass."

    @property
    def eat_mass_ratio(self) -> float:
        """Mass ratio threshold for eating: predator.mass > eat_mass_ratio * prey.mass."""
        return self.eat_ratio**2
