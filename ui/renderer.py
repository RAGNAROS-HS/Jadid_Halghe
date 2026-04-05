"""Pygame renderer — draws all world entities onto the display surface."""

from __future__ import annotations

import math
from dataclasses import dataclass

import pygame
import numpy as np

from game.world import GameState
from ui.camera import Camera


# ---------------------------------------------------------------------------
# Attention weight overlay
# ---------------------------------------------------------------------------

@dataclass
class AttentionWeights:
    """Per-entity attention weights for the glow overlay.

    Both arrays have the same length as the corresponding sub-arrays of the
    current :class:`~game.world.GameState`.  Entries are ``0.0`` for entities
    that were not included in the observed k-nearest tokens.

    Parameters
    ----------
    food_weight : ndarray, shape (n_food,)
        Normalised attention score for each food pellet in ``state.food_pos``.
    cell_weight : ndarray, shape (n_cells,)
        Normalised attention score for each enemy cell in ``state.cell_pos``
        (non-enemy cells are always ``0.0``).
    """

    food_weight: np.ndarray   # shape (n_food,), float32
    cell_weight: np.ndarray   # shape (n_cells,), float32


# ---------------------------------------------------------------------------
# Colour palette (one per player slot, up to 16 players)
# ---------------------------------------------------------------------------

_PLAYER_COLOURS: list[tuple[int, int, int]] = [
    (220,  80,  80),  # 0 — red (human)
    ( 80, 160, 220),  # 1 — blue
    ( 80, 220, 100),  # 2 — green
    (220, 200,  60),  # 3 — yellow
    (200,  80, 220),  # 4 — purple
    ( 60, 200, 200),  # 5 — cyan
    (220, 140,  60),  # 6 — orange
    (140, 220,  60),  # 7 — lime
    (220,  60, 140),  # 8 — pink
    ( 60, 100, 220),  # 9 — indigo
    (160, 100,  60),  # 10 — brown
    (100, 180, 180),  # 11 — teal
    (180, 180,  60),  # 12 — olive
    (180,  60, 180),  # 13 — violet
    ( 60, 180, 100),  # 14 — sea-green
    (180, 100, 100),  # 15 — rose
]

_FOOD_COLOUR = (80, 210, 80)
_FOOD_DARK_COLOUR = (50, 160, 50)
_VIRUS_COLOUR = (50, 200, 50)
_VIRUS_BORDER = (20, 140, 20)
_EJECTED_ALPHA = 180   # used for ejected circles (drawn slightly transparent)
_BG_COLOUR = (18, 18, 28)
_GRID_COLOUR = (30, 30, 48)
_GRID_SPACING = 500    # world units between grid lines


def _player_colour(player_id: int) -> tuple[int, int, int]:
    return _PLAYER_COLOURS[player_id % len(_PLAYER_COLOURS)]


def _dim(colour: tuple[int, int, int], factor: float) -> tuple[int, int, int]:
    return (
        int(colour[0] * factor),
        int(colour[1] * factor),
        int(colour[2] * factor),
    )


class Renderer:
    """Draws one frame to a Pygame surface.

    Parameters
    ----------
    screen : pygame.Surface
        The display surface.
    font_size : int
        Font size for cell labels.
    """

    def __init__(self, screen: pygame.Surface, font_size: int = 14) -> None:
        self.screen = screen
        self.screen_w = screen.get_width()
        self.screen_h = screen.get_height()

        pygame.font.init()
        self.font = pygame.font.SysFont("Arial", font_size, bold=True)
        self.small_font = pygame.font.SysFont("Arial", max(10, font_size - 4))

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def draw(
        self,
        state: GameState,
        camera: Camera,
        human_id: int,
        attention: AttentionWeights | None = None,
    ) -> None:
        """Render a full frame.

        Parameters
        ----------
        state : GameState
            Current world snapshot.
        camera : Camera
            Current viewport.
        human_id : int
            Player slot of the human; their cells are drawn with a border.
        attention : AttentionWeights, optional
            If provided, draws a glow halo on food/enemy cells proportional
            to their attention weight.  Computed once per simulation tick in
            ``main.py`` and passed through unchanged each render frame.
        """
        self.screen.fill(_BG_COLOUR)
        self._draw_grid(camera)
        self._draw_food(state, camera, attention)
        self._draw_viruses(state, camera)
        self._draw_ejected(state, camera)
        self._draw_cells(state, camera, human_id, attention)

    # ------------------------------------------------------------------
    # Grid
    # ------------------------------------------------------------------

    def _draw_grid(self, camera: Camera) -> None:
        """Draw a background grid aligned to world coordinates."""
        spacing = _GRID_SPACING
        w, h = camera.world_w, camera.world_h

        # Vertical lines
        x = 0.0
        while x <= w:
            sx, _ = camera.world_to_screen(x, 0.0)
            if 0 <= sx <= self.screen_w:
                pygame.draw.line(
                    self.screen, _GRID_COLOUR,
                    (int(sx), 0), (int(sx), self.screen_h), 1,
                )
            x += spacing

        # Horizontal lines
        y = 0.0
        while y <= h:
            _, sy = camera.world_to_screen(0.0, y)
            if 0 <= sy <= self.screen_h:
                pygame.draw.line(
                    self.screen, _GRID_COLOUR,
                    (0, int(sy)), (self.screen_w, int(sy)), 1,
                )
            y += spacing

    # ------------------------------------------------------------------
    # Food
    # ------------------------------------------------------------------

    def _draw_food(
        self,
        state: GameState,
        camera: Camera,
        attention: AttentionWeights | None = None,
    ) -> None:
        n = len(state.food_pos)
        if n == 0:
            return

        food_r_world = 5.0   # food radius = sqrt(food_mass=25) = 5
        radii = np.full(n, food_r_world, dtype=np.float32)
        mask = camera.visible_mask(state.food_pos, radii)
        visible_indices = np.where(mask)[0]
        visible_pos = state.food_pos[visible_indices]

        if len(visible_pos) == 0:
            return

        screen_xy = camera.world_to_screen_arr(visible_pos)
        r_px = max(1, int(camera.world_radius_to_screen(food_r_world)))

        # Pre-extract attention weights for visible food (None → all zero)
        food_w: np.ndarray | None = None
        if attention is not None and len(attention.food_weight) == n:
            food_w = attention.food_weight[visible_indices]

        for k, (sx, sy) in enumerate(screen_xy):
            ix, iy = int(sx), int(sy)

            # Attention glow: drawn first so the food circle appears on top
            if food_w is not None:
                w = float(food_w[k])
                if w > 0.05:
                    glow_r = max(r_px + 1, int(r_px * (1.0 + 2.5 * w)))
                    _draw_glow(self.screen, _FOOD_COLOUR, ix, iy, glow_r, int(180 * w))

            pygame.draw.circle(self.screen, _FOOD_COLOUR, (ix, iy), r_px)

    # ------------------------------------------------------------------
    # Viruses
    # ------------------------------------------------------------------

    def _draw_viruses(self, state: GameState, camera: Camera) -> None:
        n = len(state.virus_pos)
        if n == 0:
            return

        vir_r_world = 50.0   # sqrt(virus_mass=2500) = 50
        radii = np.full(n, vir_r_world, dtype=np.float32)
        mask = camera.visible_mask(state.virus_pos, radii)
        visible_pos = state.virus_pos[mask]

        if len(visible_pos) == 0:
            return

        screen_xy = camera.world_to_screen_arr(visible_pos)
        r_px = max(2, int(camera.world_radius_to_screen(vir_r_world)))

        for sx, sy in screen_xy:
            ix, iy = int(sx), int(sy)
            pygame.draw.circle(self.screen, _VIRUS_COLOUR, (ix, iy), r_px)
            pygame.draw.circle(self.screen, _VIRUS_BORDER, (ix, iy), r_px, max(2, r_px // 6))
            # Spiky outline: draw small triangles around the perimeter
            if r_px >= 6:
                n_spikes = 12
                for k in range(n_spikes):
                    angle = 2 * math.pi * k / n_spikes
                    tip_x = ix + int((r_px + max(3, r_px // 4)) * math.cos(angle))
                    tip_y = iy + int((r_px + max(3, r_px // 4)) * math.sin(angle))
                    pygame.draw.line(self.screen, _VIRUS_BORDER, (ix, iy), (tip_x, tip_y), 1)

    # ------------------------------------------------------------------
    # Ejected mass
    # ------------------------------------------------------------------

    def _draw_ejected(self, state: GameState, camera: Camera) -> None:
        n = len(state.ejected_pos)
        if n == 0:
            return

        ej_r_world = float(math.sqrt(13.0))   # eject_mass = 13
        radii = np.full(n, ej_r_world, dtype=np.float32)
        mask = camera.visible_mask(state.ejected_pos, radii)
        visible_pos = state.ejected_pos[mask]
        visible_owner = state.ejected_owner[mask]

        if len(visible_pos) == 0:
            return

        screen_xy = camera.world_to_screen_arr(visible_pos)
        r_px = max(1, int(camera.world_radius_to_screen(ej_r_world)))

        for i, (sx, sy) in enumerate(screen_xy):
            pid = int(visible_owner[i])
            colour = _dim(_player_colour(pid), 0.75) if pid >= 0 else (120, 120, 120)
            pygame.draw.circle(self.screen, colour, (int(sx), int(sy)), r_px)

    # ------------------------------------------------------------------
    # Cells
    # ------------------------------------------------------------------

    def _draw_cells(
        self,
        state: GameState,
        camera: Camera,
        human_id: int,
        attention: AttentionWeights | None = None,
    ) -> None:
        n = len(state.cell_pos)
        if n == 0:
            return

        mass = state.cell_mass
        owners = state.cell_owner
        radii_world = np.sqrt(mass)

        vis_indices = np.where(camera.visible_mask(state.cell_pos, radii_world))[0]
        if len(vis_indices) == 0:
            return

        vis_pos = state.cell_pos[vis_indices]
        vis_mass = mass[vis_indices]
        vis_owners = owners[vis_indices]
        vis_rad_world = radii_world[vis_indices]

        # Attention weights for visible cells (None → all zero)
        cell_w: np.ndarray | None = None
        if attention is not None and len(attention.cell_weight) == len(state.cell_pos):
            cell_w = attention.cell_weight[vis_indices]

        screen_xy = camera.world_to_screen_arr(vis_pos)

        # Draw back-to-front (smallest first so larger cells appear on top)
        order = np.argsort(vis_mass)

        for i in order:
            ox, oy = int(screen_xy[i, 0]), int(screen_xy[i, 1])
            r_px = max(2, int(camera.world_radius_to_screen(vis_rad_world[i])))
            pid = int(vis_owners[i])
            colour = _player_colour(pid)

            # Attention glow on enemy cells (drawn before the cell fill)
            if cell_w is not None and pid != human_id:
                w = float(cell_w[i])
                if w > 0.05:
                    glow_r = max(r_px + 2, int(r_px * (1.0 + 1.5 * w)))
                    _draw_glow(self.screen, colour, ox, oy, glow_r, int(160 * w))

            pygame.draw.circle(self.screen, colour, (ox, oy), r_px)

            # Border: thicker for human player
            border_w = max(2, r_px // 8) if pid == human_id else max(1, r_px // 12)
            border_col = _dim(colour, 0.6)
            pygame.draw.circle(self.screen, border_col, (ox, oy), r_px, border_w)

            # Label (mass) for large enough cells
            if r_px >= 14:
                mass_str = _mass_label(float(vis_mass[i]))
                surf = self.font.render(mass_str, True, (255, 255, 255))
                tw, th = surf.get_size()
                self.screen.blit(surf, (ox - tw // 2, oy - th // 2))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mass_label(mass: float) -> str:
    if mass >= 1_000_000:
        return f"{mass / 1_000_000:.1f}M"
    if mass >= 1_000:
        return f"{mass / 1_000:.1f}k"
    return str(int(mass))


def _draw_glow(
    surface: pygame.Surface,
    colour: tuple[int, int, int],
    cx: int,
    cy: int,
    radius: int,
    alpha: int,
) -> None:
    """Draw a semi-transparent filled circle (glow halo) onto *surface*.

    Parameters
    ----------
    surface : pygame.Surface
        Target display surface.
    colour : tuple[int, int, int]
        RGB base colour of the glow.
    cx, cy : int
        Centre of the glow in screen pixels.
    radius : int
        Radius of the glow circle in pixels.
    alpha : int
        Opacity 0–255; higher = more visible.
    """
    glow_surf = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
    pygame.draw.circle(glow_surf, (*colour, max(0, min(255, alpha))), (radius, radius), radius)
    surface.blit(glow_surf, (cx - radius, cy - radius))
