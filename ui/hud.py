"""HUD — draws leaderboard, FPS counter, and minimap over the game view."""

from __future__ import annotations

import math

import pygame
import numpy as np

from game.world import GameState
from ui.camera import Camera


# Colours
_HUD_BG = (0, 0, 0, 140)          # RGBA for semi-transparent panels
_HUD_TEXT = (240, 240, 240)
_HUD_HEADER = (255, 220, 50)
_MINIMAP_BG = (20, 20, 35, 180)
_MINIMAP_BORDER = (80, 80, 120)
_MINIMAP_VIEWPORT = (200, 200, 255, 100)

_PLAYER_COLOURS: list[tuple[int, int, int]] = [
    (220,  80,  80),
    ( 80, 160, 220),
    ( 80, 220, 100),
    (220, 200,  60),
    (200,  80, 220),
    ( 60, 200, 200),
    (220, 140,  60),
    (140, 220,  60),
    (220,  60, 140),
    ( 60, 100, 220),
    (160, 100,  60),
    (100, 180, 180),
    (180, 180,  60),
    (180,  60, 180),
    ( 60, 180, 100),
    (180, 100, 100),
]


def _player_colour(pid: int) -> tuple[int, int, int]:
    return _PLAYER_COLOURS[pid % len(_PLAYER_COLOURS)]


class HUD:
    """Draws the HUD overlay each frame.

    Parameters
    ----------
    screen : pygame.Surface
    world_w, world_h : float
        World bounds (for minimap scaling).
    max_leaderboard : int
        Number of entries in the leaderboard.
    minimap_size : int
        Pixel size (square) of the minimap.
    """

    def __init__(
        self,
        screen: pygame.Surface,
        world_w: float,
        world_h: float,
        max_leaderboard: int = 10,
        minimap_size: int = 160,
    ) -> None:
        self.screen = screen
        self.screen_w = screen.get_width()
        self.screen_h = screen.get_height()
        self.world_w = world_w
        self.world_h = world_h
        self.max_leaderboard = max_leaderboard
        self.minimap_size = minimap_size

        pygame.font.init()
        self.font = pygame.font.SysFont("Arial", 15, bold=True)
        self.small_font = pygame.font.SysFont("Arial", 12)
        self.title_font = pygame.font.SysFont("Arial", 17, bold=True)

        # Pre-bake a per-pixel-alpha surface for the minimap background
        self._mm_surf = pygame.Surface(
            (minimap_size, minimap_size), pygame.SRCALPHA
        )

    # ------------------------------------------------------------------

    def draw(
        self,
        state: GameState,
        camera: Camera,
        fps: float,
        human_id: int,
        player_names: dict[int, str] | None = None,
    ) -> None:
        """Render HUD overlay.

        Parameters
        ----------
        state : GameState
        camera : Camera
        fps : float
        human_id : int
        player_names : dict mapping player_id → display name (optional)
        """
        self._draw_leaderboard(state, player_names or {}, human_id)
        self._draw_fps(fps)
        self._draw_minimap(state, camera, human_id)

    # ------------------------------------------------------------------
    # Leaderboard
    # ------------------------------------------------------------------

    def _draw_leaderboard(
        self,
        state: GameState,
        player_names: dict[int, str],
        human_id: int,
    ) -> None:
        alive_players = np.where(state.player_alive)[0]
        if len(alive_players) == 0:
            return

        masses = state.player_mass[alive_players]
        order = np.argsort(masses)[::-1]
        top = alive_players[order][: self.max_leaderboard]
        top_masses = masses[order][: self.max_leaderboard]

        pad = 8
        line_h = 20
        header_h = 22
        w = 180
        h = header_h + len(top) * line_h + pad

        # Semi-transparent background
        bg = pygame.Surface((w, h), pygame.SRCALPHA)
        bg.fill(_HUD_BG)
        x0 = self.screen_w - w - 10
        y0 = 10
        self.screen.blit(bg, (x0, y0))

        # Header
        hdr = self.title_font.render("Leaderboard", True, _HUD_HEADER)
        self.screen.blit(hdr, (x0 + pad, y0 + 3))

        # Entries
        for rank, (pid, mass) in enumerate(zip(top.tolist(), top_masses.tolist())):
            colour = _player_colour(pid)
            name = player_names.get(pid, f"Player {pid}")
            if pid == human_id:
                name = f"[You] {name}"
            label = f"{rank + 1}. {name}"
            mass_str = _mass_label(float(mass))

            yx = y0 + header_h + rank * line_h
            # Colour dot
            pygame.draw.circle(
                self.screen, colour,
                (x0 + pad + 6, yx + line_h // 2), 5,
            )
            name_surf = self.small_font.render(label, True, _HUD_TEXT)
            self.screen.blit(name_surf, (x0 + pad + 16, yx + 3))

            mass_surf = self.small_font.render(mass_str, True, _HUD_TEXT)
            self.screen.blit(
                mass_surf, (x0 + w - mass_surf.get_width() - pad, yx + 3)
            )

    # ------------------------------------------------------------------
    # FPS
    # ------------------------------------------------------------------

    def _draw_fps(self, fps: float) -> None:
        text = f"FPS: {fps:.0f}"
        surf = self.small_font.render(text, True, _HUD_TEXT)
        self.screen.blit(surf, (8, 8))

    # ------------------------------------------------------------------
    # Minimap
    # ------------------------------------------------------------------

    def _draw_minimap(
        self, state: GameState, camera: Camera, human_id: int,
    ) -> None:
        mm = self.minimap_size
        scale_x = mm / self.world_w
        scale_y = mm / self.world_h
        x0 = 8
        y0 = self.screen_h - mm - 8

        # Background
        self._mm_surf.fill(_MINIMAP_BG)

        # Food (tiny dots)
        if len(state.food_pos) > 0:
            # Sample at most 800 food pellets for performance
            fp = state.food_pos
            if len(fp) > 800:
                step = len(fp) // 800
                fp = fp[::step]
            for fx, fy in fp:
                pygame.draw.circle(
                    self._mm_surf, (60, 180, 60),
                    (int(fx * scale_x), int(fy * scale_y)), 1,
                )

        # Cells
        if len(state.cell_pos) > 0:
            for i in range(len(state.cell_pos)):
                cx = float(state.cell_pos[i, 0])
                cy = float(state.cell_pos[i, 1])
                pid = int(state.cell_owner[i])
                r = max(2, int(math.sqrt(state.cell_mass[i]) * scale_x * 0.5))
                colour = _player_colour(pid)
                pygame.draw.circle(
                    self._mm_surf, colour,
                    (int(cx * scale_x), int(cy * scale_y)), r,
                )

        # Viewport rectangle
        vp_world_w = camera.screen_w / camera.zoom
        vp_world_h = camera.screen_h / camera.zoom
        vp_x = (camera.cx - vp_world_w / 2) * scale_x
        vp_y = (camera.cy - vp_world_h / 2) * scale_y
        vp_w = vp_world_w * scale_x
        vp_h = vp_world_h * scale_y
        pygame.draw.rect(
            self._mm_surf, _MINIMAP_VIEWPORT,
            (int(vp_x), int(vp_y), int(vp_w), int(vp_h)), 1,
        )

        # Border
        pygame.draw.rect(self._mm_surf, _MINIMAP_BORDER, (0, 0, mm, mm), 1)

        self.screen.blit(self._mm_surf, (x0, y0))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mass_label(mass: float) -> str:
    if mass >= 1_000_000:
        return f"{mass / 1_000_000:.1f}M"
    if mass >= 1_000:
        return f"{mass / 1_000:.1f}k"
    return str(int(mass))
