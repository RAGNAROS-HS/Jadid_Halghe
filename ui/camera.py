"""Camera — maps world coordinates to screen pixels and back.

The camera follows the mass-weighted centroid of the human player's cells.
Zoom is inversely proportional to the player's effective radius so that a
small cell sees a tight view and a huge cell sees more of the world.
"""

from __future__ import annotations

import numpy as np


class Camera:
    """Viewport into the world.

    Parameters
    ----------
    screen_w, screen_h : int
        Display resolution in pixels.
    world_w, world_h : float
        World bounds.
    """

    def __init__(
        self,
        screen_w: int,
        screen_h: int,
        world_w: float,
        world_h: float,
    ) -> None:
        self.screen_w = screen_w
        self.screen_h = screen_h
        self.world_w = world_w
        self.world_h = world_h

        # Camera centre in world coordinates
        self.cx: float = world_w / 2.0
        self.cy: float = world_h / 2.0
        # Pixels per world unit
        self.zoom: float = screen_h / 1_000.0

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update(
        self,
        cell_pos: np.ndarray,
        cell_mass: np.ndarray,
        cell_owner: np.ndarray,
        player_id: int,
    ) -> None:
        """Re-centre on *player_id*'s mass-weighted centroid and adjust zoom.

        Parameters
        ----------
        cell_pos : ndarray, shape (n, 2)
        cell_mass : ndarray, shape (n,)
        cell_owner : ndarray, shape (n,), int32
        player_id : int
        """
        mask = cell_owner == player_id
        if not mask.any():
            return

        pm = cell_mass[mask]
        pp = cell_pos[mask]
        total = pm.sum()
        if total <= 0:
            return

        # Mass-weighted centroid — follow instantly to avoid jitter.
        # (Lerping position causes the cell to appear to drift off-centre
        # because game state updates at 25 TPS while the camera renders at 60 FPS.)
        self.cx = float((pp[:, 0] * pm).sum() / total)
        self.cy = float((pp[:, 1] * pm).sum() / total)

        # Zoom: smoothly lerp so zoom changes don't snap after splits.
        player_radius = float(np.sqrt(total))
        view_half_h = max(600.0, 2.8 * player_radius)
        target_zoom = self.screen_h / (2.0 * view_half_h)
        self.zoom += 0.08 * (target_zoom - self.zoom)

    # ------------------------------------------------------------------
    # Coordinate transforms
    # ------------------------------------------------------------------

    def world_to_screen(self, wx: float, wy: float) -> tuple[float, float]:
        """Convert a world-space point to screen pixels."""
        sx = (wx - self.cx) * self.zoom + self.screen_w / 2.0
        sy = (wy - self.cy) * self.zoom + self.screen_h / 2.0
        return sx, sy

    def world_to_screen_arr(self, world_xy: np.ndarray) -> np.ndarray:
        """Vectorised version of :meth:`world_to_screen`.

        Parameters
        ----------
        world_xy : ndarray, shape (n, 2)

        Returns
        -------
        ndarray, shape (n, 2), float32
            Screen-space pixel coordinates.
        """
        screen_xy = (world_xy - np.array([self.cx, self.cy], dtype=np.float32)) * self.zoom
        screen_xy[:, 0] += self.screen_w / 2.0
        screen_xy[:, 1] += self.screen_h / 2.0
        return screen_xy

    def screen_to_world(self, sx: float, sy: float) -> tuple[float, float]:
        """Convert screen pixels to world-space coordinates."""
        wx = (sx - self.screen_w / 2.0) / self.zoom + self.cx
        wy = (sy - self.screen_h / 2.0) / self.zoom + self.cy
        return wx, wy

    def world_radius_to_screen(self, r: float) -> float:
        """Convert a world-unit radius to pixel radius."""
        return r * self.zoom

    # ------------------------------------------------------------------
    # Visibility culling
    # ------------------------------------------------------------------

    def is_visible(self, wx: float, wy: float, wr: float) -> bool:
        """Return True if a circle (wx, wy, wr) overlaps the viewport."""
        sx, sy = self.world_to_screen(wx, wy)
        sr = wr * self.zoom
        return (
            sx + sr >= 0
            and sx - sr <= self.screen_w
            and sy + sr >= 0
            and sy - sr <= self.screen_h
        )

    def visible_mask(self, world_xy: np.ndarray, world_radii: np.ndarray) -> np.ndarray:
        """Return boolean mask of entities within (or overlapping) the viewport.

        Parameters
        ----------
        world_xy : ndarray, shape (n, 2)
        world_radii : ndarray, shape (n,)

        Returns
        -------
        ndarray, shape (n,), bool
        """
        screen_xy = self.world_to_screen_arr(world_xy)
        sr = world_radii * self.zoom
        in_x = (screen_xy[:, 0] + sr >= 0) & (screen_xy[:, 0] - sr <= self.screen_w)
        in_y = (screen_xy[:, 1] + sr >= 0) & (screen_xy[:, 1] - sr <= self.screen_h)
        return in_x & in_y
