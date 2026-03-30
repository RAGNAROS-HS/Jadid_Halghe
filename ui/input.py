"""Input handling — translates Pygame events into game actions.

Returns an action row ``[dx, dy, split, eject]`` for the human player.
"""

from __future__ import annotations

import pygame
import numpy as np


def handle_events(
    screen_w: int,
    screen_h: int,
) -> tuple[np.ndarray, bool, bool]:
    """Process Pygame event queue and mouse/keyboard state.

    Parameters
    ----------
    screen_w, screen_h : int
        Window resolution; used to compute direction relative to centre.

    Returns
    -------
    action : ndarray, shape (4,), float32
        ``[dx, dy, split_flag, eject_flag]`` for the human player.
        ``dx``, ``dy`` are the normalised direction toward the mouse cursor.
        ``split_flag`` and ``eject_flag`` are 1.0 when the key was pressed
        this frame (handled as events, not held state).
    quit_flag : bool
        True if the user closed the window or pressed Escape.
    paused : bool
        True if the game should be paused this frame (toggled by P key).
    """
    action = np.zeros(4, dtype=np.float32)
    quit_flag = False
    paused = False

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            quit_flag = True
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                quit_flag = True
            elif event.key == pygame.K_SPACE:
                action[2] = 1.0   # split
            elif event.key == pygame.K_w:
                action[3] = 1.0   # eject
            elif event.key == pygame.K_p:
                paused = True

    # Mouse direction (relative to screen centre)
    mx, my = pygame.mouse.get_pos()
    dx = mx - screen_w / 2.0
    dy = my - screen_h / 2.0
    mag = (dx * dx + dy * dy) ** 0.5
    if mag > 1e-4:
        action[0] = dx / mag
        action[1] = dy / mag

    return action, quit_flag, paused
