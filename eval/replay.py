from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from game.config import WorldConfig
from game.world import GameState


@dataclass
class ReplayEpisode:
    """A recorded evaluation episode.

    Parameters
    ----------
    config : WorldConfig
        World configuration used during recording.
    frames : list[GameState]
        Per-tick game state snapshots.
    metadata : dict
        Arbitrary metadata (e.g. episode stats, policy name, seed).
    """

    config: WorldConfig
    frames: list[GameState]
    metadata: dict[str, Any] = field(default_factory=dict)


def save_replay(episode: ReplayEpisode, path: str | Path) -> None:
    """Serialise *episode* to *path* using pickle.

    Parameters
    ----------
    episode : ReplayEpisode
    path : str or Path
        Destination file path.  Parent directories are created if needed.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(episode, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_replay(path: str | Path) -> ReplayEpisode:
    """Load a :class:`ReplayEpisode` from *path*.

    Parameters
    ----------
    path : str or Path

    Returns
    -------
    ReplayEpisode
    """
    with open(path, "rb") as f:
        return pickle.load(f)  # noqa: S301


def replay_with_ui(
    episode: ReplayEpisode,
    fps: int = 30,
    speed: float = 1.0,
    player_id: int = 0,
    screen_w: int = 1280,
    screen_h: int = 720,
) -> None:
    """Play back a :class:`ReplayEpisode` in a Pygame window.

    Parameters
    ----------
    episode : ReplayEpisode
    fps : int
        Display frame-rate (not simulation rate — all frames are pre-computed).
    speed : float
        Playback multiplier.  ``2.0`` skips every other frame; ``0.5`` shows
        each frame twice.
    player_id : int
        The player whose centroid the camera follows.
    screen_w, screen_h : int
        Window resolution.
    """
    import pygame

    from ui.camera import Camera
    from ui.hud import HUD
    from ui.renderer import Renderer

    cfg = episode.config
    frames = episode.frames
    if not frames:
        return

    pygame.init()
    screen = pygame.display.set_mode((screen_w, screen_h))
    pygame.display.set_caption("Replay")
    clock = pygame.time.Clock()

    camera = Camera(screen_w, screen_h, cfg.width, cfg.height)
    renderer = Renderer(screen)
    hud = HUD(screen, cfg.width, cfg.height)

    frame_idx = 0
    frame_accum = 0.0
    running = True

    while running and frame_idx < len(frames):
        dt = clock.tick(fps) / 1000.0
        frame_accum += dt * speed * fps

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_RIGHT:
                    frame_idx = min(frame_idx + 60, len(frames) - 1)
                    frame_accum = 0.0
                elif event.key == pygame.K_LEFT:
                    frame_idx = max(frame_idx - 60, 0)
                    frame_accum = 0.0

        while frame_accum >= 1.0 and frame_idx < len(frames) - 1:
            frame_idx += 1
            frame_accum -= 1.0

        state = frames[frame_idx]
        camera.update(
            state.cell_pos,
            state.cell_mass,
            state.cell_owner,
            player_id,
        )

        screen.fill((20, 20, 20))
        renderer.draw(state, camera, player_id)
        hud.draw(state, camera, fps, player_id)
        pygame.display.flip()

    pygame.quit()


def plot_mass_over_time(
    episodes: list[ReplayEpisode],
    player_id: int = 0,
    save_path: str | Path | None = None,
) -> None:
    """Plot mass over time for one or more episodes.

    Parameters
    ----------
    episodes : list[ReplayEpisode]
    player_id : int
        Which player's mass to track.
    save_path : str or Path, optional
        If provided, save the figure to this path instead of displaying it.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 4))

    for i, ep in enumerate(episodes):
        masses = [
            float(frame.player_mass[player_id])
            for frame in ep.frames
        ]
        ticks = list(range(len(masses)))
        label = ep.metadata.get("label", f"episode {i}")
        ax.plot(ticks, masses, label=label, alpha=0.8)

    ax.set_xlabel("Tick")
    ax.set_ylabel("Mass")
    ax.set_title("Mass over time")
    if len(episodes) > 1:
        ax.legend(loc="upper left", fontsize=8)

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
    else:
        plt.show()
