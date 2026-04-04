from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from game.config import WorldConfig
    from game.world import GameState


# One distinct colour per player ID (cycles if more agents than colours).
_COLORS = [
    "#e74c3c", "#3498db", "#2ecc71", "#f1c40f",
    "#9b59b6", "#1abc9c", "#e67e22", "#ecf0f1",
    "#e91e63", "#00bcd4", "#8bc34a", "#ff5722",
    "#607d8b", "#795548", "#ffc107", "#03a9f4",
]


def render_episode_to_video(
    frames: list[GameState],
    cfg: WorldConfig,
    path: str | Path,
    fps: int = 15,
) -> None:
    """Render a list of :class:`~game.world.GameState` frames to a video file.

    The full world view is rendered at each tick using matplotlib circles.
    Output format is determined by the file extension: ``.gif`` uses the
    Pillow writer (no extra dependencies); ``.mp4`` requires ffmpeg.

    Parameters
    ----------
    frames : list[GameState]
        Ordered snapshots to render.
    cfg : WorldConfig
        World configuration (used for coordinate bounds).
    path : str or Path
        Destination file (``.gif`` or ``.mp4``).
    fps : int
        Playback frame rate.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib import animation

    if not frames:
        return

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    W, H = float(cfg.width), float(cfg.height)
    aspect = H / W
    fig_w = 7.0
    fig, ax = plt.subplots(figsize=(fig_w, fig_w * aspect))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout(pad=0)

    def _draw(frame_idx: int) -> None:
        ax.cla()
        ax.set_facecolor("#0d1117")
        ax.set_xlim(0, W)
        ax.set_ylim(0, H)
        ax.set_aspect("equal")
        ax.axis("off")

        state = frames[frame_idx]

        # Food — tiny grey dots
        if len(state.food_pos) > 0:
            ax.scatter(
                state.food_pos[:, 0], state.food_pos[:, 1],
                s=1, c="#4a4a6a", linewidths=0, alpha=0.6,
            )

        # Viruses — green outlines
        for vpos in state.virus_pos:
            ax.add_patch(mpatches.Circle(
                vpos, radius=np.sqrt(cfg.virus_mass),
                color="#27ae60", fill=False, linewidth=1.0, alpha=0.7,
            ))

        # Cells — sorted so larger appear on top
        if len(state.cell_pos) > 0:
            order = np.argsort(state.cell_mass)
            for i in order:
                owner = int(state.cell_owner[i])
                radius = float(np.sqrt(state.cell_mass[i]))
                color = _COLORS[owner % len(_COLORS)]
                ax.add_patch(mpatches.Circle(
                    state.cell_pos[i], radius=radius,
                    color=color, alpha=0.88,
                ))

        ax.set_title(
            f"tick {state.tick}",
            fontsize=7, color="#aaaaaa", pad=1,
        )

    anim = animation.FuncAnimation(
        fig, _draw, frames=len(frames), interval=1000 / fps,
    )

    suffix = path.suffix.lower()
    if suffix == ".mp4" and animation.FFMpegWriter.isAvailable():
        writer = animation.FFMpegWriter(
            fps=fps, extra_args=["-vcodec", "libx264", "-pix_fmt", "yuv420p"]
        )
        anim.save(str(path), writer=writer)
    else:
        path = path.with_suffix(".gif")
        anim.save(str(path), writer="pillow", fps=fps)

    plt.close(fig)
    return path


def record_video(
    policy: object,
    cfg: WorldConfig,
    n_agents: int,
    path: str | Path,
    n_ticks: int = 300,
    fps: int = 15,
    seed: int = 0,
    device: object = None,
) -> None:
    """Run a short episode with *policy* and save it as a video.

    All *n_agents* slots use the same shared policy.  Dead agents respawn
    immediately.  The recording is entirely headless (no Pygame window).

    Parameters
    ----------
    policy : MLPPolicy or AttentionPolicy
        Policy to evaluate (called with ``deterministic=True``).
    cfg : WorldConfig
    n_agents : int
        Number of agents to spawn in the world.
    path : str or Path
        Output video path.
    n_ticks : int
        Number of simulation ticks to record.
    fps : int
        Video frame rate.
    seed : int
    device : torch.device, optional
    """
    import torch

    from game.world import World
    from rl.env import build_observation

    if device is None:
        device = torch.device("cpu")

    pos_scale = float(max(cfg.width, cfg.height)) / 2.0
    large_scale = float(max(cfg.width, cfg.height))
    agent_ids = list(range(n_agents))

    world = World(cfg)
    world.reset(seed=seed)
    for aid in agent_ids:
        world.add_player(aid)

    frames = []
    policy.eval()

    with torch.no_grad():
        for _ in range(n_ticks):
            state = world.get_state()
            frames.append(state)

            world_actions = np.zeros((cfg.max_players, 4), dtype=np.float32)
            for aid in agent_ids:
                if aid in world._active_players:
                    obs = build_observation(state, aid, cfg, pos_scale)
                    obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(device)
                    z, _, _ = policy.act(obs_t, deterministic=True)
                    act = torch.tanh(z).squeeze(0).cpu().numpy().astype(np.float32)

                    mask = state.cell_owner == aid
                    if mask.any():
                        pos = state.cell_pos[mask]
                        mass = state.cell_mass[mask]
                        total = float(mass.sum())
                        c = (pos * mass[:, None]).sum(0) / total if total > 0 else pos.mean(0)
                    else:
                        c = np.array([cfg.width / 2, cfg.height / 2], dtype=np.float32)

                    world_actions[aid, 0] = c[0] + act[0] * large_scale
                    world_actions[aid, 1] = c[1] + act[1] * large_scale
                    world_actions[aid, 2] = 1.0 if act[2] > 0.0 else 0.0
                    world_actions[aid, 3] = 1.0 if act[3] > 0.0 else 0.0

            _, dones, _ = world.step(world_actions)
            for aid in agent_ids:
                if dones[aid]:
                    world.add_player(aid)

    return render_episode_to_video(frames, cfg, path, fps=fps)
