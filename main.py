"""Human-playable agar.io entry point.

Usage
-----
    python main.py                    # 4 bots, human plays
    python main.py --agents 8         # 8 random bots
    python main.py --agents 0         # solo (only human)
    python main.py --seed 7           # deterministic seed
    python main.py --width 1600 --height 900
    python main.py --checkpoint ckpt.pt           # bots use trained policy
    python main.py --checkpoint ckpt.pt --agents 7

Controls
--------
    Mouse    — movement direction
    Space    — split
    W        — eject mass
    P        — pause / unpause
    Escape   — quit
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import pygame

import torch

from game.config import WorldConfig
from game.world import GameState, World
from rl.env import build_observation
from ui.camera import Camera
from ui.hud import HUD
from ui.input import handle_events
from ui.renderer import Renderer


# ---------------------------------------------------------------------------
# Sub-tick interpolation
# ---------------------------------------------------------------------------

def _interp_state(
    prev: GameState | None,
    curr: GameState,
    alpha: float,
) -> GameState:
    """Linearly interpolate cell and ejected positions between two ticks.

    ``alpha=0`` returns positions as of *prev*; ``alpha=1`` returns *curr*.
    Falls back to *curr* on any entity-count mismatch (cells died or spawned
    this tick) — a single-frame snap that is imperceptible at 60 FPS.
    """
    if prev is None or alpha >= 1.0:
        return curr

    # Cells
    if (
        len(prev.cell_pos) == len(curr.cell_pos)
        and np.array_equal(prev.cell_owner, curr.cell_owner)
    ):
        cell_pos = prev.cell_pos + alpha * (curr.cell_pos - prev.cell_pos)
    else:
        cell_pos = curr.cell_pos

    # Ejected mass
    if len(prev.ejected_pos) == len(curr.ejected_pos):
        ejected_pos = prev.ejected_pos + alpha * (curr.ejected_pos - prev.ejected_pos)
    else:
        ejected_pos = curr.ejected_pos

    return GameState(
        tick=curr.tick,
        cell_pos=cell_pos,
        cell_mass=curr.cell_mass,
        cell_owner=curr.cell_owner,
        food_pos=curr.food_pos,
        virus_pos=curr.virus_pos,
        ejected_pos=ejected_pos,
        ejected_owner=curr.ejected_owner,
        player_alive=curr.player_alive,
        player_mass=curr.player_mass,
    )


# ---------------------------------------------------------------------------
# Random bot policy
# ---------------------------------------------------------------------------

_WANDER_DIST = 3000.0   # world units ahead of centroid to place virtual cursor


def _bot_action(
    bot_id: int,
    state: GameState,
    rng: np.random.Generator,
    config: WorldConfig,
    bot_dirs: np.ndarray,
) -> np.ndarray:
    """Simple bot: chase nearest eatable enemy; otherwise wander.

    Returns an action row ``[target_x, target_y, split, eject]`` where
    ``target_x/y`` is the world-space cursor position physics steers toward.
    """
    action = np.zeros(4, dtype=np.float32)

    mask = state.cell_owner == bot_id
    if not mask.any():
        return action

    bot_pos = state.cell_pos[mask]
    bot_mass = state.cell_mass[mask]
    bot_centroid = bot_pos.mean(axis=0)
    total_mass = float(bot_mass.sum())

    # Chase nearest eatable enemy
    enemy_mask = (state.cell_owner != bot_id) & (state.cell_owner >= 0)
    if enemy_mask.any():
        enemy_pos = state.cell_pos[enemy_mask]
        enemy_mass = state.cell_mass[enemy_mask]
        can_eat = enemy_mass * (config.eat_ratio ** 2) < total_mass
        if can_eat.any():
            targets = enemy_pos[can_eat]
            nearest = int(np.argmin(((targets - bot_centroid) ** 2).sum(axis=1)))
            action[0] = float(targets[nearest, 0])
            action[1] = float(targets[nearest, 1])
            if total_mass > config.min_split_mass * 1.5 and rng.random() < 0.01:
                action[2] = 1.0
            return action

    # Wander: project current direction to a world target position
    noise = rng.standard_normal(2).astype(np.float32)
    new_dir = bot_dirs[bot_id] + noise * 0.3
    mag = float(np.linalg.norm(new_dir))
    new_dir = new_dir / mag if mag > 1e-4 else np.array([1.0, 0.0], dtype=np.float32)
    bot_dirs[bot_id] = new_dir

    target = bot_centroid + new_dir * _WANDER_DIST
    action[0] = float(np.clip(target[0], 0.0, config.width))
    action[1] = float(np.clip(target[1], 0.0, config.height))

    return action


def _load_trained_bot(checkpoint: str) -> object:
    """Load a policy from *checkpoint* and return it in eval mode."""
    from rl.agent import load_policy
    policy, _ = load_policy(checkpoint)
    policy.eval()
    return policy


def _trained_bot_action(
    bot_id: int,
    state: GameState,
    policy: object,
    cfg: WorldConfig,
    pos_scale: float,
    large_scale: float,
) -> np.ndarray:
    """Run the trained policy for *bot_id* and return a world-space action row."""
    mask = state.cell_owner == bot_id
    if not mask.any():
        return np.zeros(4, dtype=np.float32)

    obs = build_observation(state, bot_id, cfg, pos_scale)
    with torch.no_grad():
        z, _, _ = policy.act(obs, deterministic=True)
        act = torch.tanh(z).squeeze(0).cpu().numpy().astype(np.float32)

    # Centroid of bot's cells
    pos = state.cell_pos[mask]
    mass = state.cell_mass[mask]
    total = float(mass.sum())
    centroid = (pos * mass[:, None]).sum(axis=0) / total if total > 0 else pos.mean(axis=0)

    action = np.zeros(4, dtype=np.float32)
    action[0] = centroid[0] + act[0] * large_scale
    action[1] = centroid[1] + act[1] * large_scale
    action[2] = 1.0 if act[2] > 0.0 else 0.0
    action[3] = 1.0 if act[3] > 0.0 else 0.0
    return action


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Jadid Halghe — agar.io clone")
    parser.add_argument("--agents", type=int, default=4,
                        help="Number of random bot agents (default: 4)")
    parser.add_argument("--seed", type=int, default=42,
                        help="RNG seed (default: 42)")
    parser.add_argument("--width", type=int, default=1280,
                        help="Window width in pixels (default: 1280)")
    parser.add_argument("--height", type=int, default=720,
                        help="Window height in pixels (default: 720)")
    parser.add_argument("--fps", type=int, default=60,
                        help="Display frame rate (default: 60)")
    parser.add_argument("--tps", type=int, default=25,
                        help="Simulation ticks per second (default: 25)")
    parser.add_argument("--no-human", action="store_true",
                        help="Spectator mode: all players are bots")
    parser.add_argument("--checkpoint", metavar="PATH", default=None,
                        help="Policy checkpoint — bots use the trained agent instead of the random heuristic")
    args = parser.parse_args()

    # ── Config ────────────────────────────────────────────────────────────
    n_bots = args.agents
    has_human = not args.no_human
    human_id = 0
    n_players = n_bots + (1 if has_human else 0)
    if n_players == 0:
        print("Need at least one player (--agents > 0 or omit --no-human).")
        sys.exit(1)

    max_players = max(16, n_players)
    cfg = WorldConfig(max_players=max_players)
    tick_interval = 1.0 / args.tps

    rng = np.random.default_rng(args.seed)

    # ── Trained bot policy (optional) ─────────────────────────────────────
    trained_policy = None
    pos_scale = float(max(cfg.width, cfg.height)) / 2.0
    large_scale = float(max(cfg.width, cfg.height))
    if args.checkpoint is not None:
        print(f"Loading checkpoint: {args.checkpoint}")
        trained_policy = _load_trained_bot(args.checkpoint)
        print("Trained policy loaded — bots will use it.")

    # ── World setup ────────────────────────────────────────────────────────
    world = World(cfg)
    world.reset(seed=args.seed)

    bot_ids: list[int] = []
    if has_human:
        world.add_player(human_id)
        bot_ids = list(range(1, n_bots + 1))
    else:
        bot_ids = list(range(n_bots))
        human_id = bot_ids[0] if bot_ids else 0

    for bid in bot_ids:
        world.add_player(bid)

    bot_dirs = rng.standard_normal((max_players, 2)).astype(np.float32)
    mags = np.linalg.norm(bot_dirs, axis=1, keepdims=True)
    bot_dirs /= np.where(mags > 1e-4, mags, 1.0)

    player_names: dict[int, str] = {}
    if has_human:
        player_names[human_id] = "You"
    bot_label = "Agent" if args.checkpoint is not None else "Bot"
    for bid in bot_ids:
        player_names[bid] = f"{bot_label} {bid}"

    # ── Pygame setup ───────────────────────────────────────────────────────
    pygame.init()
    screen = pygame.display.set_mode((args.width, args.height))
    pygame.display.set_caption("Jadid Halghe")
    clock = pygame.time.Clock()

    camera = Camera(args.width, args.height, cfg.width, cfg.height)
    renderer = Renderer(screen)
    hud = HUD(screen, cfg.width, cfg.height)

    pause_font = pygame.font.SysFont("Arial", 48, bold=True)

    # ── State ─────────────────────────────────────────────────────────────
    # actions[:, :2] holds world-space cursor positions (target_x, target_y).
    # Initialised to each player's spawn centroid so there is no spurious
    # drift before the first mouse/bot update.
    actions = np.zeros((cfg.max_players, 4), dtype=np.float32)
    init_state = world.get_state()
    all_ids = ([human_id] if has_human else []) + bot_ids
    for pid in all_ids:
        pmask = init_state.cell_owner == pid
        if pmask.any():
            c = init_state.cell_pos[pmask].mean(axis=0)
            actions[pid, 0] = float(c[0])
            actions[pid, 1] = float(c[1])

    prev_state: GameState | None = None
    curr_state = world.get_state()

    respawn_timers: dict[int, int] = {}
    RESPAWN_TICKS = 50  # ~2 s at 25 TPS

    accumulator = 0.0
    prev_time = time.perf_counter()
    paused = False
    running = True

    while running:
        # ── Real elapsed time since last frame ────────────────────────────
        now = time.perf_counter()
        frame_dt = min(now - prev_time, 0.1)   # cap at 100 ms (spiral-of-death guard)
        prev_time = now

        # ── Input (every render frame) ────────────────────────────────────
        if has_human:
            human_action, quit_flag, toggled_pause = handle_events(
                args.width, args.height
            )
            if quit_flag:
                running = False
                break
            if toggled_pause:
                paused = not paused

            # Convert screen mouse position → world cursor position each frame.
            wx, wy = camera.screen_to_world(human_action[0], human_action[1])
            actions[human_id, 0] = wx
            actions[human_id, 1] = wy
            # Split / eject: LATCH — set to 1 here, cleared only after a tick
            # consumes them.  This guarantees key presses are never dropped.
            if human_action[2] > 0.5:
                actions[human_id, 2] = 1.0
            if human_action[3] > 0.5:
                actions[human_id, 3] = 1.0
        else:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    break
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                        break
                    if event.key == pygame.K_p:
                        paused = not paused

        # ── Fixed-timestep simulation ─────────────────────────────────────
        if not paused:
            accumulator += frame_dt

            while accumulator >= tick_interval:
                # Bot actions are computed once per tick from the last known state
                for bid in bot_ids:
                    if bid in world._active_players:
                        if trained_policy is not None:
                            actions[bid] = _trained_bot_action(
                                bid, curr_state, trained_policy, cfg,
                                pos_scale, large_scale,
                            )
                        else:
                            actions[bid] = _bot_action(
                                bid, curr_state, rng, cfg, bot_dirs
                            )

                # Advance simulation
                prev_state = curr_state
                rewards, dones, _info = world.step(actions)
                curr_state = world.get_state()

                # Clear one-shot flags AFTER the tick has consumed them
                actions[human_id, 2] = 0.0
                actions[human_id, 3] = 0.0

                # Deaths → respawn queue
                for pid, dead in enumerate(dones):
                    if dead and (pid in bot_ids or (has_human and pid == human_id)):
                        respawn_timers[pid] = RESPAWN_TICKS

                # Respawn countdown
                for pid in list(respawn_timers):
                    respawn_timers[pid] -= 1
                    if respawn_timers[pid] <= 0:
                        del respawn_timers[pid]
                        if pid not in world._active_players:
                            try:
                                world.add_player(pid)
                            except (ValueError, RuntimeError):
                                pass  # no cell slots — retry next tick

                accumulator -= tick_interval

        # ── Sub-tick interpolation ─────────────────────────────────────────
        # alpha ∈ [0, 1]: fraction of the way through the current tick.
        # Interpolating positions here gives smooth 60 FPS visuals with a
        # 25 TPS simulation — the grid and cells glide rather than jumping.
        alpha = accumulator / tick_interval if not paused else 1.0
        render_state = _interp_state(prev_state, curr_state, alpha)

        # ── Camera ────────────────────────────────────────────────────────
        if len(render_state.cell_pos) > 0:
            camera.update(
                render_state.cell_pos,
                render_state.cell_mass,
                render_state.cell_owner,
                human_id,
            )

        # ── Render ────────────────────────────────────────────────────────
        renderer.draw(render_state, camera, human_id)
        hud.draw(curr_state, camera, clock.get_fps(), human_id, player_names)

        if paused:
            surf = pause_font.render("PAUSED", True, (255, 220, 50))
            tw, th = surf.get_size()
            screen.blit(surf, (args.width // 2 - tw // 2, args.height // 2 - th // 2))

        pygame.display.flip()
        clock.tick(args.fps)

    pygame.quit()
    sys.exit(0)


if __name__ == "__main__":
    main()
