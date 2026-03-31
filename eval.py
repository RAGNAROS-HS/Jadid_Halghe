"""Evaluation entry point for trained policies.

Usage examples::

    python eval.py --checkpoint checkpoints/run_default/ckpt_000100.pt
    python eval.py --checkpoint ckpt.pt --opponents greedy --episodes 50
    python eval.py --checkpoint ckpt.pt --save-replay replays/ep.pkl --plot
    python eval.py --replay replays/ep.pkl
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate a checkpoint or replay a saved episode."
    )

    # Evaluation mode
    eval_grp = p.add_argument_group("Evaluation")
    eval_grp.add_argument(
        "--checkpoint",
        metavar="PATH",
        help="Path to a policy checkpoint (.pt file).",
    )
    eval_grp.add_argument(
        "--opponents",
        default="random",
        choices=["random", "greedy"],
        help="Opponent bot policy (default: random).",
    )
    eval_grp.add_argument(
        "--episodes",
        type=int,
        default=20,
        metavar="N",
        help="Number of evaluation episodes (default: 20).",
    )
    eval_grp.add_argument(
        "--n-bots",
        type=int,
        default=7,
        metavar="N",
        help="Number of opponent bots per episode (default: 7).",
    )
    eval_grp.add_argument(
        "--max-ticks",
        type=int,
        default=2000,
        metavar="N",
        help="Episode truncation length (default: 2000).",
    )
    eval_grp.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Base random seed (default: 0).",
    )
    eval_grp.add_argument(
        "--device",
        default="cpu",
        help="Torch device string (default: cpu).",
    )

    # Output
    out_grp = p.add_argument_group("Output")
    out_grp.add_argument(
        "--save-replay",
        metavar="PATH",
        help="Save the first episode as a replay file.",
    )
    out_grp.add_argument(
        "--plot",
        action="store_true",
        help="Plot mass-over-time for all recorded episodes.",
    )
    out_grp.add_argument(
        "--output",
        metavar="PATH",
        help="Write evaluation summary to this text file.",
    )

    # Replay mode
    rep_grp = p.add_argument_group("Replay")
    rep_grp.add_argument(
        "--replay",
        metavar="PATH",
        help="Replay a previously saved episode in the Pygame UI.",
    )
    rep_grp.add_argument(
        "--replay-fps",
        type=int,
        default=30,
        help="Playback frame-rate (default: 30).",
    )
    rep_grp.add_argument(
        "--replay-speed",
        type=float,
        default=1.0,
        help="Playback speed multiplier (default: 1.0).",
    )

    return p.parse_args()


def main() -> None:
    """CLI entry point for evaluation and replay."""
    args = _parse_args()

    # ── Replay mode ──────────────────────────────────────────────────────────
    if args.replay is not None:
        from eval.replay import load_replay, replay_with_ui

        episode = load_replay(args.replay)
        replay_with_ui(episode, fps=args.replay_fps, speed=args.replay_speed)
        return

    # ── Evaluation mode ──────────────────────────────────────────────────────
    if args.checkpoint is None:
        print("error: --checkpoint or --replay is required.", file=sys.stderr)
        sys.exit(1)

    import torch

    from eval.harness import Harness
    from eval.replay import ReplayEpisode, plot_mass_over_time, save_replay

    device = torch.device(args.device)
    record = args.save_replay is not None or args.plot

    harness = Harness(
        policy=args.checkpoint,
        opponent=args.opponents,
        n_bots=args.n_bots,
        max_ticks=args.max_ticks,
        device=device,
    )

    result, replays = harness.run(
        n_episodes=args.episodes,
        seed=args.seed,
        record=record,
    )

    summary = result.summary()
    print(summary)

    if args.output is not None:
        Path(args.output).write_text(summary + "\n", encoding="utf-8")

    if args.save_replay is not None and replays and replays[0]:
        from game.config import WorldConfig

        cfg = harness.config
        ep = ReplayEpisode(
            config=cfg,
            frames=replays[0],
            metadata={
                "label": f"ep0 reward={result.episodes[0].total_reward:+.2f}",
                "checkpoint": args.checkpoint,
                "opponents": args.opponents,
                "seed": args.seed,
            },
        )
        save_replay(ep, args.save_replay)
        print(f"replay saved → {args.save_replay}")

    if args.plot and any(replays):
        from game.config import WorldConfig

        cfg = harness.config
        replay_episodes = [
            ReplayEpisode(
                config=cfg,
                frames=frames,
                metadata={
                    "label": f"ep{i} r={result.episodes[i].total_reward:+.0f}",
                },
            )
            for i, frames in enumerate(replays)
            if frames
        ]
        plot_mass_over_time(replay_episodes)


if __name__ == "__main__":
    main()
