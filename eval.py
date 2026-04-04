"""Evaluation entry point for trained policies.

Usage examples::

    python eval.py --checkpoint checkpoints/run_default/ckpt_000100.pt
    python eval.py --checkpoint ckpt.pt --opponents greedy --episodes 50
    python eval.py --checkpoint ckpt.pt --save-replay replays/ep.pkl --plot
    python eval.py --replay replays/ep.pkl
    python eval.py --elo --checkpoint-dir checkpoints/run_default --episodes 20
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

    # Elo tournament mode
    elo_grp = p.add_argument_group("Elo tournament")
    elo_grp.add_argument(
        "--elo",
        action="store_true",
        help="Run a round-robin Elo tournament across all checkpoints in --checkpoint-dir.",
    )
    elo_grp.add_argument(
        "--checkpoint-dir",
        metavar="DIR",
        help="Directory of checkpoints for Elo tournament.",
    )
    elo_grp.add_argument(
        "--ckpt-glob",
        default="ckpt_[0-9]*.pt",
        metavar="GLOB",
        help="Glob pattern to select checkpoints (default: ckpt_[0-9]*.pt).",
    )
    elo_grp.add_argument(
        "--elo-output",
        metavar="PATH",
        default=None,
        help="Save Elo results to JSON (default: <checkpoint-dir>/elo_results.json).",
    )
    elo_grp.add_argument(
        "--elo-bots",
        type=int,
        default=0,
        metavar="N",
        help="Extra random-bot bystanders per Elo game (default: 0 = pure 1v1).",
    )
    elo_grp.add_argument(
        "--k-factor",
        type=float,
        default=32.0,
        help="Elo K-factor (default: 32).",
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
    """CLI entry point for evaluation, Elo tournament, and replay."""
    args = _parse_args()

    # ── Elo tournament mode ──────────────────────────────────────────────────
    if args.elo:
        if args.checkpoint_dir is None:
            print("error: --checkpoint-dir is required with --elo.", file=sys.stderr)
            sys.exit(1)

        import json
        import torch
        from eval.elo import run_tournament

        ckpt_dir = Path(args.checkpoint_dir)
        paths = sorted(ckpt_dir.glob(args.ckpt_glob))
        if len(paths) < 2:
            print(
                f"error: found {len(paths)} checkpoint(s) matching "
                f"'{args.ckpt_glob}' in {ckpt_dir} — need at least 2.",
                file=sys.stderr,
            )
            sys.exit(1)

        print(f"Elo tournament: {len(paths)} checkpoints, "
              f"{args.episodes} episodes/pair, {args.elo_bots} extra bots")
        device = torch.device(args.device)
        elo = run_tournament(
            checkpoint_paths=paths,
            episodes_per_pair=args.episodes,
            n_bots=args.elo_bots,
            max_ticks=args.max_ticks,
            device=device,
            k_factor=args.k_factor,
            seed=args.seed,
        )

        print("\n" + elo.table_str())

        out_path = Path(args.elo_output) if args.elo_output else ckpt_dir / "elo_results.json"
        out_path.write_text(json.dumps(elo.to_dict(), indent=2), encoding="utf-8")
        print(f"\nElo results saved → {out_path}")
        return

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
