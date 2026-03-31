"""PPO training entry point for Jadid_Halghe.

Usage
-----
    python train.py --config configs/default.yaml
    python train.py --config configs/default.yaml --resume checkpoints/run_default/ckpt_000100.pt
    python train.py --config configs/default.yaml --device cuda
"""

from __future__ import annotations

import argparse
import logging
import random
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import yaml

from game.config import WorldConfig
from rl.agent import build_policy, load_policy
from rl.ppo import PPO
from rl.runner import Runner
from rl.vec_env import VecAgarEnv


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _to_ns(obj: object) -> object:
    """Recursively convert nested dicts to SimpleNamespace for attribute access."""
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: _to_ns(v) for k, v in obj.items()})
    return obj


def _load_config(path: str) -> SimpleNamespace:
    with open(path) as f:
        raw = yaml.safe_load(f)
    return _to_ns(raw)  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------

def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Parse CLI arguments, build environment + policy, and run the PPO training loop.

    Loads hyperparameters from a YAML config file.  Logs metrics to TensorBoard
    and saves checkpoints to ``config.train.checkpoint_dir``.  Pass ``--resume``
    to continue from an existing checkpoint.
    """
    parser = argparse.ArgumentParser(description="Train a PPO agent on the agar.io simulation.")
    parser.add_argument("--config", required=True, help="Path to YAML config file.")
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume from.")
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device string (e.g. 'cpu', 'cuda', 'cuda:0').",
    )
    args = parser.parse_args()

    cfg = _load_config(args.config)
    device = torch.device(args.device)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger("train")

    _seed_everything(cfg.train.seed)

    # ── Environment ─────────────────────────────────────────────────────────
    world_cfg = WorldConfig()
    venv = VecAgarEnv(
        n_envs=cfg.env.n_envs,
        config=world_cfg,
        n_bots=cfg.env.n_bots,
        max_ticks=cfg.env.max_ticks,
    )
    obs_dim: int = venv.single_observation_space.shape[0]
    act_dim: int = venv.single_action_space.shape[0]

    # ── Policy ───────────────────────────────────────────────────────────────
    policy_cfg = vars(cfg.policy)
    policy_type: str = policy_cfg.pop("type")

    start_step = 0
    if args.resume:
        log.info("Resuming from %s", args.resume)
        policy, meta = load_policy(args.resume)
        start_step = meta.get("step", 0)
        policy = policy.to(device)
    else:
        policy = build_policy(
            policy_type,
            obs_dim=obs_dim,
            act_dim=act_dim,
            **policy_cfg,
        ).to(device)

    n_params = sum(p.numel() for p in policy.parameters())
    log.info(
        "Policy: %s  |  params: %s  |  device: %s",
        type(policy).__name__,
        f"{n_params:,}",
        device,
    )

    # ── Optimiser + PPO ──────────────────────────────────────────────────────
    optimizer = torch.optim.Adam(policy.parameters(), lr=cfg.ppo.lr)

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    ppo = PPO(
        policy=policy,
        optimizer=optimizer,
        clip_range=cfg.ppo.clip_range,
        value_coef=cfg.ppo.value_coef,
        entropy_coef=cfg.ppo.entropy_coef,
        max_grad_norm=cfg.ppo.max_grad_norm,
    )

    runner = Runner(
        venv=venv,
        policy=policy,
        n_steps=cfg.ppo.n_steps,
        device=device,
        gamma=cfg.ppo.gamma,
        gae_lambda=cfg.ppo.gae_lambda,
    )

    # ── TensorBoard ──────────────────────────────────────────────────────────
    from torch.utils.tensorboard import SummaryWriter  # lazy import

    tb_dir = Path(cfg.train.checkpoint_dir) / "tb"
    writer = SummaryWriter(log_dir=str(tb_dir))
    log.info("TensorBoard: %s", tb_dir)

    ckpt_dir = Path(cfg.train.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ── Training loop ────────────────────────────────────────────────────────
    steps_per_rollout = cfg.ppo.n_steps * cfg.env.n_envs
    total_steps = cfg.train.total_steps
    rollout_idx = 0
    global_step = start_step

    log.info(
        "Training | total_steps=%s  steps_per_rollout=%d  n_epochs=%d",
        f"{total_steps:,}",
        steps_per_rollout,
        cfg.ppo.n_epochs,
    )

    t0 = time.perf_counter()

    while global_step < total_steps:
        # ── Collect rollout ──────────────────────────────────────────────
        buffer, rollout_info = runner.collect()

        # ── PPO update ───────────────────────────────────────────────────
        policy.train()
        metrics = ppo.update(buffer, n_epochs=cfg.ppo.n_epochs, batch_size=cfg.ppo.batch_size)

        global_step += steps_per_rollout
        rollout_idx += 1

        # ── Logging ──────────────────────────────────────────────────────
        if rollout_idx % cfg.train.log_interval == 0:
            elapsed = time.perf_counter() - t0
            fps = int(global_step / elapsed)

            writer.add_scalar("train/policy_loss", metrics["policy_loss"], global_step)
            writer.add_scalar("train/value_loss", metrics["value_loss"], global_step)
            writer.add_scalar("train/entropy", metrics["entropy"], global_step)
            writer.add_scalar("train/approx_kl", metrics["approx_kl"], global_step)
            writer.add_scalar("train/clip_fraction", metrics["clip_fraction"], global_step)
            writer.add_scalar("train/total_loss", metrics["total_loss"], global_step)

            if not np.isnan(rollout_info["mean_reward"]):
                writer.add_scalar("train/reward", rollout_info["mean_reward"], global_step)
                writer.add_scalar("train/episode_length", rollout_info["mean_episode_length"], global_step)

            log.info(
                "step %8s | fps %5d | reward %+7.3f | kl %.4f | ent %.3f | loss %.4f",
                f"{global_step:,}",
                fps,
                rollout_info["mean_reward"] if not np.isnan(rollout_info["mean_reward"]) else 0.0,
                metrics["approx_kl"],
                metrics["entropy"],
                metrics["total_loss"],
            )

        # ── Checkpoint ───────────────────────────────────────────────────
        if rollout_idx % cfg.train.save_interval == 0:
            ckpt_path = ckpt_dir / f"ckpt_{rollout_idx:06d}.pt"
            policy.save(
                ckpt_path,
                step=global_step,
                optimizer_state_dict=optimizer.state_dict(),
            )
            log.info("Saved checkpoint → %s", ckpt_path)

    # Final checkpoint
    final_path = ckpt_dir / "ckpt_final.pt"
    policy.save(final_path, step=global_step, optimizer_state_dict=optimizer.state_dict())
    log.info("Training complete. Final checkpoint → %s", final_path)

    writer.close()
    venv.close()


if __name__ == "__main__":
    main()
