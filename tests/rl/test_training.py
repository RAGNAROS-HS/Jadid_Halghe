from __future__ import annotations

import numpy as np
import pytest
import torch

from game.config import WorldConfig
from rl.agent import AttentionPolicy, MLPPolicy
from rl.buffer import RolloutBuffer
from rl.env import OBS_DIM
from rl.ppo import PPO
from rl.runner import Runner
from rl.vec_env import VecAgarEnv


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def small_cfg() -> WorldConfig:
    return WorldConfig(
        width=1000.0,
        height=1000.0,
        max_cells=64,
        max_food=128,
        max_viruses=4,
        max_ejected=16,
        max_players=4,
        target_food_count=30,
        target_virus_count=2,
    )


@pytest.fixture()
def device() -> torch.device:
    return torch.device("cpu")


@pytest.fixture()
def mlp_policy(device: torch.device) -> MLPPolicy:
    return MLPPolicy(obs_dim=OBS_DIM, act_dim=4, hidden_dim=32).to(device)


# ---------------------------------------------------------------------------
# RolloutBuffer
# ---------------------------------------------------------------------------

class TestRolloutBuffer:
    def test_add_and_full(self, device: torch.device) -> None:
        buf = RolloutBuffer(n_steps=4, n_envs=2, obs_dim=OBS_DIM, act_dim=4, device=device)
        for _ in range(4):
            buf.add(
                torch.zeros(2, OBS_DIM),
                torch.zeros(2, 4),
                torch.zeros(2),
                torch.zeros(2),
                torch.zeros(2),
                torch.zeros(2, dtype=torch.bool),
            )
        assert buf.full

    def test_not_full_before_n_steps(self, device: torch.device) -> None:
        buf = RolloutBuffer(n_steps=4, n_envs=2, obs_dim=OBS_DIM, act_dim=4, device=device)
        buf.add(
            torch.zeros(2, OBS_DIM),
            torch.zeros(2, 4),
            torch.zeros(2),
            torch.zeros(2),
            torch.zeros(2),
            torch.zeros(2, dtype=torch.bool),
        )
        assert not buf.full

    def test_gae_no_nan(self, device: torch.device) -> None:
        T, N = 8, 3
        buf = RolloutBuffer(n_steps=T, n_envs=N, obs_dim=OBS_DIM, act_dim=4, device=device)
        for _ in range(T):
            buf.add(
                torch.randn(N, OBS_DIM),
                torch.randn(N, 4),
                torch.randn(N),
                torch.randn(N),
                torch.randn(N),
                torch.zeros(N, dtype=torch.bool),
            )
        buf.compute_returns_and_advantages(
            last_values=torch.zeros(N),
            last_dones=torch.zeros(N, dtype=torch.bool),
        )
        assert not torch.any(torch.isnan(buf.returns))
        assert not torch.any(torch.isnan(buf.advantages))

    def test_get_batches_coverage(self, device: torch.device) -> None:
        """All T*N transitions should appear in the batches."""
        T, N, B = 4, 2, 3
        buf = RolloutBuffer(n_steps=T, n_envs=N, obs_dim=OBS_DIM, act_dim=4, device=device)
        for i in range(T):
            buf.add(
                torch.full((N, OBS_DIM), float(i)),
                torch.zeros(N, 4),
                torch.zeros(N),
                torch.zeros(N),
                torch.zeros(N),
                torch.zeros(N, dtype=torch.bool),
            )
        buf.compute_returns_and_advantages(torch.zeros(N), torch.zeros(N, dtype=torch.bool))

        total_seen = 0
        for batch in buf.get_batches(batch_size=B, shuffle=False):
            total_seen += batch[0].shape[0]
        assert total_seen == T * N

    def test_reset_clears_full_flag(self, device: torch.device) -> None:
        buf = RolloutBuffer(n_steps=2, n_envs=1, obs_dim=OBS_DIM, act_dim=4, device=device)
        for _ in range(2):
            buf.add(torch.zeros(1, OBS_DIM), torch.zeros(1, 4), torch.zeros(1),
                    torch.zeros(1), torch.zeros(1), torch.zeros(1, dtype=torch.bool))
        assert buf.full
        buf.reset()
        assert not buf.full


# ---------------------------------------------------------------------------
# PPO
# ---------------------------------------------------------------------------

class TestPPO:
    def test_update_returns_loss_keys(
        self, mlp_policy: MLPPolicy, device: torch.device
    ) -> None:
        optimizer = torch.optim.Adam(mlp_policy.parameters(), lr=1e-3)
        ppo = PPO(mlp_policy, optimizer)

        T, N = 4, 2
        buf = RolloutBuffer(n_steps=T, n_envs=N, obs_dim=OBS_DIM, act_dim=4, device=device)
        for _ in range(T):
            obs = torch.randn(N, OBS_DIM)
            with torch.no_grad():
                z, lp, v = mlp_policy.act(obs)
            buf.add(obs, z, lp, torch.randn(N), v, torch.zeros(N, dtype=torch.bool))
        buf.compute_returns_and_advantages(torch.zeros(N), torch.zeros(N, dtype=torch.bool))

        metrics = ppo.update(buf, n_epochs=1, batch_size=4)
        assert "policy_loss" in metrics
        assert "value_loss" in metrics
        assert "entropy" in metrics
        assert "approx_kl" in metrics

    def test_loss_values_are_finite(
        self, mlp_policy: MLPPolicy, device: torch.device
    ) -> None:
        optimizer = torch.optim.Adam(mlp_policy.parameters(), lr=1e-3)
        ppo = PPO(mlp_policy, optimizer)

        T, N = 8, 2
        buf = RolloutBuffer(n_steps=T, n_envs=N, obs_dim=OBS_DIM, act_dim=4, device=device)
        for _ in range(T):
            obs = torch.randn(N, OBS_DIM)
            with torch.no_grad():
                z, lp, v = mlp_policy.act(obs)
            buf.add(obs, z, lp, torch.randn(N), v, torch.zeros(N, dtype=torch.bool))
        buf.compute_returns_and_advantages(torch.zeros(N), torch.zeros(N, dtype=torch.bool))

        metrics = ppo.update(buf, n_epochs=2, batch_size=4)
        for k, val in metrics.items():
            assert np.isfinite(val), f"Metric {k} = {val} is not finite"

    def test_weights_change_after_update(
        self, mlp_policy: MLPPolicy, device: torch.device
    ) -> None:
        params_before = [p.clone() for p in mlp_policy.parameters()]
        optimizer = torch.optim.Adam(mlp_policy.parameters(), lr=1e-3)
        ppo = PPO(mlp_policy, optimizer)

        T, N = 4, 2
        buf = RolloutBuffer(n_steps=T, n_envs=N, obs_dim=OBS_DIM, act_dim=4, device=device)
        for _ in range(T):
            obs = torch.randn(N, OBS_DIM)
            with torch.no_grad():
                z, lp, v = mlp_policy.act(obs)
            buf.add(obs, z, lp, torch.ones(N), v, torch.zeros(N, dtype=torch.bool))
        buf.compute_returns_and_advantages(torch.zeros(N), torch.zeros(N, dtype=torch.bool))
        ppo.update(buf, n_epochs=1, batch_size=4)

        changed = any(
            not torch.allclose(p, pb)
            for p, pb in zip(mlp_policy.parameters(), params_before)
        )
        assert changed, "Policy weights should change after a PPO update"


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

class TestRunner:
    def test_collect_returns_full_buffer(
        self, small_cfg: WorldConfig, mlp_policy: MLPPolicy, device: torch.device
    ) -> None:
        venv = VecAgarEnv(n_envs=2, config=small_cfg, n_bots=1, max_ticks=50)
        runner = Runner(venv=venv, policy=mlp_policy, n_steps=8, device=device)
        buf, info = runner.collect()
        assert buf.full
        assert buf.obs.shape == (8, 2, OBS_DIM)
        assert buf.actions.shape == (8, 2, 4)
        assert isinstance(info, dict)
        assert "mean_reward" in info

    def test_collect_no_nan(
        self, small_cfg: WorldConfig, mlp_policy: MLPPolicy, device: torch.device
    ) -> None:
        venv = VecAgarEnv(n_envs=2, config=small_cfg, n_bots=1, max_ticks=50)
        runner = Runner(venv=venv, policy=mlp_policy, n_steps=8, device=device)
        buf, _ = runner.collect()
        assert not torch.any(torch.isnan(buf.obs))
        assert not torch.any(torch.isnan(buf.rewards))
        assert not torch.any(torch.isnan(buf.returns))
        assert not torch.any(torch.isnan(buf.advantages))

    def test_consecutive_collects_are_contiguous(
        self, small_cfg: WorldConfig, mlp_policy: MLPPolicy, device: torch.device
    ) -> None:
        """Second collect should continue from where the first left off (no re-reset)."""
        venv = VecAgarEnv(n_envs=1, config=small_cfg, n_bots=0, max_ticks=100)
        runner = Runner(venv=venv, policy=mlp_policy, n_steps=5, device=device)
        _, _ = runner.collect()
        # Second collect should not raise and should produce a valid buffer
        buf2, _ = runner.collect()
        assert buf2.full

    def test_attention_policy_in_runner(
        self, small_cfg: WorldConfig, device: torch.device
    ) -> None:
        policy = AttentionPolicy(
            obs_dim=OBS_DIM, act_dim=4, embed_dim=16, n_heads=2, n_layers=1
        ).to(device)
        venv = VecAgarEnv(n_envs=2, config=small_cfg, n_bots=1, max_ticks=50)
        runner = Runner(venv=venv, policy=policy, n_steps=8, device=device)
        buf, _ = runner.collect()
        assert buf.full
        assert not torch.any(torch.isnan(buf.obs))
