from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from rl.agent import (
    ACT_DIM,
    AttentionPolicy,
    MLPPolicy,
    RecurrentPolicy,
    build_policy,
    load_policy,
)
from rl.env import OBS_DIM


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def mlp() -> MLPPolicy:
    return MLPPolicy(obs_dim=OBS_DIM, act_dim=ACT_DIM, hidden_dim=64)


@pytest.fixture()
def attn() -> AttentionPolicy:
    return AttentionPolicy(obs_dim=OBS_DIM, act_dim=ACT_DIM, embed_dim=16, n_heads=2, n_layers=1)


@pytest.fixture()
def rnn() -> RecurrentPolicy:
    return RecurrentPolicy(obs_dim=OBS_DIM, act_dim=ACT_DIM, hidden_dim=64, gru_hidden=32)


@pytest.fixture()
def batch_obs() -> torch.Tensor:
    return torch.zeros(4, OBS_DIM)  # (B=4, obs_dim) — all zeros ≈ dead/empty world


# ---------------------------------------------------------------------------
# MLPPolicy
# ---------------------------------------------------------------------------

class TestMLPPolicy:
    def test_act_output_shapes(self, mlp: MLPPolicy, batch_obs: torch.Tensor) -> None:
        z, lp, v = mlp.act(batch_obs)
        assert z.shape == (4, ACT_DIM)
        assert lp.shape == (4,)
        assert v.shape == (4,)

    def test_act_single_obs(self, mlp: MLPPolicy) -> None:
        obs = np.zeros(OBS_DIM, dtype=np.float32)
        z, lp, v = mlp.act(obs)
        assert z.shape == (1, ACT_DIM)

    def test_deterministic_no_randomness(self, mlp: MLPPolicy, batch_obs: torch.Tensor) -> None:
        z1, _, _ = mlp.act(batch_obs, deterministic=True)
        z2, _, _ = mlp.act(batch_obs, deterministic=True)
        torch.testing.assert_close(z1, z2)

    def test_stochastic_has_variance(self, mlp: MLPPolicy, batch_obs: torch.Tensor) -> None:
        zs = torch.stack([mlp.act(batch_obs)[0] for _ in range(10)])
        assert zs.std() > 1e-4, "Stochastic samples should vary"

    def test_no_nan_in_outputs(self, mlp: MLPPolicy, batch_obs: torch.Tensor) -> None:
        z, lp, v = mlp.act(batch_obs)
        assert not torch.any(torch.isnan(z))
        assert not torch.any(torch.isnan(lp))
        assert not torch.any(torch.isnan(v))

    def test_evaluate_shapes(self, mlp: MLPPolicy, batch_obs: torch.Tensor) -> None:
        z, _, _ = mlp.act(batch_obs)
        lp, v, ent = mlp.evaluate(batch_obs, z)
        assert lp.shape == (4,)
        assert v.shape == (4,)
        assert ent.shape == ()  # scalar

    def test_evaluate_log_prob_consistent(self, mlp: MLPPolicy, batch_obs: torch.Tensor) -> None:
        """evaluate() log-prob must equal act() log-prob for the same z."""
        z, lp_act, _ = mlp.act(batch_obs)
        lp_eval, _, _ = mlp.evaluate(batch_obs, z.detach())
        torch.testing.assert_close(lp_act.detach(), lp_eval.detach(), atol=1e-5, rtol=1e-5)

    def test_save_load_roundtrip(self, mlp: MLPPolicy, batch_obs: torch.Tensor) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "policy.pt"
            mlp.save(path, step=42)
            loaded, meta = MLPPolicy.load(path)
        assert meta["step"] == 42
        z_orig, _, _ = mlp.act(batch_obs, deterministic=True)
        z_load, _, _ = loaded.act(batch_obs, deterministic=True)
        torch.testing.assert_close(z_orig, z_load)

    def test_load_policy_factory(self, mlp: MLPPolicy) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "policy.pt"
            mlp.save(path)
            policy, _ = load_policy(path)
        assert isinstance(policy, MLPPolicy)


# ---------------------------------------------------------------------------
# AttentionPolicy
# ---------------------------------------------------------------------------

class TestAttentionPolicy:
    def test_act_output_shapes(self, attn: AttentionPolicy, batch_obs: torch.Tensor) -> None:
        z, lp, v = attn.act(batch_obs)
        assert z.shape == (4, ACT_DIM)
        assert lp.shape == (4,)
        assert v.shape == (4,)

    def test_no_nan_on_zero_obs(self, attn: AttentionPolicy, batch_obs: torch.Tensor) -> None:
        """All-zero obs = all entities padded — must not produce NaN."""
        z, lp, v = attn.act(batch_obs)
        assert not torch.any(torch.isnan(z)), "NaN in z"
        assert not torch.any(torch.isnan(lp)), "NaN in log_prob"
        assert not torch.any(torch.isnan(v)), "NaN in value"

    def test_no_nan_on_nonzero_obs(self, attn: AttentionPolicy) -> None:
        obs = torch.randn(4, OBS_DIM)
        z, lp, v = attn.act(obs)
        assert not torch.any(torch.isnan(z))
        assert not torch.any(torch.isnan(lp))
        assert not torch.any(torch.isnan(v))

    def test_evaluate_shapes(self, attn: AttentionPolicy, batch_obs: torch.Tensor) -> None:
        z, _, _ = attn.act(batch_obs)
        lp, v, ent = attn.evaluate(batch_obs, z)
        assert lp.shape == (4,)
        assert v.shape == (4,)

    def test_evaluate_log_prob_consistent(
        self, attn: AttentionPolicy, batch_obs: torch.Tensor
    ) -> None:
        z, lp_act, _ = attn.act(batch_obs)
        lp_eval, _, _ = attn.evaluate(batch_obs, z.detach())
        torch.testing.assert_close(lp_act.detach(), lp_eval.detach(), atol=1e-5, rtol=1e-5)

    def test_save_load_roundtrip(self, attn: AttentionPolicy, batch_obs: torch.Tensor) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "attn.pt"
            attn.save(path, step=10)
            loaded, meta = AttentionPolicy.load(path)
        assert meta["step"] == 10
        z_orig, _, _ = attn.act(batch_obs, deterministic=True)
        z_load, _, _ = loaded.act(batch_obs, deterministic=True)
        torch.testing.assert_close(z_orig, z_load)

    def test_permutation_sensitivity(self, attn: AttentionPolicy) -> None:
        """Different obs should give different outputs (sanity check)."""
        obs_a = torch.zeros(1, OBS_DIM)
        obs_b = torch.ones(1, OBS_DIM) * 0.1
        z_a, _, _ = attn.act(obs_a, deterministic=True)
        z_b, _, _ = attn.act(obs_b, deterministic=True)
        assert not torch.allclose(z_a, z_b)


# ---------------------------------------------------------------------------
# RecurrentPolicy
# ---------------------------------------------------------------------------

class TestRecurrentPolicy:
    def test_act_output_shapes(self, rnn: RecurrentPolicy) -> None:
        obs = torch.zeros(4, OBS_DIM)
        h = rnn.initial_state(4)
        z, lp, v, new_h = rnn.act(obs, h)
        assert z.shape == (4, ACT_DIM)
        assert lp.shape == (4,)
        assert v.shape == (4,)
        assert new_h.shape == (1, 4, rnn.gru_hidden)

    def test_hidden_state_changes(self, rnn: RecurrentPolicy) -> None:
        obs = torch.randn(1, OBS_DIM)
        h = rnn.initial_state(1)
        _, _, _, h2 = rnn.act(obs, h)
        assert not torch.allclose(h, h2), "Hidden state should update after a step"

    def test_no_nan(self, rnn: RecurrentPolicy) -> None:
        obs = torch.randn(4, OBS_DIM)
        h = rnn.initial_state(4)
        z, lp, v, _ = rnn.act(obs, h)
        assert not torch.any(torch.isnan(z))
        assert not torch.any(torch.isnan(lp))
        assert not torch.any(torch.isnan(v))

    def test_save_load_roundtrip(self, rnn: RecurrentPolicy) -> None:
        obs = torch.zeros(1, OBS_DIM)
        h = rnn.initial_state(1)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rnn.pt"
            rnn.save(path, step=7)
            loaded, meta = RecurrentPolicy.load(path)
        assert meta["step"] == 7
        z_orig, _, _, _ = rnn.act(obs, h, deterministic=True)
        z_load, _, _, _ = loaded.act(obs, h, deterministic=True)
        torch.testing.assert_close(z_orig, z_load)


# ---------------------------------------------------------------------------
# build_policy factory
# ---------------------------------------------------------------------------

class TestBuildPolicy:
    def test_mlp(self) -> None:
        p = build_policy("mlp", hidden_dim=32)
        assert isinstance(p, MLPPolicy)

    def test_attention(self) -> None:
        p = build_policy("attention", embed_dim=16, n_heads=2, n_layers=1)
        assert isinstance(p, AttentionPolicy)

    def test_recurrent(self) -> None:
        p = build_policy("recurrent", hidden_dim=32, gru_hidden=16)
        assert isinstance(p, RecurrentPolicy)

    def test_unknown_raises(self) -> None:
        with pytest.raises(ValueError):
            build_policy("transformer_xl")
