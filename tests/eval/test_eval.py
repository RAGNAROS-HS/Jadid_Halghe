"""Tests for eval/ — harness, baselines, replay, and attention maps."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from eval.baselines import GreedyPolicy, RandomPolicy
from eval.harness import EpisodeResult, EvalResult, Harness
from eval.replay import ReplayEpisode, load_replay, plot_mass_over_time, save_replay
from game.config import WorldConfig
from rl.env import OBS_DIM, build_observation
from rl.agent import AttentionPolicy, MLPPolicy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CFG_SMALL = WorldConfig(width=2000, height=2000, max_food=100, max_players=4)


def _dummy_obs() -> np.ndarray:
    return np.zeros(OBS_DIM, dtype=np.float32)


def _random_obs(rng: np.random.Generator | None = None) -> np.ndarray:
    rng = rng or np.random.default_rng(0)
    return rng.standard_normal(OBS_DIM).astype(np.float32)


# ---------------------------------------------------------------------------
# RandomPolicy
# ---------------------------------------------------------------------------

class TestRandomPolicy:
    def test_output_shape(self):
        pol = RandomPolicy()
        a = pol.act(_dummy_obs())
        assert a.shape == (4,)
        assert a.dtype == np.float32

    def test_output_range(self):
        pol = RandomPolicy(seed=1)
        for _ in range(20):
            a = pol.act(_dummy_obs())
            assert np.all(a >= -1.0) and np.all(a <= 1.0)

    def test_seeded_reproducible(self):
        a1 = RandomPolicy(seed=42).act(_dummy_obs())
        a2 = RandomPolicy(seed=42).act(_dummy_obs())
        np.testing.assert_array_equal(a1, a2)

    def test_different_seeds_differ(self):
        a1 = RandomPolicy(seed=0).act(_dummy_obs())
        a2 = RandomPolicy(seed=1).act(_dummy_obs())
        assert not np.allclose(a1, a2)

    def test_no_nan(self):
        pol = RandomPolicy(seed=7)
        for _ in range(10):
            a = pol.act(_random_obs())
            assert np.all(np.isfinite(a))


# ---------------------------------------------------------------------------
# GreedyPolicy
# ---------------------------------------------------------------------------

class TestGreedyPolicy:
    def test_output_shape(self):
        pol = GreedyPolicy()
        a = pol.act(_dummy_obs())
        assert a.shape == (4,)

    def test_no_nan(self):
        pol = GreedyPolicy()
        rng = np.random.default_rng(0)
        for _ in range(20):
            a = pol.act(_random_obs(rng))
            assert np.all(np.isfinite(a))

    def test_no_split_or_eject(self):
        pol = GreedyPolicy()
        for _ in range(20):
            a = pol.act(_random_obs())
            assert a[2] == 0.0 and a[3] == 0.0

    def test_direction_clipped(self):
        pol = GreedyPolicy()
        for _ in range(20):
            a = pol.act(_random_obs())
            assert np.all(a[:2] >= -1.0) and np.all(a[:2] <= 1.0)

    def test_zero_obs_wanders(self):
        """On an all-zero obs, GreedyPolicy should still return a finite action."""
        pol = GreedyPolicy()
        a = pol.act(_dummy_obs())
        assert np.all(np.isfinite(a))


# ---------------------------------------------------------------------------
# Harness (fast, small world)
# ---------------------------------------------------------------------------

class TestHarness:
    def _make_harness(self, opponent="random"):
        return Harness(
            policy=RandomPolicy(),
            opponent=opponent,
            config=_CFG_SMALL,
            n_bots=2,
            max_ticks=30,
        )

    def test_run_returns_eval_result(self):
        h = self._make_harness()
        result, replays = h.run(n_episodes=2, seed=0)
        assert isinstance(result, EvalResult)
        assert result.n_episodes == 2

    def test_replays_length_matches_episodes(self):
        h = self._make_harness()
        result, replays = h.run(n_episodes=3, seed=0)
        assert len(replays) == 3

    def test_record_false_empty_frames(self):
        h = self._make_harness()
        _, replays = h.run(n_episodes=2, seed=0, record=False)
        assert all(len(r) == 0 for r in replays)

    def test_record_true_has_frames(self):
        h = self._make_harness()
        _, replays = h.run(n_episodes=2, seed=0, record=True)
        assert all(len(r) > 0 for r in replays)

    def test_mean_reward_finite(self):
        h = self._make_harness()
        result, _ = h.run(n_episodes=3, seed=1)
        assert np.isfinite(result.mean_reward)
        assert np.isfinite(result.std_reward)

    def test_survival_rate_in_range(self):
        h = self._make_harness()
        result, _ = h.run(n_episodes=5, seed=2)
        assert 0.0 <= result.survival_rate <= 1.0

    def test_mean_rank_positive(self):
        h = self._make_harness()
        result, _ = h.run(n_episodes=3, seed=3)
        assert result.mean_rank >= 1.0

    def test_greedy_opponent(self):
        h = self._make_harness(opponent="greedy")
        result, _ = h.run(n_episodes=2, seed=0)
        assert isinstance(result, EvalResult)

    def test_episode_results_stored(self):
        h = self._make_harness()
        result, _ = h.run(n_episodes=4, seed=0)
        assert len(result.episodes) == 4
        for ep in result.episodes:
            assert isinstance(ep, EpisodeResult)
            assert ep.length > 0

    def test_summary_string(self):
        h = self._make_harness()
        result, _ = h.run(n_episodes=2, seed=0)
        s = result.summary()
        assert "episodes=2" in s
        assert "survival=" in s

    def test_deterministic_with_seed(self):
        # Use seeded RandomPolicies so both harnesses produce identical results.
        h1 = Harness(
            policy=RandomPolicy(seed=0),
            opponent=RandomPolicy(seed=1),
            config=_CFG_SMALL,
            n_bots=2,
            max_ticks=30,
        )
        h2 = Harness(
            policy=RandomPolicy(seed=0),
            opponent=RandomPolicy(seed=1),
            config=_CFG_SMALL,
            n_bots=2,
            max_ticks=30,
        )
        r1, _ = h1.run(n_episodes=2, seed=99)
        r2, _ = h2.run(n_episodes=2, seed=99)
        assert r1.mean_reward == pytest.approx(r2.mean_reward, abs=1e-5)

    def test_mlp_policy(self):
        pol = MLPPolicy(obs_dim=OBS_DIM)
        h = Harness(policy=pol, config=_CFG_SMALL, n_bots=1, max_ticks=20)
        result, _ = h.run(n_episodes=1, seed=0)
        assert np.isfinite(result.mean_reward)

    def test_attention_policy(self):
        pol = AttentionPolicy(obs_dim=OBS_DIM)
        h = Harness(policy=pol, config=_CFG_SMALL, n_bots=1, max_ticks=20)
        result, _ = h.run(n_episodes=1, seed=0)
        assert np.isfinite(result.mean_reward)


# ---------------------------------------------------------------------------
# Replay save/load
# ---------------------------------------------------------------------------

class TestReplay:
    def _make_episode(self) -> ReplayEpisode:
        from game.world import World

        cfg = _CFG_SMALL
        world = World(cfg)
        world.reset(seed=0)
        world.add_player(0)
        world.add_player(1)
        frames = []
        for _ in range(10):
            frames.append(world.get_state())
            actions = np.zeros((cfg.max_players, 4), dtype=np.float32)
            world.step(actions)
        return ReplayEpisode(config=cfg, frames=frames, metadata={"label": "test"})

    def test_save_and_load_roundtrip(self, tmp_path):
        ep = self._make_episode()
        path = tmp_path / "test.pkl"
        save_replay(ep, path)
        loaded = load_replay(path)
        assert isinstance(loaded, ReplayEpisode)
        assert len(loaded.frames) == len(ep.frames)

    def test_metadata_preserved(self, tmp_path):
        ep = self._make_episode()
        path = tmp_path / "test.pkl"
        save_replay(ep, path)
        loaded = load_replay(path)
        assert loaded.metadata["label"] == "test"

    def test_creates_parent_dirs(self, tmp_path):
        ep = self._make_episode()
        nested = tmp_path / "a" / "b" / "ep.pkl"
        save_replay(ep, nested)
        assert nested.exists()

    def test_frame_states_intact(self, tmp_path):
        ep = self._make_episode()
        path = tmp_path / "test.pkl"
        save_replay(ep, path)
        loaded = load_replay(path)
        orig_mass = ep.frames[0].player_mass.copy()
        loaded_mass = loaded.frames[0].player_mass
        np.testing.assert_array_equal(orig_mass, loaded_mass)


# ---------------------------------------------------------------------------
# plot_mass_over_time (non-interactive, save to file)
# ---------------------------------------------------------------------------

class TestPlotMassOverTime:
    def _make_episode(self) -> ReplayEpisode:
        from game.world import World

        cfg = _CFG_SMALL
        world = World(cfg)
        world.reset(seed=0)
        world.add_player(0)
        frames = []
        for _ in range(15):
            frames.append(world.get_state())
            actions = np.zeros((cfg.max_players, 4), dtype=np.float32)
            world.step(actions)
        return ReplayEpisode(config=cfg, frames=frames, metadata={"label": "ep0"})

    def test_saves_figure(self, tmp_path):
        ep = self._make_episode()
        out = tmp_path / "mass.png"
        plot_mass_over_time([ep], player_id=0, save_path=out)
        assert out.exists()
        assert out.stat().st_size > 0

    def test_multiple_episodes(self, tmp_path):
        ep1 = self._make_episode()
        ep2 = self._make_episode()
        out = tmp_path / "mass2.png"
        plot_mass_over_time([ep1, ep2], save_path=out)
        assert out.exists()


# ---------------------------------------------------------------------------
# attention_maps
# ---------------------------------------------------------------------------

class TestAttentionMaps:
    def test_returns_two_outputs(self):
        pol = AttentionPolicy(obs_dim=OBS_DIM)
        pol.eval()
        obs = _random_obs()
        mean_act, weights = pol.attention_maps(obs)
        assert mean_act is not None
        assert isinstance(weights, list)

    def test_weight_list_length_equals_n_layers(self):
        pol = AttentionPolicy(obs_dim=OBS_DIM, n_layers=2)
        pol.eval()
        _, weights = pol.attention_maps(_random_obs())
        assert len(weights) == 2

    def test_weight_shape(self):
        pol = AttentionPolicy(obs_dim=OBS_DIM, n_heads=4, n_layers=2)
        pol.eval()
        _, weights = pol.attention_maps(_random_obs())
        # Each entry should be (1, n_heads, N, N) or (n_heads, N, N)
        for w in weights:
            assert w.ndim >= 3
            assert w.shape[-3] == 4  # n_heads

    def test_weights_sum_to_one_over_keys(self):
        pol = AttentionPolicy(obs_dim=OBS_DIM)
        pol.eval()
        _, weights = pol.attention_maps(_random_obs())
        for w in weights:
            # Sum over key dimension should be ~1 (softmax output)
            w_squeezed = w.squeeze(0)  # (n_heads, N, N)
            sums = w_squeezed.sum(axis=-1)  # (n_heads, N)
            np.testing.assert_allclose(sums, np.ones_like(sums), atol=1e-5)

    def test_no_nan_in_weights(self):
        pol = AttentionPolicy(obs_dim=OBS_DIM)
        pol.eval()
        _, weights = pol.attention_maps(_random_obs())
        for w in weights:
            assert np.all(np.isfinite(w))
