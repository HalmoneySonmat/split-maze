"""Unit tests for ``split_maze.train`` — Phase 1.3 산출물.

All tests use :class:`MockMazeEnv` (gym3-호환, no procgen). Verifies:
1. MockMazeEnv surface matches the gym3 contract assumed by the loop.
2. ``obs_to_tensor`` correctness (shape, dtype, contiguity, value preservation).
3. ``collect_rollout`` fills the buffer correctly + tracks episodes.
4. Full ``train`` smoke: 2 PPO updates run without NaN/Inf, log dicts are
   complete, parameters move under grad, log_callback fires per update.

The PyTorch path here is the safety net: sandbox disk pressure made
direct ``pip install torch`` impossible this session, so these tests
are written to pass cleanly the first time they run in user WSL.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from split_maze.agent import ImpalaAgent
from split_maze.ppo import PPOConfig, RolloutBuffer
from split_maze.train import (
    MockMazeEnv,
    RolloutStats,
    collect_rollout,
    obs_to_tensor,
    train,
)


# ---- MockMazeEnv surface ------------------------------------------------

def test_mock_env_observe_initial_shapes_and_first_all_true():
    env = MockMazeEnv(num=4, episode_length=8, seed=0)
    rew, obs_dict, first = env.observe()
    assert rew.shape == (4,) and rew.dtype == np.float32
    assert obs_dict["rgb"].shape == (4, 64, 64, 3)
    assert obs_dict["rgb"].dtype == np.uint8
    assert first.shape == (4,) and first.dtype == np.bool_
    assert first.all(), "initial observe should report all envs just-reset"


def test_mock_env_reward_is_action_0():
    env = MockMazeEnv(num=2, episode_length=8, seed=0)
    env.observe()  # initial
    env.act(np.array([0, 5], dtype=np.int32))
    rew, _, _ = env.observe()
    np.testing.assert_allclose(rew, [0.1, 0.0])


def test_mock_env_first_triggers_at_episode_end():
    env = MockMazeEnv(num=2, episode_length=3, seed=0)
    env.observe()  # initial reset (first=True)
    for t in range(3):
        env.act(np.zeros(2, dtype=np.int32))
        _, _, first = env.observe()
        if t < 2:
            assert not first.any(), f"first should still be False at t={t}"
        else:
            assert first.all(), "first should be True on the terminal step"


def test_mock_env_step_count_resets_after_termination():
    env = MockMazeEnv(num=1, episode_length=2, seed=0)
    env.observe()
    env.act(np.zeros(1, dtype=np.int32))   # step 1
    _, _, first = env.observe()
    assert not first.any()
    env.act(np.zeros(1, dtype=np.int32))   # step 2 → terminate
    _, _, first = env.observe()
    assert first.all()
    env.act(np.zeros(1, dtype=np.int32))   # step 1 of NEW episode
    _, _, first = env.observe()
    assert not first.any(), "step_count must reset after termination"


def test_mock_env_rejects_wrong_action_shape():
    env = MockMazeEnv(num=4, seed=0)
    with pytest.raises(ValueError):
        env.act(np.zeros(3, dtype=np.int32))


def test_mock_env_rejects_bad_init_args():
    with pytest.raises(ValueError):
        MockMazeEnv(num=0)
    with pytest.raises(ValueError):
        MockMazeEnv(episode_length=0)


# ---- obs_to_tensor ------------------------------------------------------

def test_obs_to_tensor_shape_dtype_contiguous():
    obs_dict = {"rgb": np.zeros((4, 64, 64, 3), dtype=np.uint8)}
    t = obs_to_tensor(obs_dict, "cpu")
    assert t.shape == (4, 3, 64, 64)
    assert t.dtype == torch.uint8
    assert t.is_contiguous()


def test_obs_to_tensor_preserves_per_channel_values():
    rng = np.random.RandomState(42)
    rgb = rng.randint(0, 256, size=(2, 64, 64, 3), dtype=np.uint8)
    t = obs_to_tensor({"rgb": rgb}, "cpu").numpy()
    # NCHW slice [:, c] should equal NHWC slice [..., c]
    for c in range(3):
        np.testing.assert_array_equal(t[:, c], rgb[..., c])


def test_obs_to_tensor_rejects_3d():
    with pytest.raises(ValueError):
        obs_to_tensor({"rgb": np.zeros((4, 64, 64), dtype=np.uint8)}, "cpu")


# ---- collect_rollout ----------------------------------------------------

def test_collect_rollout_fills_buffer_and_advances_obs():
    torch.manual_seed(0)
    env = MockMazeEnv(num=4, episode_length=8, seed=0)
    agent = ImpalaAgent()
    buffer = RolloutBuffer(T=8, N=4, device="cpu")
    _, obs_dict, _ = env.observe()
    obs_holder = obs_to_tensor(obs_dict, "cpu")
    ep_ret = np.zeros(4, dtype=np.float64)
    ep_len = np.zeros(4, dtype=np.int64)

    stats, next_obs = collect_rollout(
        env, agent, buffer,
        obs_holder=obs_holder,
        episode_returns=ep_ret,
        episode_lengths=ep_len,
        device="cpu",
    )

    assert buffer.obs.shape == (8, 4, 3, 64, 64)
    assert buffer.obs.dtype == torch.uint8
    assert buffer.action.shape == (8, 4) and buffer.action.dtype == torch.long
    assert buffer.reward.shape == (8, 4) and buffer.reward.dtype == torch.float32
    assert buffer.done.shape == (8, 4) and buffer.done.dtype == torch.float32
    assert next_obs.shape == (4, 3, 64, 64) and next_obs.dtype == torch.uint8
    # 8 step rollout × episode_length 8 → exactly one termination per env
    assert stats.num_completed == 4
    assert isinstance(stats, RolloutStats)


def test_collect_rollout_done_count_per_env_matches_episode_count():
    torch.manual_seed(0)
    env = MockMazeEnv(num=2, episode_length=3, seed=0)
    agent = ImpalaAgent()
    buffer = RolloutBuffer(T=6, N=2, device="cpu")
    _, obs_dict, _ = env.observe()
    obs_holder = obs_to_tensor(obs_dict, "cpu")
    collect_rollout(
        env, agent, buffer,
        obs_holder=obs_holder,
        episode_returns=np.zeros(2, dtype=np.float64),
        episode_lengths=np.zeros(2, dtype=np.int64),
        device="cpu",
    )
    # episode_length=3 over T=6 → exactly 2 terminations per env
    dones_per_env = buffer.done.numpy().sum(axis=0)
    np.testing.assert_array_equal(dones_per_env, [2, 2])


def test_collect_rollout_resets_episode_trackers_in_place():
    torch.manual_seed(0)
    env = MockMazeEnv(num=2, episode_length=3, seed=0)
    agent = ImpalaAgent()
    buffer = RolloutBuffer(T=6, N=2, device="cpu")
    _, obs_dict, _ = env.observe()
    obs_holder = obs_to_tensor(obs_dict, "cpu")
    ep_ret = np.zeros(2, dtype=np.float64)
    ep_len = np.zeros(2, dtype=np.int64)
    collect_rollout(
        env, agent, buffer,
        obs_holder=obs_holder,
        episode_returns=ep_ret,
        episode_lengths=ep_len,
        device="cpu",
    )
    # After 2 full episodes per env, the trackers must be at the start of
    # the next (third) episode — i.e. < episode_length.
    assert (ep_len < 3).all(), f"ep_len should reset, got {ep_len}"


def test_collect_rollout_rejects_env_num_mismatch():
    env = MockMazeEnv(num=4, seed=0)
    agent = ImpalaAgent()
    buffer = RolloutBuffer(T=4, N=8, device="cpu")
    _, obs_dict, _ = env.observe()
    obs_holder = obs_to_tensor(obs_dict, "cpu")
    with pytest.raises(ValueError, match="env.num"):
        collect_rollout(
            env, agent, buffer,
            obs_holder=obs_holder,
            episode_returns=np.zeros(4, dtype=np.float64),
            episode_lengths=np.zeros(4, dtype=np.int64),
            device="cpu",
        )


# ---- train (full smoke) -------------------------------------------------

def test_train_smoke_runs_two_updates_without_nan():
    torch.manual_seed(0)
    env = MockMazeEnv(num=4, episode_length=4, seed=0)
    agent = ImpalaAgent()
    # T=16, N=4 → T*N=64; total=128 → 2 updates. mini_batches=8 → mb size 8.
    logs = train(env, agent, PPOConfig(),
                 num_steps=16, total_env_steps=128, device="cpu")
    assert len(logs) == 2
    for log in logs:
        for k in ("total", "policy", "value", "entropy",
                  "approx_kl", "clipfrac"):
            assert k in log, f"missing log key: {k}"
            assert np.isfinite(log[k]), f"{k} = {log[k]} (not finite)"


def test_train_smoke_param_updates_under_ppo():
    """Direct grad-flow check: policy.weight must move after PPO updates."""
    torch.manual_seed(0)
    env = MockMazeEnv(num=4, episode_length=4, seed=0)
    agent = ImpalaAgent()
    before = agent.policy.weight.detach().clone()
    train(env, agent, PPOConfig(),
          num_steps=16, total_env_steps=128, device="cpu")
    after = agent.policy.weight.detach().clone()
    max_abs_diff = (after - before).abs().max().item()
    assert max_abs_diff > 1e-7, (
        f"policy weights did not change (max |Δ|={max_abs_diff:.2e}); "
        "PPO grad path may be broken")


def test_train_smoke_log_callback_fires_per_update():
    env = MockMazeEnv(num=4, episode_length=4, seed=0)
    agent = ImpalaAgent()
    seen: list[tuple[int, dict]] = []
    logs = train(env, agent, PPOConfig(),
                 num_steps=16, total_env_steps=128, device="cpu",
                 log_callback=lambda i, d: seen.append((i, dict(d))))
    assert len(seen) == len(logs) == 2
    assert [i for i, _ in seen] == [0, 1]


def test_train_at_least_one_update_for_tiny_budget():
    """total_env_steps < T*N must still produce one update."""
    env = MockMazeEnv(num=4, episode_length=4, seed=0)
    agent = ImpalaAgent()
    logs = train(env, agent, PPOConfig(),
                 num_steps=16, total_env_steps=10,  # < T*N=64
                 device="cpu")
    assert len(logs) == 1


def test_train_log_keys_complete():
    env = MockMazeEnv(num=4, episode_length=4, seed=0)
    agent = ImpalaAgent()
    logs = train(env, agent, PPOConfig(),
                 num_steps=16, total_env_steps=128, device="cpu")
    expected = {"update_idx", "env_steps", "ep_count", "ep_return_mean",
                "total", "policy", "value", "entropy",
                "approx_kl", "clipfrac"}
    for log in logs:
        missing = expected - set(log.keys())
        assert not missing, f"missing keys in log: {missing}"


def test_train_env_steps_increments_correctly():
    env = MockMazeEnv(num=4, episode_length=4, seed=0)
    agent = ImpalaAgent()
    logs = train(env, agent, PPOConfig(),
                 num_steps=16, total_env_steps=128, device="cpu")
    # T*N = 64 per update
    assert logs[0]["env_steps"] == 64
    assert logs[1]["env_steps"] == 128


# ---- rolling-window ep_return logging -----------------------------------

def test_train_rolling_keys_present_and_finite_after_episodes():
    """rolling fields should appear in every log dict and be finite once
    at least one episode has completed."""
    env = MockMazeEnv(num=4, episode_length=4, seed=0)
    agent = ImpalaAgent()
    logs = train(env, agent, PPOConfig(),
                 num_steps=16, total_env_steps=128, device="cpu")
    for log in logs:
        for k in ("ep_return_rolling", "ep_length_rolling",
                  "ep_return_rolling_n", "ep_length_mean"):
            assert k in log, f"missing rolling key: {k}"
        # T=16, N=4, episode_length=4 → many episodes finish each rollout
        assert log["ep_return_rolling_n"] > 0
        assert np.isfinite(log["ep_return_rolling"])
        assert np.isfinite(log["ep_length_rolling"])


def test_train_rolling_n_caps_at_window_size():
    """ep_return_rolling_n must never exceed the configured window."""
    env = MockMazeEnv(num=4, episode_length=2, seed=0)  # very short → many eps
    agent = ImpalaAgent()
    # T*N = 64 → many short eps per rollout; 3 updates → plenty to overflow K=8
    logs = train(env, agent, PPOConfig(),
                 num_steps=16, total_env_steps=192, device="cpu",
                 rolling_window=8)
    assert all(log["ep_return_rolling_n"] <= 8 for log in logs)
    # Must reach the cap given the short episode_length
    assert any(log["ep_return_rolling_n"] == 8 for log in logs), \
        "rolling window should fill to capacity"


def test_train_rolling_persists_across_empty_rollouts():
    """If a rollout completes 0 episodes, rolling must still report the
    *previous* window value — that is the entire point of this patch."""
    # Long episode_length relative to T → some rollouts will complete 0 eps.
    env = MockMazeEnv(num=2, episode_length=20, seed=0)
    agent = ImpalaAgent()
    # T*N = 32; total = 96 → 3 updates. episode_length=20 over 16 step rollout
    # means the 1st rollout completes 0 eps in some envs, mixed pattern.
    logs = train(env, agent, PPOConfig(),
                 num_steps=16, total_env_steps=96, device="cpu",
                 rolling_window=50)
    # Once any update has rolling_n > 0, all subsequent ones must also have it.
    saw_nonzero = False
    for log in logs:
        if log["ep_return_rolling_n"] > 0:
            saw_nonzero = True
            assert np.isfinite(log["ep_return_rolling"])
        elif saw_nonzero:
            pytest.fail("rolling reset to 0 after previously having entries — "
                        "deque should persist across updates")


def test_train_rejects_nonpositive_rolling_window():
    env = MockMazeEnv(num=4, seed=0)
    agent = ImpalaAgent()
    with pytest.raises(ValueError, match="rolling_window"):
        train(env, agent, PPOConfig(),
              num_steps=16, total_env_steps=64, device="cpu",
              rolling_window=0)


def test_train_rolling_mean_matches_manual_computation():
    """End-to-end sanity: the rolling mean stored in the *last* log should
    equal the arithmetic mean over the actual completed-episode returns
    captured during the run."""
    env = MockMazeEnv(num=4, episode_length=4, seed=0)
    agent = ImpalaAgent()
    captured: list[float] = []

    def cb(idx, log):
        # collect this rollout's completed eps from the rollout-only key
        # *and* re-derive the rolling mean by ourselves
        pass  # we'll just inspect the final log

    logs = train(env, agent, PPOConfig(),
                 num_steps=16, total_env_steps=128, device="cpu",
                 log_callback=cb, rolling_window=1000)
    # rolling_window=1000 ≥ all completed eps → rolling mean ==
    # arithmetic mean of the per-rollout means weighted by ep_count.
    # Reconstruct it:
    total_ret, total_n = 0.0, 0
    for log in logs:
        if log["ep_count"] > 0:
            total_ret += log["ep_return_mean"] * log["ep_count"]
            total_n += log["ep_count"]
    expected = total_ret / total_n
    assert abs(logs[-1]["ep_return_rolling"] - expected) < 1e-9
    assert logs[-1]["ep_return_rolling_n"] == total_n
