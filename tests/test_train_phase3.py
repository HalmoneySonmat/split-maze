"""Unit tests for split_maze.train_phase3 (Phase 3.4 co-training skeleton).

Uses MockMazeEnv + an injected state_extractor stub so pairs flow without
procgen sprites. Verifies:

1. Phase3Config defaults match 박제값.
2. collect_rollout_with_pairs shapes + alignment.
3. train_phase3 runs end-to-end, PPO metrics + pair collection + interpreter
   updates fire when the buffer is ready.
4. With the default (random-rgb) extractor, no sprites → no pairs → loop
   still runs without crashing (no interpreter updates).
"""

from __future__ import annotations

import random

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from split_maze.acc import ACC, ACCConfig  # noqa: E402
from split_maze.agent import ImpalaAgent  # noqa: E402
from split_maze.builds import B3Probe, B4Adapter, V2ACC  # noqa: E402
from split_maze.env import TrajectoryTracker  # noqa: E402
from split_maze.language import MazeState  # noqa: E402
from split_maze.lm import LMConfig, MazeLM, MazeTokenizer  # noqa: E402
from split_maze.paired_collect import PairBuffer, PairBufferConfig, PairedCollector  # noqa: E402
from split_maze.ppo import PPOConfig, RolloutBuffer  # noqa: E402
from split_maze.train import MockMazeEnv  # noqa: E402
from split_maze.train_phase3 import (  # noqa: E402
    Phase3Config,
    collect_rollout_with_pairs,
    default_state_extractor,
    train_phase3,
)


D_A = 256  # ImpalaAgent default embedding width.


def _stub_extractor(rgb, tracker):
    """Always return a valid MazeState (so describer_oracle yields Slots)."""
    return MazeState(
        agent_xy=(0.2, 0.2),
        cheese_xy=(0.9, 0.1),
        maze_size=(1.0, 1.0),
        recent_trajectory=((0.0, 0.0), (0.1, 0.1), (0.2, 0.2)),
    )


def _small_lm(tok):
    cfg = LMConfig.from_tokenizer(
        tok, d_model=32, n_head=4, n_layer=2, d_ff=64, max_len=32, dropout=0.0
    )
    return MazeLM(cfg)


def _make_builds(tok):
    b3 = B3Probe(tok, d_agent=D_A)
    b4 = B4Adapter(_small_lm(tok), d_agent=D_A)
    v2 = V2ACC(_small_lm(tok), ACC(ACCConfig(d_agent=D_A, d_lm=32)))
    return {"B3": b3, "B4": b4, "V2": v2}


def _smoke_setup(num=2, T=4):
    env = MockMazeEnv(num=num, episode_length=4, seed=0)
    agent = ImpalaAgent()
    tok = MazeTokenizer()
    builds = _make_builds(tok)
    collector = PairedCollector(tok, stride=2, max_token_len=16)
    buffer = PairBuffer(
        PairBufferConfig(capacity=1000, batch_size=4, d_agent=D_A, max_token_len=16),
        pad_id=tok.pad_id,
    )
    cfg = Phase3Config(
        num_steps=T,
        total_env_steps=T * num * 2,    # ~2 updates
        acc_updates_per_rl=2,
        interp_batch=4,
        stride=2,
        buffer_capacity=1000,
    )
    return env, agent, builds, collector, buffer, cfg


# ---- 1. Config ----------------------------------------------------------


class TestPhase3Config:
    def test_defaults(self):
        c = Phase3Config()
        assert c.acc_updates_per_rl == 32     # P3-2-4
        assert c.interp_batch == 128          # P3-3
        assert c.interp_lr == 3e-4
        assert c.interp_warmup == 500         # POST-HOC-4 계승
        assert c.stride == 4                  # P3-4
        assert c.buffer_capacity == 256_000
        assert c.heading_window == 4


# ---- 2. augmented rollout ----------------------------------------------


class TestCollectRolloutWithPairs:
    def test_shapes(self):
        env = MockMazeEnv(num=2, episode_length=4, seed=0)
        agent = ImpalaAgent()
        rb = RolloutBuffer(T=4, N=2, device="cpu")
        trackers = [TrajectoryTracker(4) for _ in range(2)]
        from split_maze.train import obs_to_tensor

        _r, obs_dict, _f = env.observe()
        obs_holder = obs_to_tensor(obs_dict, "cpu")
        cur_rgb = np.asarray(obs_dict["rgb"])
        ep_r = np.zeros(2)
        ep_l = np.zeros(2, dtype=np.int64)

        stats, obs2, rgb2, h_agent, maze_states = collect_rollout_with_pairs(
            env, agent, rb, trackers,
            obs_holder=obs_holder, cur_rgb=cur_rgb,
            episode_returns=ep_r, episode_lengths=ep_l,
            state_extractor=_stub_extractor, d_agent=D_A, device="cpu",
        )
        assert h_agent.shape == (4, 2, D_A)
        assert len(maze_states) == 4 and len(maze_states[0]) == 2
        # stub always returns a MazeState.
        assert all(maze_states[t][n] is not None for t in range(4) for n in range(2))
        assert rgb2.shape == (2, 64, 64, 3)


# ---- 3. full co-train loop ---------------------------------------------


class TestTrainPhase3:
    def test_runs_and_returns_logs(self):
        env, agent, builds, collector, buffer, cfg = _smoke_setup()
        logs = train_phase3(
            env, agent, builds, collector, buffer,
            config=cfg, ppo_config=PPOConfig(), device="cpu",
            state_extractor=_stub_extractor,
            surface_rng=random.Random(0),
        )
        assert len(logs) >= 1
        # PPO metrics present.
        assert "policy" in logs[0] and "value" in logs[0]
        assert "pairs_added" in logs[0] and "buffer_size" in logs[0]

    def test_pairs_collected(self):
        env, agent, builds, collector, buffer, cfg = _smoke_setup()
        train_phase3(
            env, agent, builds, collector, buffer,
            config=cfg, state_extractor=_stub_extractor,
            surface_rng=random.Random(0),
        )
        # T//stride * N = 4//2 * 2 = 4 pairs per rollout → buffer non-empty.
        assert len(buffer) > 0

    def test_interpreter_updates_fire(self):
        env, agent, builds, collector, buffer, cfg = _smoke_setup()
        logs = train_phase3(
            env, agent, builds, collector, buffer,
            config=cfg, state_extractor=_stub_extractor,
            surface_rng=random.Random(0),
        )
        # Once the buffer reaches interp_batch=4 (after update 0's extract),
        # each build logs a loss.
        fired = [lg for lg in logs if "B3/loss" in lg]
        assert fired, "expected at least one update with interpreter losses"
        lg = fired[0]
        for name in ("B3", "B4", "V2"):
            assert f"{name}/loss" in lg
            assert np.isfinite(lg[f"{name}/loss"])
            assert lg[f"{name}/n_updates"] >= cfg.acc_updates_per_rl

    def test_default_extractor_runs_end_to_end(self):
        """The real default_state_extractor (= env.extract_maze_state) wires
        through the loop without crashing.

        NB: MockEnv rgb is uniform *noise*, not a real maze frame. The sprite
        color-match (L1 ≤ tol) can *false-positive* on noise and return a
        valid MazeState → spurious pairs may be collected. That's a MockEnv
        artifact (real procgen frames have real sprites); this test only
        asserts the wiring runs, not the pair count."""
        env, agent, builds, collector, buffer, cfg = _smoke_setup()
        logs = train_phase3(
            env, agent, builds, collector, buffer,
            config=cfg, state_extractor=default_state_extractor,
            surface_rng=random.Random(0),
        )
        assert len(logs) >= 1
        assert "policy" in logs[0]

    def test_none_extractor_yields_no_pairs(self):
        """When the extractor returns None for every step, no pairs are
        collected and no interpreter updates fire (deterministic empty-buffer
        path)."""
        env, agent, builds, collector, buffer, cfg = _smoke_setup()
        logs = train_phase3(
            env, agent, builds, collector, buffer,
            config=cfg, state_extractor=lambda rgb, tracker: None,
            surface_rng=random.Random(0),
        )
        assert len(logs) >= 1
        assert len(buffer) == 0
        assert all("B3/loss" not in lg for lg in logs)


# ---- 4. default extractor wrapper --------------------------------------


class TestCLIHelpers:
    """scripts/train_phase3.py is loaded by file path (avoids the name clash
    with the library module split_maze.train_phase3)."""

    @staticmethod
    def _load_script():
        import importlib.util
        from pathlib import Path

        path = (Path(__file__).resolve().parents[1]
                / "scripts" / "train_phase3.py")
        spec = importlib.util.spec_from_file_location("train_phase3_cli", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def test_build_interpreters_fresh_lm(self):
        mod = self._load_script()
        tok = MazeTokenizer()
        builds = mod.build_interpreters(
            ["B3", "B4", "V2"], tok, None,
            d_agent=D_A, device=torch.device("cpu"),
        )
        assert set(builds) == {"B3", "B4", "V2"}
        assert isinstance(builds["B3"], B3Probe)
        assert isinstance(builds["B4"], B4Adapter)
        assert isinstance(builds["V2"], V2ACC)

    def test_build_interpreters_subset(self):
        mod = self._load_script()
        tok = MazeTokenizer()
        builds = mod.build_interpreters(
            ["B3"], tok, None, d_agent=D_A, device=torch.device("cpu"),
        )
        assert set(builds) == {"B3"}


class TestDefaultExtractor:
    def test_returns_none_or_mazestate_on_noise(self):
        # FINDING (2026-05-21): random rgb is NOT guaranteed sprite-free —
        # the L1≤tol color match can false-positive on uniform noise and
        # yield a valid MazeState. So we only assert the return *type*
        # contract (None or MazeState), not that noise → None.
        rng = np.random.RandomState(0)
        rgb = rng.randint(0, 256, size=(64, 64, 3), dtype=np.uint8)
        tracker = TrajectoryTracker(4)
        result = default_state_extractor(rgb, tracker)
        assert result is None or isinstance(result, MazeState)
