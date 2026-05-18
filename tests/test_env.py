"""Tests for src/split_maze/env.py.

Unit tests synthesize rgb frames to verify sprite detection and trajectory
tracking without needing procgen installed. An integration test against the
real procgen `maze` env is included but skipped if procgen isn't available.
"""

import numpy as np
import pytest

from split_maze.env import (
    CHEESE_COLORS, MOUSE_COLORS, SPRITE_MATCH_TOL,
    TrajectoryTracker, ExtractionResult,
    find_sprite_centroid, extract_maze_state,
)
from split_maze.language import describer_oracle


# ---- find_sprite_centroid ---------------------------------------------

def _make_rgb(size=64, bg=(190, 140, 90)):
    """Return a (size, size, 3) uint8 rgb filled with the background color."""
    rgb = np.full((size, size, 3), bg, dtype=np.uint8)
    return rgb


def test_find_sprite_centroid_known_position():
    rgb = _make_rgb()
    # Paint a 3x3 cheese-yellow blob at (x=20, y=30)
    rgb[28:31, 18:21] = CHEESE_COLORS[1]  # (254, 231, 98)
    pos = find_sprite_centroid(rgb, CHEESE_COLORS)
    assert pos is not None
    x, y = pos
    assert abs(x - 19) < 0.01 and abs(y - 29) < 0.01


def test_find_sprite_centroid_no_match_returns_none():
    rgb = _make_rgb()
    assert find_sprite_centroid(rgb, CHEESE_COLORS) is None


def test_find_sprite_centroid_tolerance():
    rgb = _make_rgb()
    # Paint a pixel slightly off from the exact cheese color (L1 dist = 15)
    rgb[10, 10] = (254 - 5, 231 - 5, 98 - 5)  # within tol=20
    assert find_sprite_centroid(rgb, CHEESE_COLORS, tol=20) is not None
    assert find_sprite_centroid(rgb, CHEESE_COLORS, tol=5) is None


def test_find_sprite_centroid_rejects_bad_shape():
    with pytest.raises(ValueError):
        find_sprite_centroid(np.zeros((5, 5), dtype=np.uint8), CHEESE_COLORS)


# ---- TrajectoryTracker ------------------------------------------------

def test_trajectory_tracker_window():
    t = TrajectoryTracker(window=3)
    for xy in [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)]:
        t.update(xy)
    traj = t.trajectory()
    assert len(traj) == 3
    assert traj == ((2.0, 2.0), (3.0, 3.0), (4.0, 4.0))


def test_trajectory_tracker_reset():
    t = TrajectoryTracker(window=3)
    t.update((1, 2))
    t.update((3, 4))
    t.reset()
    assert len(t) == 0
    assert t.trajectory() == ()


def test_trajectory_tracker_rejects_tiny_window():
    with pytest.raises(ValueError):
        TrajectoryTracker(window=1)


# ---- extract_maze_state (synthetic) -----------------------------------

def _paint_sprite(rgb, colors, center, size=3):
    """Paint a `size×size` block of `colors[0]` centered at (x,y)."""
    cx, cy = center
    rgb[cy - size//2 : cy - size//2 + size,
        cx - size//2 : cx - size//2 + size] = colors[0]


def test_extract_maze_state_both_sprites_present():
    rgb = _make_rgb()
    _paint_sprite(rgb, MOUSE_COLORS, center=(20, 30))      # agent at (20, 30)
    _paint_sprite(rgb, CHEESE_COLORS, center=(50, 10))     # cheese at (50, 10)
    tracker = TrajectoryTracker()
    res = extract_maze_state(rgb, tracker)
    assert res.maze_state is not None
    assert res.agent_pixel_count > 0
    assert res.cheese_pixel_count > 0
    state = res.maze_state
    assert abs(state.agent_xy[0] - 20) < 1
    assert abs(state.agent_xy[1] - 30) < 1
    assert abs(state.cheese_xy[0] - 50) < 1
    assert abs(state.cheese_xy[1] - 10) < 1
    assert state.maze_size == (64.0, 64.0)
    assert len(tracker) == 1


def test_extract_maze_state_missing_cheese_returns_none():
    rgb = _make_rgb()
    _paint_sprite(rgb, MOUSE_COLORS, center=(20, 30))
    # No cheese painted
    tracker = TrajectoryTracker()
    res = extract_maze_state(rgb, tracker)
    assert res.maze_state is None
    assert res.cheese_pixel_count == 0
    assert res.agent_pixel_count > 0
    assert len(tracker) == 0  # tracker not updated when state invalid


def test_extract_maze_state_updates_tracker_over_steps():
    """A multi-step rollout updates the trajectory; describer oracle
    should compute a sensible HEADING from it."""
    tracker = TrajectoryTracker(window=4)
    # Simulate agent moving from (10,10) → (15,10) → (20,10) → (25,10) (rightward)
    for ax in (10, 15, 20, 25):
        rgb = _make_rgb()
        _paint_sprite(rgb, MOUSE_COLORS, center=(ax, 10))
        _paint_sprite(rgb, CHEESE_COLORS, center=(50, 50))
        res = extract_maze_state(rgb, tracker)
        assert res.maze_state is not None
    assert len(tracker) == 4
    # Final state's describer oracle: agent moving right, cheese down-right
    slots = describer_oracle(res.maze_state)
    assert slots is not None
    assert slots.heading == "right"
    # Agent at (25,10) in 64×64: x/64 = 0.39 → col_idx=1 → "center";
    # y/64 = 0.156 → row_idx=0 → "top"
    assert slots.agent_region == ("top", "center")
    # Cheese (50,50) is down-right of agent (25,10): dx=+25, dy=+40 → down-right
    assert slots.cheese_dir == "down-right"


# ---- Integration with real procgen (sandbox / WSL) --------------------

procgen = pytest.importorskip("procgen", reason="procgen not installed")


def test_real_procgen_maze_detects_both_sprites():
    """Smoke test on actual procgen `maze` env (works in sandbox + WSL)."""
    from procgen import ProcgenGym3Env
    env = ProcgenGym3Env(num=1, env_name="maze", num_levels=1, start_level=0,
                         distribution_mode="easy", use_backgrounds=False)
    _, obs, _ = env.observe()
    rgb = obs["rgb"][0]
    env.close()

    tracker = TrajectoryTracker()
    res = extract_maze_state(rgb, tracker)
    assert res.maze_state is not None, (
        f"sprite detection failed: agent_n={res.agent_pixel_count}, "
        f"cheese_n={res.cheese_pixel_count}"
    )
    assert res.agent_pixel_count >= 3
    assert res.cheese_pixel_count >= 3
    # Sanity-check coordinates are within frame
    s = res.maze_state
    assert 0 <= s.agent_xy[0] <= 64 and 0 <= s.agent_xy[1] <= 64
    assert 0 <= s.cheese_xy[0] <= 64 and 0 <= s.cheese_xy[1] <= 64
    # describer oracle should produce valid Slots (or None if agent on cheese)
    slots = describer_oracle(s, move_threshold=0.0)
    # With only 1 trajectory point, HEADING will be 'still' but slots is still valid
    assert slots is not None
    assert slots.heading == "still"  # first frame, no displacement yet
