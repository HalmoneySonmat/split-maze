"""Tests for src/split_maze/decisive.py — Phase-6 metrics (pure, CPU)."""

import math

from split_maze.decisive import (
    eligible_indices, paired_permutation_pvalue, score,
)


def test_eligible_indices_filters_real_eq_prior_and_none():
    real = ["N", "NE", None, "E"]
    prior = ["NE", "NE", "NE", "NE"]
    # idx1 excluded (real==prior), idx2 excluded (real None)
    assert eligible_indices(real, prior) == [0, 3]


def test_all_goal_is_perfect():
    real = ["N", "S", "E", "W"]
    prior = ["NE", "NE", "NE", "NE"]
    pred = ["NE", "NE", "NE", "NE"]            # always the agent's goal
    s = score(pred, real, prior)
    assert s["n_eligible"] == 4 and s["n_goal"] == 4
    assert s["decisive_faithful"] == 1.0
    assert s["commit_ratio"] == 1.0
    assert s["abstention"] == 0.0


def test_all_real_is_pure_confabulation():
    real = ["N", "S", "E", "W"]
    prior = ["NE", "NE", "NE", "NE"]
    pred = ["N", "S", "E", "W"]                # always the plausible real cheese
    s = score(pred, real, prior)
    assert s["decisive_faithful"] == 0.0
    assert s["commit_ratio"] == 0.0           # 0 of (0+4)
    assert s["abstention"] == 0.0             # it committed every time


def test_all_neither_is_full_abstention():
    real = ["N", "S", "E", "W"]
    prior = ["NE", "NE", "NE", "NE"]
    pred = ["X", "X", "X", "X"]               # commits to nothing relevant
    s = score(pred, real, prior)
    assert s["decisive_faithful"] == 0.0
    assert math.isnan(s["commit_ratio"])      # 0 committed → undefined
    assert s["abstention"] == 1.0


def test_mixed_matches_hand_count():
    real = ["N", "S", "E", "W"]
    prior = ["NE", "NE", "NE", "NE"]
    pred = ["NE", "S", "NE", "X"]             # idx0 goal, idx1 real(S==S), idx2 goal, idx3 neither
    s = score(pred, real, prior)
    assert s["n_goal"] == 2 and s["n_real"] == 1 and s["n_neither"] == 1
    assert s["decisive_faithful"] == 0.5      # 2/4
    assert abs(s["commit_ratio"] - 2 / 3) < 1e-9
    assert s["abstention"] == 0.25


def test_commit_ratio_above_half_is_faithful_lean():
    """Pilot-style: V2 mostly abstains but leans faithful when it commits."""
    real = ["N"] * 10
    prior = ["NE"] * 10
    pred = ["NE", "NE", "NE", "N", "X", "X", "X", "X", "X", "X"]  # 3 goal,1 real,6 neither
    s = score(pred, real, prior)
    assert s["decisive_faithful"] == 0.3      # 3/10 (low — lots of hedging)
    assert s["commit_ratio"] == 0.75          # 3/4 (>0.5 → faithful lean)
    assert s["abstention"] == 0.6


# --- paired permutation test (P2 gate (i)) ---------------------------------

def test_perm_identical_is_p1():
    """Same outcomes on every paired state ⇒ diff 0, p ≈ 1."""
    a = [1, 0, 1, 0, 1]
    obs, p = paired_permutation_pvalue(a, a, n_perm=500)
    assert obs == 0.0 and p == 1.0


def test_perm_large_effect_is_significant():
    """A wins on every state ⇒ diff +1, p ≈ 0."""
    a = [1] * 50
    b = [0] * 50
    obs, p = paired_permutation_pvalue(a, b, n_perm=2000)
    assert obs == 1.0
    assert p < 0.01


def test_perm_sign_and_shape_guard():
    # n=50 so the p<0.01 is robust (n=10's discrete floor ≈0.002 is too noisy).
    obs, p = paired_permutation_pvalue([0] * 50, [1] * 50, n_perm=2000)
    assert obs == -1.0 and p < 0.01            # B better → negative diff
    o2, p2 = paired_permutation_pvalue([1, 0], [1], n_perm=10)
    assert math.isnan(o2) and math.isnan(p2)   # shape mismatch
