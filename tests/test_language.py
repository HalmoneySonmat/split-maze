"""Tests for src/split_maze/language.py."""

import random
from collections import Counter

import pytest

from split_maze.language import (
    BOS, EOS, PAD, SUM,
    DIRECTIONS_8, HEADING_VALUES, CHEESE_DIR_VALUES,
    REGION_ROWS, REGION_COLS,
    MazeState, Slots,
    vocab,
    quantize_to_3x3, quantize_8way,
    describer_oracle,
    render, parse,
    sample_slots, generate_corpus,
)


# ---- Vocabulary --------------------------------------------------------

def test_vocab_size_and_uniqueness():
    v = vocab()
    assert len(v) == 25, f"expected 25 unique tokens, got {len(v)}: {v}"
    assert len(set(v)) == len(v)


def test_vocab_contains_specials():
    v = set(vocab())
    for t in (BOS, EOS, PAD, SUM):
        assert t in v


def test_directions_8_count():
    assert len(DIRECTIONS_8) == 8
    assert len(set(DIRECTIONS_8)) == 8


def test_heading_has_still():
    assert "still" in HEADING_VALUES
    assert len(HEADING_VALUES) == 9


def test_cheese_dir_no_still():
    assert "still" not in CHEESE_DIR_VALUES
    assert len(CHEESE_DIR_VALUES) == 8


# ---- quantize_8way ----------------------------------------------------

@pytest.mark.parametrize("dx,dy,expected", [
    (1, 0, "right"),
    (0, -1, "up"),
    (1, -1, "up-right"),
    (-1, 0, "left"),
    (0, 1, "down"),
    (-1, 1, "down-left"),
    (1, 1, "down-right"),
    (-1, -1, "up-left"),
])
def test_quantize_8way_cardinals(dx, dy, expected):
    assert quantize_8way(dx, dy) == expected


def test_quantize_8way_zero_returns_none():
    assert quantize_8way(0, 0) is None


def test_quantize_8way_threshold():
    assert quantize_8way(0.1, 0.1, move_threshold=0.5) is None
    assert quantize_8way(1, 0, move_threshold=0.5) == "right"


# ---- quantize_to_3x3 --------------------------------------------------

def test_quantize_to_3x3_corners():
    W, H = 90.0, 90.0
    assert quantize_to_3x3((0, 0), (W, H)) == ("top", "left")
    assert quantize_to_3x3((89, 0), (W, H)) == ("top", "right")
    assert quantize_to_3x3((0, 89), (W, H)) == ("bottom", "left")
    assert quantize_to_3x3((89, 89), (W, H)) == ("bottom", "right")
    assert quantize_to_3x3((45, 45), (W, H)) == ("middle", "center")


def test_quantize_to_3x3_clamps_out_of_range():
    assert quantize_to_3x3((-10, -10), (90, 90)) == ("top", "left")
    assert quantize_to_3x3((9999, 9999), (90, 90)) == ("bottom", "right")


# ---- describer_oracle -------------------------------------------------

def test_describer_oracle_basic():
    """Agent at top-right moving up-right, cheese at bottom-left."""
    state = MazeState(
        agent_xy=(80, 10),
        cheese_xy=(10, 80),
        maze_size=(90, 90),
        recent_trajectory=((70, 20), (75, 15), (78, 12), (80, 10)),
    )
    slots = describer_oracle(state)
    assert slots is not None
    assert slots.agent_region == ("top", "right")
    assert slots.heading == "up-right"
    assert slots.cheese_dir == "down-left"


def test_describer_oracle_returns_none_when_on_cheese():
    state = MazeState(
        agent_xy=(50, 50),
        cheese_xy=(50, 50),
        maze_size=(90, 90),
        recent_trajectory=((50, 50), (50, 50)),
    )
    assert describer_oracle(state) is None


def test_describer_oracle_still_heading():
    state = MazeState(
        agent_xy=(50, 50),
        cheese_xy=(10, 10),
        maze_size=(90, 90),
        recent_trajectory=((50, 50), (50, 50), (50, 50), (50, 50)),
    )
    slots = describer_oracle(state, move_threshold=0.5)
    assert slots is not None
    assert slots.heading == "still"
    assert slots.cheese_dir == "up-left"


def test_describer_oracle_is_deterministic():
    state = MazeState(
        agent_xy=(70, 20),
        cheese_xy=(15, 75),
        maze_size=(90, 90),
        recent_trajectory=((60, 30), (65, 25), (70, 20)),
    )
    a = describer_oracle(state)
    b = describer_oracle(state)
    assert a == b


# ---- render + parse roundtrip ----------------------------------------

def test_render_parse_roundtrip_random():
    rng = random.Random(42)
    for _ in range(500):
        slots = sample_slots(rng)
        toks = render(slots, rng=rng)
        parsed = parse(toks)
        assert parsed.agent_region == slots.agent_region
        assert parsed.heading == slots.heading
        assert parsed.cheese_dir == slots.cheese_dir


def test_parse_robust_to_connectives():
    toks = [BOS, "agent", "top", "right", "and", "heading", "up-right",
            ",", "cheese", "down-left", EOS]
    p = parse(toks)
    assert p.agent_region == ("top", "right")
    assert p.heading == "up-right"
    assert p.cheese_dir == "down-left"


def test_parse_missing_slot_returns_none_for_that_slot():
    """If a marker is absent, only that slot becomes None."""
    toks = [BOS, "agent", "top", "right", "cheese", "down-left", EOS]
    p = parse(toks)
    assert p.agent_region == ("top", "right")
    assert p.heading is None
    assert p.cheese_dir == "down-left"


def test_parse_invalid_value_rejects_slot():
    toks = [BOS, "heading", "banana", "cheese", "up", EOS]
    p = parse(toks)
    assert p.heading is None
    assert p.cheese_dir == "up"


def test_slot_match_rate():
    gold = Slots("top", "right", "up-right", "down-left")
    # All match
    p_full = parse(render(gold, rng=random.Random(0)))
    assert p_full.slot_match_rate(gold) == 1.0
    # All wrong
    p_none = parse([BOS, EOS])
    assert p_none.slot_match_rate(gold) == 0.0
    # Partial — only heading matches
    toks = [BOS, "heading", "up-right", EOS]
    p_partial = parse(toks)
    assert abs(p_partial.slot_match_rate(gold) - 1/3) < 1e-9


# ---- Surface diversity (memorization guard — LANGUAGE_SPEC §6) -------

def test_surface_diversity_per_triple():
    """Same Slots should produce many distinct surface forms."""
    slots = Slots("top", "right", "up-right", "down-left")
    rng = random.Random(0)
    forms = {tuple(render(slots, rng=rng, include_bos_eos=False))
             for _ in range(200)}
    assert len(forms) >= 20, f"too few surface forms: {len(forms)}"


# ---- Corpus uniformity (neutrality — LANGUAGE_SPEC §7) ---------------

def test_corpus_uniformity_per_slot():
    """Slot value frequencies should be close to uniform."""
    n = 10_000
    heading_counts: Counter = Counter()
    cheese_counts: Counter = Counter()
    region_counts: Counter = Counter()
    rng = random.Random(0)
    for _ in range(n):
        s = sample_slots(rng)
        heading_counts[s.heading] += 1
        cheese_counts[s.cheese_dir] += 1
        region_counts[s.agent_region] += 1

    for v in HEADING_VALUES:
        assert abs(heading_counts[v] - n / 9) < n / 9 * 0.20
    for v in CHEESE_DIR_VALUES:
        assert abs(cheese_counts[v] - n / 8) < n / 8 * 0.20
    for r in REGION_ROWS:
        for c in REGION_COLS:
            assert abs(region_counts[(r, c)] - n / 9) < n / 9 * 0.20


def test_corpus_no_correlation_heading_cheese():
    """Joint (HEADING, CHEESE_DIR) distribution should match the product of
    marginals — neutrality property that protects against the §3.3
    rationalization prior."""
    n = 20_000
    pair_counts: Counter = Counter()
    rng = random.Random(1)
    for _ in range(n):
        s = sample_slots(rng)
        pair_counts[(s.heading, s.cheese_dir)] += 1
    expected = n / 72  # 9 × 8 pairs
    for h in HEADING_VALUES:
        for c in CHEESE_DIR_VALUES:
            assert abs(pair_counts[(h, c)] - expected) < expected * 0.30


# ---- generate_corpus integration -------------------------------------

def test_generate_corpus_count_and_format():
    sents = list(generate_corpus(50, seed=0))
    assert len(sents) == 50
    for s in sents:
        assert s[0] == BOS
        assert s[-1] == EOS
        p = parse(s)
        assert p.agent_region is not None
        assert p.heading is not None
        assert p.cheese_dir is not None
