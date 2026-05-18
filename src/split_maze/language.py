"""Synthetic maze-language + describer oracle.

Implements PLAN §3.3 (② 3슬롯 최소) — vocabulary, grammar, describer oracle,
neutral corpus generator, parser. See docs/LANGUAGE_SPEC.md for the spec.

3 slots, all independently parsed:
- AGENT_REGION = (row, col) where row in {top, middle, bottom}, col in {left, center, right}
- HEADING in 8 compass directions + 'still'
- CHEESE_DIR in 8 compass directions (excludes the agent-on-cheese state)
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Iterator, Optional

# ---- Vocabulary ---------------------------------------------------------

DIRECTIONS_8: tuple[str, ...] = (
    "up", "up-right", "right", "down-right",
    "down", "down-left", "left", "up-left",
)

HEADING_VALUES: tuple[str, ...] = DIRECTIONS_8 + ("still",)
CHEESE_DIR_VALUES: tuple[str, ...] = DIRECTIONS_8

REGION_ROWS: tuple[str, ...] = ("top", "middle", "bottom")
REGION_COLS: tuple[str, ...] = ("left", "center", "right")

# Slot markers (synonyms drive surface variation; spec §6).
MARKER_AGENT: tuple[str, ...] = ("agent", "it")
MARKER_HEADING: tuple[str, ...] = ("heading", "going", "moving")
MARKER_CHEESE: tuple[str, ...] = ("cheese",)

# Optional connectives between slot phrases. Empty string = no connective.
CONNECTIVES: tuple[str, ...] = ("", "and", ",")

# Special tokens.
BOS = "<BOS>"
EOS = "<EOS>"
PAD = "<PAD>"
SUM = "<SUM>"  # handle-B slot — placement decided in Phase 2 (§3.4)

SPECIAL_TOKENS: tuple[str, ...] = (BOS, EOS, PAD, SUM)


def vocab() -> list[str]:
    """Return the sorted, deduplicated full token vocabulary."""
    toks: set[str] = set()
    toks.update(DIRECTIONS_8)
    toks.add("still")
    toks.update(REGION_ROWS)
    toks.update(REGION_COLS)  # 'left' / 'right' overlap with DIRECTIONS_8 — fine
    toks.update(MARKER_AGENT)
    toks.update(MARKER_HEADING)
    toks.update(MARKER_CHEESE)
    toks.update(c for c in CONNECTIVES if c)  # exclude empty
    toks.update(SPECIAL_TOKENS)
    return sorted(toks)


# ---- Slot dataclasses --------------------------------------------------

@dataclass(frozen=True)
class Slots:
    """Canonical 3-slot representation (oracle / sample output).

    AGENT_REGION is conceptually one slot (row+col together).
    """
    agent_row: str
    agent_col: str
    heading: str
    cheese_dir: str

    @property
    def agent_region(self) -> tuple[str, str]:
        return (self.agent_row, self.agent_col)


@dataclass(frozen=True)
class ParsedSlots:
    """Parser output. Any slot that couldn't be extracted is None — yielding
    the natural penalty in slot-match scoring (LANGUAGE_SPEC §8)."""
    agent_region: Optional[tuple[str, str]]
    heading: Optional[str]
    cheese_dir: Optional[str]

    def matches_per_slot(self, gold: Slots) -> tuple[bool, bool, bool]:
        return (
            self.agent_region == gold.agent_region,
            self.heading == gold.heading,
            self.cheese_dir == gold.cheese_dir,
        )

    def slot_match_rate(self, gold: Slots) -> float:
        a, h, c = self.matches_per_slot(gold)
        return (int(a) + int(h) + int(c)) / 3.0


# ---- Maze state + describer oracle -------------------------------------

@dataclass(frozen=True)
class MazeState:
    """Input to describer_oracle. Image convention: y=0 is top.

    - agent_xy, cheese_xy: positions in maze coordinates.
    - maze_size: (W, H) extent.
    - recent_trajectory: last K (x, y) positions. HEADING = direction of net
      displacement between first and last entries (LANGUAGE_SPEC §5, K=4 default).
    """
    agent_xy: tuple[float, float]
    cheese_xy: tuple[float, float]
    maze_size: tuple[float, float]
    recent_trajectory: tuple[tuple[float, float], ...]


def quantize_to_3x3(xy: tuple[float, float],
                    maze_size: tuple[float, float]) -> tuple[str, str]:
    """Quantize (x, y) into the 3×3 grid: (row, col). y=0 is top."""
    x, y = xy
    W, H = maze_size
    col_idx = min(2, max(0, int(x / W * 3)))
    row_idx = min(2, max(0, int(y / H * 3)))
    return REGION_ROWS[row_idx], REGION_COLS[col_idx]


# Sector index → compass name. Sector i covers angles in [(i-0.5)·π/4, (i+0.5)·π/4).
# atan2(-dy, dx) puts 'right' at 0, 'up' at π/2 (image y inverted).
_SECTOR_TO_NAME: tuple[str, ...] = (
    "right", "up-right", "up", "up-left",
    "left", "down-left", "down", "down-right",
)


def quantize_8way(dx: float, dy: float,
                  move_threshold: float = 0.0) -> Optional[str]:
    """Quantize a 2D vector to an 8-way compass direction.

    y=0 is top, so 'up' corresponds to dy < 0. Returns None if magnitude
    ≤ move_threshold (the 'still' / undefined case).
    """
    mag = math.hypot(dx, dy)
    if mag <= move_threshold:
        return None
    angle = math.atan2(-dy, dx)
    sector = int(round(angle / (math.pi / 4))) % 8
    return _SECTOR_TO_NAME[sector]


def describer_oracle(state: MazeState,
                     move_threshold: float = 0.5) -> Optional[Slots]:
    """Deterministic narrator of *objective observable facts* (PLAN §3.3).

    Returns canonical Slots, or None if the agent is on the cheese
    (CHEESE_DIR undefined — those states are excluded per LANGUAGE_SPEC §5).
    Never narrates the agent's intent — that's what ACC has to recover.
    """
    agent_row, agent_col = quantize_to_3x3(state.agent_xy, state.maze_size)

    # HEADING from net displacement of recent trajectory.
    if len(state.recent_trajectory) >= 2:
        x0, y0 = state.recent_trajectory[0]
        x1, y1 = state.recent_trajectory[-1]
        dx, dy = x1 - x0, y1 - y0
    else:
        dx, dy = 0.0, 0.0
    heading_dir = quantize_8way(dx, dy, move_threshold=move_threshold)
    heading = heading_dir if heading_dir is not None else "still"

    # CHEESE_DIR: cheese relative to agent.
    cx, cy = state.cheese_xy
    ax, ay = state.agent_xy
    cheese_dir = quantize_8way(cx - ax, cy - ay)
    if cheese_dir is None:
        return None  # agent on cheese — excluded

    return Slots(agent_row, agent_col, heading, cheese_dir)


# ---- Render: Slots → token sequence ------------------------------------

def render(slots: Slots, *, rng: Optional[random.Random] = None,
           include_bos_eos: bool = True) -> list[str]:
    """Render Slots to a token sequence with surface variation (LANGUAGE_SPEC §6).

    Sources of variation: slot-order permutation (3! = 6), marker synonym
    choice, optional connective insertion. Internal content is deterministic
    given Slots, only the surface form is randomized.
    """
    if rng is None:
        rng = random.Random()

    agent_phrase = [rng.choice(MARKER_AGENT), slots.agent_row, slots.agent_col]
    heading_phrase = [rng.choice(MARKER_HEADING), slots.heading]
    cheese_phrase = [rng.choice(MARKER_CHEESE), slots.cheese_dir]

    phrases = [agent_phrase, heading_phrase, cheese_phrase]
    rng.shuffle(phrases)

    out: list[str] = []
    for i, phrase in enumerate(phrases):
        if i > 0:
            conn = rng.choice(CONNECTIVES)
            if conn:
                out.append(conn)
        out.extend(phrase)

    if include_bos_eos:
        out = [BOS] + out + [EOS]
    return out


# ---- Parse: token sequence → ParsedSlots --------------------------------

_FILTER_OUT: frozenset[str] = frozenset({BOS, EOS, PAD, SUM, "and", ","})


def parse(tokens: list[str]) -> ParsedSlots:
    """Parse tokens → ParsedSlots, robust to slot order, synonyms, connectives.

    Each slot is extracted independently. Invalid or missing values yield
    None for that slot (which scores 0 in slot-match — LANGUAGE_SPEC §8).
    """
    content = [t for t in tokens if t not in _FILTER_OUT]

    agent_region: Optional[tuple[str, str]] = None
    heading: Optional[str] = None
    cheese_dir: Optional[str] = None

    for i, tok in enumerate(content):
        if tok in MARKER_AGENT and agent_region is None:
            if i + 2 < len(content):
                row, col = content[i + 1], content[i + 2]
                if row in REGION_ROWS and col in REGION_COLS:
                    agent_region = (row, col)
        elif tok in MARKER_HEADING and heading is None:
            if i + 1 < len(content):
                v = content[i + 1]
                if v in HEADING_VALUES:
                    heading = v
        elif tok in MARKER_CHEESE and cheese_dir is None:
            if i + 1 < len(content):
                v = content[i + 1]
                if v in CHEESE_DIR_VALUES:
                    cheese_dir = v

    return ParsedSlots(agent_region, heading, cheese_dir)


# ---- Neutral corpus generation -----------------------------------------

def sample_slots(rng: random.Random) -> Slots:
    """Sample Slots uniformly from the grammar's full state space (648 triples)."""
    return Slots(
        agent_row=rng.choice(REGION_ROWS),
        agent_col=rng.choice(REGION_COLS),
        heading=rng.choice(HEADING_VALUES),
        cheese_dir=rng.choice(CHEESE_DIR_VALUES),
    )


def generate_corpus(n: int, seed: int = 0,
                    include_bos_eos: bool = True) -> Iterator[list[str]]:
    """Yield N grammar-sampled sentences (no maze involved).

    Triples are uniform → joint (HEADING, CHEESE_DIR) distribution has no
    correlation. The LM trained on this becomes a *neutral language substrate*;
    the goal-misgen bias lives only in the agent (PLAN §3.3).
    """
    rng = random.Random(seed)
    for _ in range(n):
        slots = sample_slots(rng)
        yield render(slots, rng=rng, include_bos_eos=include_bos_eos)
