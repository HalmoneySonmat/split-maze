"""procgenAISC maze env wrapper + describer-oracle state extraction.

Provides:
- find_sprite_centroid: L1-tolerance color matching on rgb frames.
- TrajectoryTracker: rolling window of recent agent positions (HEADING window).
- extract_maze_state: rgb → MazeState (agent_xy, cheese_xy, maze_size, trajectory).
- make_maze_env: ProcgenGym3Env constructor with SPLIT-MAZE defaults.

Sprite colors hard-coded from procgenAISC's PNG assets
(`procgen/data/assets/misc_assets/cheese.png` and
`procgen/data/assets/kenney/Enemies/mouse_move.png`). Color-distance
matching (L1 ≤ tol) is robust against the 64×64 downsampling — verified
empirically in sandbox on `maze` env (seeds 0/5/13 all detect both
sprites at distinguishable locations).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .language import MazeState

# ---- Sprite colors (extracted from procgenAISC PNG assets) ------------

CHEESE_COLORS: np.ndarray = np.array([
    [253, 155,  37],   # orange outline (most pixels)
    [254, 231,  98],   # bright yellow body (uniquely cheese)
    [181, 112,  82],   # brown shadow
], dtype=np.int16)

MOUSE_COLORS: np.ndarray = np.array([
    [187, 203, 204],   # blue-gray body (dominant)
    [243, 185, 203],   # pink (ears/nose)
    [100, 122, 123],   # dark gray (details)
], dtype=np.int16)

SPRITE_MATCH_TOL: int = 20  # L1 distance tolerance; floor/wall are ≥75 away
DEFAULT_HEADING_WINDOW: int = 4  # K-step net displacement for HEADING


# ---- Sprite detection ------------------------------------------------

def find_sprite_centroid(rgb: np.ndarray,
                         target_colors: np.ndarray,
                         tol: int = SPRITE_MATCH_TOL,
                         ) -> Optional[tuple[float, float]]:
    """Return (x, y) centroid of pixels within L1 ≤ tol of any target color.

    Args:
        rgb: (H, W, 3) uint8 array (procgen single-frame observation).
        target_colors: (K, 3) int16 array of sprite reference colors.
        tol: max per-pixel L1 distance (sum of |R−r|+|G−g|+|B−b|).

    Returns None if no pixels matched.
    """
    if rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError(f"expected (H,W,3) rgb, got shape {rgb.shape}")
    H, W, _ = rgb.shape
    pix = rgb.reshape(-1, 3).astype(np.int16)
    mask = np.zeros(len(pix), dtype=bool)
    for tc in target_colors:
        mask |= (np.abs(pix - tc).sum(axis=1) <= tol)
    if not mask.any():
        return None
    coords = np.argwhere(mask.reshape(H, W))  # (n, 2) → (y, x)
    return float(coords[:, 1].mean()), float(coords[:, 0].mean())


# ---- Trajectory tracker ----------------------------------------------

class TrajectoryTracker:
    """Maintains a rolling window of the most recent K agent positions.

    HEADING in the describer oracle is computed from the net displacement
    between the oldest and newest position in this window (LANGUAGE_SPEC §5).
    """

    def __init__(self, window: int = DEFAULT_HEADING_WINDOW):
        if window < 2:
            raise ValueError("window must be ≥ 2 to define a displacement")
        self.window = window
        self.history: deque[tuple[float, float]] = deque(maxlen=window)

    def update(self, xy: tuple[float, float]) -> None:
        self.history.append((float(xy[0]), float(xy[1])))

    def trajectory(self) -> tuple[tuple[float, float], ...]:
        return tuple(self.history)

    def reset(self) -> None:
        self.history.clear()

    def __len__(self) -> int:
        return len(self.history)


# ---- Combined extractor ----------------------------------------------

@dataclass(frozen=True)
class ExtractionResult:
    """Output of one extraction attempt on a single rgb frame."""
    maze_state: Optional[MazeState]    # None if a sprite is missing
    agent_pixel_count: int             # pixels matched as agent
    cheese_pixel_count: int            # pixels matched as cheese


def extract_maze_state(rgb: np.ndarray,
                       tracker: TrajectoryTracker,
                       maze_size: Optional[tuple[float, float]] = None,
                       tol: int = SPRITE_MATCH_TOL,
                       ) -> ExtractionResult:
    """Build a MazeState from a procgen rgb frame + trajectory tracker.

    Args:
        rgb: (H, W, 3) uint8 — a single procgen observation frame.
        tracker: TrajectoryTracker (will be updated in-place with the new
            agent position if detection succeeds).
        maze_size: (W, H) for the describer oracle's quantization;
            defaults to the rgb dimensions (which is the full-maze view
            for a `distribution_mode='easy'` maze that fits in 64×64).
        tol: sprite-match L1 tolerance.

    Returns:
        ExtractionResult. maze_state is None if either sprite is missing
        (e.g. agent off-screen with center_agent=True, or cheese reached).
    """
    if rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError(f"expected (H,W,3) rgb, got shape {rgb.shape}")
    H, W, _ = rgb.shape
    if maze_size is None:
        maze_size = (float(W), float(H))

    # Count matched pixels and locate centroids in one pass each
    pix = rgb.reshape(-1, 3).astype(np.int16)

    cheese_mask = np.zeros(len(pix), dtype=bool)
    for tc in CHEESE_COLORS:
        cheese_mask |= (np.abs(pix - tc).sum(axis=1) <= tol)
    cheese_n = int(cheese_mask.sum())

    mouse_mask = np.zeros(len(pix), dtype=bool)
    for tc in MOUSE_COLORS:
        mouse_mask |= (np.abs(pix - tc).sum(axis=1) <= tol)
    mouse_n = int(mouse_mask.sum())

    if cheese_n == 0 or mouse_n == 0:
        return ExtractionResult(None, mouse_n, cheese_n)

    cheese_coords = np.argwhere(cheese_mask.reshape(H, W))
    mouse_coords = np.argwhere(mouse_mask.reshape(H, W))
    cheese_xy = (float(cheese_coords[:, 1].mean()),
                 float(cheese_coords[:, 0].mean()))
    agent_xy = (float(mouse_coords[:, 1].mean()),
                float(mouse_coords[:, 0].mean()))

    tracker.update(agent_xy)
    state = MazeState(
        agent_xy=agent_xy,
        cheese_xy=cheese_xy,
        maze_size=maze_size,
        recent_trajectory=tracker.trajectory(),
    )
    return ExtractionResult(state, mouse_n, cheese_n)


# ---- procgenAISC env constructor -------------------------------------

def make_maze_env(env_name: str = "maze_aisc",
                  num: int = 1,
                  num_levels: int = 200,
                  start_level: int = 0,
                  distribution_mode: str = "easy",
                  use_backgrounds: bool = False,
                  **kwargs):
    """Construct a procgenAISC ProcgenGym3Env with SPLIT-MAZE defaults.

    env_name: 'maze_aisc' for goal-misgen training (cheese always
              top-right corner — PLAN §1, procgenAISC modification),
              'maze' for OOD evaluation (cheese random, base procgen).
    use_backgrounds: False keeps the floor/wall palette stable for
              robust sprite-color detection.

    Other procgen options pass through via kwargs.
    """
    from procgen import ProcgenGym3Env  # import lazily; not needed for unit tests
    return ProcgenGym3Env(
        num=num,
        env_name=env_name,
        num_levels=num_levels,
        start_level=start_level,
        distribution_mode=distribution_mode,
        use_backgrounds=use_backgrounds,
        **kwargs,
    )
