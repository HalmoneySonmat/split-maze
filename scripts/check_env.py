"""scripts/check_env.py — Phase 0 #69 deliverable.

Runs a random-policy rollout in procgenAISC maze/maze_aisc, extracts maze
state per step via rgb sprite detection, and renders describer-oracle
sentences. Verifies the full pipeline: env → extractor → oracle → sentence.

Usage (in splitmaze conda env on WSL):
    python scripts/check_env.py --env_name maze_aisc --steps 50
    python scripts/check_env.py --env_name maze --steps 50 --seed 0
"""

from __future__ import annotations

import argparse
import random
import sys

import numpy as np

from split_maze.env import (
    DEFAULT_HEADING_WINDOW,
    TrajectoryTracker,
    extract_maze_state,
    make_maze_env,
)
from split_maze.language import describer_oracle, render


def main() -> int:
    p = argparse.ArgumentParser(
        description="Phase 0 #69 — procgen rollout + describer oracle sanity"
    )
    p.add_argument("--env_name", default="maze_aisc",
                   choices=("maze", "maze_aisc"),
                   help="maze_aisc = goal-misgen training (cheese top-right);"
                        " maze = OOD eval (cheese random).")
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--seed", type=int, default=0, help="procgen start_level")
    p.add_argument("--num_levels", type=int, default=200)
    p.add_argument("--window", type=int, default=DEFAULT_HEADING_WINDOW,
                   help="HEADING window K (LANGUAGE_SPEC §5)")
    p.add_argument("--move_threshold", type=float, default=0.5,
                   help="min net displacement for non-'still' HEADING")
    p.add_argument("--print_every", type=int, default=5)
    args = p.parse_args()

    print("=== Phase 0 #69: rollout + describer oracle ===")
    print(f"  env={args.env_name}  steps={args.steps}  seed={args.seed}"
          f"  window={args.window}")
    print()

    env = make_maze_env(env_name=args.env_name, num=1,
                        num_levels=args.num_levels, start_level=args.seed,
                        use_backgrounds=False)
    tracker = TrajectoryTracker(window=args.window)
    action_rng = random.Random(args.seed)
    surface_rng = random.Random(args.seed * 31 + 1)

    sentences_generated = 0
    extraction_failures = 0
    oracle_excluded = 0   # agent on cheese (oracle returns None)
    episode_resets = 0

    for step in range(args.steps):
        # gym3: observe() returns (rew, obs, first). 'first' marks episode start.
        _, obs, first = env.observe()
        if first[0] and step > 0:
            tracker.reset()
            episode_resets += 1

        rgb = obs["rgb"][0]   # (64, 64, 3) uint8

        res = extract_maze_state(rgb, tracker)
        label: str
        if res.maze_state is None:
            extraction_failures += 1
            label = (f"[extract fail: agent_n={res.agent_pixel_count}, "
                     f"cheese_n={res.cheese_pixel_count}]")
        else:
            slots = describer_oracle(res.maze_state,
                                     move_threshold=args.move_threshold)
            if slots is None:
                oracle_excluded += 1
                label = "[oracle: agent on cheese]"
            else:
                tokens = render(slots, rng=surface_rng, include_bos_eos=False)
                label = " ".join(tokens)
                sentences_generated += 1

        if step % args.print_every == 0 or step == args.steps - 1:
            tag = "RESET" if (first[0] and step > 0) else "step"
            print(f"  {tag} {step:3d}: {label}")

        # Random action in Discrete(15)
        action = np.array([action_rng.randint(0, 14)], dtype=np.int32)
        env.act(action)

    env.close()

    print()
    print("=== summary ===")
    print(f"  sentences generated:   {sentences_generated}/{args.steps}")
    print(f"  extraction failures:   {extraction_failures}/{args.steps}")
    print(f"  oracle excluded:       {oracle_excluded}/{args.steps}"
          f"  (agent on cheese)")
    print(f"  episode resets:        {episode_resets}")
    print(f"  trajectory length end: {len(tracker)}")

    # Phase 0 #69 pass criterion: at least some sentences generated.
    if sentences_generated == 0:
        print("\nFAIL: no sentences generated — sprite detection or oracle broken.")
        return 1
    print("\n✓ PHASE 0 #69 PASS — pipeline (env → extractor → oracle → sentence) works.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
