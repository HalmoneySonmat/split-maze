"""Evaluation script for SPLIT-MAZE — Phase 1.5 산출물 (#74).

Loads a trained ImpalaAgent checkpoint and evaluates it against either
the in-distribution held-out levels or the OOD environment, computing
the metrics needed for Phase 1.6 gate judgment (PLAN §7.1):

  in-dist : success_rate ≥ 0.80
  OOD     : goal_misgen_rate ≥ 0.50

Usage — in-distribution (held-out levels of maze_aisc):
  PYTHONPATH=src python scripts/evaluate.py \\
      --checkpoint checkpoints/maze_aisc_full.pt \\
      --mode in-dist --num_episodes 500 --device cuda \\
      --output_path results/in_dist.json

Usage — OOD (cheese random):
  PYTHONPATH=src python scripts/evaluate.py \\
      --checkpoint checkpoints/maze_aisc_full.pt \\
      --mode ood --num_episodes 500 --device cuda \\
      --output_path results/ood.json

The two passes together let us judge the Phase 1 gate. Print a
PASS/FAIL line at the end if both thresholds are checkable.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from split_maze.agent import ImpalaAgent
from split_maze.evaluate import (
    compute_in_dist_metrics,
    compute_ood_metrics,
    evaluate_episodes,
)


# Phase 1 gate thresholds (PLAN §7.1) — pre-registered.
IN_DIST_SUCCESS_THRESHOLD: float = 0.80
OOD_MISGEN_THRESHOLD: float = 0.50


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate a SPLIT-MAZE agent checkpoint "
                    "(Phase 1.5 산출물).")
    p.add_argument("--checkpoint", type=Path, required=True,
                   help="Path to .pt checkpoint saved by train_agent.py.")
    p.add_argument("--mode", choices=("in-dist", "ood"), required=True,
                   help="in-dist = maze_aisc held-out; ood = maze (cheese random).")
    p.add_argument("--num_episodes", type=int, default=500,
                   help="Target completed-episode count (default 500).")
    p.add_argument("--num_envs", type=int, default=32,
                   help="Parallel envs during evaluation.")
    p.add_argument("--seed", type=int, default=0,
                   help="Seed for torch and procgen rand_seed.")
    p.add_argument("--device",
                   default=("cuda" if torch.cuda.is_available() else "cpu"))
    p.add_argument("--deterministic", action="store_true",
                   help="argmax over policy logits (no sampling). Default: sample.")
    # In-dist held-out config (cheese still top-right, levels never seen during
    # training). Training used num_levels=200, start_level=0 by default.
    p.add_argument("--in_dist_start_level", type=int, default=200,
                   help="start_level for in-dist eval (held-out).")
    p.add_argument("--in_dist_num_levels", type=int, default=0,
                   help="num_levels for in-dist eval (0 = infinite seeds).")
    # OOD config (different env, cheese random).
    p.add_argument("--ood_start_level", type=int, default=0,
                   help="start_level for OOD eval.")
    p.add_argument("--ood_num_levels", type=int, default=0,
                   help="num_levels for OOD eval (0 = infinite seeds).")
    p.add_argument("--output_path", type=Path, default=None,
                   help="Where to save metrics + per-episode rewards as JSON.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    # ---- Load checkpoint ----
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"checkpoint not found: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if "agent" not in ckpt:
        raise KeyError(f"checkpoint missing 'agent' key — got {list(ckpt.keys())}")
    agent = ImpalaAgent().to(device)
    agent.load_state_dict(ckpt["agent"])
    agent.eval()
    print(f"[evaluate] checkpoint loaded: {args.checkpoint}")
    print(f"[evaluate] agent params = {agent.num_params:,}")
    print(f"[evaluate] mode={args.mode}  num_episodes={args.num_episodes}  "
          f"num_envs={args.num_envs}  device={device}  "
          f"deterministic={args.deterministic}")

    # ---- Build env (lazy import — procgen only loaded when actually used) ----
    from split_maze.env import make_maze_env
    if args.mode == "in-dist":
        env = make_maze_env(
            env_name="maze_aisc",
            num=args.num_envs,
            num_levels=args.in_dist_num_levels,
            start_level=args.in_dist_start_level,
            distribution_mode="easy",
            use_backgrounds=False,
            rand_seed=args.seed,
        )
    else:
        env = make_maze_env(
            env_name="maze",
            num=args.num_envs,
            num_levels=args.ood_num_levels,
            start_level=args.ood_start_level,
            distribution_mode="easy",
            use_backgrounds=False,
            rand_seed=args.seed,
        )

    # ---- Roll out + compute metrics ----
    t0 = time.time()
    records = evaluate_episodes(env, agent,
                                num_episodes=args.num_episodes,
                                device=device,
                                deterministic=args.deterministic)
    elapsed = time.time() - t0

    if args.mode == "in-dist":
        metrics = compute_in_dist_metrics(records)
    else:
        metrics = compute_ood_metrics(records)
    metrics["mode"] = args.mode
    metrics["deterministic"] = args.deterministic
    metrics["eval_seconds"] = elapsed

    # ---- Print summary ----
    print(f"[evaluate] done in {elapsed:.1f}s — {len(records)} episodes")
    print(f"[evaluate] success_rate = {metrics['success_rate']:.4f} "
          f"(n_success = {metrics['n_success']} / {metrics['n_episodes']})")
    print(f"[evaluate] mean_return  = {metrics['mean_return']:.4f}")
    if args.mode == "ood":
        print(f"[evaluate] ended_top_right_rate = "
              f"{metrics['ended_top_right_rate']:.4f}")
        gm = metrics["goal_misgen_rate"]
        gm_str = f"{gm:.4f}" if np.isfinite(gm) else "nan"
        print(f"[evaluate] goal_misgen_rate = {gm_str}  "
              f"(eligible = {metrics['goal_misgen_n_eligible']})")

    # ---- Phase 1 gate verdict (per mode) ----
    if args.mode == "in-dist":
        pass_ = metrics["success_rate"] >= IN_DIST_SUCCESS_THRESHOLD
        print(f"[evaluate] in-dist gate (≥ {IN_DIST_SUCCESS_THRESHOLD:.2f}): "
              f"{'PASS' if pass_ else 'FAIL'}")
    else:
        gm = metrics["goal_misgen_rate"]
        if np.isfinite(gm):
            pass_ = gm >= OOD_MISGEN_THRESHOLD
            print(f"[evaluate] OOD goal-misgen gate (≥ {OOD_MISGEN_THRESHOLD:.2f}): "
                  f"{'PASS' if pass_ else 'FAIL'}")
        else:
            print(f"[evaluate] OOD goal-misgen gate: UNDETERMINED "
                  f"(no eligible episodes)")

    # ---- Save raw + metrics ----
    if args.output_path is not None:
        args.output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "args": {k: (str(v) if isinstance(v, Path) else v)
                     for k, v in vars(args).items()},
            "metrics": metrics,
            "records": [
                {"reward": r.reward,
                 "agent_region": list(r.agent_region) if r.agent_region else None,
                 "cheese_region": list(r.cheese_region) if r.cheese_region else None}
                for r in records
            ],
        }
        args.output_path.write_text(json.dumps(payload, indent=2))
        print(f"[evaluate] results → {args.output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
