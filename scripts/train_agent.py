"""PPO training script for SPLIT-MAZE — Phase 1.3 산출물 (#72).

Thin CLI wrapper around :func:`split_maze.train.train`. The training loop
itself lives in the library so it can be unit-tested without procgen.

Usage — sandbox / smoke (no procgen, fake env):
  PYTHONPATH=src python scripts/train_agent.py --mock \
      --num_envs 4 --num_steps 16 --total_env_steps 128

Usage — WSL with procgenAISC (full):
  PYTHONPATH=src python scripts/train_agent.py \
      --env_name maze_aisc --num_envs 64 --num_steps 256 \
      --total_env_steps 1_000_000 --device cuda \
      --save_path checkpoints/maze_aisc.pt \
      --log_path logs/maze_aisc.jsonl

Per-update log line shows the headline PPO diagnostics + episode-return
rolling mean. Full per-update logs (with all keys) optionally streamed to
``--log_path`` as JSONL.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from split_maze.agent import ImpalaAgent
from split_maze.ppo import PPOConfig
from split_maze.train import train


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="PPO training for SPLIT-MAZE (Phase 1.3).")
    p.add_argument("--env_name", default="maze_aisc",
                   help="procgen env name. 'maze_aisc' = goal-misgen train "
                        "(cheese fixed top-right). 'maze' = OOD eval.")
    p.add_argument("--mock", action="store_true",
                   help="Use MockMazeEnv — no procgen required. "
                        "For sandbox / smoke testing.")
    p.add_argument("--num_envs", type=int, default=64,
                   help="parallel envs (N). Smoke: 4. Full: 64.")
    p.add_argument("--num_steps", type=int, default=256,
                   help="rollout length per update (T). Smoke: 16. Full: 256.")
    p.add_argument("--total_env_steps", type=int, default=1_000_000,
                   help="total env transitions across all envs.")
    p.add_argument("--num_levels", type=int, default=200,
                   help="procgen num_levels (training distribution).")
    p.add_argument("--start_level", type=int, default=0,
                   help="procgen start_level.")
    p.add_argument("--distribution_mode", default="easy",
                   help="procgen distribution_mode.")
    p.add_argument("--seed", type=int, default=0,
                   help="seed for torch/numpy + procgen rand_seed.")
    p.add_argument("--device",
                   default=("cuda" if torch.cuda.is_available() else "cpu"),
                   help="torch device (auto-detects cuda when available).")
    p.add_argument("--save_path", type=Path, default=None,
                   help="Path to save final agent state_dict (.pt). Optional.")
    p.add_argument("--log_path", type=Path, default=None,
                   help="Path to stream per-update logs as JSONL. Optional.")
    p.add_argument("--log_interval", type=int, default=1,
                   help="Print every N updates to stdout (full log "
                        "still goes to --log_path).")
    p.add_argument("--rolling_window", type=int, default=100,
                   help="Size of the FIFO window for ep_return_rolling. "
                        "Smooths per-rollout noise (default 100).")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device)
    print(f"[train_agent] device={device}  env={args.env_name}  "
          f"N={args.num_envs}  T={args.num_steps}  "
          f"total_env_steps={args.total_env_steps}  mock={args.mock}")

    if args.mock:
        from split_maze.train import MockMazeEnv  # local import — kept clean
        env = MockMazeEnv(num=args.num_envs, seed=args.seed)
    else:
        # lazy import — procgen only needed when actually used
        from split_maze.env import make_maze_env
        env = make_maze_env(
            env_name=args.env_name,
            num=args.num_envs,
            num_levels=args.num_levels,
            start_level=args.start_level,
            distribution_mode=args.distribution_mode,
            rand_seed=args.seed,
        )

    agent = ImpalaAgent().to(device)
    config = PPOConfig()
    print(f"[train_agent] agent params = {agent.num_params:,}")

    log_fp = None
    if args.log_path is not None:
        args.log_path.parent.mkdir(parents=True, exist_ok=True)
        log_fp = args.log_path.open("w")

    t0 = time.time()

    def cb(idx: int, log: dict) -> None:
        if idx % args.log_interval == 0:
            elapsed = time.time() - t0
            sps = log["env_steps"] / max(elapsed, 1e-9)
            # Headline: rolling-mean ret (smooth learning-trend signal).
            # this-rollout ret is logged to JSONL but not printed (too noisy).
            roll = log["ep_return_rolling"]
            roll_str = f"{roll:+.3f}" if np.isfinite(roll) else "  nan"
            print(f"  [upd {idx:5d}] steps={log['env_steps']:9d}  "
                  f"ret(roll={log['ep_return_rolling_n']:3d})={roll_str}  "
                  f"ep_now={log['ep_count']:3d}  "
                  f"pol={log['policy']:+.4f}  val={log['value']:.4f}  "
                  f"ent={log['entropy']:.3f}  kl={log['approx_kl']:+.4f}  "
                  f"clip={log['clipfrac']:.3f}  sps={sps:.0f}")
        if log_fp is not None:
            log_fp.write(json.dumps(log) + "\n")
            log_fp.flush()

    logs = train(
        env, agent, config,
        num_steps=args.num_steps,
        total_env_steps=args.total_env_steps,
        device=device,
        log_callback=cb,
        rolling_window=args.rolling_window,
    )

    elapsed = time.time() - t0
    print(f"[train_agent] {len(logs)} updates done in {elapsed:.1f}s  "
          f"({(logs[-1]['env_steps'] / max(elapsed, 1e-9)):.0f} sps avg)")

    if log_fp is not None:
        log_fp.close()
        print(f"[train_agent] log → {args.log_path}")

    if args.save_path is not None:
        args.save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "agent": agent.state_dict(),
            "config": config.__dict__,
            "logs_tail": logs[-10:],
            "args": {k: (str(v) if isinstance(v, Path) else v)
                     for k, v in vars(args).items()},
        }, args.save_path)
        print(f"[train_agent] checkpoint → {args.save_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
