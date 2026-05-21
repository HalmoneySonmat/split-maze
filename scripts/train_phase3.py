"""Phase 3.4 co-training CLI — RL agent + B3/B4/V2 interpreters.

Thin wrapper around :func:`split_maze.train_phase3.train_phase3`. The loop
lives in the library (unit-tested without procgen); this script wires the
env, loads the Phase-2 LM (checkpoints/lm.pt) into per-build copies, builds
the interpreters, and streams per-build diagnostics.

Frozen decisions: PLAN §10.5 (P3-4-1..P3-4-4) + §10.2/§10.3 (P3-1..P3-2-5).

Usage — sandbox / smoke (no procgen):
  PYTHONPATH=src python scripts/train_phase3.py --mock \\
      --num_envs 4 --num_steps 16 --total_env_steps 256 \\
      --builds B3,B4,V2 --device cpu

Usage — WSL full (procgenAISC + Phase-2 LM):
  PYTHONPATH=src python scripts/train_phase3.py \\
      --env_name maze_aisc --num_envs 64 --num_steps 256 \\
      --total_env_steps 25_000_000 --device cuda --seed 0 \\
      --lm_checkpoint checkpoints/lm.pt \\
      --builds B3,B4,V2 \\
      --save_dir checkpoints/phase3 \\
      --log_path logs/phase3.jsonl
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from split_maze.acc import ACC, ACCConfig
from split_maze.agent import ImpalaAgent
from split_maze.builds import B3Probe, B4Adapter, Build, V2ACC
from split_maze.lm import LMConfig, MazeLM, MazeTokenizer
from split_maze.paired_collect import PairBuffer, PairBufferConfig, PairedCollector
from split_maze.ppo import PPOConfig
from split_maze.train_phase3 import Phase3Config, train_phase3


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Phase 3.4 co-training (RL + B3/B4/V2).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Env.
    p.add_argument("--env_name", default="maze_aisc",
                   help="procgen env (maze_aisc = goal-misgen train).")
    p.add_argument("--mock", action="store_true",
                   help="Use MockMazeEnv (no procgen). Smoke only.")
    p.add_argument("--num_envs", type=int, default=64)
    p.add_argument("--num_steps", type=int, default=256)
    p.add_argument("--total_env_steps", type=int, default=25_000_000)
    p.add_argument("--num_levels", type=int, default=200)
    p.add_argument("--start_level", type=int, default=0)
    p.add_argument("--distribution_mode", default="easy")
    # Builds + LM.
    p.add_argument("--builds", default="B3,B4,V2",
                   help="comma list subset of {B3,B4,V2}.")
    p.add_argument("--lm_checkpoint", type=Path, default=None,
                   help="Phase-2 LM checkpoint (.pt). Loaded into per-build "
                        "copies for B4/V2. If omitted, a fresh untrained LM "
                        "is used (warned — for smoke only).")
    # Interpreter hyperparams (PLAN §10.2 P3-3-A / §10.3 P3-2-2/4).
    p.add_argument("--acc_updates_per_rl", type=int, default=32)   # K
    p.add_argument("--interp_batch", type=int, default=128)
    p.add_argument("--interp_lr", type=float, default=3e-4)
    p.add_argument("--interp_warmup", type=int, default=500)
    p.add_argument("--interp_weight_decay", type=float, default=0.01)
    p.add_argument("--stride", type=int, default=4)
    p.add_argument("--buffer_capacity", type=int, default=256_000)
    p.add_argument("--max_token_len", type=int, default=16)
    # Run.
    p.add_argument("--device",
                   default=("cuda" if torch.cuda.is_available() else "cpu"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save_dir", type=Path, default=None,
                   help="Dir for per-build + agent checkpoints.")
    p.add_argument("--log_path", type=Path, default=None,
                   help="Per-update JSONL log.")
    p.add_argument("--log_interval", type=int, default=1)
    return p.parse_args()


def _load_lm(ckpt_path: Path | None, tokenizer: MazeTokenizer,
             device: torch.device) -> MazeLM:
    """Build a MazeLM. From ``ckpt_path`` (Phase-2 lm.pt) if given, else a
    fresh untrained LM with default architecture."""
    if ckpt_path is None:
        cfg = LMConfig.from_tokenizer(tokenizer)
        print("[train_phase3] WARNING: no --lm_checkpoint; using a FRESH "
              "untrained LM (smoke only — B4/V2 need Phase-2 weights).")
        return MazeLM(cfg).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = LMConfig(**ckpt["lm_config"])
    lm = MazeLM(cfg)
    lm.load_state_dict(ckpt["model_state"])
    return lm.to(device)


def build_interpreters(names: list[str], tokenizer: MazeTokenizer,
                       lm_checkpoint: Path | None, *, d_agent: int,
                       device: torch.device) -> dict[str, Build]:
    """Construct the requested builds. B4/V2 each get their *own* LM copy
    loaded from the same checkpoint (P3-3-3 박제)."""
    builds: dict[str, Build] = {}
    if "B3" in names:
        builds["B3"] = B3Probe(tokenizer, d_agent=d_agent).to(device)
    if "B4" in names:
        builds["B4"] = B4Adapter(
            _load_lm(lm_checkpoint, tokenizer, device), d_agent=d_agent
        ).to(device)
    if "V2" in names:
        lm_v2 = _load_lm(lm_checkpoint, tokenizer, device)
        acc = ACC(ACCConfig(d_agent=d_agent, d_lm=lm_v2.config.d_model))
        builds["V2"] = V2ACC(lm_v2, acc).to(device)
    return builds


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device)
    names = [s.strip() for s in args.builds.split(",") if s.strip()]
    print(f"[train_phase3] device={device} env={args.env_name} "
          f"N={args.num_envs} T={args.num_steps} "
          f"total={args.total_env_steps} builds={names} mock={args.mock}")

    # ---- env ----
    if args.mock:
        from split_maze.train import MockMazeEnv
        env = MockMazeEnv(num=args.num_envs, seed=args.seed)
    else:
        from split_maze.env import make_maze_env
        env = make_maze_env(
            env_name=args.env_name, num=args.num_envs,
            num_levels=args.num_levels, start_level=args.start_level,
            distribution_mode=args.distribution_mode, rand_seed=args.seed,
        )

    # ---- agent + interpreters ----
    agent = ImpalaAgent().to(device)
    tokenizer = MazeTokenizer()
    builds = build_interpreters(
        names, tokenizer, args.lm_checkpoint,
        d_agent=agent.d_a, device=device,
    )
    print(f"[train_phase3] agent params={agent.num_params:,}  "
          f"builds={list(builds)}")

    # ---- collector + shared buffer ----
    collector = PairedCollector(tokenizer, stride=args.stride,
                                max_token_len=args.max_token_len)
    buffer = PairBuffer(
        PairBufferConfig(
            capacity=args.buffer_capacity, batch_size=args.interp_batch,
            d_agent=agent.d_a, max_token_len=args.max_token_len,
        ),
        pad_id=tokenizer.pad_id,
    )

    cfg = Phase3Config(
        num_steps=args.num_steps,
        total_env_steps=args.total_env_steps,
        acc_updates_per_rl=args.acc_updates_per_rl,
        interp_batch=args.interp_batch,
        interp_lr=args.interp_lr,
        interp_warmup=args.interp_warmup,
        interp_weight_decay=args.interp_weight_decay,
        stride=args.stride,
        buffer_capacity=args.buffer_capacity,
    )

    log_fp = None
    if args.log_path is not None:
        args.log_path.parent.mkdir(parents=True, exist_ok=True)
        log_fp = args.log_path.open("w")

    t0 = time.time()

    def cb(idx: int, log: dict) -> None:
        if idx % args.log_interval == 0:
            elapsed = time.time() - t0
            sps = log["env_steps"] / max(elapsed, 1e-9)
            roll = log.get("ep_return_rolling", float("nan"))
            roll_str = f"{roll:+.3f}" if np.isfinite(roll) else "  nan"
            interp = " ".join(
                f"{n}={log[f'{n}/loss']:.3f}"
                for n in names if f"{n}/loss" in log
            )
            print(f"  [upd {idx:5d}] steps={log['env_steps']:9d} "
                  f"ret={roll_str} pol={log['policy']:+.4f} "
                  f"val={log['value']:.4f} pairs={log['pairs_added']:4d} "
                  f"buf={log['buffer_size']:7d} {interp} sps={sps:.0f}")
        if log_fp is not None:
            log_fp.write(json.dumps(log) + "\n")
            log_fp.flush()

    logs = train_phase3(
        env, agent, builds, collector, buffer,
        config=cfg, ppo_config=PPOConfig(), device=device,
        log_callback=cb,
    )

    elapsed = time.time() - t0
    print(f"[train_phase3] {len(logs)} updates in {elapsed:.1f}s")
    if log_fp is not None:
        log_fp.close()
        print(f"[train_phase3] log → {args.log_path}")

    # ---- per-build + agent checkpoints ----
    if args.save_dir is not None:
        args.save_dir.mkdir(parents=True, exist_ok=True)
        torch.save({"agent": agent.state_dict()},
                   args.save_dir / "agent.pt")
        for name, b in builds.items():
            torch.save({"build": name, "state": b.state_dict()},
                       args.save_dir / f"{name}.pt")
        print(f"[train_phase3] checkpoints → {args.save_dir}/"
              f"{{agent,{','.join(builds)}}}.pt")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
