"""POST-HOC-6 — re-train the V2 ACC bridge (W only) on the frozen Phase-3
agent + Phase-2 LM, with the collapse paths removed.

Why (PLAN §10.1 POST-HOC-6): the Phase-3 V2 collapsed (recon=0 = constant
solution; diagnose_v2). Fix = freeze the *entire* LM (interface_proj
included) + non-learnable ACC LayerNorm (affine=False) so only W learns the
bridge between two *fixed informative* backbones (= SPLIT-MNIST V2 본형). No
RL re-run is needed — the agent (checkpoints/phase3) and LM (lm.pt) are both
fixed, so this is a cheap post-hoc linear-bridge fit (a fixed→fixed mapping,
NOT an adapting interpreter, so not the SPLIT-9 failure pattern).

Run (WSL, GPU):
  PYTHONPATH=src python scripts/retrain_v2_acc.py \\
      --lm_checkpoint checkpoints/lm.pt \\
      --agent_checkpoint checkpoints/phase3/agent.pt \\
      --env_name maze_aisc --device cuda --seed 0 \\
      --rollouts 30 --steps 400 \\
      --out checkpoints/phase3/V2_postfix.pt
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch

from split_maze.acc import ACC, ACCConfig
from split_maze.agent import ImpalaAgent
from split_maze.builds import V2ACC
from split_maze.env import DEFAULT_HEADING_WINDOW, TrajectoryTracker
from split_maze.lm import LMConfig, MazeLM, MazeTokenizer
from split_maze.paired_collect import PairBuffer, PairBufferConfig, PairedCollector
from split_maze.ppo import RolloutBuffer
from split_maze.train import obs_to_tensor
from split_maze.train_phase3 import (
    Phase3Config,
    collect_rollout_with_pairs,
    default_state_extractor,
    _warmup_scale,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--lm_checkpoint", type=Path, default=Path("checkpoints/lm.pt"))
    p.add_argument("--agent_checkpoint", type=Path,
                   default=Path("checkpoints/phase3/agent.pt"))
    p.add_argument("--env_name", default="maze_aisc")
    p.add_argument("--num_envs", type=int, default=64)
    p.add_argument("--num_steps", type=int, default=256)
    p.add_argument("--rollouts", type=int, default=30,
                   help="how many rollouts to collect pairs from.")
    p.add_argument("--steps", type=int, default=3000,
                   help="ACC W gradient steps after collection (untied A2L "
                        "converges to ~0.47 cosine; needs more than warmup).")
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup", type=int, default=500)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--stride", type=int, default=4)
    p.add_argument("--device", default=("cuda" if torch.cuda.is_available() else "cpu"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=Path("checkpoints/phase3/V2_postfix2.pt"),
                   help="POST-HOC-7 untied refit output.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    tokenizer = MazeTokenizer()

    # --- LM (Phase-2, informative) + V2ACC (POST-HOC-6: LM frozen, W learns) ---
    lm_blob = torch.load(args.lm_checkpoint, map_location=device, weights_only=False)
    lm = MazeLM(LMConfig(**lm_blob["lm_config"]))
    lm.load_state_dict(lm_blob["model_state"])
    acc = ACC(ACCConfig(d_agent=256, d_lm=lm.config.d_model,
                        tied=False))   # POST-HOC-7: untied (affine=False)
    v2 = V2ACC(lm, acc).to(device)
    v2.train()  # train() keeps LM in eval (frozen); only W is trainable

    # --- agent (Phase-3, frozen) ---
    agent = ImpalaAgent().to(device)
    agent.load_state_dict(
        torch.load(args.agent_checkpoint, map_location=device,
                   weights_only=False)["agent"])
    agent.eval()
    print(f"[retrain_v2] device={device}  W params={v2.acc.W.numel():,}  "
          f"trainable={sum(p.numel() for p in v2.interpreter_parameters()):,}")

    # --- collect (h_agent, ids) pairs from the frozen agent ---
    from split_maze.env import make_maze_env
    env = make_maze_env(env_name=args.env_name, num=args.num_envs,
                        num_levels=0, start_level=0,
                        distribution_mode="easy", rand_seed=args.seed)
    collector = PairedCollector(tokenizer, stride=args.stride, max_token_len=16)
    buffer = PairBuffer(
        PairBufferConfig(capacity=256_000, batch_size=args.batch,
                         d_agent=agent.d_a, max_token_len=16),
        pad_id=tokenizer.pad_id)
    rollout_buffer = RolloutBuffer(T=args.num_steps, N=args.num_envs, device=device)
    trackers = [TrajectoryTracker(DEFAULT_HEADING_WINDOW) for _ in range(args.num_envs)]
    _r, obs_dict, _f = env.observe()
    obs_holder = obs_to_tensor(obs_dict, device)
    cur_rgb = np.asarray(obs_dict["rgb"])
    ep_r = np.zeros(args.num_envs, dtype=np.float64)
    ep_l = np.zeros(args.num_envs, dtype=np.int64)
    surface_rng = random.Random(args.seed)

    for r in range(args.rollouts):
        stats, obs_holder, cur_rgb, h_agent_TN, maze_states = (
            collect_rollout_with_pairs(
                env, agent, rollout_buffer, trackers,
                obs_holder=obs_holder, cur_rgb=cur_rgb,
                episode_returns=ep_r, episode_lengths=ep_l,
                state_extractor=default_state_extractor,
                d_agent=agent.d_a, device=device))
        added = collector.extract_into(buffer, h_agent_TN, maze_states, rng=surface_rng)
        if (r + 1) % 5 == 0:
            print(f"  [collect {r+1}/{args.rollouts}] buffer={len(buffer):,}")

    print(f"[retrain_v2] collected {len(buffer):,} pairs. Training W...")

    # --- train W only ---
    opt = torch.optim.AdamW(v2.interpreter_parameters(), lr=args.lr,
                            weight_decay=args.weight_decay, betas=(0.9, 0.95))
    for step in range(args.steps):
        batch = buffer.sample(args.batch)
        h = batch["h_agent"].to(device)
        ids = batch["ids"].to(device)
        lengths = batch["lengths"].to(device)
        out = v2.update(h, ids, lengths)
        opt.zero_grad()
        out["loss"].backward()
        for g in opt.param_groups:
            g["lr"] = args.lr * _warmup_scale(step, args.warmup)
        opt.step()
        if (step + 1) % 50 == 0 or step == 0:
            with torch.no_grad():
                hat_std = out["hat_lm"].std(dim=0).mean().item()
                nlm_std = out["n_lm"].std(dim=0).mean().item()
            print(f"  [W step {step+1:4d}] recon={out['loss'].item():.4f} "
                  f"ĥ_lm std={hat_std:.4f} ñ_lm std={nlm_std:.4f} "
                  f"(std≫0 = no collapse)")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state": v2.state_dict(), "note": "POST-HOC-6 W-only refit"},
               args.out)
    print(f"[retrain_v2] saved → {args.out}")
    print("  Next: re-run diagnose_v2.py with --v2_checkpoint this file to "
          "confirm ĥ_lm varied + generations track oracle.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
