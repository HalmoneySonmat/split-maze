"""Phase 4.0c — bridge ceiling diagnostic (POST-HOC-6 follow-up).

The post-fixed linear ACC W plateaus at cosine ~0.3 (weak; diagnose_v2 on
V2_postfix). Question: is the *linear map* the bottleneck, or is the LM
sentence-embedding genuinely not recoverable from h_agent?

This measures three ceilings on a *held-out* split of (h_agent, ñ_lm) pairs
collected from the frozen Phase-3 agent + Phase-2 LM:

  1. closed-form LINEAR (ridge normal-equations) — the best a linear map
     can do (rules out gradient-descent / undertraining as the cause).
  2. non-linear MLP (h_agent → ñ_lm) — the best an arbitrary bridge can do.
  3. for each, decode the predicted embedding (LN space — gen_ln works,
     see diagnose_v2) → slot-match vs the oracle.

Read: if MLP cosine ≫ linear (e.g. 0.7+) → linear is the bottleneck →
richer ACC justified. If MLP also ≈ linear (~0.3) → the LM-embedding target
is hard to hit from the agent rep (deeper issue; B3 shows slot *classes*
are recoverable, so the gap is the embedding manifold, not the info).

Run (WSL, GPU):
  PYTHONPATH=src python scripts/ceiling_v2.py \\
      --lm_checkpoint checkpoints/lm.pt \\
      --agent_checkpoint checkpoints/phase3/agent.pt \\
      --env_name maze_aisc --device cuda --seed 0 --rollouts 30
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from split_maze.agent import ImpalaAgent
from split_maze.env import DEFAULT_HEADING_WINDOW, TrajectoryTracker
from split_maze.language import CHEESE_DIR_VALUES, HEADING_VALUES, REGION_COLS, REGION_ROWS, parse
from split_maze.lm import LMConfig, MazeLM, MazeTokenizer
from split_maze.paired_collect import PairBuffer, PairBufferConfig, PairedCollector
from split_maze.ppo import RolloutBuffer
from split_maze.train import obs_to_tensor
from split_maze.train_phase3 import collect_rollout_with_pairs, default_state_extractor


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--lm_checkpoint", type=Path, default=Path("checkpoints/lm.pt"))
    p.add_argument("--agent_checkpoint", type=Path,
                   default=Path("checkpoints/phase3/agent.pt"))
    p.add_argument("--env_name", default="maze_aisc")
    p.add_argument("--num_envs", type=int, default=64)
    p.add_argument("--num_steps", type=int, default=256)
    p.add_argument("--rollouts", type=int, default=30)
    p.add_argument("--mlp_hidden", type=int, default=512)
    p.add_argument("--mlp_steps", type=int, default=3000)
    p.add_argument("--ridge", type=float, default=1e-3)
    p.add_argument("--device", default=("cuda" if torch.cuda.is_available() else "cpu"))
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def _ln(x: torch.Tensor) -> torch.Tensor:
    """affine-free LayerNorm over last dim (matches ACC POST-HOC-6)."""
    mu = x.mean(-1, keepdim=True)
    sd = x.std(-1, keepdim=True, unbiased=False)
    return (x - mu) / (sd + 1e-5)


def _slots_match(lm, tokenizer, pred_emb, gold_slots, device, chunk=2048):
    """Decode predicted embeddings (LN space) → parse → mean per-slot match."""
    n = pred_emb.shape[0]
    agree = 0.0
    total = 0
    for i in range(0, n, chunk):
        emb = pred_emb[i:i + chunk].to(device)
        with torch.no_grad():
            gen = lm.generate(emb, max_len=16)
        for j in range(emb.shape[0]):
            toks = tokenizer.decode([int(x) for x in gen[j].tolist()])
            ps = parse(toks)
            g = gold_slots[i + j]
            agree += (int(ps.agent_region == g[0]) + int(ps.heading == g[1])
                      + int(ps.cheese_dir == g[2]))
            total += 3
    return agree / max(total, 1)


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    tokenizer = MazeTokenizer()

    lm_blob = torch.load(args.lm_checkpoint, map_location=device, weights_only=False)
    lm = MazeLM(LMConfig(**lm_blob["lm_config"]))
    lm.load_state_dict(lm_blob["model_state"])
    lm = lm.to(device).eval()
    for p in lm.parameters():
        p.requires_grad_(False)

    agent = ImpalaAgent().to(device)
    agent.load_state_dict(torch.load(args.agent_checkpoint, map_location=device,
                                     weights_only=False)["agent"])
    agent.eval()

    # --- collect pairs from frozen agent ---
    from split_maze.env import make_maze_env
    env = make_maze_env(env_name=args.env_name, num=args.num_envs, num_levels=0,
                        start_level=0, distribution_mode="easy", rand_seed=args.seed)
    collector = PairedCollector(tokenizer, stride=4, max_token_len=16)
    buffer = PairBuffer(PairBufferConfig(capacity=300_000, batch_size=128,
                                         d_agent=agent.d_a, max_token_len=16),
                        pad_id=tokenizer.pad_id)
    rb = RolloutBuffer(T=args.num_steps, N=args.num_envs, device=device)
    trackers = [TrajectoryTracker(DEFAULT_HEADING_WINDOW) for _ in range(args.num_envs)]
    _r, obs_dict, _f = env.observe()
    obs_holder = obs_to_tensor(obs_dict, device)
    cur_rgb = np.asarray(obs_dict["rgb"])
    ep_r = np.zeros(args.num_envs); ep_l = np.zeros(args.num_envs, dtype=np.int64)
    rng = random.Random(args.seed)
    for _ in range(args.rollouts):
        _, obs_holder, cur_rgb, h_TN, ms = collect_rollout_with_pairs(
            env, agent, rb, trackers, obs_holder=obs_holder, cur_rgb=cur_rgb,
            episode_returns=ep_r, episode_lengths=ep_l,
            state_extractor=default_state_extractor, d_agent=agent.d_a, device=device)
        collector.extract_into(buffer, h_TN, ms, rng=rng)
    N = len(buffer)
    print(f"[ceiling] collected {N:,} pairs")

    # --- build matrices: ñ_agent, ñ_lm, gold slots ---
    H = buffer.h_agent_buf[:N].to(device)
    IDS = buffer.ids_buf[:N]
    LENS = buffer.lengths_buf[:N]
    h_lm_chunks = []
    with torch.no_grad():
        for i in range(0, N, 4096):
            h_lm_chunks.append(lm.encode(IDS[i:i + 4096].to(device)))
    H_lm = torch.cat(h_lm_chunks, dim=0)
    NA = _ln(H)            # (N, d_a)
    NL = _ln(H_lm)         # (N, d_lm)
    gold = []
    for i in range(N):
        ps = parse(tokenizer.decode([int(x) for x in IDS[i, :int(LENS[i])].tolist()]))
        gold.append((ps.agent_region, ps.heading, ps.cheese_dir))

    # --- train/test split ---
    g = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(N, generator=g)
    n_te = max(1, N // 5)
    te, tr = perm[:n_te], perm[n_te:]
    NA_tr, NL_tr = NA[tr], NL[tr]
    NA_te, NL_te = NA[te], NL[te]
    gold_te = [gold[int(i)] for i in te]

    def report(tag, pred_te):
        cos = F.cosine_similarity(pred_te, NL_te, dim=-1)
        sm = _slots_match(lm, tokenizer, pred_te, gold_te, device)
        print(f"  [{tag}] held-out cosine mean={cos.mean():.4f} "
              f"(min={cos.min():.3f} max={cos.max():.3f})  slot-match={sm:.4f}")

    print(f"[ceiling] train={len(tr):,} test={len(te):,}")
    # --- 1. closed-form linear (ridge normal equations) ---
    d = NA.shape[1]
    A = NA_tr.t() @ NA_tr + args.ridge * torch.eye(d, device=device)
    B = NA_tr.t() @ NL_tr
    X = torch.linalg.solve(A, B)            # (d_a, d_lm)
    report("LINEAR closed-form", NA_te @ X)

    # --- 2. non-linear MLP ---
    mlp = nn.Sequential(
        nn.Linear(NA.shape[1], args.mlp_hidden), nn.GELU(),
        nn.Linear(args.mlp_hidden, args.mlp_hidden), nn.GELU(),
        nn.Linear(args.mlp_hidden, NL.shape[1]),
    ).to(device)
    opt = torch.optim.AdamW(mlp.parameters(), lr=1e-3, weight_decay=1e-4)
    bs = 4096
    for step in range(args.mlp_steps):
        idx = torch.randint(0, len(tr), (bs,), device=device)
        pred = mlp(NA_tr[idx])
        loss = F.mse_loss(pred, NL_tr[idx])
        opt.zero_grad(); loss.backward(); opt.step()
        if (step + 1) % 500 == 0:
            with torch.no_grad():
                ct = F.cosine_similarity(mlp(NA_te), NL_te, dim=-1).mean().item()
            print(f"    [mlp {step+1}] train mse={loss.item():.4f} test cos={ct:.4f}")
    with torch.no_grad():
        report("MLP non-linear", mlp(NA_te))

    print("\n[ceiling] verdict: compare LINEAR vs MLP cosine/slot-match. "
          "MLP≫LINEAR → linear bottleneck (richer ACC). MLP≈LINEAR(~0.3) → "
          "embedding manifold hard from agent rep.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
