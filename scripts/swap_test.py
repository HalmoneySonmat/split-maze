"""Phase 4.2 — measurement #3: activation swap (PLAN §5.4 / §5.6).

Disentangles the Phase-4.1 confound: is V2's low OOD rationalization a
*principled* causal tracking of h_agent (which lacks the real cheese OOD),
or just a *weak/noisy* interpreter?

Causal test on **in-distribution** states (where the agent DOES represent
cheese_dir): pick pairs (A, B) whose oracle cheese_dir differs, interpolate
the agent hidden h(α) = (1-α)·h_A + α·h_B for α ∈ {0,.25,.5,.75,1}, and ask
whether each interpreter's generated cheese_dir flips from A's value to B's
value as α: 0→1.

  swap-following rate = fraction of pairs where the output at α=1 equals
  B's cheese_dir (the swapped-in value). A high rate ⇒ the interpreter is
  causally driven by h_agent. §5.6: V2 − B4 ≥ 0.15 predicted (if V2 tracks
  the agent while B4 leans on the prior).

Also reports the α-interpolation curve (fraction matching B at each α).

Run (WSL, GPU):
  PYTHONPATH=src python scripts/swap_test.py \\
      --lm_checkpoint checkpoints/lm.pt \\
      --agent_checkpoint checkpoints/phase3/agent.pt \\
      --v2_checkpoint checkpoints/phase3/V2_postfix2.pt \\
      --b4_checkpoint checkpoints/phase3/B4.pt \\
      --b3_checkpoint checkpoints/phase3/B3.pt \\
      --device cuda --seed 0 --rollouts 20 --n_pairs 1000
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from split_maze.acc import ACC, ACCConfig
from split_maze.agent import ImpalaAgent
from split_maze.builds import B3Probe, B4Adapter, V2ACC
from split_maze.env import DEFAULT_HEADING_WINDOW, TrajectoryTracker
from split_maze.language import (
    CHEESE_DIR_VALUES, HEADING_VALUES, REGION_COLS, REGION_ROWS,
    describer_oracle, parse,
)
from split_maze.lm import LMConfig, MazeLM, MazeTokenizer
from split_maze.ppo import RolloutBuffer
from split_maze.train import obs_to_tensor
from split_maze.train_phase3 import collect_rollout_with_pairs, default_state_extractor


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--lm_checkpoint", type=Path, default=Path("checkpoints/lm.pt"))
    p.add_argument("--agent_checkpoint", type=Path,
                   default=Path("checkpoints/phase3/agent.pt"))
    p.add_argument("--v2_checkpoint", type=Path,
                   default=Path("checkpoints/phase3/V2_postfix2.pt"))
    p.add_argument("--b4_checkpoint", type=Path, default=Path("checkpoints/phase3/B4.pt"))
    p.add_argument("--b3_checkpoint", type=Path, default=Path("checkpoints/phase3/B3.pt"))
    p.add_argument("--num_envs", type=int, default=64)
    p.add_argument("--num_steps", type=int, default=256)
    p.add_argument("--rollouts", type=int, default=20)
    p.add_argument("--n_pairs", type=int, default=1000)
    p.add_argument("--device", default=("cuda" if torch.cuda.is_available() else "cpu"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output_path", type=Path, default=Path("results/phase4_swap.json"))
    return p.parse_args()


def _fresh_lm(lm_blob, device):
    lm = MazeLM(LMConfig(**lm_blob["lm_config"]))
    lm.load_state_dict(lm_blob["model_state"])
    return lm.to(device).eval()


def _cheese_of(build, h, tokenizer, *, kind, acc=None):
    """Return list of cheese_dir strings for a batch h (already on device)."""
    with torch.no_grad():
        if kind == "V2":
            gen = build.lm.generate(acc.predict_lm_from_agent(acc.ln_agent(h)), max_len=16)
        elif kind == "B4":
            gen = build.generate(h, max_len=16)
        else:  # B3
            o = build(h)
            ch = o["cheese"].argmax(-1)
            return [CHEESE_DIR_VALUES[int(i)] for i in ch]
    out = []
    for row in gen:
        out.append(parse(tokenizer.decode([int(x) for x in row.tolist()])).cheese_dir)
    return out


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    rng = random.Random(args.seed)
    device = torch.device(args.device)
    tokenizer = MazeTokenizer()
    lm_blob = torch.load(args.lm_checkpoint, map_location=device, weights_only=False)

    agent = ImpalaAgent().to(device)
    agent.load_state_dict(torch.load(args.agent_checkpoint, map_location=device,
                                     weights_only=False)["agent"])
    agent.eval()
    acc = ACC(ACCConfig(d_agent=agent.d_a, d_lm=lm_blob["lm_config"]["d_model"], tied=False))
    v2 = V2ACC(_fresh_lm(lm_blob, device), acc).to(device)
    v2.load_state_dict(torch.load(args.v2_checkpoint, map_location=device,
                                  weights_only=False)["state"]); v2.eval()
    b4 = B4Adapter(_fresh_lm(lm_blob, device), d_agent=agent.d_a).to(device)
    b4.load_state_dict(torch.load(args.b4_checkpoint, map_location=device,
                                  weights_only=False)["state"]); b4.eval()
    b3 = B3Probe(tokenizer, d_agent=agent.d_a).to(device)
    b3.load_state_dict(torch.load(args.b3_checkpoint, map_location=device,
                                  weights_only=False)["state"]); b3.eval()
    print("[swap] loaded agent + V2 + B4 + B3")

    # --- collect in-dist states with their oracle cheese_dir ---
    from split_maze.env import make_maze_env
    env = make_maze_env(env_name="maze_aisc", num=args.num_envs, num_levels=0,
                        start_level=0, distribution_mode="easy", rand_seed=args.seed)
    rb = RolloutBuffer(T=args.num_steps, N=args.num_envs, device=device)
    trackers = [TrajectoryTracker(DEFAULT_HEADING_WINDOW) for _ in range(args.num_envs)]
    _r, obs_dict, _f = env.observe()
    obs_holder = obs_to_tensor(obs_dict, device)
    cur_rgb = np.asarray(obs_dict["rgb"])
    ep_r = np.zeros(args.num_envs); ep_l = np.zeros(args.num_envs, dtype=np.int64)
    H, CD = [], []
    for _ in range(args.rollouts):
        _, obs_holder, cur_rgb, h_TN, ms = collect_rollout_with_pairs(
            env, agent, rb, trackers, obs_holder=obs_holder, cur_rgb=cur_rgb,
            episode_returns=ep_r, episode_lengths=ep_l,
            state_extractor=default_state_extractor, d_agent=agent.d_a, device=device)
        T, N = h_TN.shape[:2]
        for t in range(T):
            for n in range(N):
                if ms[t][n] is None:
                    continue
                g = describer_oracle(ms[t][n])
                if g is None:
                    continue
                H.append(h_TN[t, n]); CD.append(g.cheese_dir)
    H = torch.stack(H)
    print(f"[swap] {len(H):,} in-dist states; cheese_dir dist: "
          f"{ {d: CD.count(d) for d in set(CD)} }")

    # --- form pairs with different cheese_dir ---
    idx_by_cd = {}
    for i, d in enumerate(CD):
        idx_by_cd.setdefault(d, []).append(i)
    cds = list(idx_by_cd.keys())
    pairs = []
    tries = 0
    while len(pairs) < args.n_pairs and tries < args.n_pairs * 20:
        tries += 1
        da, db = rng.sample(cds, 2)
        a = rng.choice(idx_by_cd[da]); b = rng.choice(idx_by_cd[db])
        pairs.append((a, b))
    A = torch.stack([H[a] for a, _ in pairs])
    B = torch.stack([H[b] for _, b in pairs])
    cdA = [CD[a] for a, _ in pairs]
    cdB = [CD[b] for _, b in pairs]
    print(f"[swap] {len(pairs)} pairs (different cheese_dir)")

    alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
    builds = {"V2": (v2, "V2", v2.acc), "B4": (b4, "B4", None), "B3": (b3, "B3", None)}
    out = {"n_pairs": len(pairs), "alphas": alphas, "curve": {}, "swap_following": {}}
    for name, (mod, kind, acc_ref) in builds.items():
        # match-to-B fraction at each α
        curve = []
        per_alpha_cd = {}
        for al in alphas:
            h = ((1 - al) * A + al * B).to(device)
            cds_pred = []
            for i in range(0, h.shape[0], 2048):
                cds_pred.extend(_cheese_of(mod, h[i:i+2048], tokenizer, kind=kind, acc=acc_ref))
            per_alpha_cd[al] = cds_pred
            match_B = np.mean([cds_pred[j] == cdB[j] for j in range(len(pairs))])
            curve.append(float(match_B))
        out["curve"][name] = curve
        # swap-following: among pairs read correctly at α=0 (==A), frac that
        # flip to B at α=1.
        a0 = per_alpha_cd[0.0]; a1 = per_alpha_cd[1.0]
        read_A = [j for j in range(len(pairs)) if a0[j] == cdA[j]]
        follow = (np.mean([a1[j] == cdB[j] for j in read_A]) if read_A else float("nan"))
        out["swap_following"][name] = {"rate": float(follow), "n_readA": len(read_A)}
        print(f"  [{name}] match-B curve(α=0..1)={['%.3f'%c for c in curve]}  "
              f"swap-following={follow:.3f} (n_readA={len(read_A)})")

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[swap] → {args.output_path}")
    print("Read: swap-following V2 vs B4 (§5.6 ≥0.15). V2 high → causal "
          "tracking (비합리화=원칙). V2 low/flat → weak/noisy (confound = 약함).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
