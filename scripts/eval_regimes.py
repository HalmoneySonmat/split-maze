"""Phase-6 — R0 baseline + echo diagnostic (PREREG §3, no retraining).

Computes the decisive-faithful / commit-ratio / abstention metrics for the
FROZEN agent (regime R0 = post-hoc read-only) on OOD eligible states, per
interpreter build (B3/B4/V2). This is the baseline R2 must beat; it should
reproduce the pilot (B4 decisive-faithful ≈ 0.50, commit-ratio ≈ 0.98).

R1 (online one-way) ≡ R0 by construction: with no feedback the agent's
trajectory and pre-injection h are identical, and the report is a function of
h, so the metric is identical. We therefore do not run R1 separately — the
{R0 ≈ R1} control is satisfied by construction (PREREG §1, fix #1).

Also reports the ECHO DIAGNOSTIC (PREREG §0.5 메아리 체크): mean cosine between
the LM-processed feedback and the bare bridge round-trip. ~1 ⇒ the R2 feedback
would just re-inject the agent's own hidden (echo, likely null R2); < 1 ⇒ the
LM's interpretation diverges (feedback carries something). Computed now, before
spending GPU on R2 training.

Run (WSL, GPU):
  bash scripts/run_eval_regimes.sh
or:
  PYTHONPATH=src python scripts/eval_regimes.py --seeds 0,1,2 --rollouts 8
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from split_maze.acc import ACC, ACCConfig
from split_maze.agent import ImpalaAgent
from split_maze.builds import B3Probe, B4Adapter, V2ACC
from split_maze.decisive import eligible_indices, score
from split_maze.env import DEFAULT_HEADING_WINDOW, TrajectoryTracker
from split_maze.feedback import echo_ratio
from split_maze.language import (
    CHEESE_DIR_VALUES, HEADING_VALUES, MazeState, REGION_COLS, REGION_ROWS,
    describer_oracle, parse,
)
from split_maze.lm import LMConfig, MazeLM, MazeTokenizer
from split_maze.ppo import RolloutBuffer
from split_maze.train import obs_to_tensor
from split_maze.train_phase3 import collect_rollout_with_pairs, default_state_extractor


# --- helpers (copied from pilot_grounding/eval_builds for an exact match) ---

def _fresh_lm(lm_blob, device):
    lm = MazeLM(LMConfig(**lm_blob["lm_config"]))
    lm.load_state_dict(lm_blob["model_state"])
    return lm.to(device).eval()


def _collect(env_name, agent, *, num_envs, num_steps, rollouts, device, seed):
    from split_maze.env import make_maze_env
    env = make_maze_env(env_name=env_name, num=num_envs, num_levels=0,
                        start_level=0, distribution_mode="easy", rand_seed=seed)
    rb = RolloutBuffer(T=num_steps, N=num_envs, device=device)
    trackers = [TrajectoryTracker(DEFAULT_HEADING_WINDOW) for _ in range(num_envs)]
    _r, obs_dict, _f = env.observe()
    obs_holder = obs_to_tensor(obs_dict, device)
    cur_rgb = np.asarray(obs_dict["rgb"])
    ep_r = np.zeros(num_envs); ep_l = np.zeros(num_envs, dtype=np.int64)
    H, states = [], []
    for _ in range(rollouts):
        _, obs_holder, cur_rgb, h_TN, ms = collect_rollout_with_pairs(
            env, agent, rb, trackers, obs_holder=obs_holder, cur_rgb=cur_rgb,
            episode_returns=ep_r, episode_lengths=ep_l,
            state_extractor=default_state_extractor, d_agent=agent.d_a, device=device)
        T, N = h_TN.shape[:2]
        for t in range(T):
            for n in range(N):
                if ms[t][n] is None or describer_oracle(ms[t][n]) is None:
                    continue
                H.append(h_TN[t, n]); states.append(ms[t][n])
    return (torch.stack(H) if H else torch.empty(0, agent.d_a)), states


def _slots_from_ids(gen, tokenizer):
    return [parse(tokenizer.decode([int(x) for x in row.tolist()])).cheese_dir
            for row in gen]


def _b3_cheese(probe, h, device, chunk=4096):
    out = []
    for i in range(0, h.shape[0], chunk):
        with torch.no_grad():
            o = probe(h[i:i+chunk].to(device))
        ch = o["cheese"].argmax(-1)
        out.extend(CHEESE_DIR_VALUES[int(j)] for j in ch)
    return out


def _gen_cheese(build, h, tokenizer, device, *, is_v2, acc=None, chunk=2048):
    out = []
    for i in range(0, h.shape[0], chunk):
        hb = h[i:i+chunk].to(device)
        with torch.no_grad():
            if is_v2:
                gen = build.lm.generate(acc.predict_lm_from_agent(acc.ln_agent(hb)), max_len=16)
            else:
                gen = build.generate(hb, max_len=16)
        out.extend(_slots_from_ids(gen, tokenizer))
    return out


def _prior_cd(state):
    w, _h = state.maze_size
    ps = describer_oracle(MazeState(agent_xy=state.agent_xy, cheese_xy=(w, 0.0),
                                    maze_size=state.maze_size,
                                    recent_trajectory=state.recent_trajectory))
    return ps.cheese_dir if ps is not None else None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--lm_checkpoint", type=Path, default=Path("checkpoints/lm.pt"))
    p.add_argument("--agent_checkpoint", type=Path, default=Path("checkpoints/phase3/agent.pt"))
    p.add_argument("--v2_checkpoint", type=Path, default=Path("checkpoints/phase3/V2_postfix2.pt"))
    p.add_argument("--b4_checkpoint", type=Path, default=Path("checkpoints/phase3/B4.pt"))
    p.add_argument("--b3_checkpoint", type=Path, default=Path("checkpoints/phase3/B3.pt"))
    p.add_argument("--num_envs", type=int, default=64)
    p.add_argument("--num_steps", type=int, default=256)
    p.add_argument("--rollouts", type=int, default=8)
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--device", default=("cuda" if torch.cuda.is_available() else "cpu"))
    p.add_argument("--output_path", type=Path, default=Path("results/regimes_baseline.json"))
    return p.parse_args()


def main() -> int:
    args = parse_args()
    seeds = [int(s) for s in str(args.seeds).split(",") if s.strip() != ""]
    device = torch.device(args.device)
    tokenizer = MazeTokenizer()
    lm_blob = torch.load(args.lm_checkpoint, map_location=device, weights_only=False)

    agent = ImpalaAgent().to(device)
    agent.load_state_dict(torch.load(args.agent_checkpoint, map_location=device,
                                     weights_only=False)["agent"]); agent.eval()
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
    print(f"[eval_regimes] loaded agent + V2 + B4 + B3 | R0 baseline | seeds={seeds}")

    builds = ("B3", "B4", "V2")
    acc_counts = {b: {"n_eligible": 0, "n_goal": 0, "n_real": 0, "n_neither": 0}
                  for b in builds}
    echo_vals = []

    for sd in seeds:
        torch.manual_seed(sd); np.random.seed(sd)
        H, states = _collect("maze", agent, num_envs=args.num_envs,
                            num_steps=args.num_steps, rollouts=args.rollouts,
                            device=device, seed=sd)
        real_cd = [describer_oracle(s).cheese_dir for s in states]
        prior_cd = [_prior_cd(s) for s in states]
        elig = eligible_indices(real_cd, prior_cd)
        preds = {
            "B3": _b3_cheese(b3, H, device),
            "B4": _gen_cheese(b4, H, tokenizer, device, is_v2=False),
            "V2": _gen_cheese(v2, H, tokenizer, device, is_v2=True, acc=v2.acc),
        }
        print(f"  [seed {sd}] OOD eligible={len(elig)}/{len(states)}")
        for b in builds:
            s = score(preds[b], real_cd, prior_cd, eligible=elig)
            for k in acc_counts[b]:
                acc_counts[b][k] += s[k]
        # echo diagnostic on this seed's hidden states (uses V2's bridge + LM)
        for i in range(0, H.shape[0], 4096):
            echo_vals.append(echo_ratio(v2.acc, v2.lm, H[i:i+4096].to(device)).cpu())

    echo = torch.cat(echo_vals) if echo_vals else torch.empty(0)
    out = {"regime": "R0", "seeds": seeds, "rollouts": args.rollouts,
           "echo_ratio_mean": float(echo.mean()) if echo.numel() else float("nan"),
           "echo_ratio_std": float(echo.std()) if echo.numel() else float("nan"),
           "builds": {}}
    for b in builds:
        c = acc_counts[b]; n = c["n_eligible"]; commit = c["n_goal"] + c["n_real"]
        out["builds"][b] = {
            **c,
            "decisive_faithful": (c["n_goal"] / n) if n else float("nan"),
            "commit_ratio": (c["n_goal"] / commit) if commit else float("nan"),
            "abstention": (c["n_neither"] / n) if n else float("nan"),
        }

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_path, "w") as f:
        json.dump(out, f, indent=2)

    print("\n========== R0 BASELINE (decisive-faithful = P2 axis) ==========")
    for b in builds:
        bb = out["builds"][b]
        print(f"  {b}: decisive-faithful={bb['decisive_faithful']:.3f}  "
              f"commit-ratio={bb['commit_ratio']:.3f}  abstention={bb['abstention']:.3f}")
    print(f"  echo-ratio (feedback vs bridge round-trip): "
          f"{out['echo_ratio_mean']:.3f} ± {out['echo_ratio_std']:.3f}")
    print(f"\n[eval_regimes] -> {args.output_path}")
    print("Read: R0 = baseline R2 must beat on decisive-faithful (pilot B4 ≈ 0.50). "
          "echo≈1 ⇒ feedback would be an echo (R2 likely null); <1 ⇒ carries interpretation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
