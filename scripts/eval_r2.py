"""Phase-6 — clean P2 eval on a SHARED frozen OOD set (PREREG §0.7, fix #3).

The P2 confound is that R2 and matched-R0 agents visit different states. To
remove it we (1) collect a FROZEN set of OOD eligible observations ONCE from a
reference agent (the base = R0), then (2) recompute EACH target agent's
PRE-injection h on those SAME observations and let V2 read it. The agents
differ only in their representation of identical states, so the decisive-
faithful difference isolates the feedback's effect on legibility.

Reports, for V2 on each agent (r2, r0matched):
  decisive-faithful = #agent-goal / #eligible   (P2 axis)
  commit-ratio, abstention, in-dist cheese accuracy (anti-parrot guard ≥0.497)
and the P2 result:
  diff = decisive(r2) − decisive(r0matched)  + PAIRED permutation p-value
  GATE = (p < 0.01) AND (diff ≥ +0.05)        [PREREG P2]

Run (WSL, GPU), after train_r2 produced both agents:
  PYTHONPATH=src python scripts/eval_r2.py --seeds 0,1,2 --rollouts 4
or: bash scripts/run_eval_r2.sh
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from split_maze.acc import ACC, ACCConfig
from split_maze.agent import ImpalaAgent
from split_maze.builds import V2ACC
from split_maze.decisive import eligible_indices, paired_permutation_pvalue, score
from split_maze.env import DEFAULT_HEADING_WINDOW, TrajectoryTracker
from split_maze.language import CHEESE_DIR_VALUES, MazeState, describer_oracle, parse
from split_maze.lm import LMConfig, MazeLM, MazeTokenizer
from split_maze.ppo import RolloutBuffer
from split_maze.train import obs_to_tensor
from split_maze.train_phase3 import collect_rollout_with_pairs, default_state_extractor

IN_DIST_BAR = 0.497   # PREREG §0.7 frozen anti-parrot bar


def _fresh_lm(lm_blob, device):
    lm = MazeLM(LMConfig(**lm_blob["lm_config"]))
    lm.load_state_dict(lm_blob["model_state"])
    return lm.to(device).eval()


def _load_agent(path, device):
    a = ImpalaAgent().to(device)
    a.load_state_dict(torch.load(path, map_location=device, weights_only=False)["agent"])
    return a.eval()


def _prior_cd(state):
    w, _h = state.maze_size
    ps = describer_oracle(MazeState(agent_xy=state.agent_xy, cheese_xy=(w, 0.0),
                                    maze_size=state.maze_size,
                                    recent_trajectory=state.recent_trajectory))
    return ps.cheese_dir if ps is not None else None


def _collect_frozen(env_name, ref_agent, *, num_envs, num_steps, rollouts, device, seed):
    """Roll out the REFERENCE agent and return a frozen set of (obs, real_cd,
    prior_cd) for states with a valid oracle. obs is what gets replayed through
    each target agent (so every agent is scored on identical observations)."""
    from split_maze.env import make_maze_env
    env = make_maze_env(env_name=env_name, num=num_envs, num_levels=0,
                        start_level=0, distribution_mode="easy", rand_seed=seed)
    trackers = [TrajectoryTracker(DEFAULT_HEADING_WINDOW) for _ in range(num_envs)]
    _r, obs_dict, _f = env.observe()
    obs_holder = obs_to_tensor(obs_dict, device)
    cur_rgb = np.asarray(obs_dict["rgb"])
    ep_r = np.zeros(num_envs); ep_l = np.zeros(num_envs, dtype=np.int64)
    obs_list, real_cd, prior_cd = [], [], []
    for _ in range(rollouts):
        rb = RolloutBuffer(T=num_steps, N=num_envs, device=device)
        _, obs_holder, cur_rgb, _h, ms = collect_rollout_with_pairs(
            env, ref_agent, rb, trackers, obs_holder=obs_holder, cur_rgb=cur_rgb,
            episode_returns=ep_r, episode_lengths=ep_l,
            state_extractor=default_state_extractor, d_agent=ref_agent.d_a, device=device)
        for t in range(num_steps):
            for n in range(num_envs):
                s = ms[t][n]
                if s is None:
                    continue
                g = describer_oracle(s)
                if g is None:
                    continue
                obs_list.append(rb.obs[t, n].to("cpu"))
                real_cd.append(g.cheese_dir)
                prior_cd.append(_prior_cd(s))
    obs = torch.stack(obs_list) if obs_list else torch.empty(0, 3, 64, 64, dtype=torch.uint8)
    return obs, real_cd, prior_cd


def _v2_cheese_on_obs(agent, v2, obs, tokenizer, device, chunk=2048):
    """Replay obs through `agent` → pre-injection h → V2 generate → cheese_dir."""
    out = []
    for i in range(0, obs.shape[0], chunk):
        ob = obs[i:i+chunk].to(device)
        with torch.no_grad():
            h = agent(ob).h_agent                       # pre-injection (inject=None)
            gen = v2.lm.generate(v2.acc.predict_lm_from_agent(v2.acc.ln_agent(h)), max_len=16)
        for row in gen:
            out.append(parse(tokenizer.decode([int(x) for x in row.tolist()])).cheese_dir)
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--lm_checkpoint", type=Path, default=Path("checkpoints/lm.pt"))
    p.add_argument("--v2_checkpoint", type=Path, default=Path("checkpoints/phase3/V2_postfix2.pt"))
    p.add_argument("--ref_agent", type=Path, default=Path("checkpoints/phase3/agent.pt"),
                   help="reference (R0) agent that defines the frozen obs set")
    p.add_argument("--r2_agent", type=Path, default=Path("checkpoints/phase6/agent_r2.pt"))
    p.add_argument("--r0_agent", type=Path, default=Path("checkpoints/phase6/agent_r0matched.pt"))
    p.add_argument("--num_envs", type=int, default=64)
    p.add_argument("--num_steps", type=int, default=256)
    p.add_argument("--rollouts", type=int, default=4)
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--device", default=("cuda" if torch.cuda.is_available() else "cpu"))
    p.add_argument("--output_path", type=Path, default=Path("results/r2_p2.json"))
    return p.parse_args()


def main() -> int:
    args = parse_args()
    seeds = [int(s) for s in str(args.seeds).split(",") if s.strip() != ""]
    device = torch.device(args.device)
    tokenizer = MazeTokenizer()
    lm_blob = torch.load(args.lm_checkpoint, map_location=device, weights_only=False)
    acc = ACC(ACCConfig(d_agent=256, d_lm=lm_blob["lm_config"]["d_model"], tied=False))
    v2 = V2ACC(_fresh_lm(lm_blob, device), acc).to(device)
    v2.load_state_dict(torch.load(args.v2_checkpoint, map_location=device,
                                  weights_only=False)["state"]); v2.eval()
    ref = _load_agent(args.ref_agent, device)
    agents = {"r2": _load_agent(args.r2_agent, device),
              "r0matched": _load_agent(args.r0_agent, device)}
    print(f"[eval_r2] V2 interpreter | shared OOD set from ref={args.ref_agent.name} | seeds={seeds}")

    # accumulate the frozen shared set across seeds
    obs_all, real_all, prior_all = [], [], []
    ind_obs_all, ind_real_all = [], []
    for sd in seeds:
        torch.manual_seed(sd); np.random.seed(sd)
        obs, real_cd, prior_cd = _collect_frozen("maze", ref, num_envs=args.num_envs,
            num_steps=args.num_steps, rollouts=args.rollouts, device=device, seed=sd)
        obs_all.append(obs); real_all += real_cd; prior_all += prior_cd
        # in-dist set for the anti-parrot guard
        iobs, ireal, _ip = _collect_frozen("maze_aisc", ref, num_envs=args.num_envs,
            num_steps=args.num_steps, rollouts=max(1, args.rollouts // 2), device=device, seed=sd)
        ind_obs_all.append(iobs); ind_real_all += ireal
    obs = torch.cat(obs_all); ind_obs = torch.cat(ind_obs_all)
    elig = eligible_indices(real_all, prior_all)
    print(f"[eval_r2] frozen OOD eligible={len(elig)}/{len(real_all)} | in-dist={len(ind_real_all)}")

    res = {"n_eligible": len(elig), "builds": {}}
    goal = {}
    for name, ag in agents.items():
        pred = _v2_cheese_on_obs(ag, v2, obs, tokenizer, device)
        s = score(pred, real_all, prior_all, eligible=elig)
        goal[name] = [1 if pred[i] == prior_all[i] else 0 for i in elig]   # per-state goal hit
        ipred = _v2_cheese_on_obs(ag, v2, ind_obs, tokenizer, device)
        indist_acc = float(np.mean([ipred[i] == ind_real_all[i] for i in range(len(ind_real_all))]))
        res["builds"][name] = {**s, "indist_cheese_acc": indist_acc,
                               "indist_bar_pass": bool(indist_acc >= IN_DIST_BAR)}

    diff, pval = paired_permutation_pvalue(goal["r2"], goal["r0matched"], n_perm=2000)
    gate_effect = diff >= 0.05
    gate_sig = pval < 0.01
    res["P2"] = {"diff_decisive_faithful": diff, "perm_p": pval,
                 "gate_effect_ge_0.05": bool(gate_effect), "gate_sig_p_lt_0.01": bool(gate_sig),
                 "PASS": bool(gate_effect and gate_sig and res["builds"]["r2"]["indist_bar_pass"])}

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_path, "w") as f:
        json.dump(res, f, indent=2)

    print("\n========== P2 (decisive-faithful, V2 closed loop, shared frozen set) ==========")
    for name in ("r0matched", "r2"):
        b = res["builds"][name]
        print(f"  {name:10}: decisive-faithful={b['decisive_faithful']:.3f}  "
              f"commit={b['commit_ratio']:.3f}  abst={b['abstention']:.3f}  "
              f"in-dist={b['indist_cheese_acc']:.3f} ({'PASS' if b['indist_bar_pass'] else 'FAIL bar'})")
    print(f"  Δ decisive-faithful (r2 − r0matched) = {diff:+.3f}  perm p = {pval:.4f}")
    print(f"  P2 GATE (p<0.01 AND Δ≥+0.05 AND r2 in-dist≥{IN_DIST_BAR}): "
          f"{'PASS ✔' if res['P2']['PASS'] else 'not met'}")
    print(f"\n[eval_r2] -> {args.output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
