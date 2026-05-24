"""Phase 5 CCM — step 0: frozen plumbing sanity (PLAN §10.1 Phase 5 CCM, CCM-4).

Before fitting any co-activation memory, confirm the injection+generation path
works on the *real* frozen backbones: load the Phase-3 agent + Phase-2 LM, build
a :class:`CCMBridge`, install an *unfit* W (random and, when d_lm==d_agent,
identity), and generate maze-language from real in-dist agent embeddings.

Pre-registered step0 PASS (no advantage over random required — this is a
plumbing check only):
  - **생성 비상수**: the bridge produces more than one distinct sentence
    (it has NOT collapsed to a constant), AND
  - **n_readA>0**: at least one generated cheese_dir matches the state's true
    cheese_dir (the pipeline emits valid, varied, parseable slots).

A PASS means step1 (`fit_ccm.py`) can fit the W ladder on this same plumbing.

Run (WSL, GPU):
  PYTHONPATH=src python scripts/ccm_sanity.py \\
      --lm_checkpoint checkpoints/lm.pt \\
      --agent_checkpoint checkpoints/phase3/agent.pt \\
      --device cuda --seed 0 --rollouts 5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from split_maze.agent import ImpalaAgent
from split_maze.ccm import CCMBridge
from split_maze.env import DEFAULT_HEADING_WINDOW, TrajectoryTracker, make_maze_env
from split_maze.language import describer_oracle, parse
from split_maze.lm import LMConfig, MazeLM, MazeTokenizer
from split_maze.ppo import RolloutBuffer
from split_maze.train import obs_to_tensor
from split_maze.train_phase3 import collect_rollout_with_pairs, default_state_extractor


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--lm_checkpoint", type=Path, default=Path("checkpoints/lm.pt"))
    p.add_argument("--agent_checkpoint", type=Path,
                   default=Path("checkpoints/phase3/agent.pt"))
    p.add_argument("--num_envs", type=int, default=64)
    p.add_argument("--num_steps", type=int, default=256)
    p.add_argument("--rollouts", type=int, default=5)
    p.add_argument("--w_scale", type=float, default=0.05)
    p.add_argument("--device", default=("cuda" if torch.cuda.is_available() else "cpu"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output_path", type=Path,
                   default=Path("results/phase5_ccm_sanity.json"))
    return p.parse_args()


def _fresh_lm(lm_blob, device):
    lm = MazeLM(LMConfig(**lm_blob["lm_config"]))
    lm.load_state_dict(lm_blob["model_state"])
    return lm.to(device).eval()


def _collect_indist(agent, *, num_envs, num_steps, rollouts, device, seed):
    """Collect in-dist (maze_aisc) agent embeddings + true cheese_dir."""
    env = make_maze_env(env_name="maze_aisc", num=num_envs, num_levels=0,
                        start_level=0, distribution_mode="easy", rand_seed=seed)
    rb = RolloutBuffer(T=num_steps, N=num_envs, device=device)
    trackers = [TrajectoryTracker(DEFAULT_HEADING_WINDOW) for _ in range(num_envs)]
    _r, obs_dict, _f = env.observe()
    obs_holder = obs_to_tensor(obs_dict, device)
    cur_rgb = np.asarray(obs_dict["rgb"])
    ep_r = np.zeros(num_envs); ep_l = np.zeros(num_envs, dtype=np.int64)
    H, CD = [], []
    for _ in range(rollouts):
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
    return torch.stack(H), CD


def _generate_slots(bridge, H, tokenizer, device, *, chunk=2048):
    """Generate per-state (sentence_string, cheese_dir) from the bridge."""
    sents, cds = [], []
    for i in range(0, H.shape[0], chunk):
        hb = H[i:i + chunk].to(device)
        with torch.no_grad():
            gen = bridge.generate(hb, max_len=16)
        for row in gen:
            toks = tokenizer.decode([int(x) for x in row.tolist()])
            sents.append(" ".join(toks))
            cds.append(parse(toks).cheese_dir)
    return sents, cds


def _eval_W(bridge, W, b, H, CD, tokenizer, device, *, label):
    bridge.set_W(W, b)
    sents, cds = _generate_slots(bridge, H, tokenizer, device)
    n = len(CD)
    n_distinct = len(set(sents))
    n_correct = int(sum(cds[i] == CD[i] for i in range(n)))
    varied = n_distinct > 1
    readA = n_correct > 0
    passed = bool(varied and readA)
    print(f"  [{label}] distinct_sentences={n_distinct}  "
          f"cheese_dir_correct={n_correct}/{n}  "
          f"varied={varied}  n_readA>0={readA}  → {'PASS' if passed else 'FAIL'}")
    return {"label": label, "n": n, "n_distinct_sentences": n_distinct,
            "n_cheese_correct": n_correct, "varied": varied,
            "n_readA_gt0": readA, "pass": passed}


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device(args.device)
    tokenizer = MazeTokenizer()
    lm_blob = torch.load(args.lm_checkpoint, map_location=device, weights_only=False)

    agent = ImpalaAgent().to(device)
    agent.load_state_dict(torch.load(args.agent_checkpoint, map_location=device,
                                     weights_only=False)["agent"])
    agent.eval()
    lm = _fresh_lm(lm_blob, device)
    bridge = CCMBridge(lm, d_agent=agent.d_a).to(device)
    d_model, d_agent = lm.config.d_model, agent.d_a
    print(f"[ccm_sanity] loaded agent (d_a={d_agent}) + LM (d_model={d_model})")

    H, CD = _collect_indist(agent, num_envs=args.num_envs, num_steps=args.num_steps,
                            rollouts=args.rollouts, device=device, seed=args.seed)
    print(f"[ccm_sanity] {len(H):,} in-dist states; cheese_dir dist: "
          f"{ {d: CD.count(d) for d in sorted(set(CD))} }")

    configs = {}
    # Random unfit W (matched small scale) + zero bias.
    Wr = torch.randn(d_model, d_agent, device=device) * args.w_scale
    configs["random"] = _eval_W(bridge, Wr, None, H, CD, tokenizer, device,
                                label="random")
    # Identity (only meaningful when dims match).
    if d_model == d_agent:
        configs["identity"] = _eval_W(bridge, torch.eye(d_model, device=device), None,
                                      H, CD, tokenizer, device, label="identity")

    overall = any(c["pass"] for c in configs.values())
    out = {"d_model": d_model, "d_agent": d_agent, "n_states": len(H),
           "configs": configs, "step0_pass": overall}
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[ccm_sanity] step0_pass={overall} → {args.output_path}")
    print("PASS ⇒ plumbing alive (varied + n_readA>0); proceed to step1 "
          "fit_ccm.py. FAIL ⇒ generation collapsed/broken — debug before fitting.")
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())
