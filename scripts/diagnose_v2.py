"""Phase 4.0 — V2 generation sanity diagnostic.

Phase 3 ended with V2 recon loss → 0.000 (POST-HOC-5). This script answers
two questions on the *fixed* Phase-3 V2 checkpoint, before building the full
Phase-4 measurement harness:

  Q1 (collapse vs alignment): does the ACC produce *varied* ĥ_lm across
      different agent states (alignment), or has it collapsed to a near-
      constant (degenerate, which would make recon=0 meaningless)?
      → reported via per-dim std of ĥ_lm / ñ_lm and per-state
        cosine(ĥ_lm, ñ_lm).

  Q2 (space mismatch): the ACC reconstructs ñ_lm = LayerNorm(h_lm), but
      MazeLM.generate was trained to decode from the *raw* interface_proj
      output h_lm (Phase-2 autoencoding). Does decode work in the LN space?
      → compares three generations per state:
          gen_raw  = generate(h_lm)            [Phase-2 autoenc path — sanity]
          gen_ln   = generate(ñ_lm)            [LN'd target — does LN decode?]
          gen_acc  = generate(ĥ_lm)            [V2's actual agent→language]
        and parses each to slots vs the oracle.

Run (WSL, GPU):
  PYTHONPATH=src python scripts/diagnose_v2.py \\
      --lm_checkpoint checkpoints/lm.pt \\
      --v2_checkpoint checkpoints/phase3/V2.pt \\
      --agent_checkpoint checkpoints/phase3/agent.pt \\
      --device cuda --seed 0 --n_states 24
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from split_maze.acc import ACC, ACCConfig
from split_maze.agent import ImpalaAgent
from split_maze.builds import V2ACC
from split_maze.env import TrajectoryTracker, extract_maze_state
from split_maze.language import describer_oracle, parse, render
from split_maze.lm import LMConfig, MazeLM, MazeTokenizer
from split_maze.train import obs_to_tensor


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--lm_checkpoint", type=Path, default=Path("checkpoints/lm.pt"))
    p.add_argument("--v2_checkpoint", type=Path,
                   default=Path("checkpoints/phase3/V2.pt"))
    p.add_argument("--agent_checkpoint", type=Path,
                   default=Path("checkpoints/phase3/agent.pt"))
    p.add_argument("--device", default=("cuda" if torch.cuda.is_available() else "cpu"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num_envs", type=int, default=16)
    p.add_argument("--rollout_steps", type=int, default=32)
    p.add_argument("--n_states", type=int, default=24,
                   help="how many collected states to print generations for.")
    p.add_argument("--heading_window", type=int, default=4)
    return p.parse_args()


def build_v2(lm_ckpt: Path, v2_ckpt: Path, tokenizer: MazeTokenizer,
             device: torch.device) -> V2ACC:
    lm_blob = torch.load(lm_ckpt, map_location=device, weights_only=False)
    lm_cfg = LMConfig(**lm_blob["lm_config"])
    lm = MazeLM(lm_cfg)
    acc = ACC(ACCConfig(d_agent=256, d_lm=lm_cfg.d_model, tied=False))  # POST-HOC-7
    v2 = V2ACC(lm, acc)
    state = torch.load(v2_ckpt, map_location=device, weights_only=False)["state"]
    v2.load_state_dict(state)
    return v2.to(device).eval()


def collect_states(env_name: str, agent: ImpalaAgent, tokenizer: MazeTokenizer,
                   *, num_envs: int, steps: int, window: int,
                   device: torch.device, seed: int):
    """Roll out a fixed agent and collect (h_agent, oracle Slots, ids) for
    states where the describer oracle is defined."""
    from split_maze.env import make_maze_env
    env = make_maze_env(env_name=env_name, num=num_envs,
                        num_levels=0, start_level=0,
                        distribution_mode="easy", rand_seed=seed)
    trackers = [TrajectoryTracker(window) for _ in range(num_envs)]
    _r, obs_dict, _f = env.observe()
    cur_rgb = np.asarray(obs_dict["rgb"])
    obs_holder = obs_to_tensor(obs_dict, device)

    h_list, slot_list, id_list = [], [], []
    rng = random.Random(seed)
    for _ in range(steps):
        with torch.no_grad():
            out = agent(obs_holder)
            logits = out.logits
            action = torch.distributions.Categorical(logits=logits).sample()
        for n in range(num_envs):
            res = extract_maze_state(cur_rgb[n], trackers[n])
            ms = res.maze_state
            if ms is None:
                continue
            slots = describer_oracle(ms)
            if slots is None:
                continue
            tokens = render(slots, rng=rng, include_bos_eos=True)
            ids = tokenizer.encode(tokens)
            h_list.append(out.h_agent[n].detach().cpu())
            slot_list.append(slots)
            id_list.append(ids)
        env.act(action.detach().cpu().numpy().astype(np.int32))
        _r, obs_dict, _f = env.observe()
        cur_rgb = np.asarray(obs_dict["rgb"])
        obs_holder = obs_to_tensor(obs_dict, device)

    return h_list, slot_list, id_list


def analyze(tag: str, v2: V2ACC, tokenizer: MazeTokenizer,
            h_list, slot_list, id_list, *, n_print: int, device):
    if not h_list:
        print(f"\n[{tag}] NO valid states collected.")
        return
    h_agent = torch.stack(h_list).to(device)              # (S, d_a)
    ids = tokenizer.collate(id_list, device=device)        # (S, T)

    with torch.no_grad():
        n_agent = v2.acc.ln_agent(h_agent)                 # ñ_agent (S, d_a)
        hat_lm = v2.acc.predict_lm_from_agent(n_agent)     # ĥ_lm  (S, d_lm) LN-space
        h_lm = v2.lm.encode(ids)                           # raw   (S, d_lm)
        n_lm = v2.acc.ln_lm(h_lm)                           # ñ_lm  (S, d_lm) LN-space

    S = h_agent.shape[0]
    # --- Q1: collapse vs alignment ---
    std_hat = hat_lm.std(dim=0).mean().item()
    std_nlm = n_lm.std(dim=0).mean().item()
    std_hag = h_agent.std(dim=0).mean().item()
    cos = F.cosine_similarity(hat_lm, n_lm, dim=-1)
    print(f"\n[{tag}] states={S}")
    print(f"  h_agent per-dim std (avg) = {std_hag:.4f}  "
          f"(low → agent hidden near-constant)")
    print(f"  ĥ_lm   per-dim std (avg) = {std_hat:.4f}  "
          f"(low → ACC output collapsed)")
    print(f"  ñ_lm   per-dim std (avg) = {std_nlm:.4f}  "
          f"(low → LM encodings collapsed)")
    print(f"  cosine(ĥ_lm, ñ_lm): mean={cos.mean():.4f} "
          f"min={cos.min():.4f} max={cos.max():.4f}  "
          f"(high+varied → meaningful alignment)")

    # --- Q2: generation in three spaces ---
    k = min(n_print, S)
    print(f"  -- generations (first {k}) : oracle | gen_raw(h_lm) | "
          f"gen_ln(ñ_lm) | gen_acc(ĥ_lm) --")
    with torch.no_grad():
        gen_raw = v2.lm.generate(h_lm[:k], max_len=16)
        gen_ln = v2.lm.generate(n_lm[:k], max_len=16)
        gen_acc = v2.lm.generate(hat_lm[:k], max_len=16)

    def slots_of(id_row):
        toks = tokenizer.decode([int(x) for x in id_row.tolist()])
        ps = parse(toks)
        return (ps.agent_region, ps.heading, ps.cheese_dir)

    for i in range(k):
        gold = slot_list[i]
        gold_t = ((gold.agent_row, gold.agent_col), gold.heading, gold.cheese_dir)
        print(f"   [{i}] oracle={gold_t}")
        print(f"        raw={slots_of(gen_raw[i])}")
        print(f"        ln ={slots_of(gen_ln[i])}")
        print(f"        acc={slots_of(gen_acc[i])}")


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    tokenizer = MazeTokenizer()
    v2 = build_v2(args.lm_checkpoint, args.v2_checkpoint, tokenizer, device)

    agent = ImpalaAgent().to(device)
    agent.load_state_dict(
        torch.load(args.agent_checkpoint, map_location=device,
                   weights_only=False)["agent"])
    agent.eval()
    print(f"[diagnose_v2] device={device}  loaded agent + V2 (lm+acc)")

    for tag, env_name in (("IN-DIST maze_aisc", "maze_aisc"),
                          ("OOD maze", "maze")):
        h_list, slot_list, id_list = collect_states(
            env_name, agent, tokenizer,
            num_envs=args.num_envs, steps=args.rollout_steps,
            window=args.heading_window, device=device, seed=args.seed)
        analyze(tag, v2, tokenizer, h_list, slot_list, id_list,
                n_print=args.n_states, device=device)

    print("\n[diagnose_v2] done. Read: (Q1) is ĥ_lm varied + cosine high "
          "→ alignment, not collapse? (Q2) does gen_ln/gen_acc match oracle "
          "→ LN-space decode works?")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
