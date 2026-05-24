"""Phase 5 CCM — step 2: closed-loop fine-tune (PLAN §10.1 Phase 5 CCM, CCM-4 step2).

"The bridge grows": from the frozen Phase-3 backbones, run a *short* closed loop
where the **sender brain (agent)** adapts so that the *recorded* co-activation
correspondence W lets the *frozen* LM decoder read it. PPO keeps the agent on the
maze task (anchor / collapse guard); W is recorded via EMA (a buffer, never a
gradient target); the LM core stays fully frozen (the receiver is protected, as
throughout Phase 3 — and interface_proj is not even on the decode path).

Per update:
  1. augmented rollout (agent acts) → RolloutBuffer + h_agent (T,N) + maze_states.
  2. PPO update on the agent (pure RL — task anchor).
  3. EMA-record moments from (LN(h_agent.detach()), lm.encode(oracle_ids)) over
     the rollout's valid states; refit ridge W every --refit_every (after warmup).
  4. once W is set: a few bridge steps — sample (obs, ids), re-forward the agent
     WITH grad, ĥ_lm = W·LN(h_agent) (W buffer), next-token CE via decode_logits,
     backprop into the AGENT only (small --bridge_lr); W & LM get no gradient.

Collapse guards (loss↓≠success): PPO is the dominant signal (small bridge_lr);
rolling return is logged each update (a crater ⇒ the bridge is destroying the
task — flagged, not a success). The real before/after verdict comes from re-running
fit_ccm.py on the adapted agent (per-slot AND swap), not from the bridge loss.

Output: an adapted agent checkpoint. Compare via:
  fit_ccm.py --agent_checkpoint checkpoints/phase3/agent.pt      (before / step1)
  fit_ccm.py --agent_checkpoint checkpoints/phase5/agent_ccm.pt  (after / step2)

Run (WSL, GPU) — start with a tiny smoke (--updates 3), then a short run:
  PYTHONPATH=src python scripts/train_ccm_step2.py \\
      --lm_checkpoint checkpoints/lm.pt \\
      --agent_checkpoint checkpoints/phase3/agent.pt \\
      --device cuda --seed 0 --updates 100
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from split_maze.agent import ImpalaAgent
from split_maze.ccm import CCMBridge, CoActAccumulator, fit_W
from split_maze.env import DEFAULT_HEADING_WINDOW, TrajectoryTracker, make_maze_env
from split_maze.language import Slots, describer_oracle, render
from split_maze.lm import LMConfig, MazeLM, MazeTokenizer
from split_maze.ppo import PPOConfig, PPOUpdater, RolloutBuffer
from split_maze.train import obs_to_tensor
from split_maze.train_phase3 import collect_rollout_with_pairs, default_state_extractor


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--lm_checkpoint", type=Path, default=Path("checkpoints/lm.pt"))
    p.add_argument("--agent_checkpoint", type=Path,
                   default=Path("checkpoints/phase3/agent.pt"))
    p.add_argument("--num_envs", type=int, default=64)
    p.add_argument("--num_steps", type=int, default=256)
    p.add_argument("--updates", type=int, default=100)
    p.add_argument("--bridge_lr", type=float, default=1e-5,
                   help="v2 default 1e-5 (v1 used 1e-4 and crashed the task)")
    p.add_argument("--bridge_steps", type=int, default=4,
                   help="agent bridge-loss steps per RL update")
    p.add_argument("--bridge_batch", type=int, default=512)
    p.add_argument("--freeze_conv", type=int, default=1,
                   help="v2: freeze the IMPALA conv stack (preserve vision); adapt embed only")
    p.add_argument("--anchor_lambda", type=float, default=1.0,
                   help="v2: L2 pull of h_agent toward the original (frozen) agent (task anchor)")
    p.add_argument("--refit_every", type=int, default=10)
    p.add_argument("--ema_decay", type=float, default=0.99)
    p.add_argument("--warmup_updates", type=int, default=5,
                   help="EMA-record only (no bridge loss) for this many updates")
    p.add_argument("--device", default=("cuda" if torch.cuda.is_available() else "cpu"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out_agent", type=Path,
                   default=Path("checkpoints/phase5/agent_ccm.pt"))
    p.add_argument("--log_path", type=Path, default=Path("logs/phase5_ccm_step2.jsonl"))
    return p.parse_args()


def _fresh_lm(lm_blob, device):
    lm = MazeLM(LMConfig(**lm_blob["lm_config"]))
    lm.load_state_dict(lm_blob["model_state"])
    return lm.to(device).eval()


def _valid_pairs(h_TN, maze_states, tokenizer, rng):
    """From a rollout, gather valid (t, n) indices, padded oracle ids, lengths,
    and the detached h_agent rows for those states.

    Returns (tn_idx (M,2) long, ids (M,T) long, lengths (M,) long, h (M,d_a))."""
    T = len(maze_states)
    N = len(maze_states[0]) if T else 0
    tn, ids_list, lens, hs = [], [], [], []
    for t in range(T):
        for n in range(N):
            s = maze_states[t][n]
            if s is None:
                continue
            g = describer_oracle(s)
            if g is None:
                continue
            slots = Slots(agent_row=g.agent_row, agent_col=g.agent_col,
                          heading=g.heading, cheese_dir=g.cheese_dir)
            ids = tokenizer.encode(render(slots, rng=rng, include_bos_eos=True))
            tn.append((t, n)); ids_list.append(ids); lens.append(len(ids))
            hs.append(h_TN[t, n])
    if not tn:
        return (torch.empty(0, 2, dtype=torch.long), torch.empty(0, 0, dtype=torch.long),
                torch.empty(0, dtype=torch.long), torch.empty(0, h_TN.shape[-1]))
    Tmax = max(lens)
    ids = torch.full((len(ids_list), Tmax), tokenizer.pad_id, dtype=torch.long)
    for i, x in enumerate(ids_list):
        ids[i, :len(x)] = torch.tensor(x, dtype=torch.long)
    return (torch.tensor(tn, dtype=torch.long), ids,
            torch.tensor(lens, dtype=torch.long), torch.stack(hs))


@torch.no_grad()
def _refit(acc, bridge, device):
    """Refit ridge W from current EMA moments → install into the bridge."""
    W, b = fit_W(acc.moments(), "ridge")
    bridge.set_W(W.to(device), b.to(device))


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    rng = random.Random(args.seed)
    surface_rng = random.Random(args.seed + 1)
    device = torch.device(args.device)
    gen_dev = torch.Generator(device=device).manual_seed(args.seed)   # PPO (buffer device)
    gen_cpu = torch.Generator(device="cpu").manual_seed(args.seed)     # bridge sampling
    tokenizer = MazeTokenizer()
    lm_blob = torch.load(args.lm_checkpoint, map_location=device, weights_only=False)

    ckpt = torch.load(args.agent_checkpoint, map_location=device,
                      weights_only=False)["agent"]
    agent = ImpalaAgent().to(device)
    agent.load_state_dict(ckpt)
    agent.train()
    d_agent, d_model = agent.d_a, lm_blob["lm_config"]["d_model"]

    # v2 task-preservation: freeze the conv stack (vision fixed), adapt embed only.
    if args.freeze_conv:
        for p in agent.blocks.parameters():
            p.requires_grad_(False)
    # Frozen reference agent for the L2 task anchor (||h - h_ref||²).
    frozen_ref = ImpalaAgent().to(device)
    frozen_ref.load_state_dict(ckpt)
    frozen_ref.eval()
    for p in frozen_ref.parameters():
        p.requires_grad_(False)

    bridge = CCMBridge(_fresh_lm(lm_blob, device), d_agent=d_agent).to(device)
    acc = CoActAccumulator(d_agent, d_model, decay=args.ema_decay, device=device)
    trainable = [p for p in agent.parameters() if p.requires_grad]
    bridge_opt = torch.optim.AdamW(trainable, lr=args.bridge_lr,
                                   betas=(0.9, 0.95), weight_decay=0.0)
    print(f"[step2-v2] agent d_a={d_agent} (conv frozen={bool(args.freeze_conv)}) + LM "
          f"d_model={d_model} (frozen) | bridge_lr={args.bridge_lr} "
          f"anchor_λ={args.anchor_lambda} ema={args.ema_decay} refit_every={args.refit_every}")

    # ---- env + PPO ----
    env = make_maze_env(env_name="maze_aisc", num=args.num_envs, num_levels=0,
                        start_level=0, distribution_mode="easy", rand_seed=args.seed)
    rb = RolloutBuffer(T=args.num_steps, N=args.num_envs, device=device)
    ppo_cfg = PPOConfig()
    updater = PPOUpdater(agent, ppo_cfg)
    trackers = [TrajectoryTracker(DEFAULT_HEADING_WINDOW) for _ in range(args.num_envs)]
    _r, obs_dict, _f = env.observe()
    obs_holder = obs_to_tensor(obs_dict, device)
    cur_rgb = np.asarray(obs_dict["rgb"])
    ep_r = np.zeros(args.num_envs); ep_l = np.zeros(args.num_envs, dtype=np.int64)
    from collections import deque
    rolling = deque(maxlen=64)

    args.log_path.parent.mkdir(parents=True, exist_ok=True)
    logf = open(args.log_path, "w")
    w_ready = False
    baseline_ret = None
    EC = 4096  # encode chunk

    for upd in range(args.updates):
        stats, obs_holder, cur_rgb, h_TN, maze_states = collect_rollout_with_pairs(
            env, agent, rb, trackers, obs_holder=obs_holder, cur_rgb=cur_rgb,
            episode_returns=ep_r, episode_lengths=ep_l,
            state_extractor=default_state_extractor, d_agent=d_agent, device=device)
        rolling.extend(stats.completed_returns)
        roll_ret = float(np.mean(rolling)) if rolling else float("nan")
        if baseline_ret is None and rolling:
            baseline_ret = roll_ret

        # PPO (RL anchor) — pure task signal on the agent.
        with torch.no_grad():
            last_value = agent(obs_holder).value
        rb.compute_advantages_and_returns(last_value, gamma=ppo_cfg.gamma,
                                          gae_lambda=ppo_cfg.gae_lambda)
        ppo_log = updater.update(rb, generator=gen_dev)

        # Gather this rollout's valid (obs,ids,h) pairs.
        tn, ids, lens, h_valid = _valid_pairs(h_TN, maze_states, tokenizer, surface_rng)
        n_valid = tn.shape[0]

        # EMA-record moments (LN(h_agent.detach()), lm.encode(ids)).
        if n_valid > 0:
            x = bridge.ln_agent(h_valid.to(device))
            a2 = []
            with torch.no_grad():
                for i in range(0, n_valid, EC):
                    a2.append(bridge.lm.encode(ids[i:i + EC].to(device)))
            a2 = torch.cat(a2, dim=0)
            acc.update(x, a2)

        # Refit W after warmup.
        if n_valid > 0 and upd >= args.warmup_updates and (
                not w_ready or upd % args.refit_every == 0):
            _refit(acc, bridge, device)
            w_ready = True

        # Bridge loss (+ task anchor) → AGENT only (W buffer / LM frozen get no grad).
        bridge_loss = float("nan")
        anchor_val = float("nan")
        if w_ready and n_valid > 0:
            for _ in range(args.bridge_steps):
                m = min(args.bridge_batch, n_valid)
                sel = torch.randint(0, n_valid, (m,), generator=gen_cpu)  # cpu idx
                bt = tn[sel]                                  # cpu (m,2)
                obs_b = rb.obs[bt[:, 0].to(device), bt[:, 1].to(device)]  # (m,3,64,64)
                ids_b = ids[sel].to(device)
                lens_b = lens[sel].to(device)
                h_b = agent(obs_b).h_agent                    # grad-carrying
                out = bridge.update(h_b, ids_b, lens_b)       # CE via decode_logits
                with torch.no_grad():
                    h_ref = frozen_ref(obs_b).h_agent         # original representation
                anchor = args.anchor_lambda * (h_b - h_ref).pow(2).mean()
                total = out["loss"] + anchor                  # task-preserving
                bridge_opt.zero_grad()
                total.backward()
                bridge_opt.step()
                bridge_loss = float(out["loss"].detach().item())
                anchor_val = float(anchor.detach().item())

        log = {"update": upd, "env_steps": (upd + 1) * args.num_steps * args.num_envs,
               "ep_return_rolling": roll_ret, "n_valid": int(n_valid),
               "w_ready": w_ready, "bridge_loss": bridge_loss, "anchor": anchor_val,
               "ppo_policy_loss": float(ppo_log.get("policy_loss", float("nan")))}
        logf.write(json.dumps(log) + "\n"); logf.flush()
        if upd % 5 == 0 or upd == args.updates - 1:
            print(f"  [upd {upd:4d}] ret={roll_ret:7.3f} bridge_loss={bridge_loss:.4f} "
                  f"anchor={anchor_val:.4f} n_valid={n_valid} w_ready={w_ready}")

    logf.close()
    # Collapse guard report.
    final_ret = float(np.mean(rolling)) if rolling else float("nan")
    if baseline_ret is not None and final_ret < 0.8 * baseline_ret:
        print(f"[step2] ⚠ rolling return dropped {baseline_ret:.2f}→{final_ret:.2f} "
              f"(>20%) — possible task-destroying collapse; check swap in fit_ccm.")
    else:
        print(f"[step2] return held: {baseline_ret}→{final_ret:.2f}")

    args.out_agent.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"agent": agent.state_dict(), "ccm_step2": True,
                "updates": args.updates, "seed": args.seed}, args.out_agent)
    print(f"[step2] adapted agent → {args.out_agent}")
    print("Next: fit_ccm.py with --agent_checkpoint pointing here (after) vs "
          "checkpoints/phase3/agent.pt (before) → compare swap/slot (rung1·rung2).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
