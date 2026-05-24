"""Phase 5 CCM — step 3: the *growing* bridge (PLAN §10.1 step3 사전등록).

The original grand vision: two brains co-adapt and **the bridge grows**. step2
(recorded-W, agent-only) was a clean negative (v1 collapsed the task, v2 held the
task but the bridge did not grow). step3 changes two things:

  1. ``W`` is no longer a recorded buffer — it becomes a *plastic* trainable
     parameter, **warm-started from the recorded ridge W** ("기억이 씨앗": the
     memory is the seed, then it grows by gradient).
  2. "양방향" — the LM also meets the agent halfway. lm.py confirms the bridge
     generate path injects ĥ as the position-0 hidden read by the *decoder core*
     (``interface_proj`` is encode-only, which is why step2's interface handle was
     inert). So the LM-side handle is a single decoder block ``blocks[0]``, kept
     gentle and paired with a **language anchor** (KL to a frozen reference LM on
     real sentences) so the LM cannot abandon language to collude.

Staged (pre-registered):
  --phase a1 : agent + LM frozen; train ONLY the plastic W,b (next-token CE).
               = a *trained translator* on the frozen agent — the fair ceiling /
               control (expected ≈ B4Thin swap 0.778), and a guaranteed
               "memory → translator" growth demo. Saves --out_bridge.
  --phase a2 : load the A1 bridge; co-adapt the agent (conv frozen + small lr +
               PPO + repr anchor), the LM ``blocks[0]`` (small lr + language KL
               anchor), and W,b together. Saves --out_bridge / --out_agent /
               --out_lm.

loss↓≠success guard (TOP priority): three things (W, agent, LM) minimize one loss
→ collusion risk is maximal. The verdict is NOT the training loss — it is held-out
swap/slot via ``fit_ccm.py --bridge_checkpoint`` (A1: frozen agent; A2: adapted
agent + adapted LM), with task anchors (agent return, LM language CE) and a
non-degeneracy check (n_readA). Training loss going down is *not* evidence.

Run (WSL, GPU) — A1 then A2:
  PYTHONPATH=src python scripts/train_ccm_step3.py --phase a1 \\
      --lm_checkpoint checkpoints/lm.pt --agent_checkpoint checkpoints/phase3/agent.pt \\
      --device cuda --seed 0 --a1_updates 60
  PYTHONPATH=src python scripts/train_ccm_step3.py --phase a2 \\
      --lm_checkpoint checkpoints/lm.pt --agent_checkpoint checkpoints/phase3/agent.pt \\
      --in_bridge checkpoints/phase5/bridge_a1.pt --device cuda --seed 0 --a2_updates 100
"""

from __future__ import annotations

import argparse
import json
import random
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

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
    p.add_argument("--phase", choices=["a1", "a2"], required=True)
    p.add_argument("--lm_checkpoint", type=Path, default=Path("checkpoints/lm.pt"))
    p.add_argument("--agent_checkpoint", type=Path,
                   default=Path("checkpoints/phase3/agent.pt"))
    p.add_argument("--num_envs", type=int, default=64)
    p.add_argument("--num_steps", type=int, default=256)
    # A1 (warm W on frozen brains)
    p.add_argument("--a1_updates", type=int, default=60)
    p.add_argument("--warmup_record", type=int, default=3,
                   help="A1: rollouts used to record the ridge warm-start W")
    p.add_argument("--w_lr", type=float, default=3e-4, help="plastic W,b lr")
    # A2 (co-adapt)
    p.add_argument("--a2_updates", type=int, default=100)
    p.add_argument("--in_bridge", type=Path, default=Path("checkpoints/phase5/bridge_a1.pt"))
    p.add_argument("--bridge_lr", type=float, default=1e-5, help="A2 agent lr (gentle, v2)")
    p.add_argument("--w_lr_a2", type=float, default=1e-4, help="A2 plastic W,b lr")
    p.add_argument("--lm_lr", type=float, default=1e-5, help="A2 LM blocks[idx] lr (gentle)")
    p.add_argument("--lm_block", type=int, default=0, help="which decoder block the LM adapts")
    p.add_argument("--freeze_conv", type=int, default=1)
    p.add_argument("--anchor_lambda", type=float, default=1.0,
                   help="agent repr anchor ||h-h_ref||² (task preserve)")
    p.add_argument("--lm_anchor_lambda", type=float, default=1.0,
                   help="LM language KL anchor to frozen ref (collusion guard)")
    # shared
    p.add_argument("--bridge_steps", type=int, default=8)
    p.add_argument("--bridge_batch", type=int, default=512)
    p.add_argument("--device", default=("cuda" if torch.cuda.is_available() else "cpu"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out_bridge", type=Path, default=None)
    p.add_argument("--out_agent", type=Path,
                   default=Path("checkpoints/phase5/agent_ccm_a2.pt"))
    p.add_argument("--out_lm", type=Path, default=Path("checkpoints/phase5/lm_a2.pt"))
    p.add_argument("--log_path", type=Path, default=None)
    return p.parse_args()


def _fresh_lm(lm_blob, device):
    lm = MazeLM(LMConfig(**lm_blob["lm_config"]))
    lm.load_state_dict(lm_blob["model_state"])
    return lm.to(device).eval()


def _valid_pairs(h_TN, maze_states, tokenizer, rng):
    """From a rollout, gather valid (t,n), padded oracle ids, lengths, h rows."""
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


def _lm_kl_anchor(lm, ref_lm, ids, pad_id):
    """Forward-KL distillation on REAL sentences: keep the adapting LM's
    next-token distribution close to the frozen reference (language preserve)."""
    with torch.no_grad():
        ref_logits, _ = ref_lm.forward(ids)
        ref_logp = F.log_softmax(ref_logits, dim=-1)
        ref_p = ref_logp.exp()
    cur_logits, _ = lm.forward(ids)
    cur_logp = F.log_softmax(cur_logits, dim=-1)
    ids_full = lm._append_sum(ids)                       # (B, T+1)
    mask = (ids_full != pad_id).float().unsqueeze(-1)    # (B, T+1, 1)
    kl = (ref_p * (ref_logp - cur_logp)).sum(-1, keepdim=True)
    return (kl * mask).sum() / mask.sum().clamp(min=1.0)


def _collect(env, agent, rb, trackers, holder, cur_rgb, ep_r, ep_l, d_agent, device):
    return collect_rollout_with_pairs(
        env, agent, rb, trackers, obs_holder=holder, cur_rgb=cur_rgb,
        episode_returns=ep_r, episode_lengths=ep_l,
        state_extractor=default_state_extractor, d_agent=d_agent, device=device)


# ----------------------------------------------------------------------------


def run_a1(args, device, tokenizer, lm_blob, agent, bridge, gen_cpu, surface_rng):
    """A1: agent + LM frozen; train plastic W,b only (trained-translator ceiling)."""
    d_agent, d_model = agent.d_a, lm_blob["lm_config"]["d_model"]
    agent.eval()
    for p in agent.parameters():
        p.requires_grad_(False)
    # frozen LM already set by CCMBridge.__init__.

    env = make_maze_env(env_name="maze_aisc", num=args.num_envs, num_levels=0,
                        start_level=0, distribution_mode="easy", rand_seed=args.seed)
    rb = RolloutBuffer(T=args.num_steps, N=args.num_envs, device=device)
    trackers = [TrajectoryTracker(DEFAULT_HEADING_WINDOW) for _ in range(args.num_envs)]
    _r, obs_dict, _f = env.observe()
    holder = obs_to_tensor(obs_dict, device)
    cur_rgb = np.asarray(obs_dict["rgb"])
    ep_r = np.zeros(args.num_envs); ep_l = np.zeros(args.num_envs, dtype=np.int64)
    EC = 4096

    # ---- warm-start: record ridge W on the frozen agent, then make W plastic ----
    acc = CoActAccumulator(d_agent, d_model, device=device)
    for _ in range(args.warmup_record):
        _s, holder, cur_rgb, h_TN, ms = _collect(
            env, agent, rb, trackers, holder, cur_rgb, ep_r, ep_l, d_agent, device)
        tn, ids, lens, h_valid = _valid_pairs(h_TN, ms, tokenizer, surface_rng)
        if tn.shape[0] == 0:
            continue
        x = bridge.ln_agent(h_valid.to(device))
        a2 = []
        with torch.no_grad():
            for i in range(0, x.shape[0], EC):
                a2.append(bridge.lm.encode(ids[i:i + EC].to(device)))
        acc.update(x, torch.cat(a2, dim=0))
    W, b = fit_W(acc.moments(), "ridge")
    bridge.set_W(W.to(device), b.to(device))
    bridge.set_plastic(True)                              # 기억이 씨앗 → 학습으로 성장
    opt = torch.optim.AdamW(list(bridge.interpreter_parameters()),
                            lr=args.w_lr, betas=(0.9, 0.95), weight_decay=0.01)
    print(f"[step3-a1] plastic W warm-started from recorded ridge | w_lr={args.w_lr} "
          f"a1_updates={args.a1_updates}")

    logf = _open_log(args)
    for upd in range(args.a1_updates):
        _s, holder, cur_rgb, h_TN, ms = _collect(
            env, agent, rb, trackers, holder, cur_rgb, ep_r, ep_l, d_agent, device)
        tn, ids, lens, h_valid = _valid_pairs(h_TN, ms, tokenizer, surface_rng)
        n_valid = tn.shape[0]
        bl = float("nan")
        if n_valid > 0:
            h_valid = h_valid.cpu()                       # index on CPU, move batch to device
            for _ in range(args.bridge_steps):
                m = min(args.bridge_batch, n_valid)
                sel = torch.randint(0, n_valid, (m,), generator=gen_cpu)
                h_b = h_valid[sel].to(device).detach()    # frozen agent → no agent grad
                out = bridge.update(h_b, ids[sel].to(device), lens[sel].to(device))
                opt.zero_grad(); out["loss"].backward(); opt.step()
                bl = float(out["loss"].detach().item())
        _log(logf, {"phase": "a1", "update": upd, "n_valid": int(n_valid), "bridge_loss": bl})
        if upd % 5 == 0 or upd == args.a1_updates - 1:
            print(f"  [a1 {upd:4d}] bridge_loss={bl:.4f} n_valid={n_valid}")
    if logf:
        logf.close()

    out_bridge = args.out_bridge or Path("checkpoints/phase5/bridge_a1.pt")
    _save_bridge(bridge, out_bridge, {"phase": "a1", "seed": args.seed})
    print(f"[step3-a1] trained-translator bridge → {out_bridge}")
    print("Next: fit_ccm.py --bridge_checkpoint <this> --agent_checkpoint <frozen agent> "
          "→ A1 ceiling swap (control).")
    return 0


def run_a2(args, device, tokenizer, lm_blob, agent, bridge, gen_dev, gen_cpu, surface_rng):
    """A2: co-adapt agent (gentle) + LM blocks[idx] (gentle + language anchor) + W."""
    d_agent, d_model = agent.d_a, lm_blob["lm_config"]["d_model"]
    ckpt = agent.state_dict()

    # warm-start the plastic bridge from A1.
    blob = torch.load(args.in_bridge, map_location=device, weights_only=False)
    bridge.set_W(blob["W"].to(device), blob["b"].to(device))
    bridge.set_plastic(True)

    # agent: conv frozen (v2), embed+heads adapt.
    agent.train()
    if args.freeze_conv:
        for p in agent.blocks.parameters():
            p.requires_grad_(False)
    frozen_ref = ImpalaAgent().to(device); frozen_ref.load_state_dict(ckpt)
    frozen_ref.eval()
    for p in frozen_ref.parameters():
        p.requires_grad_(False)

    # LM: unfreeze ONE decoder block (the bridge generate path) + frozen ref LM.
    bridge.set_lm_trainable_block(args.lm_block, True)
    ref_lm = _fresh_lm(lm_blob, device)
    for p in ref_lm.parameters():
        p.requires_grad_(False)

    agent_train = [p for p in agent.parameters() if p.requires_grad]
    lm_block_params = [p for p in bridge.lm.blocks[args.lm_block].parameters()]
    bridge_opt = torch.optim.AdamW([
        {"params": agent_train, "lr": args.bridge_lr},
        {"params": list(bridge.interpreter_parameters()), "lr": args.w_lr_a2},
        {"params": lm_block_params, "lr": args.lm_lr},
    ], betas=(0.9, 0.95), weight_decay=0.0)

    env = make_maze_env(env_name="maze_aisc", num=args.num_envs, num_levels=0,
                        start_level=0, distribution_mode="easy", rand_seed=args.seed)
    rb = RolloutBuffer(T=args.num_steps, N=args.num_envs, device=device)
    ppo_cfg = PPOConfig()
    updater = PPOUpdater(agent, ppo_cfg)
    trackers = [TrajectoryTracker(DEFAULT_HEADING_WINDOW) for _ in range(args.num_envs)]
    _r, obs_dict, _f = env.observe()
    holder = obs_to_tensor(obs_dict, device)
    cur_rgb = np.asarray(obs_dict["rgb"])
    ep_r = np.zeros(args.num_envs); ep_l = np.zeros(args.num_envs, dtype=np.int64)
    rolling = deque(maxlen=64)
    baseline_ret = None
    print(f"[step3-a2] co-adapt | agent_lr={args.bridge_lr}(conv frozen={bool(args.freeze_conv)}) "
          f"w_lr={args.w_lr_a2} lm_block[{args.lm_block}]_lr={args.lm_lr} "
          f"anchor_λ={args.anchor_lambda} lm_anchor_λ={args.lm_anchor_lambda}")

    logf = _open_log(args)
    for upd in range(args.a2_updates):
        _s, holder, cur_rgb, h_TN, ms = _collect(
            env, agent, rb, trackers, holder, cur_rgb, ep_r, ep_l, d_agent, device)
        rolling.extend(_s.completed_returns)
        roll_ret = float(np.mean(rolling)) if rolling else float("nan")
        if baseline_ret is None and rolling:
            baseline_ret = roll_ret

        # PPO (task anchor on the agent).
        with torch.no_grad():
            last_value = agent(holder).value
        rb.compute_advantages_and_returns(last_value, gamma=ppo_cfg.gamma,
                                          gae_lambda=ppo_cfg.gae_lambda)
        updater.update(rb, generator=gen_dev)

        tn, ids, lens, _h = _valid_pairs(h_TN, ms, tokenizer, surface_rng)
        n_valid = tn.shape[0]
        bl = anch = lmk = float("nan")
        if n_valid > 0:
            for _ in range(args.bridge_steps):
                m = min(args.bridge_batch, n_valid)
                sel = torch.randint(0, n_valid, (m,), generator=gen_cpu)   # CPU idx
                bt = tn[sel]                                     # CPU
                obs_b = rb.obs[bt[:, 0].to(device), bt[:, 1].to(device)]
                ids_b, lens_b = ids[sel].to(device), lens[sel].to(device)
                h_b = agent(obs_b).h_agent                       # grad → agent + W
                out = bridge.update(h_b, ids_b, lens_b)          # CE → W + LM block + agent
                with torch.no_grad():
                    h_ref = frozen_ref(obs_b).h_agent
                anchor = args.anchor_lambda * (h_b - h_ref).pow(2).mean()
                lm_kl = args.lm_anchor_lambda * _lm_kl_anchor(
                    bridge.lm, ref_lm, ids_b, tokenizer.pad_id)
                total = out["loss"] + anchor + lm_kl
                bridge_opt.zero_grad(); total.backward(); bridge_opt.step()
                bl = float(out["loss"].detach().item())
                anch = float(anchor.detach().item())
                lmk = float(lm_kl.detach().item())
        _log(logf, {"phase": "a2", "update": upd, "ep_return_rolling": roll_ret,
                    "n_valid": int(n_valid), "bridge_loss": bl, "anchor": anch, "lm_kl": lmk})
        if upd % 5 == 0 or upd == args.a2_updates - 1:
            print(f"  [a2 {upd:4d}] ret={roll_ret:7.3f} bridge_loss={bl:.4f} "
                  f"anchor={anch:.4f} lm_kl={lmk:.4f} n_valid={n_valid}")
    if logf:
        logf.close()

    final_ret = float(np.mean(rolling)) if rolling else float("nan")
    if baseline_ret is not None and final_ret < 0.8 * baseline_ret:
        print(f"[step3-a2] ⚠ return dropped {baseline_ret:.2f}→{final_ret:.2f} (>20%) — "
              f"task-destroying collapse; verdict still from fit_ccm swap.")
    else:
        print(f"[step3-a2] return held: {baseline_ret}→{final_ret:.2f}")

    out_bridge = args.out_bridge or Path("checkpoints/phase5/bridge_a2.pt")
    _save_bridge(bridge, out_bridge, {"phase": "a2", "seed": args.seed})
    args.out_agent.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"agent": agent.state_dict(), "ccm_step3_a2": True, "seed": args.seed},
               args.out_agent)
    # save the adapted LM in the fit_ccm-compatible blob format.
    torch.save({"lm_config": lm_blob["lm_config"],
                "model_state": bridge.lm.state_dict()}, args.out_lm)
    print(f"[step3-a2] bridge → {out_bridge} | agent → {args.out_agent} | adapted LM → {args.out_lm}")
    print("Next: fit_ccm.py --lm_checkpoint <out_lm> --agent_checkpoint <out_agent> "
          "--bridge_checkpoint <out_bridge> → A2 swap; compare to A1 ceiling (Claim 2).")
    return 0


# ---- small io helpers ----

def _open_log(args):
    if args.log_path is None:
        return None
    args.log_path.parent.mkdir(parents=True, exist_ok=True)
    return open(args.log_path, "w")


def _log(logf, d):
    if logf:
        logf.write(json.dumps(d) + "\n"); logf.flush()


def _save_bridge(bridge, path, meta):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"W": bridge.W.detach().cpu(), "b": bridge.b.detach().cpu(), **meta}, path)


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    surface_rng = random.Random(args.seed + 1)
    device = torch.device(args.device)
    gen_dev = torch.Generator(device=device).manual_seed(args.seed)
    gen_cpu = torch.Generator(device="cpu").manual_seed(args.seed)
    tokenizer = MazeTokenizer()
    lm_blob = torch.load(args.lm_checkpoint, map_location=device, weights_only=False)

    ckpt = torch.load(args.agent_checkpoint, map_location=device,
                      weights_only=False)["agent"]
    agent = ImpalaAgent().to(device)
    agent.load_state_dict(ckpt)
    bridge = CCMBridge(_fresh_lm(lm_blob, device), d_agent=agent.d_a).to(device)

    if args.phase == "a1":
        return run_a1(args, device, tokenizer, lm_blob, agent, bridge, gen_cpu, surface_rng)
    return run_a2(args, device, tokenizer, lm_blob, agent, bridge, gen_dev, gen_cpu, surface_rng)


if __name__ == "__main__":
    raise SystemExit(main())
