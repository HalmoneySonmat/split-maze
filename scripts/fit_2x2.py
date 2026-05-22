"""CTRL-2x2 — interface × loss controlled ablation (PLAN §10.1 CTRL-2x2).

Disentangles the Phase-4.2 confound ("V2 < B4 충실도"): is V2's defeat due to
the *learning signal* (separated reconstruction vs next-token) or the
*interface* (single summary vector vs K-latent distributed cross-attn)?

Two knobs, four cells — all fit POST-HOC on the **frozen** Phase-3 agent + LM,
on *identical* in-dist pairs (RL re-training NOT needed; the headline
co-trained V2 vs B4 lives in results/phase4_builds.json):

                   reconstruction        next-token
    thin  (1 vec)  V2  (ACC W)           B4Thin
    rich  (K xattn) V2Rich               B4 (adapter)

Then the same Phase-4 harness (per-slot fidelity in/OOD + activation swap) and
the 2×2 decomposition:

  loss main effect      = mean_interface( recon − next_token )
  interface main effect = mean_loss( rich − thin )

Pre-registered revival of the original hypothesis (CTRL-3, 둘 다):
  (recon − next) swap-following ≥ +0.10  AND  per-slot fidelity ≥ +0.05.

Run (WSL, GPU):
  PYTHONPATH=src python scripts/fit_2x2.py \\
      --lm_checkpoint checkpoints/lm.pt \\
      --agent_checkpoint checkpoints/phase3/agent.pt \\
      --device cuda --seed 0 --rollouts 20 \\
      --fit_steps 3000 --n_pairs 1000
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
from split_maze.builds import B4Adapter, B4Thin, V2ACC, V2Rich
from split_maze.env import DEFAULT_HEADING_WINDOW, TrajectoryTracker
from split_maze.language import (
    CHEESE_DIR_VALUES, HEADING_VALUES, REGION_COLS, REGION_ROWS,
    describer_oracle, parse, render,
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
    p.add_argument("--num_envs", type=int, default=64)
    p.add_argument("--num_steps", type=int, default=256)
    p.add_argument("--rollouts", type=int, default=20)
    p.add_argument("--fit_steps", type=int, default=3000)
    p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--n_pairs", type=int, default=1000)
    p.add_argument("--device", default=("cuda" if torch.cuda.is_available() else "cpu"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output_path", type=Path,
                   default=Path("results/phase4_ctrl2x2.json"))
    return p.parse_args()


def _fresh_lm(lm_blob, device):
    lm = MazeLM(LMConfig(**lm_blob["lm_config"]))
    lm.load_state_dict(lm_blob["model_state"])
    return lm.to(device).eval()


# ---- collection: (h_agent, oracle-Slots) from frozen agent rollouts -------


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
                if ms[t][n] is None:
                    continue
                if describer_oracle(ms[t][n]) is None:
                    continue
                H.append(h_TN[t, n]); states.append(ms[t][n])
    H = torch.stack(H) if H else torch.empty(0, agent.d_a)
    return H, states


def _ids_pool(states, tokenizer, rng):
    """Render each oracle Slots → ids (surface variation via rng), like Phase 2."""
    pool = []
    for s in states:
        g = describer_oracle(s)
        toks = render(g, rng=rng, include_bos_eos=True)
        pool.append(tokenizer.encode(toks))
    return pool


def _pad_batch(ids_list, idx, pad_id, device):
    rows = [ids_list[i] for i in idx]
    T = max(len(r) for r in rows)
    out = torch.full((len(rows), T), pad_id, dtype=torch.long)
    lens = torch.empty(len(rows), dtype=torch.long)
    for i, r in enumerate(rows):
        out[i, :len(r)] = torch.tensor(r, dtype=torch.long)
        lens[i] = len(r)
    return out.to(device), lens.to(device)


# ---- fitting (post-hoc, frozen backbones) ---------------------------------


def _fit(cell, H, ids_list, *, pad_id, steps, warmup, batch, lr, device, seed, tag):
    cell.train()
    opt = torch.optim.AdamW(cell.interpreter_parameters(), lr=lr, betas=(0.9, 0.95),
                            weight_decay=0.01)
    rng = np.random.default_rng(seed)
    N = H.shape[0]
    last = float("nan")
    for step in range(steps):
        # linear warmup then flat (POST-HOC-6/7 protocol).
        for g in opt.param_groups:
            g["lr"] = lr * min(1.0, (step + 1) / max(1, warmup))
        idx = rng.integers(0, N, size=batch)
        h_b = H[idx].to(device)
        ids_b, len_b = _pad_batch(ids_list, idx, pad_id, device)
        out = cell.update(h_b, ids_b, len_b)
        opt.zero_grad(); out["loss"].backward(); opt.step()
        last = float(out["loss"].item())
        if step % 500 == 0 or step == steps - 1:
            print(f"    [{tag}] step {step:5d}  loss {last:.4f}")
    cell.eval()
    return last


# ---- generation → slots ----------------------------------------------------


def _gen_cheese(cell, h, tokenizer, *, kind, acc=None):
    """cheese_dir strings for batch h (already on device)."""
    with torch.no_grad():
        if kind == "V2":
            gen = cell.lm.generate(acc.predict_lm_from_agent(acc.ln_agent(h)), max_len=16)
        else:  # B4Thin / B4 / V2Rich all expose .generate(h_agent)
            gen = cell.generate(h, max_len=16)
    return [parse(tokenizer.decode([int(x) for x in row.tolist()])).cheese_dir
            for row in gen]


def _gen_slots(cell, H, tokenizer, device, *, kind, acc=None, chunk=2048):
    out = []
    for i in range(0, H.shape[0], chunk):
        hb = H[i:i + chunk].to(device)
        with torch.no_grad():
            if kind == "V2":
                gen = cell.lm.generate(acc.predict_lm_from_agent(acc.ln_agent(hb)),
                                       max_len=16)
            else:
                gen = cell.generate(hb, max_len=16)
        for row in gen:
            ps = parse(tokenizer.decode([int(x) for x in row.tolist()]))
            out.append((ps.agent_region, ps.heading, ps.cheese_dir))
    return out


def _per_slot(preds, gold_t, n):
    reg = float(np.mean([preds[i][0] == gold_t[i][0] for i in range(n)]))
    hd = float(np.mean([preds[i][1] == gold_t[i][1] for i in range(n)]))
    ch = float(np.mean([preds[i][2] == gold_t[i][2] for i in range(n)]))
    return {"agent_region": reg, "heading": hd, "cheese_dir": ch,
            "mean": (reg + hd + ch) / 3}


# ---- main ------------------------------------------------------------------


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    rng = random.Random(args.seed)
    device = torch.device(args.device)
    tokenizer = MazeTokenizer()
    lm_blob = torch.load(args.lm_checkpoint, map_location=device, weights_only=False)
    d_lm = lm_blob["lm_config"]["d_model"]

    agent = ImpalaAgent().to(device)
    agent.load_state_dict(torch.load(args.agent_checkpoint, map_location=device,
                                     weights_only=False)["agent"])
    agent.eval()
    print("[ctrl2x2] agent + lm loaded")

    # --- collect pairs: in-dist (fit + eval + swap), OOD (eval) ---
    H_in, states_in = _collect("maze_aisc", agent, num_envs=args.num_envs,
                               num_steps=args.num_steps, rollouts=args.rollouts,
                               device=device, seed=args.seed)
    H_ood, states_ood = _collect("maze", agent, num_envs=args.num_envs,
                                num_steps=args.num_steps, rollouts=args.rollouts,
                                device=device, seed=args.seed + 1)
    ids_in = _ids_pool(states_in, tokenizer, rng)
    print(f"[ctrl2x2] in-dist pairs={len(states_in):,}  OOD pairs={len(states_ood):,}")

    # --- build the four cells (fresh LM each; all frozen except interpreter) ---
    cells = {}
    acc = ACC(ACCConfig(d_agent=agent.d_a, d_lm=d_lm, tied=False))
    cells["V2"] = (V2ACC(_fresh_lm(lm_blob, device), acc).to(device), "V2", acc,
                   "thin", "recon")
    cells["B4Thin"] = (B4Thin(_fresh_lm(lm_blob, device), d_agent=agent.d_a).to(device),
                       "OTH", None, "thin", "next")
    cells["B4"] = (B4Adapter(_fresh_lm(lm_blob, device), d_agent=agent.d_a).to(device),
                   "OTH", None, "rich", "next")
    cells["V2Rich"] = (V2Rich(_fresh_lm(lm_blob, device), d_agent=agent.d_a).to(device),
                       "OTH", None, "rich", "recon")

    # --- fit each cell on identical in-dist pairs ---
    print("[ctrl2x2] fitting 4 cells (post-hoc, frozen agent+LM) ...")
    for name, (mod, kind, acc_ref, iface, loss) in cells.items():
        _fit(mod, H_in, ids_in, pad_id=tokenizer.pad_id, steps=args.fit_steps,
             warmup=args.warmup, batch=args.batch, lr=args.lr, device=device,
             seed=args.seed, tag=name)

    # --- per-slot fidelity (in-dist + OOD) ---
    results = {"cells": {n: {"interface": c[3], "loss": c[4]} for n, c in cells.items()},
               "per_slot": {}, "swap_following": {}}
    for tag, (H, states) in (("in_dist", (H_in, states_in)),
                             ("ood", (H_ood, states_ood))):
        gold = [describer_oracle(s) for s in states]
        gold_t = [((g.agent_row, g.agent_col), g.heading, g.cheese_dir) for g in gold]
        n = len(states)
        results["per_slot"][tag] = {}
        print(f"\n[{tag}] n={n}")
        print(f"  {'cell':8} {'iface':5} {'loss':6} {'region':>7} {'heading':>7} "
              f"{'cheese':>7} {'mean':>7}")
        for name, (mod, kind, acc_ref, iface, loss) in cells.items():
            preds = _gen_slots(mod, H, tokenizer, device, kind=kind, acc=acc_ref)
            ps = _per_slot(preds, gold_t, n)
            results["per_slot"][tag][name] = ps
            print(f"  {name:8} {iface:5} {loss:6} {ps['agent_region']:7.3f} "
                  f"{ps['heading']:7.3f} {ps['cheese_dir']:7.3f} {ps['mean']:7.3f}")

    # --- activation swap (in-dist) ---
    CD_in = [describer_oracle(s).cheese_dir for s in states_in]
    idx_by_cd = {}
    for i, d in enumerate(CD_in):
        idx_by_cd.setdefault(d, []).append(i)
    cds = list(idx_by_cd.keys())
    pairs, tries = [], 0
    while len(pairs) < args.n_pairs and tries < args.n_pairs * 20:
        tries += 1
        da, db = rng.sample(cds, 2)
        pairs.append((rng.choice(idx_by_cd[da]), rng.choice(idx_by_cd[db])))
    A = torch.stack([H_in[a] for a, _ in pairs])
    B = torch.stack([H_in[b] for _, b in pairs])
    cdA = [CD_in[a] for a, _ in pairs]; cdB = [CD_in[b] for _, b in pairs]
    alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
    print(f"\n[swap] {len(pairs)} pairs")
    for name, (mod, kind, acc_ref, iface, loss) in cells.items():
        per_alpha = {}
        for al in alphas:
            h = ((1 - al) * A + al * B).to(device)
            cds_pred = []
            for i in range(0, h.shape[0], 2048):
                cds_pred.extend(_gen_cheese(mod, h[i:i + 2048], tokenizer,
                                            kind=kind, acc=acc_ref))
            per_alpha[al] = cds_pred
        a0, a1 = per_alpha[0.0], per_alpha[1.0]
        read_A = [j for j in range(len(pairs)) if a0[j] == cdA[j]]
        follow = (float(np.mean([a1[j] == cdB[j] for j in read_A]))
                  if read_A else float("nan"))
        curve = [float(np.mean([per_alpha[al][j] == cdB[j] for j in range(len(pairs))]))
                 for al in alphas]
        results["swap_following"][name] = {"rate": follow, "n_readA": len(read_A),
                                           "curve": curve}
        print(f"  {name:8} swap-following={follow:.3f} (n_readA={len(read_A)})  "
              f"curve={['%.2f' % c for c in curve]}")

    # --- 2x2 decomposition + CTRL-3 verdict ---
    def slot_mean(cell, cond="in_dist"):
        return results["per_slot"][cond][cell]["mean"]

    def swap(cell):
        return results["swap_following"][cell]["rate"]

    dec = {}
    # loss effect (recon − next) at each interface, and mean.
    dec["loss_effect_swap_thin"] = swap("V2") - swap("B4Thin")
    dec["loss_effect_swap_rich"] = swap("V2Rich") - swap("B4")
    dec["loss_effect_swap_mean"] = 0.5 * (dec["loss_effect_swap_thin"]
                                          + dec["loss_effect_swap_rich"])
    dec["loss_effect_slot_thin"] = slot_mean("V2") - slot_mean("B4Thin")
    dec["loss_effect_slot_rich"] = slot_mean("V2Rich") - slot_mean("B4")
    dec["loss_effect_slot_mean"] = 0.5 * (dec["loss_effect_slot_thin"]
                                          + dec["loss_effect_slot_rich"])
    # interface effect (rich − thin) at each loss, and mean.
    dec["iface_effect_swap_recon"] = swap("V2Rich") - swap("V2")
    dec["iface_effect_swap_next"] = swap("B4") - swap("B4Thin")
    dec["iface_effect_swap_mean"] = 0.5 * (dec["iface_effect_swap_recon"]
                                           + dec["iface_effect_swap_next"])
    dec["iface_effect_slot_recon"] = slot_mean("V2Rich") - slot_mean("V2")
    dec["iface_effect_slot_next"] = slot_mean("B4") - slot_mean("B4Thin")
    dec["iface_effect_slot_mean"] = 0.5 * (dec["iface_effect_slot_recon"]
                                           + dec["iface_effect_slot_next"])

    # CTRL-3 (사전등록): revive original hypothesis iff loss main effect meets
    # BOTH thresholds (swap ≥ +0.10 AND per-slot ≥ +0.05).
    revive_swap = dec["loss_effect_swap_mean"] >= 0.10
    revive_slot = dec["loss_effect_slot_mean"] >= 0.05
    dec["CTRL3_revive_original_hypothesis"] = bool(revive_swap and revive_slot)
    dec["CTRL3_thresholds"] = {"swap_min": 0.10, "slot_min": 0.05,
                               "swap_pass": bool(revive_swap),
                               "slot_pass": bool(revive_slot)}
    results["decomposition"] = dec

    print("\n[2x2 decomposition]")
    print(f"  LOSS effect (recon−next)  swap: thin {dec['loss_effect_swap_thin']:+.3f} "
          f"rich {dec['loss_effect_swap_rich']:+.3f}  mean {dec['loss_effect_swap_mean']:+.3f}")
    print(f"  LOSS effect (recon−next)  slot: thin {dec['loss_effect_slot_thin']:+.3f} "
          f"rich {dec['loss_effect_slot_rich']:+.3f}  mean {dec['loss_effect_slot_mean']:+.3f}")
    print(f"  IFACE effect (rich−thin)  swap: mean {dec['iface_effect_swap_mean']:+.3f}  "
          f"slot: mean {dec['iface_effect_slot_mean']:+.3f}")
    print(f"\n  CTRL-3 (둘 다: swap≥+0.10 AND slot≥+0.05) → "
          f"REVIVE={dec['CTRL3_revive_original_hypothesis']} "
          f"(swap_pass={revive_swap}, slot_pass={revive_slot})")

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[ctrl2x2] → {args.output_path}")
    print("Read: LOSS effect>0 ⇒ 재구성이 핵심(원가설 부활). IFACE effect>0 ⇒ "
          "B4 우위는 분산 인터페이스 덕. 사전등록 임계는 PLAN §10.1 CTRL-2x2.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
