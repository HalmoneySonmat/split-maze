"""Phase 5 CCM — step 1: frozen co-activation fit + per-slot/swap eval
(PLAN §10.1 Phase 5 CCM, CCM-1/CCM-4).

On the *frozen* Phase-3 agent + Phase-2 LM, record the co-activation
correspondence W from in-dist pairs and score it — does the *memory alone*
(no gradient) transfer the agent's representation through the LM?

Pairs for the fit: for each in-dist state s, x = LN(h_agent(s)); y =
lm.encode(oracle_sentence(s)) — the same scene in the LM's modality. The W
ladder (CCM-1), all closed-form from the same recorded moments:

  rung1 hebbian  — raw E[y xᵀ]            (un-normalized; headline baseline)
  rung2 ridge    — y on x least-squares   (정규화 칸)
  rung3 procrustes — semi-orthogonal align (정규화 칸)

Baselines/ceiling: random-W (matched scale, b=mean_y) = pre-registered floor;
identity-W (when d_lm==d_agent) = "free" structured reference; B4 = trained-
adapter ceiling.

Each config is scored with the Phase-4 harness: per-slot fidelity
(region/heading/cheese) on in-dist + OOD, and activation swap-following (the
real discriminator). Pre-registered verdicts (results-blind, PLAN §10.1):
  - step1 "memory transfers": rung1 in-dist slot ≥ random +0.10 AND swap ≥ +0.10.
  - "normalization key": (best of ridge/procrustes) − rung1 ≥ +0.10 swap AND
    +0.05 slot (both) ⇒ normalization needed; else pure memory suffices.
  - ceiling: best rung swap / B4 swap (descriptive).

All 1 RL seed, descriptive (statistics = multi-seed, §5.7).

Run (WSL, GPU):
  PYTHONPATH=src python scripts/fit_ccm.py \\
      --lm_checkpoint checkpoints/lm.pt \\
      --agent_checkpoint checkpoints/phase3/agent.pt \\
      --b4_checkpoint checkpoints/phase3/B4.pt \\
      --device cuda --seed 0 --rollouts 15 --n_pairs 1000
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from split_maze.agent import ImpalaAgent
from split_maze.builds import B4Adapter
from split_maze.ccm import CCMBridge, CoActAccumulator, fit_W
from split_maze.env import DEFAULT_HEADING_WINDOW, TrajectoryTracker, make_maze_env
from split_maze.language import (
    CHEESE_DIR_VALUES, HEADING_VALUES, REGION_COLS, REGION_ROWS,
    Slots, describer_oracle, parse, render,
)
from split_maze.lm import LMConfig, MazeLM, MazeTokenizer
from split_maze.ppo import RolloutBuffer
from split_maze.train import obs_to_tensor
from split_maze.train_phase3 import collect_rollout_with_pairs, default_state_extractor


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--lm_checkpoint", type=Path, default=Path("checkpoints/lm.pt"))
    p.add_argument("--agent_checkpoint", type=Path,
                   default=Path("checkpoints/phase3/agent.pt"))
    p.add_argument("--b4_checkpoint", type=Path, default=Path("checkpoints/phase3/B4.pt"))
    p.add_argument("--bridge_checkpoint", type=Path, default=None,
                   help="step3: a *trained* plastic bridge (W,b) to eval as 'trained_bridge' "
                        "with the SAME harness (A1 frozen / A2 adapted agent+LM)")
    p.add_argument("--num_envs", type=int, default=64)
    p.add_argument("--num_steps", type=int, default=256)
    p.add_argument("--rollouts", type=int, default=15)
    p.add_argument("--n_pairs", type=int, default=1000)
    p.add_argument("--max_eval", type=int, default=40000,
                   help="cap states used for per-slot generation (fit uses all)")
    p.add_argument("--ridge_lambda", type=float, default=None)
    p.add_argument("--ablation", type=int, default=0,
                   help="add whitening-ablation corners (centering_only / "
                        "whitening_only) + decisive-ingredient verdict (PLAN §10.1)")
    p.add_argument("--device", default=("cuda" if torch.cuda.is_available() else "cpu"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output_path", type=Path, default=Path("results/phase5_ccm.json"))
    return p.parse_args()


def _fresh_lm(lm_blob, device):
    lm = MazeLM(LMConfig(**lm_blob["lm_config"]))
    lm.load_state_dict(lm_blob["model_state"])
    return lm.to(device).eval()


def _collect(env_name, agent, *, num_envs, num_steps, rollouts, device, seed):
    """Return (H (n,d_a) on device, states list, cheese_dir list)."""
    env = make_maze_env(env_name=env_name, num=num_envs, num_levels=0,
                        start_level=0, distribution_mode="easy", rand_seed=seed)
    rb = RolloutBuffer(T=num_steps, N=num_envs, device=device)
    trackers = [TrajectoryTracker(DEFAULT_HEADING_WINDOW) for _ in range(num_envs)]
    _r, obs_dict, _f = env.observe()
    obs_holder = obs_to_tensor(obs_dict, device)
    cur_rgb = np.asarray(obs_dict["rgb"])
    ep_r = np.zeros(num_envs); ep_l = np.zeros(num_envs, dtype=np.int64)
    H, states, CD = [], [], []
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
                H.append(h_TN[t, n]); states.append(ms[t][n]); CD.append(g.cheese_dir)
    return torch.stack(H), states, CD


def _oracle_ids(states, tokenizer, rng):
    """Render each state's oracle slots → token ids; return padded (n,T) + lens."""
    ids_list, lens = [], []
    for s in states:
        g = describer_oracle(s)
        slots = Slots(agent_row=g.agent_row, agent_col=g.agent_col,
                      heading=g.heading, cheese_dir=g.cheese_dir)
        ids = tokenizer.encode(render(slots, rng=rng, include_bos_eos=True))
        ids_list.append(ids); lens.append(len(ids))
    T = max(lens)
    padded = torch.full((len(ids_list), T), tokenizer.pad_id, dtype=torch.long)
    for i, ids in enumerate(ids_list):
        padded[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)
    return padded


@torch.no_grad()
def _encode_targets(lm, ids, device, chunk=4096):
    """a2 = lm.encode(ids) in chunks → (n, d_model) on device."""
    out = []
    for i in range(0, ids.shape[0], chunk):
        out.append(lm.encode(ids[i:i + chunk].to(device)))
    return torch.cat(out, dim=0)


def _slots_from_gen(gen, tokenizer):
    out = []
    for row in gen:
        ps = parse(tokenizer.decode([int(x) for x in row.tolist()]))
        out.append((ps.agent_region, ps.heading, ps.cheese_dir))
    return out


@torch.no_grad()
def _gen_slots(build, H, tokenizer, device, chunk=2048):
    out = []
    for i in range(0, H.shape[0], chunk):
        gen = build.generate(H[i:i + chunk].to(device), max_len=16)
        out.extend(_slots_from_gen(gen, tokenizer))
    return out


def _per_slot(preds, gold_t):
    n = len(gold_t)
    reg = float(np.mean([preds[i][0] == gold_t[i][0] for i in range(n)]))
    hd = float(np.mean([preds[i][1] == gold_t[i][1] for i in range(n)]))
    ch = float(np.mean([preds[i][2] == gold_t[i][2] for i in range(n)]))
    return {"agent_region": reg, "heading": hd, "cheese_dir": ch,
            "mean": (reg + hd + ch) / 3}


@torch.no_grad()
def _cheese_of(build, h, tokenizer, device, chunk=2048):
    out = []
    for i in range(0, h.shape[0], chunk):
        gen = build.generate(h[i:i + chunk].to(device), max_len=16)
        for row in gen:
            out.append(parse(tokenizer.decode([int(x) for x in row.tolist()])).cheese_dir)
    return out


def _swap_following(build, A, B, cdA, cdB, tokenizer, device, alphas):
    """Return dict {rate, n_readA, curve} — fraction following the swapped-in
    cheese_dir at α=1, among pairs read correctly at α=0."""
    per_alpha = {}
    curve = []
    for al in alphas:
        h = (1 - al) * A + al * B
        cds = _cheese_of(build, h, tokenizer, device)
        per_alpha[al] = cds
        curve.append(float(np.mean([cds[j] == cdB[j] for j in range(len(cdB))])))
    a0, a1 = per_alpha[0.0], per_alpha[1.0]
    read_A = [j for j in range(len(cdA)) if a0[j] == cdA[j]]
    rate = float(np.mean([a1[j] == cdB[j] for j in read_A])) if read_A else float("nan")
    return {"rate": rate, "n_readA": len(read_A), "curve": curve}


def _subsample(H, states, gold_t, max_n, rng):
    n = len(states)
    if n <= max_n:
        return H, states, gold_t
    idx = rng.sample(range(n), max_n)
    return H[idx], [states[i] for i in idx], [gold_t[i] for i in idx]


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    rng = random.Random(args.seed)
    gen_t = torch.Generator(device="cpu").manual_seed(args.seed)
    device = torch.device(args.device)
    tokenizer = MazeTokenizer()
    lm_blob = torch.load(args.lm_checkpoint, map_location=device, weights_only=False)

    agent = ImpalaAgent().to(device)
    agent.load_state_dict(torch.load(args.agent_checkpoint, map_location=device,
                                     weights_only=False)["agent"])
    agent.eval()
    d_agent = agent.d_a
    d_model = lm_blob["lm_config"]["d_model"]

    bridge = CCMBridge(_fresh_lm(lm_blob, device), d_agent=d_agent).to(device)
    # B4 ceiling is an OPTIONAL reference (not used in the A1/A2 verdict). Skip it
    # when --b4_checkpoint is empty/"none"/missing — e.g. new RL-seed "brains" that
    # have no per-agent B4 (RL-seed generalization runs).
    b4 = None
    b4p = str(args.b4_checkpoint) if args.b4_checkpoint is not None else ""
    # NOTE: argparse type=Path turns --b4_checkpoint "" into Path(".") whose str()
    # is "." and .exists() is True (it's the cwd). So we skip "." explicitly AND
    # require an actual file (.is_file()) — a directory must never be torch.load'ed.
    has_b4 = b4p not in ("", ".", "none", "None") and Path(b4p).is_file()
    if has_b4:
        b4 = B4Adapter(_fresh_lm(lm_blob, device), d_agent=d_agent).to(device)
        b4.load_state_dict(torch.load(args.b4_checkpoint, map_location=device,
                                      weights_only=False)["state"]); b4.eval()
    print(f"[fit_ccm] agent d_a={d_agent} + LM d_model={d_model}"
          f"{' + B4(ceiling)' if has_b4 else ' (no B4 ceiling)'}")

    # ---- collect in-dist (fit + eval) and OOD (eval) ----
    H_id, st_id, CD_id = _collect("maze_aisc", agent, num_envs=args.num_envs,
                                  num_steps=args.num_steps, rollouts=args.rollouts,
                                  device=device, seed=args.seed)
    H_ood, st_ood, _CD_ood = _collect("maze", agent, num_envs=args.num_envs,
                                      num_steps=args.num_steps, rollouts=args.rollouts,
                                      device=device, seed=args.seed)
    print(f"[fit_ccm] in-dist {len(st_id):,} states, OOD {len(st_ood):,} states")
    print(f"[fit_ccm] in-dist cheese_dir: { {d: CD_id.count(d) for d in sorted(set(CD_id))} }")

    # ---- record co-activation moments on ALL in-dist pairs ----
    ids_id = _oracle_ids(st_id, tokenizer, rng)
    a2 = _encode_targets(bridge.lm, ids_id, device)               # (n, d_model)
    acc = CoActAccumulator(d_agent, d_model, device=device)
    CH = 8192
    for i in range(0, H_id.shape[0], CH):
        x = bridge.ln_agent(H_id[i:i + CH])
        acc.update(x, a2[i:i + CH])
    moments = acc.moments()
    print(f"[fit_ccm] recorded moments over {acc.count:,.0f} pairs")

    # ---- fit the W ladder + baselines ----
    Ws = {}
    Wh, bh = fit_W(moments, "hebbian")
    Ws["rung1_hebbian"] = (Wh, bh)
    Wr, br = fit_W(moments, "ridge", ridge_lambda=args.ridge_lambda)
    Ws["rung2_ridge"] = (Wr, br)
    Wp, bp = fit_W(moments, "procrustes")
    Ws["rung3_procrustes"] = (Wp, bp)
    if args.ablation:
        # whitening ablation 2×2 corners (PLAN §10.1 화이트닝 ablation):
        #   hebbian = no-center/no-whiten ; ridge = center+whiten (already above)
        #   centering_only = center, no whiten ; whitening_only = whiten, no center
        Wc, bc = fit_W(moments, "centering_only")
        Ws["centering_only"] = (Wc, bc)
        Ww, bw = fit_W(moments, "whitening_only", ridge_lambda=args.ridge_lambda)
        Ws["whitening_only"] = (Ww, bw)
    # random-W floor: matched scale, b = mean target.
    sigma = Wr.std().item()
    W_rand = torch.randn(d_model, d_agent, generator=gen_t).to(device) * sigma
    b_mean = moments["m_y"].to(torch.float32)
    Ws["random"] = (W_rand, b_mean)
    if d_model == d_agent:
        Ws["identity"] = (torch.eye(d_model), b_mean)
    # step3: a trained plastic bridge (W,b), evaluated through the SAME harness.
    if args.bridge_checkpoint is not None:
        bb = torch.load(args.bridge_checkpoint, map_location=device, weights_only=False)
        Ws["trained_bridge"] = (bb["W"].to(torch.float32), bb["b"].to(torch.float32))
        print(f"[fit_ccm] + trained_bridge from {args.bridge_checkpoint}")

    # ---- per-slot eval (subsampled) + swap (in-dist pairs) ----
    gold_id = [((g.agent_row, g.agent_col), g.heading, g.cheese_dir)
               for g in (describer_oracle(s) for s in st_id)]
    gold_ood = [((g.agent_row, g.agent_col), g.heading, g.cheese_dir)
                for g in (describer_oracle(s) for s in st_ood)]
    Hs_id, _sti, gs_id = _subsample(H_id, st_id, gold_id, args.max_eval, rng)
    Hs_ood, _sto, gs_ood = _subsample(H_ood, st_ood, gold_ood, args.max_eval, rng)

    # swap pairs from in-dist (different cheese_dir)
    idx_by_cd = {}
    for i, d in enumerate(CD_id):
        idx_by_cd.setdefault(d, []).append(i)
    cds = list(idx_by_cd.keys())
    pairs = []
    tries = 0
    while len(pairs) < args.n_pairs and tries < args.n_pairs * 50 and len(cds) >= 2:
        tries += 1
        da, db = rng.sample(cds, 2)
        pairs.append((rng.choice(idx_by_cd[da]), rng.choice(idx_by_cd[db])))
    A = torch.stack([H_id[a] for a, _ in pairs])
    B = torch.stack([H_id[b] for _, b in pairs])
    cdA = [CD_id[a] for a, _ in pairs]
    cdB = [CD_id[b] for _, b in pairs]
    alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
    print(f"[fit_ccm] {len(pairs)} swap pairs; eval states "
          f"in={len(gs_id):,} ood={len(gs_ood):,}")

    results = {"n_fit": int(acc.count), "d_model": d_model, "d_agent": d_agent,
               "cheese_dir_dist": {d: CD_id.count(d) for d in sorted(set(CD_id))},
               "ridge_lambda": args.ridge_lambda,
               "per_slot": {"in_dist": {}, "ood": {}}, "swap_following": {}}

    def _eval_config(name, gen_build):
        ps_in = _per_slot(_gen_slots(gen_build, Hs_id, tokenizer, device), gs_id)
        ps_ood = _per_slot(_gen_slots(gen_build, Hs_ood, tokenizer, device), gs_ood)
        sw = _swap_following(gen_build, A, B, cdA, cdB, tokenizer, device, alphas)
        results["per_slot"]["in_dist"][name] = ps_in
        results["per_slot"]["ood"][name] = ps_ood
        results["swap_following"][name] = sw
        print(f"  [{name:16}] in slot={ps_in['mean']:.3f} "
              f"(reg{ps_in['agent_region']:.2f}/hd{ps_in['heading']:.2f}/"
              f"ch{ps_in['cheese_dir']:.2f}) ood slot={ps_ood['mean']:.3f} "
              f"(ch{ps_ood['cheese_dir']:.2f}) swap={sw['rate']:.3f} "
              f"(n_readA={sw['n_readA']})")
        return ps_in, ps_ood, sw

    metrics = {}
    for name, (W, b) in Ws.items():
        bridge.set_W(W.to(device), b.to(device))
        metrics[name] = _eval_config(name, bridge)
    if has_b4:
        metrics["B4_ceiling"] = _eval_config("B4_ceiling", b4)

    # ---- pre-registered verdicts ----
    def slot_in(n): return metrics[n][0]["mean"]
    def swap(n): return metrics[n][2]["rate"]

    r1, rnd = "rung1_hebbian", "random"
    mt_slot = slot_in(r1) - slot_in(rnd)
    mt_swap = swap(r1) - swap(rnd)
    memory_transfer = bool(mt_slot >= 0.10 and mt_swap >= 0.10)

    norm_cfgs = ["rung2_ridge", "rung3_procrustes"]
    best_norm = max(norm_cfgs, key=lambda n: (swap(n) if swap(n) == swap(n) else -1))
    nk_swap = swap(best_norm) - swap(r1)
    nk_slot = slot_in(best_norm) - slot_in(r1)
    normalization_key = bool(nk_swap >= 0.10 and nk_slot >= 0.05)

    rungs = ["rung1_hebbian", "rung2_ridge", "rung3_procrustes"]
    best_rung = max(rungs, key=lambda n: (swap(n) if swap(n) == swap(n) else -1))
    b4s = swap("B4_ceiling") if has_b4 else None
    b4sl = slot_in("B4_ceiling") if has_b4 else None
    results["verdicts"] = {
        "step1_memory_transfer": {
            "rung1_slot_in": slot_in(r1), "random_slot_in": slot_in(rnd),
            "delta_slot": mt_slot, "rung1_swap": swap(r1), "random_swap": swap(rnd),
            "delta_swap": mt_swap, "pass": memory_transfer},
        "normalization_key": {
            "best_norm": best_norm, "delta_swap": nk_swap, "delta_slot": nk_slot,
            "pass": normalization_key},
        "ceiling": {
            "best_rung": best_rung, "best_rung_swap": swap(best_rung),
            "b4_swap": b4s, "swap_pct_of_b4": (swap(best_rung) / b4s if b4s else None),
            "best_rung_slot_in": slot_in(best_rung), "b4_slot_in": b4sl,
            "slot_pct_of_b4": (slot_in(best_rung) / b4sl if b4sl else None)},
    }

    # ---- whitening-ablation verdict (PLAN §10.1, frozen pre-reg) ----
    if args.ablation:
        co, wo = "centering_only", "whitening_only"
        rg = "rung2_ridge"
        wo_swap, co_swap = swap(wo), swap(co)
        wo_slot, co_slot = slot_in(wo), slot_in(co)
        d_swap = wo_swap - co_swap
        d_slot = wo_slot - co_slot
        rg_swap = swap(rg)
        recover = (wo_swap / rg_swap) if rg_swap else None
        whitening_decisive = bool(d_swap >= 0.10 and d_slot >= 0.05)
        results["verdicts"]["whitening_decisive"] = {
            "whitening_only_swap": wo_swap, "centering_only_swap": co_swap,
            "delta_swap": d_swap, "whitening_only_slot_in": wo_slot,
            "centering_only_slot_in": co_slot, "delta_slot": d_slot,
            "ridge_swap": rg_swap, "whitening_only_recover_pct_of_ridge": recover,
            "pass": whitening_decisive}

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_path, "w") as f:
        json.dump(results, f, indent=2)
    print("\n[fit_ccm] verdicts (1 seed, descriptive):")
    print(f"  memory transfers (rung1 vs random): slot Δ={mt_slot:+.3f} "
          f"swap Δ={mt_swap:+.3f} → {'YES' if memory_transfer else 'no'}")
    print(f"  normalization key ({best_norm} vs rung1): swap Δ={nk_swap:+.3f} "
          f"slot Δ={nk_slot:+.3f} → {'YES' if normalization_key else 'no'}")
    if b4s:
        print(f"  ceiling: best={best_rung} swap={swap(best_rung):.3f} = "
              f"{swap(best_rung)/b4s*100:.0f}% of B4 ({b4s:.3f})")
    else:
        print(f"  ceiling: best={best_rung} swap={swap(best_rung):.3f} (no B4 reference)")
    if args.ablation:
        wv = results["verdicts"]["whitening_decisive"]
        rec = wv["whitening_only_recover_pct_of_ridge"]
        rec_str = f"{rec*100:.0f}% of ridge {wv['ridge_swap']:.3f}" if rec is not None else "n/a"
        print(f"  whitening decisive (whitening_only vs centering_only): "
              f"swap Δ={wv['delta_swap']:+.3f} slot Δ={wv['delta_slot']:+.3f} → "
              f"{'YES' if wv['pass'] else 'no'}  "
              f"(whiten-only swap={wv['whitening_only_swap']:.3f} = {rec_str})")
    print(f"[fit_ccm] → {args.output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
