"""Aggregate RL-seed generalization runs (PLAN §10.1 RL-seed 사전등록).

Reads the per-brain fit_ccm eval JSONs produced by run_brains_step3.sh:
  results/phase5_ccm_brain_a2_s{k}.json      (A2: adapted system + trained bridge)
  results/phase5_ccm_brain_a1long_s{k}.json  (A1-long: frozen brain, W-budget control)

For each *brain* (a different RL-seed agent) computes the co-adaptation effect
= swap(A2) − swap(A1-long), and reports the FROZEN pre-registered verdict:
  confirmed across brains ⇔ ≥(N−1) brains effect>0 AND mean(effect) ≥ +0.03.
Also reports the W-independent corroboration (adapted recorded-ridge swap vs the
frozen baseline 0.445) per brain. Note: absolute swaps differ across brains (each
agent represents differently); the *within-brain paired* effect is the comparable
quantity.

Run:
  PYTHONPATH=src python scripts/agg_brains.py --seeds "1 2 3"
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def _swap(blob, name):
    try:
        v = blob["swap_following"][name]["rate"]
        return float(v) if v is not None else float("nan")
    except (KeyError, TypeError):
        return float("nan")


def _slot_in(blob, name):
    try:
        return float(blob["per_slot"]["in_dist"][name]["mean"])
    except (KeyError, TypeError):
        return float("nan")


def _mean_std(xs):
    xs = [x for x in xs if x == x]
    if not xs:
        return float("nan"), float("nan")
    m = sum(xs) / len(xs)
    if len(xs) < 2:
        return m, 0.0
    return m, math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="1 2 3")
    ap.add_argument("--results_dir", type=Path, default=Path("results"))
    ap.add_argument("--frozen_recorded_ridge", type=float, default=0.445)
    args = ap.parse_args()
    seeds = [s for s in args.seeds.split() if s]

    rows = []
    for k in seeds:
        fa2 = args.results_dir / f"phase5_ccm_brain_a2_s{k}.json"
        fal = args.results_dir / f"phase5_ccm_brain_a1long_s{k}.json"
        if not fa2.exists() or not fal.exists():
            print(f"  [brain {k}] MISSING ({fa2.name} / {fal.name}) — skipped")
            continue
        a2 = json.loads(fa2.read_text())
        al = json.loads(fal.read_text())
        rows.append({
            "seed": k,
            "a2_swap": _swap(a2, "trained_bridge"),
            "a2_slot": _slot_in(a2, "trained_bridge"),
            "a1long_swap": _swap(al, "trained_bridge"),
            "a2_recorded_ridge_swap": _swap(a2, "rung2_ridge"),
        })

    if not rows:
        print("No brain results found. Train brains + run run_brains_step3.sh first.")
        return 1

    print(f"\nRL-seed generalization (N={len(rows)} brains = different RL agents; shared LM)\n")
    print(f"{'brain':>5} | {'A2 swap':>8} | {'A1long swap':>11} | {'effect=A2-A1long':>16} | "
          f"{'A2 slot':>8} | {'A2 recd-ridge':>13}")
    print("-" * 78)
    effects = []
    for r in rows:
        eff = r["a2_swap"] - r["a1long_swap"]
        effects.append(eff)
        print(f"{r['seed']:>5} | {r['a2_swap']:>8.3f} | {r['a1long_swap']:>11.3f} | "
              f"{eff:>+16.3f} | {r['a2_slot']:>8.3f} | {r['a2_recorded_ridge_swap']:>13.3f}")

    ef_m, ef_s = _mean_std(effects)
    rr_m, _ = _mean_std([r["a2_recorded_ridge_swap"] for r in rows])
    n = len([e for e in effects if e == e])
    n_pos = sum(1 for e in effects if e == e and e > 0)
    print("-" * 78)
    print(f"{'mean':>5} | {'':>8} | {'':>11} | {ef_m:>+16.3f} | {'':>8} | {rr_m:>13.3f}")
    print(f"{'std':>5} | {'':>8} | {'':>11} | {ef_s:>16.3f} |")

    # ---- frozen pre-registered cross-brain verdict ----
    confirmed = (n > 0 and n_pos >= (n - 1) and ef_m >= 0.03)
    print(f"\n[verdict] cross-brain co-adaptation (frozen pre-reg: ≥N−1 brains >0 AND mean ≥ +0.03):")
    print(f"  positive brains: {n_pos}/{n}   mean effect: {ef_m:+.3f} (±{ef_s:.3f})")
    print(f"  → {'GENERALIZES' if confirmed else 'does NOT generalize (per frozen criterion)'}")
    print(f"[corroboration] adapted recorded-ridge swap mean {rr_m:.3f} vs frozen "
          f"{args.frozen_recorded_ridge:.3f} → {'↑ (W-independent alignment holds across brains)' if rr_m > args.frozen_recorded_ridge else 'not raised'}")
    print(f"[note] N={n} brains is a probe; absolute swaps differ per brain — the paired effect is the comparable quantity. Shared LM (interpreter not varied).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
