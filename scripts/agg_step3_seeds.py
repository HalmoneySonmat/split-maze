"""Aggregate step3 multi-seed runs (PLAN §10.1 step3 multi-seed 사전등록).

Reads the per-seed fit_ccm eval JSONs produced by run_step3_multiseed.sh:
  results/phase5_ccm_step3_a2_s{k}.json      (A2: adapted system + trained bridge)
  results/phase5_ccm_step3_a1long_s{k}.json  (A1-long: frozen system, W-budget control)

For each seed computes the co-adaptation effect = swap(A2) − swap(A1-long), and
reports mean±std + the FROZEN pre-registered verdict:
  confirmed ⇔ ≥4/5 seeds effect>0 AND mean(effect) ≥ +0.03.
Also reports the W-independent corroboration: the recorded-ridge swap on the A2
(adapted) system vs the frozen baseline 0.445 (a rise ⇒ representations genuinely
became more alignable, gradient-free).

Run:
  PYTHONPATH=src python scripts/agg_step3_seeds.py --seeds "1 2 3 4 5"
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
    xs = [x for x in xs if x == x]  # drop nan
    if not xs:
        return float("nan"), float("nan")
    m = sum(xs) / len(xs)
    if len(xs) < 2:
        return m, 0.0
    var = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return m, math.sqrt(var)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="1 2 3 4 5")
    ap.add_argument("--results_dir", type=Path, default=Path("results"))
    ap.add_argument("--frozen_recorded_ridge", type=float, default=0.445,
                    help="step1 frozen recorded-ridge swap (W-indep baseline)")
    args = ap.parse_args()
    seeds = [s for s in args.seeds.split() if s]

    rows = []
    for k in seeds:
        fa2 = args.results_dir / f"phase5_ccm_step3_a2_s{k}.json"
        fal = args.results_dir / f"phase5_ccm_step3_a1long_s{k}.json"
        if not fa2.exists() or not fal.exists():
            print(f"  [seed {k}] MISSING ({fa2.name} / {fal.name}) — skipped")
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
        print("No seed results found. Run run_step3_multiseed.sh first.")
        return 1

    print(f"\nstep3 multi-seed (N={len(rows)}, same frozen agent+LM; varies step3 procedure+eval)\n")
    print(f"{'seed':>4} | {'A2 swap':>8} | {'A1long swap':>11} | {'effect=A2-A1long':>16} | "
          f"{'A2 slot':>8} | {'A2 recd-ridge':>13}")
    print("-" * 78)
    effects = []
    for r in rows:
        eff = r["a2_swap"] - r["a1long_swap"]
        effects.append(eff)
        print(f"{r['seed']:>4} | {r['a2_swap']:>8.3f} | {r['a1long_swap']:>11.3f} | "
              f"{eff:>+16.3f} | {r['a2_slot']:>8.3f} | {r['a2_recorded_ridge_swap']:>13.3f}")

    a2_m, a2_s = _mean_std([r["a2_swap"] for r in rows])
    al_m, al_s = _mean_std([r["a1long_swap"] for r in rows])
    ef_m, ef_s = _mean_std(effects)
    rr_m, rr_s = _mean_std([r["a2_recorded_ridge_swap"] for r in rows])
    n_pos = sum(1 for e in effects if e == e and e > 0)
    n = len([e for e in effects if e == e])

    print("-" * 78)
    print(f"{'mean':>4} | {a2_m:>8.3f} | {al_m:>11.3f} | {ef_m:>+16.3f} | "
          f"{'':>8} | {rr_m:>13.3f}")
    print(f"{'std':>4} | {a2_s:>8.3f} | {al_s:>11.3f} | {ef_s:>16.3f} | {'':>8} | {rr_s:>13.3f}")

    # ---- frozen pre-registered verdict ----
    confirmed = (n > 0 and n_pos >= math.ceil(0.8 * n) and ef_m >= 0.03)
    print(f"\n[verdict] co-adaptation effect (frozen pre-reg: ≥4/5 seeds >0 AND mean ≥ +0.03):")
    print(f"  positive seeds: {n_pos}/{n}   mean effect: {ef_m:+.3f} (±{ef_s:.3f})")
    print(f"  → {'CONFIRMED' if confirmed else 'NOT confirmed'} "
          f"(Claim 2 공동적응 순수 이득 {'실재' if confirmed else '1-seed 운 재해석→plastic W 성장으로 하향'})")
    print(f"[corroboration] adapted recorded-ridge swap mean {rr_m:.3f} vs frozen "
          f"{args.frozen_recorded_ridge:.3f} → {'↑ 표상 정렬 실재(W-무관)' if rr_m > args.frozen_recorded_ridge else '미상승'}")
    print(f"[note] same frozen agent+LM — '이 뇌에서 진짜냐'까지. 다른 뇌 일반화는 RL-seed(②, 큐).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
