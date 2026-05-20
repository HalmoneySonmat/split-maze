"""Training CLI for the maze-language LM (Phase 2.2 산출물).

Trains a from-scratch decoder transformer on the neutral synthetic-maze
corpus (PLAN §3.3) and runs the Phase 2 gate (PLAN P2-5..P2-8 frozen
2026-05-19):

- corpus N = 50_000 (PLAN P2-4)
- AdamW(lr=3e-4, weight_decay=0.01) + grad clip 1.0 (PLAN P2-5)
- 10 epochs, batch=64                             (PLAN P2-6)
- λ_ae = 1.0                                       (PLAN P2-3)
- gate: held-out sequence-exact ≥ 0.95             (PLAN P2-7)
        AND 72/72 (HEADING, CHEESE_DIR) combos     (PLAN P2-8)

Typical usage (WSL, ~수분 on CPU; faster on CUDA):

  PYTHONPATH=src python scripts/train_lm.py \\
      --corpus_size 50000 --epochs 10 --batch 64 \\
      --device cuda --seed 0 \\
      --save_path checkpoints/lm.pt \\
      --log_path  logs/lm.jsonl \\
      --gate_path results/lm_gate.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from split_maze.lm import LMConfig, MazeLM, MazeTokenizer
from split_maze.lm_train import (
    LMTrainConfig,
    build_corpus_ids,
    evaluate_72_combinations,
    evaluate_roundtrip,
    gate_pass,
    split_train_held,
    train_lm,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    # Data.
    p.add_argument("--corpus_size", type=int, default=50_000)
    p.add_argument("--train_frac", type=float, default=0.9)
    # Optimization (PLAN P2-5..P2-6).
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--lambda_ae", type=float, default=1.0)
    p.add_argument("--warmup_steps", type=int, default=500,
                   help="Linear LR warm-up steps (POST-HOC-4).")
    # LM architecture (PLAN P2-2).
    p.add_argument("--d_model", type=int, default=256)
    p.add_argument("--n_head", type=int, default=4)
    p.add_argument("--n_layer", type=int, default=3)
    p.add_argument("--d_ff", type=int, default=1024)
    p.add_argument("--max_len", type=int, default=32)
    p.add_argument("--dropout", type=float, default=0.1)
    # Run config.
    p.add_argument("--device", type=str, default="cpu",
                   choices=("cpu", "cuda"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save_path", type=str, default=None,
                   help="Checkpoint output (.pt).")
    p.add_argument("--log_path", type=str, default=None,
                   help="Per-epoch JSONL log.")
    p.add_argument("--gate_path", type=str, default=None,
                   help="Final gate-result JSON output.")
    p.add_argument("--roundtrip_n", type=int, default=1000,
                   help="Number of held-out sentences for the final "
                        "round-trip gate evaluation.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    device = torch.device(args.device)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested but CUDA is unavailable")

    # ---- Tokenizer + LM ----
    tokenizer = MazeTokenizer()
    lm_cfg = LMConfig.from_tokenizer(
        tokenizer,
        d_model=args.d_model,
        n_head=args.n_head,
        n_layer=args.n_layer,
        d_ff=args.d_ff,
        max_len=args.max_len,
        dropout=args.dropout,
    )
    model = MazeLM(lm_cfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[init] MazeLM: {n_params:_} params, "
          f"d_model={args.d_model} n_layer={args.n_layer} "
          f"n_head={args.n_head} vocab={tokenizer.vocab_size}")

    # ---- Corpus ----
    t0 = time.time()
    all_ids = build_corpus_ids(args.corpus_size, seed=args.seed,
                                tokenizer=tokenizer)
    train_ids, held_ids = split_train_held(
        all_ids, train_frac=args.train_frac, seed=args.seed,
    )
    print(f"[data] corpus={len(all_ids):_}  "
          f"train={len(train_ids):_}  held={len(held_ids):_}  "
          f"({time.time() - t0:.1f}s to build)")

    # ---- Train ----
    train_cfg = LMTrainConfig(
        epochs=args.epochs,
        batch_size=args.batch,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        lambda_ae=args.lambda_ae,
        train_frac=args.train_frac,
        device=args.device,
        seed=args.seed,
        warmup_steps=args.warmup_steps,
    )
    save_path = Path(args.save_path) if args.save_path else None
    log_path = Path(args.log_path) if args.log_path else None
    epoch_metrics = train_lm(
        model, tokenizer, train_ids, held_ids, train_cfg,
        log_path=log_path, save_path=save_path,
        print_each_epoch=True,
    )

    # ---- Final gate evaluation ----
    rt_subset = held_ids[:min(args.roundtrip_n, len(held_ids))]
    rt = evaluate_roundtrip(model, tokenizer, rt_subset, device=device)
    combo = evaluate_72_combinations(model, tokenizer, device=device)
    verdict = gate_pass(rt, combo)

    print()
    print("=== Phase 2 gate (PLAN P2-7 post-hoc 2026-05-19) ===")
    print(f"  roundtrip slot match  : {rt['slot_match_rate']:.4f}"
          f"  (≥ {verdict['slot_threshold']:.2f} ?"
          f" {'PASS' if verdict['slot_pass'] else 'FAIL'})"
          f"  [n={rt['num_sentences']:_}]")
    print(f"    agent_region         : {rt['agent_match_rate']:.4f}"
          f"  (row={rt['agent_row_match_rate']:.4f},"
          f" col={rt['agent_col_match_rate']:.4f})")
    print(f"    heading              : {rt['heading_match_rate']:.4f}")
    print(f"    cheese_dir           : {rt['cheese_dir_match_rate']:.4f}")
    print(f"  combo 72 pass rate    : {combo['pass_rate']:.4f}"
          f"  (= {verdict['combo_threshold']:.2f} ?"
          f" {'PASS' if verdict['combo_pass'] else 'FAIL'})"
          f"  [{combo['num_passed']}/{combo['num_total']}]")
    print(f"  (diagnostic) exact    : {rt['exact_match_rate']:.4f}"
          f"   combo_exact={combo['exact_rate']:.4f}")
    print(f"  -> Phase 2 verdict   : "
          f"{'PASS' if verdict['pass'] else 'FAIL'}")
    if not verdict["pass"] and combo["failed_examples"]:
        print("\nFirst few combo failures (heading, cheese_dir):")
        for ex in combo["failed_examples"]:
            print(f"  ({ex['heading']:>10}, {ex['cheese_dir']:>10}) "
                  f"-> parsed=({ex['parsed_heading']!r}, "
                  f"{ex['parsed_cheese_dir']!r}) gen={ex['generated_tokens']}")

    # ---- Save gate JSON ----
    if args.gate_path:
        Path(args.gate_path).parent.mkdir(parents=True, exist_ok=True)
        result = {
            "args": vars(args),
            "verdict": verdict,
            "roundtrip": rt,
            "combo_72": combo,
            "epoch_metrics": epoch_metrics,
            "params": n_params,
        }
        with open(args.gate_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nWrote gate result to {args.gate_path}")


if __name__ == "__main__":
    main()
