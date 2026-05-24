#!/usr/bin/env bash
# Phase 5 CCM — step3 절차 multi-seed driver (PLAN §10.1 step3 multi-seed 사전등록).
#
# Same frozen Phase-3 agent + Phase-2 LM; vary the step3 seed (rollouts, bridge
# sampling, PPO, LM-block adaptation, eval pairs). Per seed: A1 → A2 (warm from A1)
# → A1-long (W-budget control), then eval A2 (adapted LM+agent+bridge) and A1-long
# (frozen agent+LM+bridge). The aggregator (agg_step3_seeds.py) reads the per-seed
# eval JSONs and computes effect_k = swap(A2) − swap(A1-long).
#
# Usage (WSL, from repo root, splitmaze env):
#   bash scripts/run_step3_multiseed.sh "1 2 3 4 5"
# Tip: run one seed first to confirm the pipeline + time it, then the rest:
#   bash scripts/run_step3_multiseed.sh "1"
set -euo pipefail

SEEDS="${1:-1 2 3 4 5}"
LM="checkpoints/lm.pt"
AGENT="checkpoints/phase3/agent.pt"
B4="checkpoints/phase3/B4.pt"
CKPT="checkpoints/phase5/ms"
RES="results"
mkdir -p "$CKPT" "$RES" logs/ms

for k in $SEEDS; do
  echo "================  seed $k  ================"
  ba1="$CKPT/bridge_a1_s${k}.pt"
  ba2="$CKPT/bridge_a2_s${k}.pt"
  ag2="$CKPT/agent_a2_s${k}.pt"
  lm2="$CKPT/lm_a2_s${k}.pt"
  bal="$CKPT/bridge_a1long_s${k}.pt"

  echo "--- [seed $k] A1 (warm W, 60 upd) ---"
  PYTHONPATH=src python scripts/train_ccm_step3.py --phase a1 \
    --lm_checkpoint "$LM" --agent_checkpoint "$AGENT" --device cuda --seed "$k" \
    --a1_updates 60 --warmup_record 3 --out_bridge "$ba1" \
    --log_path "logs/ms/a1_s${k}.jsonl"

  echo "--- [seed $k] A2 (co-adapt, 100 upd) ---"
  PYTHONPATH=src python scripts/train_ccm_step3.py --phase a2 \
    --lm_checkpoint "$LM" --agent_checkpoint "$AGENT" --in_bridge "$ba1" \
    --device cuda --seed "$k" --a2_updates 100 \
    --out_bridge "$ba2" --out_agent "$ag2" --out_lm "$lm2" \
    --log_path "logs/ms/a2_s${k}.jsonl"

  echo "--- [seed $k] A1-long (W-budget control, 120 upd) ---"
  PYTHONPATH=src python scripts/train_ccm_step3.py --phase a1 \
    --lm_checkpoint "$LM" --agent_checkpoint "$AGENT" --device cuda --seed "$k" \
    --a1_updates 120 --warmup_record 3 --out_bridge "$bal" \
    --log_path "logs/ms/a1long_s${k}.jsonl"

  echo "--- [seed $k] eval A2 (adapted system) ---"
  PYTHONPATH=src python scripts/fit_ccm.py \
    --lm_checkpoint "$lm2" --agent_checkpoint "$ag2" --b4_checkpoint "$B4" \
    --bridge_checkpoint "$ba2" --device cuda --seed "$k" --rollouts 15 --n_pairs 1000 \
    --output_path "$RES/phase5_ccm_step3_a2_s${k}.json"

  echo "--- [seed $k] eval A1-long (frozen system, control) ---"
  PYTHONPATH=src python scripts/fit_ccm.py \
    --lm_checkpoint "$LM" --agent_checkpoint "$AGENT" --b4_checkpoint "$B4" \
    --bridge_checkpoint "$bal" --device cuda --seed "$k" --rollouts 15 --n_pairs 1000 \
    --output_path "$RES/phase5_ccm_step3_a1long_s${k}.json"

  echo "--- [seed $k] done ---"
done

echo "All seeds done. Aggregate with:"
echo "  PYTHONPATH=src python scripts/agg_step3_seeds.py --seeds \"$SEEDS\""
