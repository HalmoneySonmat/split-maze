#!/usr/bin/env bash
# Phase 5 CCM — RL-seed generalization driver (PLAN §10.1 RL-seed 사전등록).
#
# Runs the step3 pipeline (A1 → A2 → A1-long control) + eval on each NEW RL-seed
# agent ("brain"). Tests whether the co-adaptation effect (A2 − A1-long) holds
# across *different decider brains*, not just across step3-procedure seeds on one
# brain. The LM (interpreter) is shared — it is neutral (no agent prior) — so the
# varied variable is the RL agent. B4 ceiling is skipped (no per-brain B4).
#
# Prereq — train the new brains first (each ~5–8h, run overnight):
#   PYTHONPATH=src python scripts/train_agent.py --env_name maze_aisc \
#     --num_envs 64 --num_steps 256 --total_env_steps 25000000 --num_levels 200 \
#     --seed K --save_path checkpoints/brains/agent_sK.pt --log_path logs/brains/agent_sK.jsonl
#
# Usage (after brains are trained):
#   bash scripts/run_brains_step3.sh "1 2 3"
set -euo pipefail

SEEDS="${1:-1 2 3}"
LM="checkpoints/lm.pt"
CKPT="checkpoints/phase5/brains"
RES="results"
mkdir -p "$CKPT" "$RES" logs/brains

for k in $SEEDS; do
  AG="checkpoints/brains/agent_s${k}.pt"
  if [ ! -f "$AG" ]; then
    echo "MISSING brain agent $AG — train it first (train_agent.py --seed $k)"; exit 1
  fi
  echo "================  brain $k ($AG)  ================"
  ba1="$CKPT/bridge_a1_s${k}.pt"; ba2="$CKPT/bridge_a2_s${k}.pt"
  ag2="$CKPT/agent_a2_s${k}.pt"; lm2="$CKPT/lm_a2_s${k}.pt"; bal="$CKPT/bridge_a1long_s${k}.pt"

  echo "--- [brain $k] A1 (warm W, 60 upd) ---"
  PYTHONPATH=src python scripts/train_ccm_step3.py --phase a1 --lm_checkpoint "$LM" \
    --agent_checkpoint "$AG" --device cuda --seed "$k" --a1_updates 60 --warmup_record 3 \
    --out_bridge "$ba1" --log_path "logs/brains/a1_s${k}.jsonl"

  echo "--- [brain $k] A2 (co-adapt, 100 upd) ---"
  PYTHONPATH=src python scripts/train_ccm_step3.py --phase a2 --lm_checkpoint "$LM" \
    --agent_checkpoint "$AG" --in_bridge "$ba1" --device cuda --seed "$k" --a2_updates 100 \
    --out_bridge "$ba2" --out_agent "$ag2" --out_lm "$lm2" --log_path "logs/brains/a2_s${k}.jsonl"

  echo "--- [brain $k] A1-long (W-budget control, 120 upd) ---"
  PYTHONPATH=src python scripts/train_ccm_step3.py --phase a1 --lm_checkpoint "$LM" \
    --agent_checkpoint "$AG" --device cuda --seed "$k" --a1_updates 120 --warmup_record 3 \
    --out_bridge "$bal" --log_path "logs/brains/a1long_s${k}.jsonl"

  echo "--- [brain $k] eval A2 (adapted system) ---"
  PYTHONPATH=src python scripts/fit_ccm.py --lm_checkpoint "$lm2" --agent_checkpoint "$ag2" \
    --b4_checkpoint none --bridge_checkpoint "$ba2" --device cuda --seed "$k" --rollouts 15 --n_pairs 1000 \
    --output_path "$RES/phase5_ccm_brain_a2_s${k}.json"

  echo "--- [brain $k] eval A1-long (frozen system, control) ---"
  PYTHONPATH=src python scripts/fit_ccm.py --lm_checkpoint "$LM" --agent_checkpoint "$AG" \
    --b4_checkpoint none --bridge_checkpoint "$bal" --device cuda --seed "$k" --rollouts 15 --n_pairs 1000 \
    --output_path "$RES/phase5_ccm_brain_a1long_s${k}.json"

  echo "--- [brain $k] done ---"
done

echo "All brains done. Aggregate with:"
echo "  PYTHONPATH=src python scripts/agg_brains.py --seeds \"$SEEDS\""
