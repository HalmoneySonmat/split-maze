#!/usr/bin/env bash
# Phase 6 — R0 baseline + echo diagnostic (PREREG §3). Eval-only, no retraining.
# Spec: docs/PREREG_grounded_confab.md
#
# Usage:
#   bash scripts/run_eval_regimes.sh                # seeds 0,1,2  rollouts 8
#   bash scripts/run_eval_regimes.sh "0,1,2,3,4" 12
set -euo pipefail

# --- activate project env (torch/numpy/procgen live in conda env 'splitmaze') ---
if [ "${CONDA_DEFAULT_ENV:-}" != "splitmaze" ]; then
  if command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate splitmaze
    echo "[run_eval_regimes] activated conda env: splitmaze"
  else
    echo "WARN: conda not found — activate the env with torch/numpy/procgen first." >&2
  fi
fi

SEEDS="${1:-0,1,2}"
ROLLOUTS="${2:-8}"
mkdir -p results

PYTHONPATH=src python scripts/eval_regimes.py \
    --lm_checkpoint checkpoints/lm.pt \
    --agent_checkpoint checkpoints/phase3/agent.pt \
    --v2_checkpoint checkpoints/phase3/V2_postfix2.pt \
    --b4_checkpoint checkpoints/phase3/B4.pt \
    --b3_checkpoint checkpoints/phase3/B3.pt \
    --device cuda --seeds "$SEEDS" --rollouts "$ROLLOUTS" \
    --output_path results/regimes_baseline.json
