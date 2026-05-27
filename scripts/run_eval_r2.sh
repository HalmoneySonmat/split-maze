#!/usr/bin/env bash
# Phase 6 — clean P2 eval on a SHARED frozen OOD set (PREREG §0.7, fix #3).
# Needs agent_r2.pt + agent_r0matched.pt from train_r2.
#
# Usage:
#   bash scripts/run_eval_r2.sh 0 1        # quick pipeline smoke (1 seed, 1 rollout)
#   bash scripts/run_eval_r2.sh 0,1,2 4    # full eval
set -euo pipefail

if [ "${CONDA_DEFAULT_ENV:-}" != "splitmaze" ]; then
  if command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate splitmaze
    echo "[run_eval_r2] activated conda env: splitmaze"
  else
    echo "WARN: conda not found — activate the env first." >&2
  fi
fi

SEEDS="${1:-0,1,2}"
ROLLOUTS="${2:-4}"
mkdir -p results

PYTHONPATH=src python scripts/eval_r2.py \
    --seeds "$SEEDS" --rollouts "$ROLLOUTS" --device cuda \
    --output_path results/r2_p2.json
