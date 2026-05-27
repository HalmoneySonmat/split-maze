#!/usr/bin/env bash
# Phase 6 PILOT — one-line runner for the grounded-confabulation pre-reg.
# Spec: docs/PREREG_grounded_confab.md  (PLAN §10.6)
#
# Eval-only (NO retraining): reuses frozen Phase-3 checkpoints to freeze the
# gate numbers — (0a) premise check (does h encode the OOD goal?),
# (0b) floor/ceiling/in-dist bar/variance/null, (0c) commit-ratio reinterpret.
#
# Usage:
#   bash scripts/run_pilot.sh                # seeds 0,1,2  rollouts 8
#   bash scripts/run_pilot.sh "0,1,2,3,4" 12 # custom seeds + rollouts (tighter CI)
set -euo pipefail

# --- activate the project env (torch/numpy/procgen live in conda env 'splitmaze') ---
# numpy ModuleNotFoundError in (base) => you forgot to activate splitmaze.
if [ "${CONDA_DEFAULT_ENV:-}" != "splitmaze" ]; then
  if command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate splitmaze
    echo "[run_pilot] activated conda env: splitmaze"
  else
    echo "WARN: conda not found — activate the env with torch/numpy/procgen first." >&2
  fi
fi

SEEDS="${1:-0,1,2}"
ROLLOUTS="${2:-8}"
mkdir -p results

PYTHONPATH=src python scripts/pilot_grounding.py \
    --lm_checkpoint checkpoints/lm.pt \
    --agent_checkpoint checkpoints/phase3/agent.pt \
    --v2_checkpoint checkpoints/phase3/V2_postfix2.pt \
    --b4_checkpoint checkpoints/phase3/B4.pt \
    --b3_checkpoint checkpoints/phase3/B3.pt \
    --device cuda --seeds "$SEEDS" --rollouts "$ROLLOUTS" \
    --output_path results/pilot_grounding.json
