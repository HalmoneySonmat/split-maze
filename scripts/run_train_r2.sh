#!/usr/bin/env bash
# Phase 6 — train R2 (V2 closed loop) AND matched-R0 with identical PPO budget.
# Spec: docs/PREREG_grounded_confab.md §0.7 / P2.
#
# Both modes start from the SAME base agent; only the feedback differs
# (matched-training control). Run a SHORT smoke first (e.g. 20) to confirm the
# GPU path, then the full budget.
#
# Usage:
#   bash scripts/run_train_r2.sh 20     # GPU smoke (fast)
#   bash scripts/run_train_r2.sh 300    # full budget
set -euo pipefail

if [ "${CONDA_DEFAULT_ENV:-}" != "splitmaze" ]; then
  if command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate splitmaze
    echo "[run_train_r2] activated conda env: splitmaze"
  else
    echo "WARN: conda not found — activate the env first." >&2
  fi
fi

UPDATES="${1:-300}"
mkdir -p checkpoints/phase6 logs/phase6

echo "=== R2 (V2 closed loop, feedback on) — $UPDATES updates ==="
PYTHONPATH=src python scripts/train_r2.py --mode r2 --num_updates "$UPDATES" --device cuda

echo "=== matched-R0 (no feedback, same budget) — $UPDATES updates ==="
PYTHONPATH=src python scripts/train_r2.py --mode r0matched --num_updates "$UPDATES" --device cuda

echo ""
echo "Done. Next: eval decisive-faithful (V2) on each agent:"
echo "  PYTHONPATH=src python scripts/eval_regimes.py --agent_checkpoint checkpoints/phase6/agent_r2.pt        --output_path results/r2_eval.json"
echo "  PYTHONPATH=src python scripts/eval_regimes.py --agent_checkpoint checkpoints/phase6/agent_r0matched.pt --output_path results/r0matched_eval.json"
echo "  P2 = decisive_faithful(V2, r2) − decisive_faithful(V2, r0matched)   [gate: p<0.01 AND ≥+0.05]"
echo "  (NOTE: those evals use per-agent states; the SHARED-frozen-set eval (PREREG fix #3) is a separate step.)"
