"""Phase-6 R2 training (V2 closed loop) + matched-R0. PREREG §0.7 / P2.

Co-adapts the base Phase-3 agent under the V2 feedback gate (--mode r2) or with
NO feedback for the same PPO budget (--mode r0matched). Both start from the SAME
base agent, so the only difference is the feedback (matched-training control,
PREREG §4). The bridge + LM (V2) are frozen — only the agent updates.

Output: checkpoints/phase6/agent_{mode}.pt  (+ logs/phase6/r2_{mode}.jsonl)

Run BOTH with identical --num_updates, then eval decisive-faithful on each agent
(scripts/eval_regimes-style) with V2 as the interpreter (PREREG §0.7).

Run (WSL, GPU):
  PYTHONPATH=src python scripts/train_r2.py --mode r2        --num_updates 300
  PYTHONPATH=src python scripts/train_r2.py --mode r0matched --num_updates 300
or: bash scripts/run_train_r2.sh 300
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from split_maze.acc import ACC, ACCConfig
from split_maze.agent import ImpalaAgent
from split_maze.builds import V2ACC
from split_maze.lm import LMConfig, MazeLM, MazeTokenizer
from split_maze.ppo import PPOConfig
from split_maze.train_phase3 import train_r2


def _fresh_lm(lm_blob, device):
    lm = MazeLM(LMConfig(**lm_blob["lm_config"]))
    lm.load_state_dict(lm_blob["model_state"])
    return lm.to(device).eval()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["r2", "r0matched"], required=True)
    p.add_argument("--lm_checkpoint", type=Path, default=Path("checkpoints/lm.pt"))
    p.add_argument("--base_agent", type=Path, default=Path("checkpoints/phase3/agent.pt"),
                   help="base agent both R2 and matched-R0 start from")
    p.add_argument("--v2_checkpoint", type=Path, default=Path("checkpoints/phase3/V2_postfix2.pt"))
    p.add_argument("--env_name", default="maze_aisc")
    p.add_argument("--num_envs", type=int, default=64)
    p.add_argument("--num_steps", type=int, default=256)
    p.add_argument("--num_levels", type=int, default=0)
    p.add_argument("--num_updates", type=int, default=300)
    p.add_argument("--lam", type=float, default=0.3, help="fixed feedback gate λ (primary=0.3)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=("cuda" if torch.cuda.is_available() else "cpu"))
    p.add_argument("--save_path", type=Path, default=None)
    p.add_argument("--log_path", type=Path, default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device(args.device)
    feedback_on = (args.mode == "r2")
    save_path = args.save_path or Path(f"checkpoints/phase6/agent_{args.mode}.pt")
    log_path = args.log_path or Path(f"logs/phase6/r2_{args.mode}.jsonl")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    lm_blob = torch.load(args.lm_checkpoint, map_location=device, weights_only=False)
    # base agent — both modes start identical
    agent = ImpalaAgent().to(device)
    agent.load_state_dict(torch.load(args.base_agent, map_location=device,
                                     weights_only=False)["agent"])
    # frozen V2 interpreter (bidirectional bridge for the feedback)
    acc = ACC(ACCConfig(d_agent=agent.d_a, d_lm=lm_blob["lm_config"]["d_model"], tied=False))
    v2 = V2ACC(_fresh_lm(lm_blob, device), acc).to(device)
    v2.load_state_dict(torch.load(args.v2_checkpoint, map_location=device,
                                  weights_only=False)["state"]); v2.eval()

    from split_maze.env import make_maze_env
    env = make_maze_env(env_name=args.env_name, num=args.num_envs, num_levels=args.num_levels,
                        start_level=0, distribution_mode="easy", rand_seed=args.seed)

    print(f"[train_r2] mode={args.mode} feedback_on={feedback_on} λ={args.lam} "
          f"updates={args.num_updates} env={args.env_name} | base={args.base_agent}")

    log_f = open(log_path, "w")

    def _log(upd, log):
        log_f.write(json.dumps(log) + "\n"); log_f.flush()
        if upd % 10 == 0 or upd == args.num_updates - 1:
            print(f"  upd {upd:4d}  ret={log['mean_return']:.3f}  "
                  f"policy={log['policy']:+.3f} value={log['value']:.3f} "
                  f"entropy={log['entropy']:.3f} kl={log['approx_kl']:+.4f}")

    logs = train_r2(env, agent, v2, ppo_config=PPOConfig(), num_updates=args.num_updates,
                    num_steps=args.num_steps, lam=args.lam, feedback_on=feedback_on,
                    device=device, log_callback=_log)
    log_f.close()

    torch.save({"agent": agent.state_dict(), "mode": args.mode, "lam": args.lam,
                "num_updates": args.num_updates, "base_agent": str(args.base_agent)},
               save_path)
    last = logs[-1] if logs else {}
    print(f"\n[train_r2] {args.mode} done. final ret={last.get('mean_return', float('nan')):.3f}")
    print(f"  agent → {save_path}")
    print(f"  log   → {log_path}")
    print("Next: run BOTH modes with same --num_updates, then eval decisive-faithful "
          "(V2 interpreter) on each agent → P2 = decisive(r2) − decisive(r0matched).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
