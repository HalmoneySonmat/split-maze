"""PPO algorithm: rollout buffer, GAE(λ), clipped-surrogate loss, multi-epoch updater.

Standard procgen PPO defaults (Cobbe et al. 2020, joonleesky/train-procgen-pytorch):
- γ = 0.999, λ = 0.95
- clip ε = 0.2, entropy coef = 0.01, value coef = 0.5
- N=64 parallel envs × T=256 steps per rollout
- ppo_epochs=3, mini_batches/epoch=8
- lr 5e-4 (Adam), grad clip 0.5
- Advantage normalized per mini-batch.

gym3 convention: ``first[t+1]`` flags that the step at t was terminal (env reset
between t and t+1). The buffer stores this as ``done[t]`` for clean GAE math.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F

from .agent import ImpalaAgent


# ---- Config ------------------------------------------------------------

@dataclass(frozen=True)
class PPOConfig:
    gamma: float = 0.999
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    learning_rate: float = 5e-4
    ppo_epochs: int = 3
    mini_batches_per_epoch: int = 8
    max_grad_norm: float = 0.5
    normalize_advantage: bool = True


# ---- Rollout buffer ----------------------------------------------------

class RolloutBuffer:
    """Stores T-step rollout over N parallel envs and computes GAE.

    Per-step storage (filled in by the rollout loop):
        obs       (T, N, *obs_shape) uint8
        action    (T, N) long
        log_prob  (T, N) float32
        value     (T, N) float32
        reward    (T, N) float32
        done      (T, N) float32     (= gym3 first[t+1] cast to float)

    After ``compute_advantages_and_returns(last_value)`` is called:
        advantages (T, N) float32
        returns    (T, N) float32 = advantages + value
    """

    def __init__(self, T: int, N: int,
                 obs_shape: tuple[int, ...] = (3, 64, 64),
                 device: torch.device | str = "cpu"):
        self.T, self.N = T, N
        self.obs_shape = obs_shape
        self.device = torch.device(device)
        # Storage
        self.obs = torch.zeros((T, N, *obs_shape), dtype=torch.uint8, device=self.device)
        self.action = torch.zeros((T, N), dtype=torch.long, device=self.device)
        self.log_prob = torch.zeros((T, N), dtype=torch.float32, device=self.device)
        self.value = torch.zeros((T, N), dtype=torch.float32, device=self.device)
        self.reward = torch.zeros((T, N), dtype=torch.float32, device=self.device)
        self.done = torch.zeros((T, N), dtype=torch.float32, device=self.device)
        # Filled by compute_advantages_and_returns
        self.advantages: torch.Tensor | None = None
        self.returns: torch.Tensor | None = None

    def store_step(self, t: int,
                   obs: torch.Tensor,
                   action: torch.Tensor,
                   log_prob: torch.Tensor,
                   value: torch.Tensor) -> None:
        """Store pre-action quantities at step t."""
        self.obs[t] = obs
        self.action[t] = action
        self.log_prob[t] = log_prob
        self.value[t] = value

    def store_post(self, t: int,
                   reward: torch.Tensor,
                   done: torch.Tensor) -> None:
        """Store reward earned at step t and ``done[t]`` = gym3 first[t+1]."""
        self.reward[t] = reward
        self.done[t] = done.float()

    def compute_advantages_and_returns(self,
                                       last_value: torch.Tensor,
                                       gamma: float = 0.999,
                                       gae_lambda: float = 0.95) -> None:
        """GAE(λ) — standard PPO computation.

        last_value: (N,) — V(obs_T), the bootstrap value at the rollout's end.
        """
        if last_value.shape != (self.N,):
            raise ValueError(f"last_value shape must be (N={self.N},), got {tuple(last_value.shape)}")
        adv = torch.zeros((self.T, self.N), dtype=torch.float32, device=self.device)
        last_adv = torch.zeros(self.N, dtype=torch.float32, device=self.device)
        next_v = last_value.to(self.device).float()
        for t in reversed(range(self.T)):
            not_done = 1.0 - self.done[t]
            delta = self.reward[t] + gamma * next_v * not_done - self.value[t]
            last_adv = delta + gamma * gae_lambda * last_adv * not_done
            adv[t] = last_adv
            next_v = self.value[t]
        self.advantages = adv
        self.returns = adv + self.value

    def flatten(self) -> dict[str, torch.Tensor]:
        """Return (T*N, ...) flattened tensors for mini-batching."""
        if self.advantages is None or self.returns is None:
            raise RuntimeError("call compute_advantages_and_returns() first")
        T, N = self.T, self.N
        return {
            "obs":       self.obs.reshape(T * N, *self.obs_shape),
            "action":    self.action.reshape(T * N),
            "log_prob":  self.log_prob.reshape(T * N),
            "value":     self.value.reshape(T * N),
            "advantage": self.advantages.reshape(T * N),
            "return":    self.returns.reshape(T * N),
        }

    def iter_minibatches(self, num_minibatches: int,
                         generator: torch.Generator | None = None
                         ) -> Iterator[dict[str, torch.Tensor]]:
        """Yield ``num_minibatches`` dicts of mini-batch tensors (random partition)."""
        flat = self.flatten()
        total = self.T * self.N
        if total % num_minibatches != 0:
            raise ValueError(f"T*N={total} not divisible by num_minibatches={num_minibatches}")
        mb_size = total // num_minibatches
        perm = torch.randperm(total, generator=generator, device=self.device)
        for i in range(num_minibatches):
            idx = perm[i * mb_size : (i + 1) * mb_size]
            yield {k: v[idx] for k, v in flat.items()}


# ---- Sampling ----------------------------------------------------------

def sample_action(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample actions from policy logits.

    Returns (action, log_prob), shapes (B,) and (B,).
    """
    dist = torch.distributions.Categorical(logits=logits)
    action = dist.sample()
    log_prob = dist.log_prob(action)
    return action, log_prob


# ---- PPO loss ----------------------------------------------------------

def ppo_loss(agent: ImpalaAgent,
             mb: dict[str, torch.Tensor],
             config: PPOConfig,
             ) -> dict[str, torch.Tensor]:
    """Compute PPO total loss + diagnostics for one mini-batch.

    Returns dict with:
      total      (scalar with grad)  — to backward
      policy     (scalar, detached)
      value      (scalar, detached)
      entropy    (scalar, detached)
      approx_kl  (scalar, detached)  — diagnostic
      clipfrac   (scalar, detached)  — fraction of ratios outside clip range
    """
    out = agent(mb["obs"])
    dist = torch.distributions.Categorical(logits=out.logits)
    log_prob_new = dist.log_prob(mb["action"])
    entropy = dist.entropy().mean()

    advantage = mb["advantage"]
    if config.normalize_advantage and advantage.numel() > 1:
        # numel>1 guard: std on a single-sample mini-batch is NaN (unbiased)
        # → would poison logits via the (a-mean)/(std+eps) path.
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

    ratio = (log_prob_new - mb["log_prob"]).exp()
    surr1 = ratio * advantage
    surr2 = torch.clamp(ratio, 1.0 - config.clip_range, 1.0 + config.clip_range) * advantage
    policy_loss = -torch.min(surr1, surr2).mean()

    value_loss = 0.5 * (out.value - mb["return"]).pow(2).mean()

    total = (policy_loss
             + config.value_coef * value_loss
             - config.entropy_coef * entropy)

    with torch.no_grad():
        approx_kl = (mb["log_prob"] - log_prob_new).mean()
        clipfrac = ((ratio - 1.0).abs() > config.clip_range).float().mean()

    return {
        "total": total,
        "policy": policy_loss.detach(),
        "value": value_loss.detach(),
        "entropy": entropy.detach(),
        "approx_kl": approx_kl,
        "clipfrac": clipfrac,
    }


# ---- Updater ----------------------------------------------------------

class PPOUpdater:
    """Wraps optimizer + multi-epoch mini-batched PPO update on a rollout buffer."""

    def __init__(self, agent: ImpalaAgent, config: PPOConfig):
        self.agent = agent
        self.config = config
        self.optimizer = torch.optim.Adam(agent.parameters(), lr=config.learning_rate)

    def update(self, buffer: RolloutBuffer,
               generator: torch.Generator | None = None,
               ) -> dict[str, float]:
        """Run config.ppo_epochs × config.mini_batches_per_epoch SGD steps.

        Returns averaged loss component values over all mini-batches (for logging).
        """
        sums: dict[str, float] = {k: 0.0 for k in
                                  ("total", "policy", "value", "entropy",
                                   "approx_kl", "clipfrac")}
        n = 0
        for _ in range(self.config.ppo_epochs):
            for mb in buffer.iter_minibatches(self.config.mini_batches_per_epoch,
                                              generator=generator):
                losses = ppo_loss(self.agent, mb, self.config)
                self.optimizer.zero_grad()
                losses["total"].backward()
                nn.utils.clip_grad_norm_(self.agent.parameters(),
                                         self.config.max_grad_norm)
                self.optimizer.step()
                for k in sums:
                    sums[k] += float(losses[k].item())
                n += 1
        return {k: v / max(n, 1) for k, v in sums.items()}
