"""PPO training loop for SPLIT-MAZE — Phase 1.3 산출물 (PLAN §7.1).

Modular structure (kept testable without procgen):
- ``obs_to_tensor``  : (N,H,W,3) uint8 numpy → (N,3,H,W) uint8 torch on device.
- ``collect_rollout``: fill a RolloutBuffer over T steps, track per-env
                       episode returns / lengths in-place.
- ``train``          : full PPO loop (rollout → bootstrap V(s_T) → GAE →
                       PPO update), returns a list of per-update log dicts.

Decoupled from procgen via a gym3-호환 protocol — the loop runs with
``procgen.ProcgenGym3Env`` in WSL and with :class:`MockMazeEnv` in unit
tests / sandbox.

gym3 convention (kept consistent with ``ppo.py`` docstring +
``docs/PROCGEN_ENV.md`` §7):

    env.observe() → (reward (N,) float32,
                     obs_dict {'rgb': (N,64,64,3) uint8},
                     first (N,) bool)
    env.act(action: np.ndarray (N,) int32) → None

    ``first[t+1] == True`` iff step *t* was terminal (env auto-reset
    between *t* and *t+1*). In the buffer we therefore store
    ``done[t] = first[t+1]`` so that GAE's "not_done = 1 - done[t]"
    correctly zeros the bootstrap across the reset boundary.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Protocol, runtime_checkable

import numpy as np
import torch

from .agent import ImpalaAgent
from .ppo import PPOConfig, PPOUpdater, RolloutBuffer, sample_action


# Default rolling window for per-update episode-return logging.
# Smooths the noisy per-rollout signal (where some rollouts complete 0
# episodes → nan) without hiding learning trend over short horizons.
DEFAULT_ROLLING_WINDOW: int = 100


# ---- gym3-compatible env protocol ----------------------------------------

@runtime_checkable
class VecEnvLike(Protocol):
    """Minimal gym3 surface used by the training loop.

    Any object exposing ``num: int``, ``observe()``, and ``act(action)`` with
    the shapes described in the module docstring is accepted — both the
    real :class:`procgen.ProcgenGym3Env` and :class:`MockMazeEnv` qualify.
    """

    num: int

    def observe(self) -> tuple[Any, Any, Any]: ...
    def act(self, action: np.ndarray) -> None: ...


# ---- obs tensor conversion ----------------------------------------------

def obs_to_tensor(obs_dict: dict, device: torch.device | str) -> torch.Tensor:
    """procgen rgb obs dict → (N,3,64,64) uint8 :class:`torch.Tensor` on ``device``.

    ``obs_dict["rgb"]`` is the standard procgen output: (N,H,W,3) uint8.
    PyTorch expects NCHW, so we transpose, force contiguity (transpose is a
    view), and move to the requested device.
    """
    rgb = obs_dict["rgb"]
    if not isinstance(rgb, np.ndarray):
        rgb = np.asarray(rgb)
    if rgb.ndim != 4 or rgb.shape[-1] != 3:
        raise ValueError(f"obs['rgb'] must be (N,H,W,3), got {rgb.shape}")
    arr = np.ascontiguousarray(rgb.transpose(0, 3, 1, 2))
    return torch.from_numpy(arr).to(device)


# ---- rollout collection -------------------------------------------------

@dataclass
class RolloutStats:
    """Per-rollout summary: episodes that terminated within this rollout."""
    completed_returns: list[float] = field(default_factory=list)
    completed_lengths: list[int] = field(default_factory=list)

    @property
    def num_completed(self) -> int:
        return len(self.completed_returns)

    @property
    def mean_return(self) -> float:
        if not self.completed_returns:
            return float("nan")
        return float(np.mean(self.completed_returns))

    @property
    def mean_length(self) -> float:
        if not self.completed_lengths:
            return float("nan")
        return float(np.mean(self.completed_lengths))


def collect_rollout(
    env: VecEnvLike,
    agent: ImpalaAgent,
    buffer: RolloutBuffer,
    *,
    obs_holder: torch.Tensor,
    episode_returns: np.ndarray,
    episode_lengths: np.ndarray,
    device: torch.device | str,
) -> tuple[RolloutStats, torch.Tensor]:
    """Roll out ``T = buffer.T`` steps on ``env`` and store transitions.

    The agent forward + action sampling run under ``torch.no_grad()`` so the
    autograd graph isn't built during rollout collection (PPO recomputes a
    fresh graph during :func:`ppo.ppo_loss`).

    Args:
      env: gym3-호환 vec env, must satisfy ``env.num == buffer.N``.
      agent: :class:`ImpalaAgent`, already on ``device``.
      buffer: A :class:`RolloutBuffer` of shape ``(T, N, …)`` to fill.
      obs_holder: ``(N,3,64,64)`` uint8 tensor — the *current* observation
                  tensor on ``device``. Caller obtains the initial one via
                  ``obs_to_tensor(env.observe()[1], device)``.
      episode_returns: ``(N,)`` float64 numpy, running per-env reward sum.
                       Mutated in-place: reset to 0 on terminations.
      episode_lengths: ``(N,)`` int64 numpy, running per-env step count.
                       Mutated in-place: reset to 0 on terminations.
      device: torch device.

    Returns:
      stats: :class:`RolloutStats` for episodes that ended *within* this rollout.
      next_obs_holder: the observation tensor reached *after* storing step
                       ``T-1`` — used for the V(s_T) bootstrap.
    """
    if env.num != buffer.N:
        raise ValueError(f"env.num={env.num} != buffer.N={buffer.N}")
    if episode_returns.shape != (env.num,):
        raise ValueError(f"episode_returns shape must be ({env.num},), "
                         f"got {episode_returns.shape}")
    if episode_lengths.shape != (env.num,):
        raise ValueError(f"episode_lengths shape must be ({env.num},), "
                         f"got {episode_lengths.shape}")

    stats = RolloutStats()
    N, T = buffer.N, buffer.T

    for t in range(T):
        with torch.no_grad():
            out = agent(obs_holder)
            action, log_prob = sample_action(out.logits)
        buffer.store_step(t, obs=obs_holder, action=action,
                          log_prob=log_prob, value=out.value)
        action_np = action.detach().cpu().numpy().astype(np.int32)
        env.act(action_np)
        rew, obs_dict, first = env.observe()
        rew_np = np.asarray(rew, dtype=np.float32).reshape(N)
        first_np = np.asarray(first, dtype=np.bool_).reshape(N)
        # done[t] := first[t+1]   (gym3 convention — see module docstring)
        buffer.store_post(
            t,
            reward=torch.from_numpy(rew_np).to(device),
            done=torch.from_numpy(first_np).to(device),
        )
        # Per-env episode bookkeeping (CPU)
        episode_returns += rew_np.astype(np.float64)
        episode_lengths += 1
        for i in range(N):
            if first_np[i]:
                stats.completed_returns.append(float(episode_returns[i]))
                stats.completed_lengths.append(int(episode_lengths[i]))
                episode_returns[i] = 0.0
                episode_lengths[i] = 0
        obs_holder = obs_to_tensor(obs_dict, device)

    return stats, obs_holder


# ---- main training loop -------------------------------------------------

def train(
    env: VecEnvLike,
    agent: ImpalaAgent,
    config: Optional[PPOConfig] = None,
    *,
    num_steps: int = 256,
    total_env_steps: int = 10_000,
    device: torch.device | str = "cpu",
    log_callback: Optional[Callable[[int, dict], None]] = None,
    generator: Optional[torch.Generator] = None,
    rolling_window: int = DEFAULT_ROLLING_WINDOW,
) -> list[dict]:
    """Full PPO training loop. Returns one log dict per update.

    Args:
      env: gym3-호환 vec env (procgen or :class:`MockMazeEnv`).
      agent: :class:`ImpalaAgent`, already moved to ``device``.
      config: :class:`PPOConfig` (default: ``PPOConfig()``).
      num_steps: ``T`` — rollout length per update (default 256, smoke 16).
      total_env_steps: total transitions to run across all envs.
                       num_updates = ``max(1, total_env_steps // (T*N))``.
                       The ``max(1, …)`` ensures at least one update even
                       for tiny smoke budgets.
      device: torch device.
      log_callback: optional ``fn(update_idx, log_dict)`` called after each
                    update.
      generator: optional :class:`torch.Generator` for minibatch-shuffle
                 reproducibility (passed to :meth:`PPOUpdater.update`).
      rolling_window: K — size of the FIFO window over completed episodes
                      used for ``ep_return_rolling`` / ``ep_length_rolling``.
                      Smooths the noisy per-rollout signal.

    Returns:
      ``logs``: list of dicts, one per update. Keys::
          update_idx, env_steps,
          ep_count, ep_return_mean, ep_length_mean,          # this rollout only
          ep_return_rolling, ep_length_rolling,
          ep_return_rolling_n,                                # window over last K
          total, policy, value, entropy, approx_kl, clipfrac
    """
    if config is None:
        config = PPOConfig()
    if rolling_window <= 0:
        raise ValueError(f"rolling_window must be positive, got {rolling_window}")
    device = torch.device(device)
    N = env.num
    T = num_steps

    buffer = RolloutBuffer(T=T, N=N, device=device)
    updater = PPOUpdater(agent, config)

    # Initial observe — gym3 vec envs are pre-reset on construction;
    # rew and first from this call are discarded (no step yet).
    _rew0, obs_dict, _first0 = env.observe()
    obs_holder = obs_to_tensor(obs_dict, device)

    episode_returns = np.zeros(N, dtype=np.float64)
    episode_lengths = np.zeros(N, dtype=np.int64)

    # Rolling window over completed episodes (across all rollouts).
    # Persisted across updates so rollouts that complete 0 episodes still
    # surface a meaningful learning-trend signal from earlier ones.
    rolling_returns: deque[float] = deque(maxlen=rolling_window)
    rolling_lengths: deque[int] = deque(maxlen=rolling_window)

    num_updates = max(1, total_env_steps // (T * N))
    logs: list[dict] = []

    for update_idx in range(num_updates):
        stats, obs_holder = collect_rollout(
            env, agent, buffer,
            obs_holder=obs_holder,
            episode_returns=episode_returns,
            episode_lengths=episode_lengths,
            device=device,
        )
        # Push this rollout's completed episodes into the rolling window
        rolling_returns.extend(stats.completed_returns)
        rolling_lengths.extend(stats.completed_lengths)

        # Bootstrap V(s_T) for GAE
        with torch.no_grad():
            last_value = agent(obs_holder).value
        buffer.compute_advantages_and_returns(
            last_value, gamma=config.gamma, gae_lambda=config.gae_lambda)
        update_log = updater.update(buffer, generator=generator)
        update_log["update_idx"] = update_idx
        update_log["env_steps"] = (update_idx + 1) * T * N
        # This-rollout-only (preserved for compatibility / reproducibility)
        update_log["ep_count"] = stats.num_completed
        update_log["ep_return_mean"] = stats.mean_return
        update_log["ep_length_mean"] = stats.mean_length
        # Rolling window (smoother learning-trend signal)
        update_log["ep_return_rolling"] = (
            float(np.mean(rolling_returns)) if rolling_returns else float("nan"))
        update_log["ep_length_rolling"] = (
            float(np.mean(rolling_lengths)) if rolling_lengths else float("nan"))
        update_log["ep_return_rolling_n"] = len(rolling_returns)
        logs.append(update_log)
        if log_callback is not None:
            log_callback(update_idx, update_log)

    return logs


# ---- gym3-compatible mock env for tests + sandbox smoke -----------------

class MockMazeEnv:
    """gym3-compatible vec env mock for unit tests / sandbox smoke runs.

    Mimics :class:`procgen.ProcgenGym3Env`'s surface exactly:
        ``num`` (int), ``observe()`` → (reward, obs_dict, first),
        ``act(action)``.

    Behavior (designed to give the training loop a finite, non-degenerate
    learning signal even without real procgen):

    - Each parallel env runs episodes of ``episode_length`` steps and
      auto-resets at the boundary.
    - ``reward`` per step = 0.1 if ``action == 0`` else 0.0. PPO can
      learn to prefer action 0 → small but real grad signal.
    - ``obs`` is fresh pseudo-random uint8 each step (env_idx-independent
      RNG draw). The obs *doesn't* encode "best action" — that would make
      the policy state-conditional. The smoke check is that PPO updates
      run without numerical issues + parameters move under grad.
    - ``first[t+1] == True`` iff step *t* was terminal (procgen semantics).

    Args:
      num: parallel env count (= ``N``).
      episode_length: steps per episode before auto-reset.
      seed: RNG seed for reproducible obs.
      num_actions: action space size (matches procgen Discrete(15)).
    """

    def __init__(self,
                 num: int = 4,
                 episode_length: int = 8,
                 seed: int = 0,
                 num_actions: int = 15):
        if num <= 0 or episode_length <= 0:
            raise ValueError("num and episode_length must be positive")
        self.num = int(num)
        self.episode_length = int(episode_length)
        self.num_actions = int(num_actions)
        self._rng = np.random.RandomState(int(seed))
        self._step_count = np.zeros(self.num, dtype=np.int64)
        self._just_reset = np.ones(self.num, dtype=np.bool_)   # initial reset
        self._pending_reward = np.zeros(self.num, dtype=np.float32)
        self._obs = self._render_obs()

    # ---- gym3 surface ---------------------------------------------------

    def observe(self) -> tuple[np.ndarray, dict, np.ndarray]:
        return (
            self._pending_reward.copy(),
            {"rgb": self._obs.copy()},
            self._just_reset.copy(),
        )

    def act(self, action: np.ndarray) -> None:
        action = np.asarray(action).reshape(-1)
        if action.shape != (self.num,):
            raise ValueError(
                f"action shape must be ({self.num},), got {action.shape}")
        # Reward for *this* step
        rew = np.where(action == 0, 0.1, 0.0).astype(np.float32)
        self._pending_reward = rew
        self._step_count += 1
        end = self._step_count >= self.episode_length
        self._just_reset = end.astype(np.bool_)
        if end.any():
            self._step_count[end] = 0
        # Fresh obs (whether reset or not — obs is opaque to MockEnv logic)
        self._obs = self._render_obs()

    # ---- internals ------------------------------------------------------

    def _render_obs(self) -> np.ndarray:
        return self._rng.randint(
            0, 256, size=(self.num, 64, 64, 3), dtype=np.uint8)


__all__ = [
    "VecEnvLike",
    "RolloutStats",
    "MockMazeEnv",
    "obs_to_tensor",
    "collect_rollout",
    "train",
]
