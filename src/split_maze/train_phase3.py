"""Phase 3.4 co-training loop — RL agent + B3/B4/V2 interpreters.

Frozen decisions (PLAN §10.5 P3-4-1..P3-4-4, 2026-05-21):

- **P3-4-1**: a *self-contained augmented rollout* (``collect_rollout_with_pairs``)
  modelled on ``train.collect_rollout`` but additionally (a) keeping the
  agent's ``out.h_agent`` and (b) extracting a ``MazeState`` per env per step.
  ``train.py`` is left untouched (P3-2-5).
- **P3-4-2**: one independent AdamW optimizer per build (B3/B4/V2) over its
  ``interpreter_parameters()``, plus the agent's own ``PPOUpdater`` — four
  separate optimizers so the (C-thin) signals never mix.
- **P3-4-3**: 1 seed, smoke→mid→full(25M); Phase-1.4 ladder.
- **P3-4-4**: ``MazeState`` via ``env.extract_maze_state(rgb, tracker)``;
  per-build JSONL diagnostics + per-build checkpoints (the CLI in
  ``scripts/train_phase3.py`` — sub-step 3.4.2 — wires logging/ckpt).

Co-training step per RL update:
    1. augmented rollout (T steps) → RolloutBuffer (PPO) + h_agent (T,N,d_a)
       + maze_states (T×N).
    2. PPO update on the agent (pure RL — (C-thin) boundary).
    3. PairedCollector.extract_into(buffer, h_agent, maze_states) — stride=4.
    4. once the shared PairBuffer is ready: each build runs K=32 mini-batch
       interpreter updates (P3-2-4), sampling from the shared buffer (the
       3 builds therefore see the *same agent* — P3-5 통제).

The 3 builds share **one** PairBuffer (same agent run → identical pair
population). Each build samples its own uniform-random batch each step.
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import torch

from .agent import ImpalaAgent
from .builds import Build
from .env import DEFAULT_HEADING_WINDOW, TrajectoryTracker, extract_maze_state
from .language import MazeState
from .paired_collect import PairBuffer, PairedCollector
from .feedback import compute_inject
from .ppo import PPOConfig, PPOUpdater, RolloutBuffer, sample_action
from .train import DEFAULT_ROLLING_WINDOW, RolloutStats, obs_to_tensor


# ---- config -------------------------------------------------------------


@dataclass
class Phase3Config:
    """Phase-3.4 co-training hyperparameters (PLAN §10.5 박제값 default)."""

    num_steps: int = 256                 # T per rollout
    total_env_steps: int = 25_000_000    # P3-4-3 full run
    # interpreter (P3-3-A / P3-2-4 박제)
    acc_updates_per_rl: int = 32         # K — interpreter mini-batches / RL update
    interp_batch: int = 128
    interp_lr: float = 3e-4
    interp_warmup: int = 500
    interp_weight_decay: float = 0.01
    # paired collection (P3-4 / P3-2-2 박제)
    stride: int = 4
    buffer_capacity: int = 256_000
    heading_window: int = DEFAULT_HEADING_WINDOW   # K=4
    rolling_window: int = DEFAULT_ROLLING_WINDOW


# Default state extractor — wraps env.extract_maze_state (P3-4-4). Injectable
# so unit tests can feed synthetic MazeStates without procgen sprites.
StateExtractor = Callable[[np.ndarray, TrajectoryTracker], Optional[MazeState]]


def default_state_extractor(
    rgb: np.ndarray, tracker: TrajectoryTracker
) -> Optional[MazeState]:
    return extract_maze_state(rgb, tracker).maze_state


# ---- augmented rollout --------------------------------------------------


def collect_rollout_with_pairs(
    env,
    agent: ImpalaAgent,
    buffer: RolloutBuffer,
    trackers: list[TrajectoryTracker],
    *,
    obs_holder: torch.Tensor,
    cur_rgb: np.ndarray,
    episode_returns: np.ndarray,
    episode_lengths: np.ndarray,
    state_extractor: StateExtractor,
    d_agent: int,
    device: torch.device | str,
    feedback_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
) -> tuple[RolloutStats, torch.Tensor, np.ndarray, torch.Tensor, list]:
    """Like ``train.collect_rollout`` but also returns per-step ``h_agent``
    and ``MazeState`` grid for paired collection.

    Alignment: ``h_agent[t]`` and ``maze_states[t][n]`` both describe the
    state the agent *saw* at step t (``cur_rgb`` / ``obs_holder`` at the top
    of the step), so the (h_agent, sentence) pair is consistent.

    feedback_fn (Phase-6 R2): optional callable ``h_agent (N, d_a) → inject
    (N, d_a)``. When given, the inject computed from step t's PRE-injection
    ``h_agent`` is added to the agent's hidden at step t+1
    (``agent.forward(obs, inject=...)``) — the "corpus callosum" feedback
    (PREREG §1). ``None`` (regimes R0/R1) is byte-identical to the original:
    ``agent(obs, inject=None) ≡ agent(obs)``, no trajectory change. NOTE:
    ``h_agent_steps`` always stores the PRE-injection h (clean-read).

    Returns:
        stats, next_obs_holder, next_cur_rgb,
        h_agent_steps: (T, N, d_agent) cpu float,
        maze_states:   T×N nested list of Optional[MazeState].
    """
    if env.num != buffer.N:
        raise ValueError(f"env.num={env.num} != buffer.N={buffer.N}")
    N, T = buffer.N, buffer.T
    if len(trackers) != N:
        raise ValueError(f"need {N} trackers, got {len(trackers)}")

    stats = RolloutStats()
    h_agent_steps = torch.empty(T, N, d_agent)
    maze_states: list[list[Optional[MazeState]]] = [
        [None] * N for _ in range(T)
    ]
    inject: Optional[torch.Tensor] = None   # R2 feedback from step t-1 (None=R0/R1)

    for t in range(T):
        with torch.no_grad():
            out = agent(obs_holder, inject=inject)
            action, log_prob = sample_action(out.logits)
        buffer.store_step(t, obs=obs_holder, action=action,
                          log_prob=log_prob, value=out.value, inject=inject)
        h_agent_steps[t] = out.h_agent.detach().to("cpu")
        # R2: compute next-step inject from this step's PRE-injection h_agent.
        if feedback_fn is not None:
            with torch.no_grad():
                inject = feedback_fn(out.h_agent)

        # MazeState from the state the agent just saw (cur_rgb), updating each
        # env's trajectory tracker in-place (HEADING needs sequential history).
        rgb = np.asarray(cur_rgb)
        for n in range(N):
            maze_states[t][n] = state_extractor(rgb[n], trackers[n])

        action_np = action.detach().cpu().numpy().astype(np.int32)
        env.act(action_np)
        rew, obs_dict, first = env.observe()
        rew_np = np.asarray(rew, dtype=np.float32).reshape(N)
        first_np = np.asarray(first, dtype=np.bool_).reshape(N)
        buffer.store_post(
            t,
            reward=torch.from_numpy(rew_np).to(device),
            done=torch.from_numpy(first_np).to(device),
        )
        episode_returns += rew_np.astype(np.float64)
        episode_lengths += 1
        for i in range(N):
            if first_np[i]:
                stats.completed_returns.append(float(episode_returns[i]))
                stats.completed_lengths.append(int(episode_lengths[i]))
                episode_returns[i] = 0.0
                episode_lengths[i] = 0
                # Episode boundary → clear trajectory so HEADING doesn't span
                # the teleport to the new start.
                trackers[i].history.clear()

        cur_rgb = np.asarray(obs_dict["rgb"])
        obs_holder = obs_to_tensor(obs_dict, device)

    return stats, obs_holder, cur_rgb, h_agent_steps, maze_states


# ---- warmup helper ------------------------------------------------------


def _warmup_scale(step: int, warmup: int) -> float:
    """Linear LR warmup multiplier in [0, 1] (POST-HOC-4 계승)."""
    if warmup <= 0:
        return 1.0
    return min(1.0, (step + 1) / float(warmup))


# ---- co-training loop ---------------------------------------------------


def train_phase3(
    env,
    agent: ImpalaAgent,
    builds: dict[str, Build],
    collector: PairedCollector,
    buffer: PairBuffer,
    *,
    config: Optional[Phase3Config] = None,
    ppo_config: Optional[PPOConfig] = None,
    state_extractor: StateExtractor = default_state_extractor,
    device: torch.device | str = "cpu",
    log_callback: Optional[Callable[[int, dict], None]] = None,
    generator: Optional[torch.Generator] = None,
    surface_rng: Optional[random.Random] = None,
) -> list[dict]:
    """Co-train the shared agent (PPO) and the B3/B4/V2 interpreters.

    Args:
        env:        gym3-호환 vec env (procgen or MockMazeEnv).
        agent:      :class:`ImpalaAgent` on ``device``.
        builds:     ``{"B3": B3Probe, "B4": B4Adapter, "V2": V2ACC}`` (any
                    subset). Each must implement :class:`Build`.
        collector:  :class:`PairedCollector` (stride=4).
        buffer:     a single shared :class:`PairBuffer` (all builds sample it).
        config:     :class:`Phase3Config`.
        ppo_config: :class:`PPOConfig` for the agent.
        state_extractor: rgb+tracker → MazeState (injectable for tests).
        device, log_callback, generator: as in :func:`train.train`.
        surface_rng: ``random.Random`` for describer surface-form variation.

    Returns:
        ``logs``: one dict per RL update with PPO metrics + per-build
        interpreter losses (``"<name>/loss"``) + ``"<name>/n_updates"``.
    """
    config = config or Phase3Config()
    ppo_config = ppo_config or PPOConfig()
    device = torch.device(device)
    surface_rng = surface_rng or random.Random(0)

    N = env.num
    T = config.num_steps
    d_agent = agent.d_a

    rollout_buffer = RolloutBuffer(T=T, N=N, device=device)
    updater = PPOUpdater(agent, ppo_config)

    # One independent AdamW per build (P3-4-2).
    interp_opts = {
        name: torch.optim.AdamW(
            b.interpreter_parameters(),
            lr=config.interp_lr,
            weight_decay=config.interp_weight_decay,
            betas=(0.9, 0.95),
        )
        for name, b in builds.items()
    }
    interp_steps = {name: 0 for name in builds}

    trackers = [TrajectoryTracker(config.heading_window) for _ in range(N)]

    _r0, obs_dict, _f0 = env.observe()
    obs_holder = obs_to_tensor(obs_dict, device)
    cur_rgb = np.asarray(obs_dict["rgb"])

    episode_returns = np.zeros(N, dtype=np.float64)
    episode_lengths = np.zeros(N, dtype=np.int64)
    rolling_returns: deque[float] = deque(maxlen=config.rolling_window)

    num_updates = max(1, config.total_env_steps // (T * N))
    logs: list[dict] = []

    for update_idx in range(num_updates):
        stats, obs_holder, cur_rgb, h_agent_TN, maze_states = (
            collect_rollout_with_pairs(
                env, agent, rollout_buffer, trackers,
                obs_holder=obs_holder, cur_rgb=cur_rgb,
                episode_returns=episode_returns,
                episode_lengths=episode_lengths,
                state_extractor=state_extractor,
                d_agent=d_agent, device=device,
            )
        )
        rolling_returns.extend(stats.completed_returns)

        # ---- PPO update (agent — pure RL) ----
        with torch.no_grad():
            last_value = agent(obs_holder).value
        rollout_buffer.compute_advantages_and_returns(
            last_value, gamma=ppo_config.gamma, gae_lambda=ppo_config.gae_lambda)
        ppo_log = updater.update(rollout_buffer, generator=generator)

        # ---- paired collection (shared buffer) ----
        n_added = collector.extract_into(
            buffer, h_agent_TN, maze_states, rng=surface_rng)

        # ---- interpreter updates (K each, if buffer ready) ----
        interp_logs: dict[str, float] = {}
        if buffer.is_ready(config.interp_batch):
            for name, b in builds.items():
                opt = interp_opts[name]
                last_loss = float("nan")
                for _ in range(config.acc_updates_per_rl):
                    batch = buffer.sample(config.interp_batch, generator=generator)
                    h = batch["h_agent"].to(device)
                    ids = batch["ids"].to(device)
                    lengths = batch["lengths"].to(device)
                    out = b.update(h, ids, lengths)
                    opt.zero_grad()
                    out["loss"].backward()
                    # LR warmup (POST-HOC-4 계승).
                    scale = _warmup_scale(interp_steps[name], config.interp_warmup)
                    for g in opt.param_groups:
                        g["lr"] = config.interp_lr * scale
                    opt.step()
                    interp_steps[name] += 1
                    last_loss = float(out["loss"].detach().item())
                interp_logs[f"{name}/loss"] = last_loss
                interp_logs[f"{name}/n_updates"] = interp_steps[name]

        # ---- log ----
        log = dict(ppo_log)
        log["update_idx"] = update_idx
        log["env_steps"] = (update_idx + 1) * T * N
        log["ep_return_rolling"] = (
            float(np.mean(rolling_returns)) if rolling_returns else float("nan"))
        log["pairs_added"] = int(n_added)
        log["buffer_size"] = len(buffer)
        log.update(interp_logs)
        logs.append(log)
        if log_callback is not None:
            log_callback(update_idx, log)

    return logs


def train_r2(
    env,
    agent: ImpalaAgent,
    v2,
    *,
    ppo_config: PPOConfig,
    num_updates: int,
    num_steps: int,
    lam: float = 0.3,
    feedback_on: bool = True,
    device: torch.device | str = "cpu",
    state_extractor: Optional[StateExtractor] = None,
    log_callback=None,
) -> list[dict]:
    """Phase-6 R2 training (V2 closed loop) — PREREG §0.7 / P2.

    Co-adapt the agent by PPO while, in R2, the V2 interpreter's reading of
    step t's PRE-injection ``h`` is bridged back and added to the agent's
    hidden at step t+1 (``compute_inject`` = λ·Wᵀ·LN(LM_summary(W·LN(h))),
    fixed λ). The bridge + LM are FROZEN — only the agent's weights update
    ((C-thin) on the RL side; the inject enters the PPO update as a stored
    constant, so no grad flows into V2). ``feedback_on=False`` is the matched-R0
    control: identical PPO budget, no feedback.

    Args:
        env: gym3-like vec env (``num``, ``observe``, ``act``).
        agent: the :class:`ImpalaAgent` to co-adapt (start from the base agent).
        v2: a frozen :class:`V2ACC` (provides ``acc`` bridge + ``lm``).
        ppo_config, num_updates, num_steps: PPO schedule.
        lam: fixed feedback gate λ (PREREG primary = 0.3).
        feedback_on: True = R2 (V2 closed loop); False = matched-R0.
        state_extractor: unused-but-required by the rollout (pairs ignored).
    Returns:
        list of per-update log dicts (mean_return + PPO loss components).
    """
    device = torch.device(device)
    d_a = agent.d_a
    v2.eval()
    for p in v2.parameters():
        p.requires_grad_(False)
    updater = PPOUpdater(agent, ppo_config)
    extractor = state_extractor or default_state_extractor

    feedback_fn = None
    if feedback_on:
        def feedback_fn(h):
            return compute_inject(v2.acc, v2.lm, h, lam)

    N = env.num
    trackers = [TrajectoryTracker(DEFAULT_HEADING_WINDOW) for _ in range(N)]
    _r, obs_dict, _f = env.observe()
    obs_holder = obs_to_tensor(obs_dict, device)
    cur_rgb = np.asarray(obs_dict["rgb"])
    ep_r = np.zeros(N); ep_l = np.zeros(N, dtype=np.int64)

    logs: list[dict] = []
    for upd in range(num_updates):
        buffer = RolloutBuffer(T=num_steps, N=N, device=device,
                               inject_dim=(d_a if feedback_on else None))
        stats, obs_holder, cur_rgb, _h, _ms = collect_rollout_with_pairs(
            env, agent, buffer, trackers, obs_holder=obs_holder, cur_rgb=cur_rgb,
            episode_returns=ep_r, episode_lengths=ep_l,
            state_extractor=extractor, d_agent=d_a, device=device,
            feedback_fn=feedback_fn)
        with torch.no_grad():
            last_value = agent(obs_holder).value     # boundary bootstrap (inject=None)
        buffer.compute_advantages_and_returns(
            last_value, ppo_config.gamma, ppo_config.gae_lambda)
        losses = updater.update(buffer)
        log = {"update": upd, "mean_return": stats.mean_return,
               "feedback": feedback_on, **losses}
        logs.append(log)
        if log_callback is not None:
            log_callback(upd, log)
    return logs


__all__ = [
    "Phase3Config",
    "StateExtractor",
    "default_state_extractor",
    "collect_rollout_with_pairs",
    "train_phase3",
    "train_r2",
]
