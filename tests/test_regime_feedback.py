"""Phase-6 R2 feedback wiring in collect_rollout_with_pairs(feedback_fn=...).

MockMazeEnv obs are action-INDEPENDENT (fresh RNG each step), so feedback's
effect on actions is not observable via the returned pre-injection h. We
therefore test the WIRING directly (the agent-side application of inject is
covered by test_agent; the inject computation by test_feedback):

  - feedback_fn=None (R0/R1) runs and returns correctly-shaped h_steps.
  - feedback_fn is called once per step with the PRE-injection h (== h_steps[t]).
  - the inject from step t's h is carried into step t+1's agent.forward;
    step 0 gets inject=None (PREREG §1 next-step timing).
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from split_maze.agent import ImpalaAgent
from split_maze.env import DEFAULT_HEADING_WINDOW, TrajectoryTracker
from split_maze.ppo import RolloutBuffer
from split_maze.train import MockMazeEnv, obs_to_tensor
from split_maze.train_phase3 import collect_rollout_with_pairs


def _none_extractor(rgb, tracker):
    return None


class SpyAgent(ImpalaAgent):
    """Records the ``inject`` argument seen at each forward call."""
    def forward(self, obs, inject=None):
        if not hasattr(self, "seen_injects"):
            self.seen_injects = []
        self.seen_injects.append(None if inject is None else inject.detach().clone())
        return super().forward(obs, inject=inject)


def _run(agent, feedback_fn, *, num=2, T=5, device="cpu"):
    torch.manual_seed(0)
    env = MockMazeEnv(num=num, episode_length=8, seed=0)
    buffer = RolloutBuffer(T=T, N=num, device=device)
    trackers = [TrajectoryTracker(DEFAULT_HEADING_WINDOW) for _ in range(num)]
    _r, obs_dict, _f = env.observe()
    obs_holder = obs_to_tensor(obs_dict, device)
    cur_rgb = np.asarray(obs_dict["rgb"])
    ep_r = np.zeros(num); ep_l = np.zeros(num, dtype=np.int64)
    _, _, _, h_steps, _ = collect_rollout_with_pairs(
        env, agent, buffer, trackers, obs_holder=obs_holder, cur_rgb=cur_rgb,
        episode_returns=ep_r, episode_lengths=ep_l,
        state_extractor=_none_extractor, d_agent=agent.d_a, device=device,
        feedback_fn=feedback_fn)
    return h_steps


def test_none_feedback_runs_and_shapes():
    agent = ImpalaAgent().eval()
    h = _run(agent, None, num=2, T=5)
    assert h.shape == (5, 2, agent.d_a)


def test_feedback_fn_receives_pre_injection_h():
    """feedback_fn is called once per step with the PRE-injection h_agent."""
    agent = ImpalaAgent().eval()
    seen = []
    def fb(h):
        seen.append(h.detach().clone())
        return torch.zeros_like(h)
    h_steps = _run(agent, fb, num=2, T=5)
    assert len(seen) == 5                       # one call per step
    for t in range(5):
        assert torch.equal(seen[t].cpu(), h_steps[t])   # == recorded pre-injection h


def test_inject_carried_to_next_step():
    """inject from step t (= feedback_fn(h_t)) is fed at step t+1; step 0 = None."""
    agent = SpyAgent().eval()
    h_steps = _run(agent, lambda h: h * 2.0, num=2, T=5)
    inj = agent.seen_injects
    assert len(inj) == 5
    assert inj[0] is None                                   # next-step timing
    for t in range(1, 5):
        assert inj[t] is not None
        assert torch.allclose(inj[t].cpu(), 2.0 * h_steps[t - 1], atol=1e-5)
