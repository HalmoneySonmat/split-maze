"""Phase-6 R2 training smoke (CPU, MockEnv) — src/split_maze/train_phase3.train_r2.

Verifies the V2-closed-loop training loop runs in both modes (R2 feedback on /
matched-R0 feedback off), updates the agent, and leaves the frozen V2 untouched.
The real effect (decisive-faithful gain) is a GPU experiment; here we check the
loop mechanics + the (C-thin) RL boundary.
"""

import pytest

torch = pytest.importorskip("torch")

from split_maze.acc import ACC, ACCConfig
from split_maze.agent import ImpalaAgent
from split_maze.builds import V2ACC
from split_maze.lm import LMConfig, MazeLM, MazeTokenizer
from split_maze.ppo import PPOConfig
from split_maze.train import MockMazeEnv
from split_maze.train_phase3 import train_r2


def _v2(tok):
    lm = MazeLM(LMConfig.from_tokenizer(tok, d_model=32, n_head=4, n_layer=2,
                                        d_ff=64, max_len=32, dropout=0.0))
    return V2ACC(lm, ACC(ACCConfig(d_agent=256, d_lm=32)))


def _cfg():
    return PPOConfig(ppo_epochs=1, mini_batches_per_epoch=2)


def test_train_r2_feedback_on_runs_and_updates_agent():
    env = MockMazeEnv(num=2, episode_length=8, seed=0)
    agent = ImpalaAgent()
    v2 = _v2(MazeTokenizer())
    p0 = agent.embed.weight.detach().clone()
    logs = train_r2(env, agent, v2, ppo_config=_cfg(), num_updates=2,
                    num_steps=4, lam=0.3, feedback_on=True, device="cpu")
    assert len(logs) == 2
    assert all("mean_return" in lg and lg["feedback"] is True for lg in logs)
    assert not torch.equal(agent.embed.weight, p0)        # agent co-adapted


def test_train_r2_matched_r0_runs():
    env = MockMazeEnv(num=2, episode_length=8, seed=0)
    agent = ImpalaAgent()
    v2 = _v2(MazeTokenizer())
    logs = train_r2(env, agent, v2, ppo_config=_cfg(), num_updates=2,
                    num_steps=4, feedback_on=False, device="cpu")
    assert len(logs) == 2
    assert all(lg["feedback"] is False for lg in logs)


def test_train_r2_freezes_v2():
    """(C-thin) RL boundary: the V2 interpreter must NOT change during R2."""
    env = MockMazeEnv(num=2, episode_length=8, seed=0)
    agent = ImpalaAgent()
    v2 = _v2(MazeTokenizer())
    before = [p.detach().clone() for p in v2.parameters()]
    train_r2(env, agent, v2, ppo_config=_cfg(), num_updates=2, num_steps=4,
             lam=0.3, feedback_on=True, device="cpu")
    for p, b in zip(v2.parameters(), before):
        assert torch.equal(p, b), "V2 changed during R2 — (C-thin) boundary broken"
