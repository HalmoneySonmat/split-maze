"""Tests for src/split_maze/ppo.py — rollout buffer + GAE + PPO loss."""

import pytest
import torch

from split_maze.agent import D_A, NUM_ACTIONS, ImpalaAgent
from split_maze.ppo import (
    PPOConfig, PPOUpdater, RolloutBuffer,
    ppo_loss, sample_action,
)


# ---- RolloutBuffer basic shapes --------------------------------------

def test_buffer_shapes():
    buf = RolloutBuffer(T=4, N=3)
    assert buf.obs.shape == (4, 3, 3, 64, 64)
    assert buf.obs.dtype == torch.uint8
    assert buf.action.shape == (4, 3)
    assert buf.action.dtype == torch.long
    for name in ("log_prob", "value", "reward", "done"):
        t = getattr(buf, name)
        assert t.shape == (4, 3)
        assert t.dtype == torch.float32


def test_buffer_store_and_flatten():
    buf = RolloutBuffer(T=2, N=3)
    for t in range(2):
        buf.store_step(t,
                       obs=torch.zeros(3, 3, 64, 64, dtype=torch.uint8),
                       action=torch.tensor([1, 2, 3]),
                       log_prob=torch.tensor([0.1, 0.2, 0.3]),
                       value=torch.tensor([0.5, 0.6, 0.7]))
        buf.store_post(t,
                       reward=torch.tensor([1.0, 0.0, -0.5]),
                       done=torch.tensor([0.0, 0.0, 0.0]))
    buf.compute_advantages_and_returns(last_value=torch.zeros(3))
    flat = buf.flatten()
    assert flat["obs"].shape == (6, 3, 64, 64)
    assert flat["action"].shape == (6,)
    assert flat["advantage"].shape == (6,)
    assert flat["return"].shape == (6,)


def test_flatten_before_compute_raises():
    buf = RolloutBuffer(T=2, N=3)
    with pytest.raises(RuntimeError):
        buf.flatten()


# ---- GAE — known cases ------------------------------------------------

def test_gae_constant_reward_no_done():
    """γ=1, λ=1, r_t=1, V=0, no done → A_t = sum of remaining rewards."""
    T, N = 3, 1
    buf = RolloutBuffer(T=T, N=N)
    buf.reward[:] = 1.0
    buf.value[:] = 0.0
    buf.done[:] = 0.0
    buf.compute_advantages_and_returns(last_value=torch.zeros(N),
                                       gamma=1.0, gae_lambda=1.0)
    # A_2 = 1, A_1 = 1+1=2, A_0 = 1+2=3
    assert torch.allclose(buf.advantages.squeeze(),
                          torch.tensor([3.0, 2.0, 1.0]))
    assert torch.allclose(buf.returns.squeeze(),
                          torch.tensor([3.0, 2.0, 1.0]))


def test_gae_discount_factor():
    """γ=0.5, λ=1, r=1, V=0 → A_t = 1 + 0.5 + 0.25 + ... (geometric)."""
    T, N = 4, 1
    buf = RolloutBuffer(T=T, N=N)
    buf.reward[:] = 1.0
    buf.value[:] = 0.0
    buf.done[:] = 0.0
    buf.compute_advantages_and_returns(last_value=torch.zeros(N),
                                       gamma=0.5, gae_lambda=1.0)
    # A_3 = 1
    # A_2 = 1 + 0.5*1 = 1.5
    # A_1 = 1 + 0.5*1.5 = 1.75
    # A_0 = 1 + 0.5*1.75 = 1.875
    expected = torch.tensor([1.875, 1.75, 1.5, 1.0])
    assert torch.allclose(buf.advantages.squeeze(), expected, atol=1e-6)


def test_gae_done_resets_bootstrap():
    """A done at step t should cut off discounted reward propagation."""
    T, N = 3, 1
    buf = RolloutBuffer(T=T, N=N)
    buf.reward[:] = 1.0
    buf.value[:] = 0.0
    buf.done[0] = 1.0   # episode ended after step 0
    buf.done[1] = 0.0
    buf.done[2] = 0.0
    buf.compute_advantages_and_returns(last_value=torch.zeros(N),
                                       gamma=1.0, gae_lambda=1.0)
    # A_2 = 1
    # A_1 = 1 + 1 = 2
    # delta_0 = 1 + 1*0*(1-1) - 0 = 1; A_0 = 1 + 1*1*A_1*(1-1) = 1
    # because done[0]=1, the discounted future is cut off.
    assert torch.allclose(buf.advantages.squeeze(),
                          torch.tensor([1.0, 2.0, 1.0]))


# ---- Minibatch iteration ---------------------------------------------

def test_iter_minibatches_partitions():
    T, N = 4, 4
    buf = RolloutBuffer(T=T, N=N)
    for t in range(T):
        buf.store_step(t,
                       obs=torch.zeros(N, 3, 64, 64, dtype=torch.uint8),
                       action=torch.arange(N) + t * N,  # unique action per slot
                       log_prob=torch.zeros(N),
                       value=torch.zeros(N))
        buf.store_post(t, reward=torch.zeros(N), done=torch.zeros(N))
    buf.compute_advantages_and_returns(last_value=torch.zeros(N))

    gen = torch.Generator()
    gen.manual_seed(0)
    mbs = list(buf.iter_minibatches(num_minibatches=4, generator=gen))
    assert len(mbs) == 4
    # All actions across mini-batches together = original 0..15 (permutation)
    seen = torch.cat([mb["action"] for mb in mbs])
    assert torch.equal(torch.sort(seen).values, torch.arange(16))


def test_iter_minibatches_rejects_indivisible():
    buf = RolloutBuffer(T=3, N=4)
    buf.compute_advantages_and_returns(last_value=torch.zeros(4))
    with pytest.raises(ValueError):
        list(buf.iter_minibatches(num_minibatches=5))  # 12 % 5 != 0


# ---- sample_action ----------------------------------------------------

def test_sample_action_shapes():
    logits = torch.randn(7, NUM_ACTIONS)
    a, lp = sample_action(logits)
    assert a.shape == (7,)
    assert lp.shape == (7,)
    assert a.dtype == torch.long
    assert (a >= 0).all() and (a < NUM_ACTIONS).all()


# ---- ppo_loss ---------------------------------------------------------

@pytest.fixture
def agent():
    return ImpalaAgent()


@pytest.fixture
def small_buffer(agent):
    """Tiny filled rollout buffer for loss testing."""
    T, N = 4, 2
    buf = RolloutBuffer(T=T, N=N)
    torch.manual_seed(0)
    for t in range(T):
        obs = torch.randint(0, 256, (N, 3, 64, 64), dtype=torch.uint8)
        with torch.no_grad():
            out = agent(obs)
        a, lp = sample_action(out.logits)
        buf.store_step(t, obs=obs, action=a, log_prob=lp, value=out.value)
        buf.store_post(t,
                       reward=torch.randn(N),
                       done=torch.zeros(N))
    with torch.no_grad():
        last_out = agent(torch.randint(0, 256, (N, 3, 64, 64), dtype=torch.uint8))
    buf.compute_advantages_and_returns(last_value=last_out.value)
    return buf


def test_ppo_loss_scalar_and_backprop(agent, small_buffer):
    config = PPOConfig()
    mb = next(iter(small_buffer.iter_minibatches(num_minibatches=2)))
    losses = ppo_loss(agent, mb, config)
    assert losses["total"].dim() == 0          # scalar
    assert losses["total"].requires_grad        # has grad
    # Backward populates agent params
    losses["total"].backward()
    has_grad = any(p.grad is not None and p.grad.abs().max() > 0
                   for p in agent.parameters())
    assert has_grad, "no agent param received gradient"


def test_ppo_loss_components_are_detached(agent, small_buffer):
    """Diagnostic components should not be in autograd graph."""
    config = PPOConfig()
    mb = next(iter(small_buffer.iter_minibatches(num_minibatches=2)))
    losses = ppo_loss(agent, mb, config)
    for key in ("policy", "value", "entropy", "approx_kl", "clipfrac"):
        assert not losses[key].requires_grad, (
            f"{key} should be detached (diagnostic only)"
        )


# ---- PPOUpdater ------------------------------------------------------

def test_updater_runs_and_decreases_loss_on_overfit(agent, small_buffer):
    """Run several PPO updates on the same tiny buffer; loss should drop
    (we're overfitting that buffer)."""
    config = PPOConfig(ppo_epochs=2, mini_batches_per_epoch=2,
                       learning_rate=1e-3)
    updater = PPOUpdater(agent, config)
    log1 = updater.update(small_buffer)
    log2 = updater.update(small_buffer)
    log3 = updater.update(small_buffer)
    # Total loss should drop monotonically-ish on this trivial overfit.
    assert log3["total"] < log1["total"], (
        f"loss did not drop: {log1['total']:.4f} -> {log3['total']:.4f}"
    )


def test_updater_grad_clipping_doesnt_break(agent, small_buffer):
    """Even with a very tight grad clip, update should not crash and params
    should still change."""
    config = PPOConfig(max_grad_norm=1e-4)
    updater = PPOUpdater(agent, config)
    p0 = next(agent.parameters()).clone()
    updater.update(small_buffer)
    p1 = next(agent.parameters())
    assert not torch.equal(p0, p1)  # some change
