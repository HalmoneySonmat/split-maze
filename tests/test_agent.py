"""Tests for src/split_maze/agent.py — IMPALA-CNN."""

import pytest
import torch

from split_maze.agent import D_A, NUM_ACTIONS, AgentOutput, ImpalaAgent


@pytest.fixture
def agent() -> ImpalaAgent:
    return ImpalaAgent()


def test_forward_shapes_uint8(agent):
    obs = torch.randint(0, 256, (4, 3, 64, 64), dtype=torch.uint8)
    out = agent(obs)
    assert isinstance(out, AgentOutput)
    assert out.logits.shape == (4, NUM_ACTIONS)
    assert out.value.shape == (4,)
    assert out.h_agent.shape == (4, D_A)


def test_forward_shapes_float(agent):
    obs = torch.rand(2, 3, 64, 64)  # in [0,1]
    out = agent(obs)
    assert out.logits.shape == (2, NUM_ACTIONS)
    assert out.value.shape == (2,)
    assert out.h_agent.shape == (2, D_A)


def test_h_agent_post_relu_nonnegative(agent):
    """h_agent = ReLU(Linear(...)), so all values must be ≥ 0."""
    obs = torch.rand(8, 3, 64, 64)
    out = agent(obs)
    assert (out.h_agent >= 0).all()


def test_gradients_flow(agent):
    obs = torch.rand(4, 3, 64, 64, requires_grad=False)
    out = agent(obs)
    loss = out.logits.sum() + out.value.sum() + out.h_agent.sum()
    loss.backward()
    # Every parameter that requires grad must receive a non-None gradient.
    for name, p in agent.named_parameters():
        assert p.grad is not None, f"no grad for {name}"
        assert torch.isfinite(p.grad).all(), f"non-finite grad in {name}"


def test_h_agent_gradient_isolatable(agent):
    """The PLAN §4.3 (C-thin) detach blocks grad to the agent backbone.

    Simulates ACC's reconstruction path: a learnable W (the ACC matrix)
    operates on h_agent.detach() and the loss backprops. W should receive
    gradient (it's in the graph); the agent's parameters should not.
    """
    obs = torch.rand(2, 3, 64, 64)
    out = agent(obs)

    # Stand-in for the ACC's W matrix — needs grad like a learnable param.
    W_fake = torch.randn(D_A, D_A, requires_grad=True)

    # Reconstruction-style loss with h_agent detached (C-thin: agent side).
    recon = out.h_agent.detach() @ W_fake.T
    loss = (recon ** 2).sum()
    loss.backward()

    # The "ACC" stand-in must have received gradient
    assert W_fake.grad is not None
    assert W_fake.grad.abs().max() > 0, "W_fake (ACC stand-in) got no gradient"

    # No agent parameter should have any gradient (detach blocked everything)
    for name, p in agent.named_parameters():
        assert p.grad is None or p.grad.abs().max() == 0, (
            f"(C-thin) detach failed: agent param {name} got nonzero grad"
        )


def test_rejects_bad_obs_shape(agent):
    with pytest.raises(ValueError):
        agent(torch.rand(2, 3, 32, 32))
    with pytest.raises(ValueError):
        agent(torch.rand(3, 64, 64))


def test_param_count_reasonable(agent):
    # Hand-computed expectation ~620k for channels=(16,32,32), d_a=256.
    n = agent.num_params
    assert 500_000 < n < 800_000, f"unexpected param count {n}"


def test_policy_head_small_init():
    """Policy head should be near-uniform at init (small gain=0.01)."""
    agent = ImpalaAgent()
    obs = torch.zeros(1, 3, 64, 64)
    out = agent(obs)
    # logits should be small in magnitude
    assert out.logits.abs().max() < 0.5, (
        f"policy logits too large at init: max={out.logits.abs().max().item():.3f}"
    )


def test_value_head_zero_at_zero_input():
    """Value head with zero input + zero-bias init should be exactly zero."""
    agent = ImpalaAgent()
    obs = torch.zeros(1, 3, 64, 64)
    out = agent(obs)
    # With zero conv biases, the conv outputs aren't strictly zero (since
    # weight init is orthogonal not zero) but h_agent goes through ReLU → 0
    # (since the Linear's bias is zero and pre-ReLU value is zero...). Actually
    # the conv layers will produce nonzero outputs from random weights even on
    # zero input — NO, conv on zero input gives bias (which we init to 0) so
    # output is 0. Through all blocks, output stays 0. Then h_agent = ReLU(0) = 0,
    # value = 0.
    assert torch.allclose(out.value, torch.zeros_like(out.value), atol=1e-6)
    assert torch.allclose(out.h_agent, torch.zeros_like(out.h_agent), atol=1e-6)


def test_d_a_is_256():
    assert D_A == 256  # PLAN §3.4 박제


# --- R2 feedback gate (PREREG §1: h' = h + λ·Wᵀ·ñ_lm) ----------------------

def test_inject_none_is_default(agent):
    """inject=None must reproduce the no-arg forward byte-for-byte (R0/R1)."""
    obs = torch.rand(4, 3, 64, 64)
    a = agent(obs)
    b = agent(obs, inject=None)
    assert torch.equal(a.logits, b.logits)
    assert torch.equal(a.value, b.value)
    assert torch.equal(a.h_agent, b.h_agent)


def test_inject_zero_identical(agent):
    """Zero injection leaves logits/value/h_agent unchanged (h' = h + 0)."""
    obs = torch.rand(4, 3, 64, 64)
    base = agent(obs)
    z = agent(obs, inject=torch.zeros(4, D_A))
    assert torch.allclose(z.logits, base.logits, atol=1e-6)
    assert torch.allclose(z.value, base.value, atol=1e-6)
    assert torch.equal(z.h_agent, base.h_agent)


def test_inject_changes_action_not_h_agent(agent):
    """Nonzero injection modulates logits/value (load-bearing) but h_agent
    stays PRE-injection (clean-read for eval — PREREG fix #2)."""
    obs = torch.rand(4, 3, 64, 64)
    base = agent(obs)
    inj = torch.randn(4, D_A)
    out = agent(obs, inject=inj)
    assert torch.equal(out.h_agent, base.h_agent)        # pre-injection, unchanged
    assert not torch.allclose(out.logits, base.logits)   # action modulated
    assert not torch.allclose(out.value, base.value)


def test_inject_shape_mismatch_raises(agent):
    obs = torch.rand(4, 3, 64, 64)
    with pytest.raises(ValueError):
        agent(obs, inject=torch.randn(4, D_A + 1))   # wrong feature dim
    with pytest.raises(ValueError):
        agent(obs, inject=torch.randn(3, D_A))        # wrong batch


def test_inject_grad_flows_into_inject(agent):
    """Grad must reach inject so the bridge can train via the feedback path;
    the caller detaches it for the pure-RL PPO update ((C-thin) boundary)."""
    obs = torch.rand(2, 3, 64, 64)
    inj = torch.randn(2, D_A, requires_grad=True)
    out = agent(obs, inject=inj)
    out.logits.sum().backward()
    assert inj.grad is not None
    assert inj.grad.abs().max() > 0
