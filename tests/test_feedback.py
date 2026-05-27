"""Tests for the Phase-6 R2 feedback path (src/split_maze/feedback.py) and
MazeLM.summarize_vector — PREREG §1 (lm → agent)."""

import pytest
import torch

from split_maze.acc import ACC, ACCConfig
from split_maze.feedback import compute_inject, echo_ratio
from split_maze.lm import LMConfig, MazeLM, MazeTokenizer


@pytest.fixture
def kit():
    """Small ACC + MazeLM with matching d_lm. eval() so dropout is off
    (deterministic — needed for the linearity check)."""
    tok = MazeTokenizer()
    d_a, d_lm = 8, 16
    acc = ACC(ACCConfig(d_agent=d_a, d_lm=d_lm, tied=False)).eval()
    lm = MazeLM(LMConfig.from_tokenizer(
        tok, d_model=d_lm, n_head=2, n_layer=1, d_ff=32, max_len=8)).eval()
    return acc, lm, d_a, d_lm


def test_summarize_vector_shape_and_grad(kit):
    _, lm, _, d_lm = kit
    h_in = torch.randn(5, d_lm, requires_grad=True)
    h_lm = lm.summarize_vector(h_in)
    assert h_lm.shape == (5, d_lm)
    h_lm.sum().backward()
    assert lm.interface_proj.weight.grad is not None   # differentiable read
    assert h_in.grad is not None


def test_compute_inject_shape(kit):
    acc, lm, d_a, _ = kit
    inj = compute_inject(acc, lm, torch.randn(5, d_a), lam=0.3)
    assert inj.shape == (5, d_a)


def test_inject_zero_lambda_is_zero(kit):
    """λ=0 ⇒ no feedback ⇒ byte-identical to R0/R1 (gate off)."""
    acc, lm, d_a, _ = kit
    inj = compute_inject(acc, lm, torch.randn(5, d_a), lam=0.0)
    assert torch.allclose(inj, torch.zeros_like(inj))


def test_inject_linear_in_lambda(kit):
    """inject scales linearly with the fixed gate λ (λ only multiplies)."""
    acc, lm, d_a, _ = kit
    h = torch.randn(5, d_a)
    a = compute_inject(acc, lm, h, lam=0.3)
    b = compute_inject(acc, lm, h, lam=0.6)
    assert torch.allclose(b, 2.0 * a, atol=1e-5)


def test_echo_ratio_range(kit):
    """Echo-check is a per-sample cosine ⇒ in [-1, 1], one value per row."""
    acc, lm, d_a, _ = kit
    r = echo_ratio(acc, lm, torch.randn(5, d_a))
    assert r.shape == (5,)
    assert (r >= -1.0 - 1e-5).all() and (r <= 1.0 + 1e-5).all()


def test_inject_feeds_agent_gate(kit):
    """End-to-end shape contract: compute_inject output is a valid
    ImpalaAgent.forward(inject=...) argument (matches d_agent)."""
    acc, lm, d_a, _ = kit
    from split_maze.agent import ImpalaAgent
    agent = ImpalaAgent(d_a=d_a).eval()
    h = torch.randn(3, d_a)
    inj = compute_inject(acc, lm, h, lam=0.3)
    obs = torch.rand(3, 3, 64, 64)
    out = agent(obs, inject=inj)          # must not raise (shape matches)
    assert out.logits.shape[0] == 3
