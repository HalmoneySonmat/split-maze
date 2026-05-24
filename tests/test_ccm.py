"""Unit tests for split_maze.ccm (Phase 5 — CCM co-activation memory bridge).

Organised by concern:

1. CoActAccumulator — cumulative running mean exactness, EMA mode, centered
   moments, shape/validation guards.
2. fit_W — the three rungs: hebbian (raw outer product, b=0), ridge (recovers a
   known linear map / beats random W), procrustes (semi-orthogonal, finite).
3. CCMBridge — construction (LM frozen, W/b are buffers not params,
   interpreter_parameters empty), set_W, bridge shape, generate validity,
   update finite loss, set_lm_trainable_interface toggle.
4. End-to-end — fit W from (LN(h_agent), lm.encode(ids)) pairs and confirm the
   recorded bridge reconstructs lm.encode better than a random W (the
   "memory transfers at the vector level" sanity, step1's premise).
"""

from __future__ import annotations

import random

import pytest

torch = pytest.importorskip("torch")

import torch.nn as nn  # noqa: E402

from split_maze.ccm import CCMBridge, CoActAccumulator, fit_W  # noqa: E402
from split_maze.language import (  # noqa: E402
    CHEESE_DIR_VALUES,
    HEADING_VALUES,
    REGION_COLS,
    REGION_ROWS,
    Slots,
    render,
)
from split_maze.lm import LMConfig, MazeLM, MazeTokenizer  # noqa: E402


D_AGENT = 16  # small agent width for fast tests


def _small_lm(tok: MazeTokenizer) -> MazeLM:
    cfg = LMConfig.from_tokenizer(
        tok, d_model=32, n_head=4, n_layer=2, d_ff=64, max_len=32, dropout=0.0
    )
    return MazeLM(cfg)


def _sentence_batch(tok: MazeTokenizer, B: int = 8, seed: int = 0):
    """A padded (ids, lengths) batch of random describer sentences."""
    rng = random.Random(seed)
    ids_list, lens = [], []
    for _ in range(B):
        slots = Slots(
            agent_row=rng.choice(REGION_ROWS),
            agent_col=rng.choice(REGION_COLS),
            heading=rng.choice(HEADING_VALUES),
            cheese_dir=rng.choice(CHEESE_DIR_VALUES),
        )
        ids = tok.encode(render(slots, rng=rng, include_bos_eos=True))
        ids_list.append(ids)
        lens.append(len(ids))
    T = max(lens)
    padded = torch.full((B, T), tok.pad_id, dtype=torch.long)
    for i, ids in enumerate(ids_list):
        padded[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
    return padded, torch.tensor(lens)


# ---- 1. CoActAccumulator ------------------------------------------------


def test_accumulator_cumulative_matches_full_batch():
    torch.manual_seed(0)
    d_x, d_y = 5, 7
    X = torch.randn(40, d_x)
    Y = torch.randn(40, d_y)
    acc = CoActAccumulator(d_x, d_y)  # cumulative
    # Feed in 4 uneven chunks.
    for sl in (slice(0, 7), slice(7, 20), slice(20, 33), slice(33, 40)):
        acc.update(X[sl], Y[sl])
    assert acc.count == 40
    m = acc.moments()
    # Compare to direct full-batch moments. The chunked running mean is exact in
    # exact arithmetic; allow float64 rounding noise (~1e-8) from the sequential
    # weighted updates vs a single batched matmul.
    torch.testing.assert_close(m["m_x"], X.mean(0).double(), atol=1e-6, rtol=1e-4)
    torch.testing.assert_close(m["m_y"], Y.mean(0).double(), atol=1e-6, rtol=1e-4)
    exp_yx = (Y.t() @ X / 40).double()
    torch.testing.assert_close(m["m_yx"], exp_yx, atol=1e-6, rtol=1e-4)
    # Centered covariance c_xx = cov_uncorrected.
    Xc = X - X.mean(0)
    exp_cxx = (Xc.t() @ Xc / 40).double()
    torch.testing.assert_close(m["c_xx"], exp_cxx, atol=1e-6, rtol=1e-4)


def test_accumulator_ema_runs_and_tracks_recent():
    torch.manual_seed(1)
    d_x, d_y = 4, 4
    acc = CoActAccumulator(d_x, d_y, decay=0.9)
    # Early batches around mean 0, later batches shifted to mean +5.
    for _ in range(5):
        acc.update(torch.randn(16, d_x), torch.randn(16, d_y))
    for _ in range(30):
        acc.update(torch.randn(16, d_x) + 5.0, torch.randn(16, d_y))
    m = acc.moments()
    # EMA should sit near the recent (shifted) regime, not the old one.
    assert m["m_x"].mean().item() > 3.0


def test_accumulator_empty_raises_and_validates():
    acc = CoActAccumulator(3, 4)
    with pytest.raises(RuntimeError):
        acc.moments()
    with pytest.raises(ValueError):
        acc.update(torch.randn(2, 3), torch.randn(3, 4))  # batch mismatch
    with pytest.raises(ValueError):
        acc.update(torch.randn(2, 9), torch.randn(2, 4))  # wrong d_x
    with pytest.raises(ValueError):
        CoActAccumulator(3, 4, decay=1.5)  # bad decay


# ---- 2. fit_W rungs -----------------------------------------------------


def _moments_from(X: torch.Tensor, Y: torch.Tensor) -> dict:
    acc = CoActAccumulator(X.shape[1], Y.shape[1])
    acc.update(X, Y)
    return acc.moments()


def test_fit_hebbian_shape_and_zero_bias():
    torch.manual_seed(2)
    X = torch.randn(50, 6)
    Y = torch.randn(50, 9)
    W, b = fit_W(_moments_from(X, Y), "hebbian")
    assert W.shape == (9, 6)
    assert b.shape == (9,)
    assert torch.allclose(b, torch.zeros_like(b))
    # Equals the raw mean outer product.
    torch.testing.assert_close(W.double(), (Y.t() @ X / 50).double(),
                               atol=1e-6, rtol=1e-4)


def test_fit_ridge_recovers_linear_map():
    torch.manual_seed(3)
    d_x, d_y = 8, 10
    M = torch.randn(d_y, d_x)
    bias = torch.randn(d_y)
    X = torch.randn(400, d_x)
    Y = X @ M.t() + bias + 0.01 * torch.randn(400, d_y)
    W, b = fit_W(_moments_from(X, Y), "ridge", ridge_lambda=1e-4)
    # Recorded ridge map should approximate the true (M, bias).
    assert (W - M).norm() / M.norm() < 0.1
    Y_hat = X @ W.t() + b
    mse = (Y_hat - Y).pow(2).mean().item()
    assert mse < 0.05


def test_fit_ridge_beats_random_W():
    torch.manual_seed(4)
    d_x, d_y = 8, 12
    M = torch.randn(d_y, d_x)
    X = torch.randn(300, d_x)
    Y = X @ M.t() + 0.1 * torch.randn(300, d_y)
    W, b = fit_W(_moments_from(X, Y), "ridge")
    Wr = torch.randn(d_y, d_x)
    mse_fit = (X @ W.t() + b - Y).pow(2).mean().item()
    mse_rand = (X @ Wr.t() - Y).pow(2).mean().item()
    assert mse_fit < mse_rand


def test_fit_procrustes_semi_orthogonal():
    torch.manual_seed(5)
    d_x, d_y = 6, 6
    X = torch.randn(200, d_x)
    Y = torch.randn(200, d_y)
    W, b = fit_W(_moments_from(X, Y), "procrustes", procrustes_scale=False)
    assert W.shape == (d_y, d_x)
    # Pure semi-orthogonal: W Wᵀ ≈ I on the (square) overlap.
    prod = W @ W.t()
    torch.testing.assert_close(prod, torch.eye(d_y), atol=1e-4, rtol=1e-3)
    # Finite with scaling too.
    Ws, _ = fit_W(_moments_from(X, Y), "procrustes", procrustes_scale=True)
    assert torch.isfinite(Ws).all()


def test_fit_centering_only_is_centered_cov_with_bias():
    """Ablation corner — centering, no whitening: W is exactly the centered
    cross-covariance c_yx (no inverse) and the bias closes the means."""
    torch.manual_seed(8)
    X = torch.randn(120, 6)
    Y = torch.randn(120, 9)
    m = _moments_from(X, Y)
    W, b = fit_W(m, "centering_only")
    assert W.shape == (9, 6) and b.shape == (9,)
    torch.testing.assert_close(W.double(), m["c_yx"], atol=1e-6, rtol=1e-4)
    torch.testing.assert_close(
        b.double(), m["m_y"] - m["c_yx"] @ m["m_x"], atol=1e-6, rtol=1e-4)


def test_fit_whitening_only_uncentered_ridge_zero_bias():
    """Ablation corner — whitening, no centering: solves the *un-centered*
    normal equations (m_xx + λI) Wᵀ = m_yxᵀ with zero bias."""
    torch.manual_seed(9)
    X = torch.randn(150, 7)
    Y = torch.randn(150, 5)
    m = _moments_from(X, Y)
    lam = 1e-3
    W, b = fit_W(m, "whitening_only", ridge_lambda=lam)
    assert W.shape == (5, 7) and b.shape == (5,)
    assert torch.allclose(b, torch.zeros_like(b))  # no centering bias
    A = m["m_xx"] + lam * torch.eye(7, dtype=torch.float64)
    # Normal equations are (m_xx + λI) Wᵀ = m_yxᵀ → compare against m_yx transposed.
    resid = (A @ W.t().double() - m["m_yx"].t()).abs().max().item()
    assert resid < 1e-3
    # Differs from centered ridge (centering changes the map).
    Wr, _ = fit_W(m, "ridge", ridge_lambda=lam)
    assert (W - Wr).abs().max().item() > 1e-6


def test_fit_W_unknown_method():
    with pytest.raises(ValueError):
        fit_W(_moments_from(torch.randn(10, 3), torch.randn(10, 4)), "nope")


# ---- 3. CCMBridge -------------------------------------------------------


def test_bridge_construction_lm_frozen_and_buffers():
    tok = MazeTokenizer()
    lm = _small_lm(tok)
    bridge = CCMBridge(lm, d_agent=D_AGENT)
    d_model = lm.config.d_model
    # LM fully frozen.
    assert all(not p.requires_grad for p in bridge.lm.parameters())
    # W, b are buffers — present in state_dict but NOT trainable params.
    sd = dict(bridge.named_buffers())
    assert "W" in sd and "b" in sd
    assert sd["W"].shape == (d_model, D_AGENT)
    param_ids = {id(p) for p in bridge.parameters()}
    assert id(bridge.W) not in param_ids and id(bridge.b) not in param_ids
    # No trainable interpreter params (W is recorded, not learned).
    assert list(bridge.interpreter_parameters()) == []


def test_bridge_set_W_and_shape():
    tok = MazeTokenizer()
    lm = _small_lm(tok)
    bridge = CCMBridge(lm, d_agent=D_AGENT)
    d_model = lm.config.d_model
    W = torch.randn(d_model, D_AGENT)
    b = torch.randn(d_model)
    bridge.set_W(W, b)
    torch.testing.assert_close(bridge.W, W)
    torch.testing.assert_close(bridge.b, b)
    h = torch.randn(5, D_AGENT)
    out = bridge.bridge(h)
    assert out.shape == (5, d_model)
    # set_W with wrong shape rejected.
    with pytest.raises(ValueError):
        bridge.set_W(torch.randn(d_model, D_AGENT + 1))


def test_bridge_generate_valid_ids():
    tok = MazeTokenizer()
    lm = _small_lm(tok)
    bridge = CCMBridge(lm, d_agent=D_AGENT)
    bridge.set_W(torch.randn(lm.config.d_model, D_AGENT) * 0.1)
    h = torch.randn(4, D_AGENT)
    seq = bridge.generate(h, max_len=12)
    assert seq.dim() == 2 and seq.shape[0] == 4
    assert 1 <= seq.shape[1] <= 12
    assert (seq >= 0).all() and (seq < tok.vocab_size).all()


def test_bridge_update_finite_loss():
    tok = MazeTokenizer()
    lm = _small_lm(tok)
    bridge = CCMBridge(lm, d_agent=D_AGENT)
    bridge.set_W(torch.randn(lm.config.d_model, D_AGENT) * 0.1)
    ids, lengths = _sentence_batch(tok, B=6)
    h = torch.randn(6, D_AGENT)
    out = bridge.update(h, ids, lengths)
    assert "loss" in out
    assert torch.isfinite(out["loss"]) and out["loss"].item() > 0.0


def test_bridge_loss_grads_to_input_not_W():
    """step2 wiring: the bridge next-token loss must flow into the agent-side
    input (h_agent) but NOT into W/b (recorded buffers) or the frozen LM."""
    tok = MazeTokenizer()
    lm = _small_lm(tok)
    bridge = CCMBridge(lm, d_agent=D_AGENT)
    bridge.set_W(torch.randn(lm.config.d_model, D_AGENT) * 0.1)
    ids, lengths = _sentence_batch(tok, B=6)
    h = torch.randn(6, D_AGENT, requires_grad=True)  # stands in for agent output
    out = bridge.update(h, ids, lengths)
    out["loss"].backward()
    # Gradient reaches the (agent-side) input.
    assert h.grad is not None and torch.isfinite(h.grad).any()
    # W/b are buffers — they never carry gradient (memory, not learning).
    assert bridge.W.grad is None and bridge.b.grad is None
    # The frozen LM core received no gradient.
    assert all(p.grad is None for p in bridge.lm.parameters())


def test_set_lm_trainable_interface_toggle():
    tok = MazeTokenizer()
    lm = _small_lm(tok)
    bridge = CCMBridge(lm, d_agent=D_AGENT)
    assert all(not p.requires_grad for p in bridge.lm.interface_proj.parameters())
    bridge.set_lm_trainable_interface(True)
    assert all(p.requires_grad for p in bridge.lm.interface_proj.parameters())
    # Rest of the LM stays frozen.
    assert not bridge.lm.tok_embed.weight.requires_grad
    bridge.set_lm_trainable_interface(False)
    assert all(not p.requires_grad for p in bridge.lm.interface_proj.parameters())


def test_set_lm_trainable_block_toggle():
    """step3 (A2) — unfreeze one decoder block (the bridge generate path) while
    the rest of the LM core stays frozen."""
    tok = MazeTokenizer()
    lm = _small_lm(tok)  # n_layer=2
    bridge = CCMBridge(lm, d_agent=D_AGENT)
    assert all(not p.requires_grad for p in bridge.lm.blocks[0].parameters())
    bridge.set_lm_trainable_block(0, True)
    assert all(p.requires_grad for p in bridge.lm.blocks[0].parameters())
    assert all(not p.requires_grad for p in bridge.lm.blocks[1].parameters())  # others frozen
    assert not bridge.lm.lm_head.weight.requires_grad
    bridge.set_lm_trainable_block(0, False)
    assert all(not p.requires_grad for p in bridge.lm.blocks[0].parameters())


def test_set_plastic_warm_start_and_param_toggle():
    """step3 — recorded W/b buffers become trainable Parameters warm-started from
    the recorded values, and revert cleanly."""
    tok = MazeTokenizer()
    lm = _small_lm(tok)
    bridge = CCMBridge(lm, d_agent=D_AGENT)
    d_model = lm.config.d_model
    W = torch.randn(d_model, D_AGENT) * 0.1
    b = torch.randn(d_model) * 0.1
    bridge.set_W(W, b)
    assert not isinstance(bridge.W, nn.Parameter)
    assert list(bridge.interpreter_parameters()) == []
    # buffer → Parameter, values preserved (warm start)
    bridge.set_plastic(True)
    assert isinstance(bridge.W, nn.Parameter) and isinstance(bridge.b, nn.Parameter)
    torch.testing.assert_close(bridge.W.data, W)
    torch.testing.assert_close(bridge.b.data, b)
    assert len(list(bridge.interpreter_parameters())) == 2
    pid = {id(p) for p in bridge.parameters()}
    assert id(bridge.W) in pid and id(bridge.b) in pid
    bridge.set_plastic(True)  # idempotent
    assert isinstance(bridge.W, nn.Parameter)
    # Parameter → buffer, values preserved
    bridge.set_plastic(False)
    assert not isinstance(bridge.W, nn.Parameter)
    assert list(bridge.interpreter_parameters()) == []
    torch.testing.assert_close(bridge.W, W)
    # set_W still works after toggling
    bridge.set_W(W * 2, b * 2)
    torch.testing.assert_close(bridge.W, W * 2)


def test_plastic_bridge_loss_grads_to_W_even_with_frozen_agent():
    """step3 A1: agent frozen (h detached) but plastic W must still receive
    gradient so the bridge can grow; the frozen LM core gets none."""
    tok = MazeTokenizer()
    lm = _small_lm(tok)
    bridge = CCMBridge(lm, d_agent=D_AGENT)
    bridge.set_W(torch.randn(lm.config.d_model, D_AGENT) * 0.1)
    bridge.set_plastic(True)
    ids, lengths = _sentence_batch(tok, B=6)
    h = torch.randn(6, D_AGENT)  # frozen-agent output
    out = bridge.update(h.detach(), ids, lengths)
    out["loss"].backward()
    assert bridge.W.grad is not None and torch.isfinite(bridge.W.grad).any()
    assert bridge.b.grad is not None
    assert all(p.grad is None for p in bridge.lm.parameters())  # LM core frozen


# ---- 4. end-to-end: recorded bridge beats random W ----------------------


def test_recorded_bridge_reconstructs_better_than_random():
    """Fit W from (LN(h_agent), lm.encode(ids)) pairs; the recorded ridge
    bridge should reconstruct lm.encode(ids) better than a random W. This is
    step1's premise at the vector level (does the co-activation memory carry
    a usable correspondence)."""
    torch.manual_seed(7)
    tok = MazeTokenizer()
    lm = _small_lm(tok)
    bridge = CCMBridge(lm, d_agent=D_AGENT)
    d_model = lm.config.d_model

    # A fixed (random) "agent" embedding per sentence, correlated with the LM
    # target via a hidden linear map so a correspondence exists to record.
    ids, lengths = _sentence_batch(tok, B=64, seed=11)
    with torch.no_grad():
        a2 = lm.encode(ids)                        # (B, d_model) target
    Mtrue = torch.randn(D_AGENT, d_model) * 0.3
    h_agent = a2 @ Mtrue.t() + 0.05 * torch.randn(a2.shape[0], D_AGENT)

    # Record W on LN(h_agent) → a2.
    x = bridge.ln_agent(h_agent).detach()
    acc = CoActAccumulator(D_AGENT, d_model)
    acc.update(x, a2)
    W, b = fit_W(acc.moments(), "ridge")
    bridge.set_W(W, b)

    mse_fit = (bridge.bridge(h_agent) - a2).pow(2).mean().item()

    bridge.set_W(torch.randn(d_model, D_AGENT) * W.std().item())
    mse_rand = (bridge.bridge(h_agent) - a2).pow(2).mean().item()

    assert mse_fit < mse_rand
