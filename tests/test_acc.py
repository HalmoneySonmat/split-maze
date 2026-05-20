"""Unit tests for split_maze.acc (ACC — Artificial Corpus Callosum).

Tests organised by concern:

1. Config defaults match Phase-3 박제값 (PLAN §10.2 P3-3-A).
2. Module construction — W shape, LayerNorm presence, parameter count.
3. W initialisation — orthogonal columns orthonormal, xavier bounded,
   reset_W functions.
4. Forward helpers — shapes of normalize, predict_lm_from_agent,
   predict_agent_from_lm.
5. recon_loss — output keys, scalar/finite, decomposition into
   a2l + l2a.
6. **(C-thin) detach policy — critical**. Grad flow:
     * h_agent receives no grad (boundary 1)
     * h_lm receives grad (boundary 2 allowed)
     * W and LayerNorms receive grad
7. cross_cosine — shapes, no_grad context.
8. acc_parameters accessor — yields all and only ACC params.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

import torch.nn as nn  # noqa: E402

from split_maze.acc import ACC, ACCConfig  # noqa: E402


# ---- 1. Config defaults --------------------------------------------------


class TestACCConfigDefaults:
    def test_default_d_agent_is_256(self):
        # PROCGEN_ENV.md §7: IMPALA-CNN final dense = 256.
        assert ACCConfig().d_agent == 256

    def test_default_d_lm_is_256(self):
        # Phase 2 박제 P2-2: MazeLM d_model = 256.
        assert ACCConfig().d_lm == 256

    def test_default_init_is_orthogonal(self):
        # PLAN §10.2 P3-3-A.
        assert ACCConfig().init == "orthogonal"

    def test_layernorm_eps_default(self):
        assert ACCConfig().layernorm_eps == 1e-5

    def test_unknown_init_raises(self):
        with pytest.raises(ValueError, match="unknown W init"):
            ACC(ACCConfig(init="kaiming"))


# ---- 2. Module construction ---------------------------------------------


class TestACCConstruction:
    def test_default_construction(self):
        acc = ACC()
        assert isinstance(acc, nn.Module)

    def test_W_shape_is_d_lm_by_d_agent(self):
        acc = ACC(ACCConfig(d_agent=128, d_lm=64))
        assert acc.W.shape == (64, 128)

    def test_W_is_a_parameter(self):
        acc = ACC()
        assert isinstance(acc.W, nn.Parameter)
        assert acc.W.requires_grad

    def test_layernorms_present_with_correct_sizes(self):
        acc = ACC(ACCConfig(d_agent=128, d_lm=64))
        assert isinstance(acc.ln_agent, nn.LayerNorm)
        assert isinstance(acc.ln_lm, nn.LayerNorm)
        assert acc.ln_agent.normalized_shape == (128,)
        assert acc.ln_lm.normalized_shape == (64,)

    def test_param_count_matches_W_plus_two_layernorms(self):
        # Expected = d_lm·d_a + 2·(d_a + d_a) + 2·(d_lm + d_lm)
        #          = W + ln_agent(weight,bias) + ln_lm(weight,bias)
        acc = ACC(ACCConfig(d_agent=128, d_lm=64))
        expected = 64 * 128 + 2 * 128 + 2 * 64
        actual = sum(p.numel() for p in acc.parameters())
        assert actual == expected

    def test_param_count_at_default_under_70k(self):
        # PLAN §3.4 박제: ACC ~ 65k params. 256·256 + 2·256·2 = 65536 + 1024.
        acc = ACC()
        n = sum(p.numel() for p in acc.parameters())
        assert n == 256 * 256 + 4 * 256
        assert n < 70_000


# ---- 3. W initialisation -------------------------------------------------


class TestWInit:
    def test_orthogonal_init_yields_orthonormal_rows_or_cols(self):
        # For a square matrix nn.init.orthogonal_ gives W @ W.T = I.
        torch.manual_seed(0)
        acc = ACC(ACCConfig(d_agent=8, d_lm=8, init="orthogonal"))
        # 8x8 orthogonal: W W.T = I.
        prod = acc.W @ acc.W.t()
        assert torch.allclose(prod, torch.eye(8), atol=1e-5)

    def test_orthogonal_init_nonsquare(self):
        # Nonsquare: orthogonal_ makes shorter dim orthonormal.
        torch.manual_seed(0)
        acc = ACC(ACCConfig(d_agent=16, d_lm=8, init="orthogonal"))
        # rows orthonormal: W @ W.T = I_8.
        prod = acc.W @ acc.W.t()
        assert torch.allclose(prod, torch.eye(8), atol=1e-5)

    def test_xavier_init_bounded(self):
        # Xavier uniform_ bounds = ±sqrt(6/(d_a + d_lm)).
        torch.manual_seed(0)
        acc = ACC(ACCConfig(d_agent=128, d_lm=128, init="xavier"))
        bound = (6.0 / (128 + 128)) ** 0.5
        assert acc.W.abs().max().item() <= bound + 1e-6

    def test_orthogonal_and_xavier_differ(self):
        torch.manual_seed(0)
        acc_o = ACC(ACCConfig(d_agent=64, d_lm=64, init="orthogonal"))
        torch.manual_seed(0)
        acc_x = ACC(ACCConfig(d_agent=64, d_lm=64, init="xavier"))
        assert not torch.allclose(acc_o.W, acc_x.W)

    def test_reset_W_in_place_changes_values(self):
        torch.manual_seed(0)
        acc = ACC(ACCConfig(d_agent=16, d_lm=16, init="orthogonal"))
        before = acc.W.detach().clone()
        torch.manual_seed(123)  # different seed → different orthogonal matrix
        acc.reset_W()
        assert not torch.allclose(acc.W, before)

    def test_reset_W_with_kind_override(self):
        acc = ACC(ACCConfig(d_agent=64, d_lm=64, init="orthogonal"))
        acc.reset_W(kind="xavier")
        bound = (6.0 / (64 + 64)) ** 0.5
        assert acc.W.abs().max().item() <= bound + 1e-6


# ---- 4. Forward helpers --------------------------------------------------


class TestForwardHelpers:
    def test_normalize_shape(self):
        acc = ACC(ACCConfig(d_agent=128, d_lm=64))
        h_a = torch.randn(4, 128)
        h_l = torch.randn(4, 64)
        n_a, n_l = acc.normalize(h_a, h_l)
        assert n_a.shape == (4, 128)
        assert n_l.shape == (4, 64)

    def test_normalize_layernorm_applies(self):
        # After LayerNorm with default eps, the per-sample mean ≈ 0 and
        # std ≈ 1 (modulo the learnable affine, which starts as
        # weight=1, bias=0).
        acc = ACC(ACCConfig(d_agent=64, d_lm=64))
        h_a = torch.randn(8, 64) * 5.0 + 3.0  # arbitrary scale/shift
        n_a, _ = acc.normalize(h_a, torch.zeros(8, 64))
        # Per-row stats:
        per_row_mean = n_a.mean(dim=-1)
        per_row_std = n_a.std(dim=-1, unbiased=False)
        assert torch.allclose(per_row_mean, torch.zeros(8), atol=1e-5)
        assert torch.allclose(per_row_std, torch.ones(8), atol=1e-2)

    def test_predict_lm_from_agent_shape(self):
        acc = ACC(ACCConfig(d_agent=128, d_lm=64))
        n_a = torch.randn(4, 128)
        hat_lm = acc.predict_lm_from_agent(n_a)
        assert hat_lm.shape == (4, 64)

    def test_predict_agent_from_lm_shape(self):
        acc = ACC(ACCConfig(d_agent=128, d_lm=64))
        n_l = torch.randn(4, 64)
        hat_agent = acc.predict_agent_from_lm(n_l)
        assert hat_agent.shape == (4, 128)

    def test_predict_uses_W_and_W_t(self):
        # Sanity that W and W.t() are used in the right directions.
        # If we substitute identity W and identity LN, predict should be
        # near-identity (modulo LayerNorm reshaping).
        acc = ACC(ACCConfig(d_agent=4, d_lm=4))
        with torch.no_grad():
            acc.W.copy_(torch.eye(4))
        x = torch.randn(2, 4)
        # Wᵀ for eye is the same eye, so predict_agent_from_lm(x) == x.
        out = acc.predict_agent_from_lm(x)
        assert torch.allclose(out, x, atol=1e-5)


# ---- 5. recon_loss structure --------------------------------------------


class TestReconLossStructure:
    def _make_inputs(self, B=4, d_a=128, d_lm=64, requires_grad=False):
        h_a = torch.randn(B, d_a, requires_grad=requires_grad)
        h_l = torch.randn(B, d_lm, requires_grad=requires_grad)
        acc = ACC(ACCConfig(d_agent=d_a, d_lm=d_lm))
        return acc, h_a, h_l

    def test_returns_dict_with_expected_keys(self):
        acc, h_a, h_l = self._make_inputs()
        out = acc.recon_loss(h_a, h_l)
        keys = {"loss", "loss_a2l", "loss_l2a",
                "n_agent", "n_lm", "hat_lm", "hat_agent"}
        assert set(out) == keys

    def test_loss_is_scalar_and_finite(self):
        acc, h_a, h_l = self._make_inputs()
        out = acc.recon_loss(h_a, h_l)
        assert out["loss"].ndim == 0
        assert torch.isfinite(out["loss"])

    def test_loss_equals_a2l_plus_l2a(self):
        acc, h_a, h_l = self._make_inputs()
        out = acc.recon_loss(h_a, h_l)
        assert torch.isclose(
            out["loss"], out["loss_a2l"] + out["loss_l2a"], atol=1e-6
        )

    def test_intermediate_shapes(self):
        acc, h_a, h_l = self._make_inputs(B=3, d_a=16, d_lm=8)
        out = acc.recon_loss(h_a, h_l)
        assert out["n_agent"].shape == (3, 16)
        assert out["n_lm"].shape == (3, 8)
        assert out["hat_lm"].shape == (3, 8)
        assert out["hat_agent"].shape == (3, 16)


# ---- 6. (C-thin) detach policy — critical -------------------------------


class TestCThinGradBoundary:
    """PLAN §4.3 — the two grad boundaries.

    Boundary 1: h_agent receives NO grad — agent core is fully insulated.
    Boundary 2: h_lm DOES receive grad — LM interface (and ACC params) are
                updated. (Caller is responsible for stop-gradding LM core
                via optimizer setup; this is not ACC's job.)
    """

    def _setup(self, B=4, d_a=128, d_lm=64):
        torch.manual_seed(0)
        acc = ACC(ACCConfig(d_agent=d_a, d_lm=d_lm))
        h_a = torch.randn(B, d_a, requires_grad=True)
        h_l = torch.randn(B, d_lm, requires_grad=True)
        return acc, h_a, h_l

    def test_h_agent_receives_no_grad(self):
        """Boundary 1: h_agent.grad must be None after loss.backward()."""
        acc, h_a, h_l = self._setup()
        out = acc.recon_loss(h_a, h_l)
        out["loss"].backward()
        # h_a was provided with requires_grad=True but ACC detaches it
        # internally — no grad accumulates.
        assert h_a.grad is None

    def test_h_lm_receives_grad(self):
        """Boundary 2: h_lm.grad must be non-zero after loss.backward()."""
        acc, h_a, h_l = self._setup()
        out = acc.recon_loss(h_a, h_l)
        out["loss"].backward()
        assert h_l.grad is not None
        assert h_l.grad.abs().sum().item() > 0.0

    def test_W_receives_grad(self):
        acc, h_a, h_l = self._setup()
        out = acc.recon_loss(h_a, h_l)
        out["loss"].backward()
        assert acc.W.grad is not None
        assert acc.W.grad.abs().sum().item() > 0.0

    def test_ln_agent_receives_grad(self):
        # ln_agent is on the "agent side" but its affine params are
        # ACC-side — they should be updated.
        acc, h_a, h_l = self._setup()
        out = acc.recon_loss(h_a, h_l)
        out["loss"].backward()
        assert acc.ln_agent.weight.grad is not None
        assert acc.ln_agent.weight.grad.abs().sum().item() > 0.0

    def test_ln_lm_receives_grad(self):
        acc, h_a, h_l = self._setup()
        out = acc.recon_loss(h_a, h_l)
        out["loss"].backward()
        assert acc.ln_lm.weight.grad is not None
        assert acc.ln_lm.weight.grad.abs().sum().item() > 0.0

    def test_h_lm_grad_includes_both_directions(self):
        """Sanity: ñ_lm appears as A2L *target* and as L2A *prediction*
        input. Both should contribute to h_lm.grad."""
        # Compute only the A2L term and check h_lm gets nonzero grad
        # (target side).
        torch.manual_seed(0)
        acc = ACC(ACCConfig(d_agent=16, d_lm=8))
        h_a = torch.randn(2, 16, requires_grad=True)
        h_l = torch.randn(2, 8, requires_grad=True)
        out = acc.recon_loss(h_a, h_l)
        # Backward only on loss_a2l first
        a2l_only_grad = torch.autograd.grad(
            out["loss_a2l"], h_l, retain_graph=True
        )[0]
        l2a_only_grad = torch.autograd.grad(out["loss_l2a"], h_l)[0]
        # Both should be nonzero.
        assert a2l_only_grad.abs().sum().item() > 0.0
        assert l2a_only_grad.abs().sum().item() > 0.0

    def test_h_agent_grad_zero_even_when_l2a_only(self):
        # Even if we drive only the L2A loss (which uses ñ_agent as target),
        # h_agent.grad must remain None because n_agent was computed from
        # h_agent.detach().
        torch.manual_seed(0)
        acc = ACC(ACCConfig(d_agent=16, d_lm=8))
        h_a = torch.randn(2, 16, requires_grad=True)
        h_l = torch.randn(2, 8, requires_grad=True)
        out = acc.recon_loss(h_a, h_l)
        # autograd.grad with allow_unused — h_a is not connected to the
        # graph at all on ACC's side.
        grads = torch.autograd.grad(
            out["loss_l2a"], h_a, allow_unused=True
        )
        assert grads[0] is None


# ---- 7. cross_cosine (eval) ---------------------------------------------


class TestCrossCosine:
    def test_returns_dict_with_expected_keys(self):
        acc = ACC(ACCConfig(d_agent=64, d_lm=32))
        h_a = torch.randn(4, 64)
        h_l = torch.randn(4, 32)
        out = acc.cross_cosine(h_a, h_l)
        assert set(out) == {"cos_a2l", "cos_l2a",
                            "mean_cos_a2l", "mean_cos_l2a"}

    def test_per_sample_cosine_shape(self):
        acc = ACC(ACCConfig(d_agent=64, d_lm=32))
        h_a = torch.randn(5, 64)
        h_l = torch.randn(5, 32)
        out = acc.cross_cosine(h_a, h_l)
        assert out["cos_a2l"].shape == (5,)
        assert out["cos_l2a"].shape == (5,)

    def test_means_are_scalar(self):
        acc = ACC(ACCConfig(d_agent=64, d_lm=32))
        h_a = torch.randn(5, 64)
        h_l = torch.randn(5, 32)
        out = acc.cross_cosine(h_a, h_l)
        assert out["mean_cos_a2l"].ndim == 0
        assert out["mean_cos_l2a"].ndim == 0

    def test_cosine_in_unit_range(self):
        acc = ACC(ACCConfig(d_agent=64, d_lm=32))
        h_a = torch.randn(8, 64)
        h_l = torch.randn(8, 32)
        out = acc.cross_cosine(h_a, h_l)
        assert (out["cos_a2l"].abs() <= 1.0 + 1e-5).all()
        assert (out["cos_l2a"].abs() <= 1.0 + 1e-5).all()

    def test_cosine_runs_under_no_grad(self):
        # Outputs should have no grad_fn (decorator forces no_grad).
        acc = ACC(ACCConfig(d_agent=64, d_lm=32))
        h_a = torch.randn(4, 64, requires_grad=True)
        h_l = torch.randn(4, 32, requires_grad=True)
        out = acc.cross_cosine(h_a, h_l)
        for key in ("cos_a2l", "cos_l2a", "mean_cos_a2l", "mean_cos_l2a"):
            assert out[key].grad_fn is None


# ---- 8. acc_parameters accessor -----------------------------------------


class TestACCParameters:
    def test_yields_all_module_parameters(self):
        acc = ACC()
        all_params = list(acc.parameters())
        acc_params = list(acc.acc_parameters())
        assert len(acc_params) == len(all_params)
        # Same objects, in same order.
        for a, b in zip(acc_params, all_params):
            assert a is b

    def test_acc_parameters_excludes_nothing_under_current_config(self):
        # ACC has no "frozen" sub-modules — every param is trainable.
        acc = ACC()
        for p in acc.acc_parameters():
            assert p.requires_grad


# ---- 9. Integration / smoke ---------------------------------------------


class TestIntegrationSmoke:
    def test_one_optim_step_decreases_loss(self):
        """ACC + AdamW on a fixed (h_agent, h_lm) batch should reduce
        loss in 50 steps. Sanity for the loss function shape."""
        torch.manual_seed(0)
        acc = ACC(ACCConfig(d_agent=16, d_lm=8))
        h_a = torch.randn(32, 16)
        h_l = torch.randn(32, 8)

        opt = torch.optim.AdamW(acc.parameters(), lr=1e-2)

        with torch.no_grad():
            loss0 = acc.recon_loss(h_a, h_l)["loss"].item()

        for _ in range(50):
            opt.zero_grad()
            out = acc.recon_loss(h_a, h_l)
            out["loss"].backward()
            opt.step()

        with torch.no_grad():
            loss1 = acc.recon_loss(h_a, h_l)["loss"].item()

        assert loss1 < loss0

    def test_default_p3_hyperparams_match_session_handoff(self):
        # Catch accidental drift between PLAN §10.2 P3-3-A 박제값 and code.
        cfg = ACCConfig()
        assert cfg.d_agent == 256
        assert cfg.d_lm == 256
        assert cfg.init == "orthogonal"
