"""Unit tests for split_maze.adapter (Flamingo-style B4 adapter, Phase 3.3.0).

Organised by class:

1. PerceiverBlock — shape preservation, residual structure.
2. AgentResampler — output shape, n_kv validation, h_agent flows through,
   per-sample latent distinctness, param count.
3. GatedCrossAttentionBlock — gate-init identity (the Flamingo property),
   shape preservation, shape validation, gate_values diagnostic.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from split_maze.adapter import (  # noqa: E402
    AgentResampler,
    GatedCrossAttentionBlock,
    PerceiverBlock,
)


# ---- 1. PerceiverBlock --------------------------------------------------


class TestPerceiverBlock:
    def test_shape_preserved(self):
        blk = PerceiverBlock(d_model=32, n_heads=4)
        latents = torch.randn(2, 16, 32)
        kv = torch.randn(2, 8, 32)
        out = blk(latents, kv)
        assert out.shape == (2, 16, 32)

    def test_changes_latents(self):
        # With nonzero attention, the block should modify the latents.
        torch.manual_seed(0)
        blk = PerceiverBlock(d_model=32, n_heads=4)
        latents = torch.randn(2, 16, 32)
        kv = torch.randn(2, 8, 32)
        out = blk(latents, kv)
        assert not torch.allclose(out, latents)


# ---- 2. AgentResampler --------------------------------------------------


class TestAgentResampler:
    def test_output_shape(self):
        r = AgentResampler(d_agent=256, d_model=256, n_latents=16, n_kv=8)
        h = torch.randn(4, 256)
        out = r(h)
        assert out.shape == (4, 16, 256)

    def test_custom_dims(self):
        r = AgentResampler(d_agent=64, d_model=32, n_latents=10, n_kv=4)
        out = r(torch.randn(3, 64))
        assert out.shape == (3, 10, 32)

    def test_n_kv_too_small_raises(self):
        with pytest.raises(ValueError, match="n_kv must be"):
            AgentResampler(d_agent=64, d_model=32, n_kv=1)

    def test_h_agent_wrong_shape_raises(self):
        r = AgentResampler(d_agent=256, d_model=256)
        with pytest.raises(ValueError, match="h_agent must be"):
            r(torch.randn(4, 99))

    def test_h_agent_flows_through(self):
        """Different h_agent → different adapter tokens."""
        torch.manual_seed(0)
        r = AgentResampler(d_agent=64, d_model=32, n_latents=8, n_kv=4)
        a = r(torch.randn(1, 64))
        b = r(torch.randn(1, 64))
        assert not torch.allclose(a, b)

    def test_latents_distinct_within_sample(self):
        """The n_latents tokens should not all be identical."""
        torch.manual_seed(0)
        r = AgentResampler(d_agent=64, d_model=32, n_latents=8, n_kv=4)
        out = r(torch.randn(1, 64))  # (1, 8, 32)
        # Compare token 0 vs token 1.
        assert not torch.allclose(out[0, 0], out[0, 1])

    def test_grad_flows_to_h_agent(self):
        # The resampler itself doesn't detach — the build does. Verify the
        # path is differentiable end to end.
        r = AgentResampler(d_agent=64, d_model=32, n_latents=8, n_kv=4)
        h = torch.randn(2, 64, requires_grad=True)
        out = r(h)
        out.sum().backward()
        assert h.grad is not None
        assert h.grad.abs().sum().item() > 0.0

    def test_num_parameters_positive(self):
        r = AgentResampler(d_agent=256, d_model=256)
        assert r.num_parameters() > 0


# ---- 3. GatedCrossAttentionBlock ---------------------------------------


class TestGatedCrossAttentionBlock:
    def test_gate_init_is_identity(self):
        """At init both gates are 0 → tanh(0)=0 → output == hidden.

        This is the Flamingo property: the LM behaves exactly as before
        until training opens the gates."""
        torch.manual_seed(0)
        blk = GatedCrossAttentionBlock(d_model=32, n_heads=4)
        hidden = torch.randn(2, 5, 32)
        adapter = torch.randn(2, 8, 32)
        out = blk(hidden, adapter)
        assert torch.allclose(out, hidden, atol=1e-6)

    def test_gate_values_init_zero(self):
        blk = GatedCrossAttentionBlock(d_model=32, n_heads=4)
        g_attn, g_ffn = blk.gate_values
        assert g_attn == pytest.approx(0.0)
        assert g_ffn == pytest.approx(0.0)

    def test_nonzero_gate_changes_output(self):
        torch.manual_seed(0)
        blk = GatedCrossAttentionBlock(d_model=32, n_heads=4)
        # Open the gates manually.
        with torch.no_grad():
            blk.gate_attn.fill_(1.0)
            blk.gate_ffn.fill_(1.0)
        hidden = torch.randn(2, 5, 32)
        adapter = torch.randn(2, 8, 32)
        out = blk(hidden, adapter)
        assert not torch.allclose(out, hidden)

    def test_shape_preserved(self):
        blk = GatedCrossAttentionBlock(d_model=32, n_heads=4)
        with torch.no_grad():
            blk.gate_attn.fill_(0.5)
        hidden = torch.randn(3, 7, 32)
        adapter = torch.randn(3, 4, 32)
        assert blk(hidden, adapter).shape == (3, 7, 32)

    def test_hidden_must_be_3d(self):
        blk = GatedCrossAttentionBlock(d_model=32, n_heads=4)
        with pytest.raises(ValueError, match="hidden must be 3D"):
            blk(torch.randn(3, 32), torch.randn(3, 4, 32))

    def test_adapter_must_be_3d(self):
        blk = GatedCrossAttentionBlock(d_model=32, n_heads=4)
        with pytest.raises(ValueError, match="adapter_tokens must be 3D"):
            blk(torch.randn(3, 7, 32), torch.randn(3, 32))

    def test_batch_mismatch_raises(self):
        blk = GatedCrossAttentionBlock(d_model=32, n_heads=4)
        with pytest.raises(ValueError, match="batch mismatch"):
            blk(torch.randn(2, 7, 32), torch.randn(3, 4, 32))

    def test_d_model_mismatch_raises(self):
        blk = GatedCrossAttentionBlock(d_model=32, n_heads=4)
        with pytest.raises(ValueError, match="d_model mismatch"):
            blk(torch.randn(2, 7, 64), torch.randn(2, 4, 64))

    def test_adapter_mask_runs(self):
        blk = GatedCrossAttentionBlock(d_model=32, n_heads=4)
        with torch.no_grad():
            blk.gate_attn.fill_(0.5)
        hidden = torch.randn(2, 5, 32)
        adapter = torch.randn(2, 4, 32)
        mask = torch.tensor([[True, True, False, False],
                            [True, True, True, False]])
        out = blk(hidden, adapter, adapter_mask=mask)
        assert out.shape == (2, 5, 32)
