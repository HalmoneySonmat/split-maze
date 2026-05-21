"""Flamingo-style adapter for the B4 build (Phase 3.3).

Reused (and adapted) from split_brain_go's ``adapter/`` package, per PLAN
§7.2 "재사용 가능한 자산" and the P3-3-2 박제 (B4 = SPLIT-9 패턴 충실 재현).

Two pieces copied verbatim (only ``d_model`` differs — here 256):

- :class:`PerceiverBlock` — one cross-attention (latents ← KV) + FFN.
- :class:`GatedCrossAttentionBlock` — Flamingo gated cross-attn block,
  inserted between the LM's transformer blocks. Both gates init at 0 so
  ``tanh(0)=0`` → identity at start (LM behaviour fully preserved).

One piece **adapted** for SPLIT-MAZE's IMPALA agent:

- :class:`AgentResampler` — split_brain_go's ``PerceiverResampler`` mapped
  a 9×9 *spatial* Go-Net activation into tokens. The SPLIT-MAZE agent
  exposes a single 256-d vector ``h_agent`` (PROCGEN_ENV §7), not a
  spatial map. We therefore project ``h_agent`` into ``n_kv`` pseudo-token
  KV slots (a learned linear, each slot tagged with an id embedding), then
  cross-attend ``n_latents`` learned query tokens to them. Why ``n_kv > 1``:
  with a single KV token the cross-attention output (softmax over one
  element) is *identical* for every query, so the only thing that flows
  from ``h_agent`` is one broadcast vector — the residual still keeps the
  latents distinct (their random init differs), but they can only carry a
  broadcast of ``h_agent``. Multiple KV slots let different latents extract
  different aspects of ``h_agent``, which is the point of a resampler.

(C-thin) note: nothing here detaches ``h_agent`` — that is the *build's*
responsibility (B4Adapter feeds ``h_agent.detach()``), keeping the agent
core on pure RL.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


# ============================================================== PerceiverBlock


class PerceiverBlock(nn.Module):
    """One pass of cross-attention (latents ← KV) followed by an FFN.

    No self-attention on the latents — they're few, so mixing them adds
    cost without much benefit. Pre-norm: norm before each sublayer,
    residual after. Copied from split_brain_go.adapter.projection.
    """

    def __init__(self, d_model: int, n_heads: int, ffn_mult: int = 2) -> None:
        super().__init__()
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(
            d_model, n_heads, batch_first=True
        )
        self.norm_ffn = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * ffn_mult),
            nn.GELU(),
            nn.Linear(d_model * ffn_mult, d_model),
        )

    def forward(self, latents: Tensor, kv: Tensor) -> Tensor:
        q = self.norm_q(latents)
        k = self.norm_kv(kv)
        attn_out, _ = self.cross_attn(q, k, k, need_weights=False)
        latents = latents + attn_out
        latents = latents + self.ffn(self.norm_ffn(latents))
        return latents


# ============================================================ AgentResampler


class AgentResampler(nn.Module):
    """``h_agent`` (B, d_agent) → ``(B, n_latents, d_model)`` adapter tokens.

    Single-vector adaptation of Flamingo's PerceiverResampler:

        1. Linear-project ``h_agent`` into ``n_kv`` pseudo-token KV slots:
           (B, d_agent) → (B, n_kv·d_model) → (B, n_kv, d_model).
        2. Cross-attend ``n_latents`` learned query tokens to those KV
           slots through ``n_blocks`` PerceiverBlocks.

    Args:
        d_agent:   input width (IMPALA-CNN final dense, 256).
        d_model:   output width — must match the LM's d_model (256).
        n_latents: number of adapter tokens emitted. P3-3-2 박제 → 16.
        n_kv:      number of pseudo-token KV slots (>1 so latents
                   differentiate). Default 8.
        n_heads:   attention heads per PerceiverBlock.
        n_blocks:  PerceiverBlock stack depth.
    """

    def __init__(
        self,
        d_agent: int = 256,
        d_model: int = 256,
        n_latents: int = 16,
        n_kv: int = 8,
        n_heads: int = 4,
        n_blocks: int = 2,
    ) -> None:
        super().__init__()
        if n_kv < 2:
            raise ValueError(
                "n_kv must be ≥ 2 (a single KV token collapses all latents)"
            )
        self.d_agent = d_agent
        self.d_model = d_model
        self.n_latents = n_latents
        self.n_kv = n_kv

        self.to_kv = nn.Linear(d_agent, n_kv * d_model)
        # Per-KV-slot id embedding so the slots aren't permutation-identical.
        self.kv_emb = nn.Embedding(n_kv, d_model)
        # Learned latent queries; small init for a stable start.
        self.latents = nn.Parameter(torch.randn(n_latents, d_model) * 0.02)
        self.blocks = nn.ModuleList(
            [PerceiverBlock(d_model, n_heads) for _ in range(n_blocks)]
        )

    def forward(self, h_agent: Tensor) -> Tensor:
        """Return ``(B, n_latents, d_model)`` adapter tokens."""
        if h_agent.dim() != 2 or h_agent.shape[1] != self.d_agent:
            raise ValueError(
                f"h_agent must be (B, {self.d_agent}); got {tuple(h_agent.shape)}"
            )
        B = h_agent.shape[0]
        kv = self.to_kv(h_agent).view(B, self.n_kv, self.d_model)
        # Add KV-slot id embeddings (broadcast over batch).
        kv_ids = torch.arange(self.n_kv, device=h_agent.device)
        kv = kv + self.kv_emb(kv_ids).unsqueeze(0)

        latents = self.latents.unsqueeze(0).expand(B, -1, -1).contiguous()
        for block in self.blocks:
            latents = block(latents, kv)
        return latents

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ===================================================== GatedCrossAttentionBlock


class GatedCrossAttentionBlock(nn.Module):
    """Single gated cross-attention layer + gated FFN (Flamingo, Alayrac 2022).

    Inserted between the LM's transformer blocks. Reads ``adapter_tokens``
    (from :class:`AgentResampler`) and merges them into the LM hidden state.
    Both gates init at 0 → ``tanh(0)=0`` → identity at start, so the LM's
    Phase-2 behaviour is preserved until training opens the gates. Copied
    from split_brain_go.adapter.xattn (only d_model differs).

    Layout (pre-norm):
        out = hidden + tanh(g_attn) · CrossAttn(LN(hidden), LN(adapter))
        out = out    + tanh(g_ffn)  · FFN(LN(out))
    """

    def __init__(self, d_model: int, n_heads: int, ffn_mult: int = 4) -> None:
        super().__init__()
        self.d_model = d_model

        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(
            d_model, n_heads, batch_first=True
        )
        self.gate_attn = nn.Parameter(torch.zeros(1))

        self.norm_ffn = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * ffn_mult),
            nn.GELU(),
            nn.Linear(d_model * ffn_mult, d_model),
        )
        self.gate_ffn = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        hidden: Tensor,
        adapter_tokens: Tensor,
        adapter_mask: Tensor | None = None,
    ) -> Tensor:
        """Augment ``hidden`` with information from ``adapter_tokens``.

        Args:
            hidden:         (B, T, d_model) from a preceding LM block.
            adapter_tokens: (B, N, d_model) from :class:`AgentResampler`.
            adapter_mask:   optional (B, N) bool; True = attend, False =
                            mask out. None → all attend.

        Returns:
            (B, T, d_model) — same shape as ``hidden``.
        """
        if hidden.dim() != 3:
            raise ValueError(f"hidden must be 3D, got {tuple(hidden.shape)}")
        if adapter_tokens.dim() != 3:
            raise ValueError(
                f"adapter_tokens must be 3D, got {tuple(adapter_tokens.shape)}"
            )
        if hidden.shape[0] != adapter_tokens.shape[0]:
            raise ValueError(
                f"batch mismatch: hidden {hidden.shape[0]} vs adapter "
                f"{adapter_tokens.shape[0]}"
            )
        if hidden.shape[2] != self.d_model or adapter_tokens.shape[2] != self.d_model:
            raise ValueError(
                f"d_model mismatch: expected {self.d_model}, got hidden "
                f"{hidden.shape[2]}, adapter {adapter_tokens.shape[2]}"
            )

        q = self.norm_q(hidden)
        kv = self.norm_kv(adapter_tokens)
        key_padding_mask = None if adapter_mask is None else ~adapter_mask
        attn_out, _ = self.cross_attn(
            q, kv, kv, key_padding_mask=key_padding_mask, need_weights=False
        )
        hidden = hidden + torch.tanh(self.gate_attn) * attn_out

        ffn_out = self.ffn(self.norm_ffn(hidden))
        hidden = hidden + torch.tanh(self.gate_ffn) * ffn_out
        return hidden

    @property
    def gate_values(self) -> tuple[float, float]:
        """Current ``(tanh(gate_attn), tanh(gate_ffn))`` for diagnostics."""
        with torch.no_grad():
            return (
                float(torch.tanh(self.gate_attn).item()),
                float(torch.tanh(self.gate_ffn).item()),
            )


__all__ = [
    "PerceiverBlock",
    "AgentResampler",
    "GatedCrossAttentionBlock",
]
