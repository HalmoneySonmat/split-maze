"""ACC (Artificial Corpus Callosum) — V2 build, Phase 3.

Implements PLAN §4.1–§4.3 with the Phase-3 decisions frozen on 2026-05-20
(PLAN §10.2 P3-2, P3-3):

- W: single tied parameter (d_lm × d_a). Used as ``W`` for the
  agent→lm direction and as ``W.t()`` for the lm→agent direction
  ("tied W + Wᵀ", PLAN §3.2(6) and §4.2).
- W initialisation: ``orthogonal`` by default (P3-3-A). Inherited from
  SPLIT-MNIST V2; matters for the §5.5 measurement-#4 random baseline.
- LayerNorm on both sides before W (cross-domain scale alignment,
  PLAN §4.2 "차원·스케일 격차 대응").
- Bidirectional MSE reconstruction loss (PLAN §4.2 equation).
- **(C-thin) grad boundary** (PLAN §4.3):

    * Boundary 1 — *agent core never receives ACC grad*. ``h_agent`` is
      detached internally (both as the *input* to the agent→lm prediction
      and as the *target* for the lm→agent reconstruction).
    * Boundary 2 — *LM core is stop-grad'd*. This module does NOT detach
      ``h_lm`` (so reconstruction grad can flow back through it). The
      caller is responsible for keeping the LM core frozen by only
      stepping the optimizer over ``MazeLM.interface_parameters()`` (i.e.
      ``interface_proj``) and ACC's own parameters.

The exact loss formula matches PLAN §4.2 verbatim::

    ñ_agent = LayerNorm(h_agent)
    ñ_lm    = LayerNorm(h_lm)
    ĥ_lm    = W   · ñ_agent.detach()
    ĥ_agent = Wᵀ · ñ_lm
    L_recon = ‖ĥ_lm − ñ_lm‖² + ‖ĥ_agent − ñ_agent.detach()‖²
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---- Config -------------------------------------------------------------


@dataclass
class ACCConfig:
    """Phase-3 P3-3-A default hyperparameters.

    Attributes:
        d_agent: IMPALA-CNN final dense width (Phase 0 #67 → 256).
        d_lm:    MazeLM ``d_model`` (Phase 2 박제 P2-2 → 256).
        init:    W initialisation. ``"orthogonal"`` (P3-3-A default) or
                 ``"xavier"`` (P3-3-B alternative, kept available for
                 ablation per PLAN §10.2 P3-3 fallback notes).
        layernorm_eps: LayerNorm ``eps``.
    """

    d_agent: int = 256
    d_lm: int = 256
    init: str = "orthogonal"
    layernorm_eps: float = 1e-5
    # POST-HOC-6 (2026-05-21): LayerNorm is non-learnable by default. A
    # learnable affine (γ,β) is a representational-collapse path — γ→0 makes
    # ñ a constant, enabling the trivial recon=0 solution that sank the
    # Phase-3 V2 (diagnose_v2). With affine=False, ñ is forced to zero-mean
    # unit-variance, so the constant-collapse minimum disappears and only W
    # carries the alignment (= SPLIT-MNIST V2 본형). The scale-alignment
    # purpose (PLAN §4.2) is kept by the normalization itself.
    layernorm_affine: bool = False
    # POST-HOC-7 (2026-05-22): when False, untie the bridge into two
    # asymmetric matrices — W_a2l (agent→lm, generation) and W_l2a
    # (lm→agent) — instead of one tied W + Wᵀ. The ceiling diagnostic
    # showed the tied-bidirectional constraint capped the *generation*
    # direction (cosine 0.27) far below the one-directional linear ceiling
    # (~0.47). Untying lets each direction reach its own ceiling. This is
    # PLAN §9 Deferred's "비대칭 W" ablation, adopted on the ceiling data.
    # PLAN §4.2 default was tied=True; V2/Phase-4 use tied=False.
    tied: bool = True


# ---- Module -------------------------------------------------------------


class ACC(nn.Module):
    """Artificial Corpus Callosum — tied W (d_lm × d_a) + bidirectional MSE.

    Forward usage (Phase 3 co-training loop)::

        acc = ACC(ACCConfig(d_agent=256, d_lm=256))
        out = acc.recon_loss(h_agent, h_lm)
        out["loss"].backward()   # updates ACC.W, ln_agent, ln_lm,
                                  # and LM.interface_proj — but NOT the
                                  # agent (boundary 1) nor the LM core
                                  # (boundary 2 enforced by the caller's
                                  # optimizer setup).

    Inputs to :meth:`recon_loss`:
        h_agent: (B, d_a). The IMPALA-CNN ``out.hidden`` (no detach needed;
                 this module detaches internally).
        h_lm:    (B, d_lm). Output of ``MazeLM.encode(ids)`` (= last-position
                 ``interface_proj`` activation). Must carry grad if you want
                 the LM interface to be updated.
    """

    def __init__(self, config: Optional[ACCConfig] = None):
        super().__init__()
        cfg = config or ACCConfig()
        self.config = cfg

        # LayerNorm pre-projection — scale alignment for the two domains
        # (PLAN §4.2). Both have learnable affine params; both are ACC-side.
        self.ln_agent = nn.LayerNorm(cfg.d_agent, eps=cfg.layernorm_eps,
                                     elementwise_affine=cfg.layernorm_affine)
        self.ln_lm = nn.LayerNorm(cfg.d_lm, eps=cfg.layernorm_eps,
                                  elementwise_affine=cfg.layernorm_affine)

        # Agent→lm projection (generation direction). Shape (d_lm, d_a) so
        # ``F.linear(x_agent, W)`` maps (B, d_a) → (B, d_lm).
        self.W = nn.Parameter(torch.empty(cfg.d_lm, cfg.d_agent))
        # POST-HOC-7: lm→agent direction. tied → reuse W.t(); untied → its
        # own matrix (d_a, d_lm). Untying frees the generation direction
        # from the tied compromise (ceiling 0.27 → ~0.47).
        if cfg.tied:
            self.W_l2a: Optional[nn.Parameter] = None
        else:
            self.W_l2a = nn.Parameter(torch.empty(cfg.d_agent, cfg.d_lm))
        self._init_W(cfg.init)

    # ---- init / reset ----

    def _init_one(self, t: torch.Tensor, kind: str) -> None:
        if kind == "orthogonal":
            nn.init.orthogonal_(t)
        elif kind == "xavier":
            nn.init.xavier_uniform_(t)
        else:
            raise ValueError(f"unknown W init: {kind!r}")

    def _init_W(self, kind: str) -> None:
        self._init_one(self.W, kind)
        if self.W_l2a is not None:
            self._init_one(self.W_l2a, kind)

    def _l2a_weight(self) -> torch.Tensor:
        """The lm→agent weight: W.t() if tied, else the untied W_l2a."""
        return self.W.t() if self.W_l2a is None else self.W_l2a

    @torch.no_grad()
    def reset_W(self, kind: Optional[str] = None) -> None:
        """Re-initialise W (and W_l2a if untied) in place (§5.5 baseline)."""
        self._init_W(kind or self.config.init)

    # ---- forward helpers ----

    def normalize(
        self,
        h_agent: torch.Tensor,
        h_lm: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """LayerNorm both sides. Returns ``(ñ_agent, ñ_lm)``.

        Does NOT apply the (C-thin) agent detach — that's the caller's
        responsibility if they use this helper outside :meth:`recon_loss`.
        """
        return self.ln_agent(h_agent), self.ln_lm(h_lm)

    def predict_lm_from_agent(self, n_agent: torch.Tensor) -> torch.Tensor:
        """ĥ_lm = W · ñ_agent.

        ``n_agent``: (B, d_a). Returns (B, d_lm).
        """
        return F.linear(n_agent, self.W)

    def predict_agent_from_lm(self, n_lm: torch.Tensor) -> torch.Tensor:
        """ĥ_agent = Wᵀ · ñ_lm.

        ``n_lm``: (B, d_lm). Returns (B, d_a). Uses ``W.t()`` (tied) or the
        untied ``W_l2a`` (POST-HOC-7).
        """
        return F.linear(n_lm, self._l2a_weight())

    # ---- the (C-thin) loss ----

    def recon_loss(
        self,
        h_agent: torch.Tensor,
        h_lm: torch.Tensor,
    ) -> dict:
        """Bidirectional reconstruction loss with the (C-thin) detach policy.

        Implements PLAN §4.2 verbatim::

            ñ_agent = LN(h_agent.detach())
            ñ_lm    = LN(h_lm)
            ĥ_lm    = W   · ñ_agent           (n_agent already grad-free)
            ĥ_agent = Wᵀ · ñ_lm
            L_recon = ‖ĥ_lm − ñ_lm‖² + ‖ĥ_agent − ñ_agent‖²

        Grad flow summary:

        * ``h_agent`` receives **no** grad (boundary 1).
        * ``h_lm`` receives grad through *both* directions — through the
          A2L target ``ñ_lm`` and through the L2A prediction ``Wᵀ · ñ_lm``.
        * ``self.W`` receives grad through both predictions.
        * ``self.ln_agent.{weight,bias}`` and ``self.ln_lm.{weight,bias}``
          receive grad as ACC-side parameters.

        Args:
            h_agent: (B, d_a) — agent hidden. Detached internally.
            h_lm:    (B, d_lm) — LM handle-B output. NOT detached.

        Returns:
            Dict with keys::

              "loss"      scalar — L_recon (= loss_a2l + loss_l2a),
              "loss_a2l"  scalar — agent→lm direction MSE,
              "loss_l2a"  scalar — lm→agent direction MSE,
              "n_agent"   (B, d_a)  — LN(h_agent.detach()),  grad-free input,
              "n_lm"      (B, d_lm) — LN(h_lm),               carries lm-grad,
              "hat_lm"    (B, d_lm) — W · n_agent,
              "hat_agent" (B, d_a)  — Wᵀ · n_lm,
        """
        # Boundary 1 — agent core grad is always cut.
        h_agent_det = h_agent.detach()
        n_agent = self.ln_agent(h_agent_det)
        # n_agent has no grad path to h_agent. But it DOES have grad path to
        # ln_agent.{weight,bias} — those are ACC-side, intended.

        n_lm = self.ln_lm(h_lm)
        # n_lm carries grad back into h_lm (and from there into LM
        # interface_proj; LM core protection is the caller's responsibility).

        hat_lm = self.predict_lm_from_agent(n_agent)
        hat_agent = self.predict_agent_from_lm(n_lm)

        # PLAN §4.2 targets:
        #   A2L target = ñ_lm        (NOT detached — grad flows to interface)
        #   L2A target = ñ_agent     (already grad-free via h_agent.detach())
        loss_a2l = F.mse_loss(hat_lm, n_lm)
        loss_l2a = F.mse_loss(hat_agent, n_agent)
        loss = loss_a2l + loss_l2a

        return {
            "loss": loss,
            "loss_a2l": loss_a2l,
            "loss_l2a": loss_l2a,
            "n_agent": n_agent,
            "n_lm": n_lm,
            "hat_lm": hat_lm,
            "hat_agent": hat_agent,
        }

    # ---- evaluation (Phase 4 측정 #2) ----

    @torch.no_grad()
    def cross_cosine(
        self,
        h_agent: torch.Tensor,
        h_lm: torch.Tensor,
    ) -> dict:
        """Cosine similarity between hats and normalised targets (eval-only).

        Used by PLAN §5.3 measurement #2 (representation-level cosine).
        Always runs under ``torch.no_grad`` — pure evaluation.

        Returns a dict with per-sample cosine vectors ``cos_a2l`` and
        ``cos_l2a`` plus their batch means.
        """
        n_agent = self.ln_agent(h_agent)
        n_lm = self.ln_lm(h_lm)
        hat_lm = self.predict_lm_from_agent(n_agent)
        hat_agent = self.predict_agent_from_lm(n_lm)

        cos_a2l = F.cosine_similarity(hat_lm, n_lm, dim=-1)
        cos_l2a = F.cosine_similarity(hat_agent, n_agent, dim=-1)
        return {
            "cos_a2l": cos_a2l,
            "cos_l2a": cos_l2a,
            "mean_cos_a2l": cos_a2l.mean(),
            "mean_cos_l2a": cos_l2a.mean(),
        }

    # ---- parameter accessor ----

    def acc_parameters(self) -> Iterator[nn.Parameter]:
        """All ACC-side trainable parameters (W + both LayerNorms).

        Convenience for building the ACC optimizer in the Phase-3 co-training
        loop — combined with ``MazeLM.interface_parameters()`` they form the
        complete set of parameters updated by the reconstruction loss.
        """
        yield from self.parameters()
