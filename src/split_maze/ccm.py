"""CCM — Co-activation Callosal Memory (Phase 5).

The follow-up direction after V2 (separation-reconstruction) was rejected.
User's idea (PLAN §10.1 *Phase 5 — CCM*, AskUserQuestion 2026-05-22): the bridge
between the two heterogeneous brains is neither a *trained translator* (B4) nor a
*thin reconstruction* (V2). Instead, when both nets see the *same scene*, we
**record (remember) which nodes co-activate** and use that record alone to drive
one net (agent) from the other (LM).

Concretely the record is a linear correspondence ``W`` (d_lm × d_a):

    a1 = h_agent                    # (B, d_a)   agent embedding
    ñ  = LN(a1)                     # non-affine LN — V2/B4Thin interface
    a2 = lm.encode(ids)             # (B, d_lm)  LM summary vector (the target)
    ĥ_lm = W · ñ (+ b)              # bridged vector
    generate = lm.generate(ĥ_lm)    # byte-identical to B4Thin's path

So **CCM = B4Thin's interface with ``W`` filled by *recorded co-activation
statistics* instead of *gradient*.** ``W`` is a buffer, never an ``nn.Parameter``
— that "memory, not learning" identity is what separates CCM from B4 and from
model-stitching (which trains the affine layer).

Three closed-form "rungs" fill ``W`` from the same recorded moments (CCM-1
사다리 전체); they differ only in how much they account for the geometry of the
agent features:

- **rung1 — Hebbian (raw, un-normalized):** ``W = E[a2 ñᵀ]`` (mean outer product).
  Pure "fire together"; the *headline baseline* the user wanted (no "정규화").
- **rung2 — ridge (normalized / least-squares):** ``W = C_yx (C_xx + λI)⁻¹``.
  Down-weights always-on / common directions — the ML analog of the callosum's
  inhibitory normalization (PV→SNR sharpening).
- **rung3 — Procrustes (orthogonal shape-match):** semi-orthogonal ``W`` from the
  SVD of the cross-covariance, plus a global scale. PLAN §5 #4.

Biological grounding (조사 2026-05-22): the real corpus callosum is not a clean
relay — it applies inhibitory normalization (sharpens tuning, enforces SNR) and
its correspondence is refined by neural activity (Hebbian pruning). The "정규화
칸" (rung2/3) is the brain-like correction; rung1 is the un-normalized memory.

NOTE on the two "보정"s that confused the design (PLAN §10.1 CCM-3): the **EMA**
in :class:`CoActAccumulator` is a *temporal* recency weighting (forget stale pages
while the nets drift during co-training); the **ridge λ** is a *spatial*
normalization of the mapping. Different axes — either can be on without the other.

(C-thin) boundaries:
- step0/step1 use a frozen agent + frozen LM (W fit / generate only).
- step2 (closed-loop) *intentionally* lets the bridge loss reshape the two nets
  (agent + LM ``interface_proj``) while ``W`` stays a recorded statistic — a
  pre-registered departure from boundary 1 for the "two brains adapt to the
  bridge" experiment only.
"""

from __future__ import annotations

from typing import Iterator, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .builds import Build
from .lm import MazeLM


# ---- co-activation moment accumulator -----------------------------------


class CoActAccumulator:
    """Accumulate the moments needed to fit ``W`` from co-activation pairs.

    Stores running *means* of the four moments over feature/target pairs
    ``(x, y)`` (``x`` = bridge input features, e.g. ``LN(h_agent)``; ``y`` =
    LM target, e.g. ``lm.encode(ids)``):

        m_x  = E[x]              (d_x,)
        m_y  = E[y]              (d_y,)
        m_xx = E[x xᵀ]           (d_x, d_x)
        m_yx = E[y xᵀ]           (d_y, d_x)

    Two regimes (PLAN §10.1 CCM-3):
    - ``decay=None`` — *cumulative* exact running mean over all seen samples
      (step1: frozen backbone, one pass over in-dist pairs).
    - ``decay∈(0,1)`` — *exponential moving average* of per-batch means
      (step2: co-training, recent pages weighted more as representations drift).

    Moments are kept in ``float64`` for numerical stability over many batches;
    :meth:`moments` / fitters return ``float32`` (model dtype) results.

    This is a plain object (not an ``nn.Module``) — it holds statistics, not
    parameters. ``W`` is never trained by gradient.
    """

    def __init__(
        self,
        d_x: int,
        d_y: int,
        *,
        decay: Optional[float] = None,
        device: Optional[torch.device] = None,
    ):
        if decay is not None and not (0.0 < decay < 1.0):
            raise ValueError(f"decay must be in (0,1) or None, got {decay}")
        self.d_x = int(d_x)
        self.d_y = int(d_y)
        self.decay = decay
        dev = device if device is not None else torch.device("cpu")
        self.device = dev
        self._n = 0.0  # samples seen (cumulative) / 1.0 once warm (ema)
        self.m_x = torch.zeros(d_x, dtype=torch.float64, device=dev)
        self.m_y = torch.zeros(d_y, dtype=torch.float64, device=dev)
        self.m_xx = torch.zeros(d_x, d_x, dtype=torch.float64, device=dev)
        self.m_yx = torch.zeros(d_y, d_x, dtype=torch.float64, device=dev)

    @property
    def count(self) -> float:
        """Samples seen (cumulative) — 0 until the first update."""
        return self._n

    @torch.no_grad()
    def update(self, x: torch.Tensor, y: torch.Tensor) -> None:
        """Fold a batch ``x`` (B, d_x), ``y`` (B, d_y) into the moments."""
        if x.dim() != 2 or y.dim() != 2:
            raise ValueError("x, y must be 2-D (B, d)")
        if x.shape[0] != y.shape[0]:
            raise ValueError("x, y batch sizes differ")
        if x.shape[1] != self.d_x or y.shape[1] != self.d_y:
            raise ValueError("x/y feature dims do not match accumulator")
        B = x.shape[0]
        # Device-safe: fold onto the buffer device/dtype regardless of where the
        # caller's x/y live (e.g. CPU agent embeddings + CUDA LM targets).
        xd = x.to(self.m_x.device, torch.float64)
        yd = y.to(self.m_y.device, torch.float64)
        # Per-batch means of the four moments.
        b_x = xd.mean(dim=0)
        b_y = yd.mean(dim=0)
        b_xx = (xd.t() @ xd) / B
        b_yx = (yd.t() @ xd) / B

        if self.decay is None:
            # Exact cumulative running mean.
            n_new = self._n + B
            w_old = self._n / n_new
            w_new = B / n_new
            self.m_x = w_old * self.m_x + w_new * b_x
            self.m_y = w_old * self.m_y + w_new * b_y
            self.m_xx = w_old * self.m_xx + w_new * b_xx
            self.m_yx = w_old * self.m_yx + w_new * b_yx
            self._n = n_new
        else:
            d = self.decay
            if self._n == 0.0:
                # First batch initializes the EMA (no bias-toward-zero warmup).
                self.m_x, self.m_y, self.m_xx, self.m_yx = b_x, b_y, b_xx, b_yx
            else:
                self.m_x = d * self.m_x + (1 - d) * b_x
                self.m_y = d * self.m_y + (1 - d) * b_y
                self.m_xx = d * self.m_xx + (1 - d) * b_xx
                self.m_yx = d * self.m_yx + (1 - d) * b_yx
            self._n += B

    def moments(self) -> dict:
        """Return the raw + centered moments (float64) for fitting.

        ``c_xx = E[x xᵀ] - m_x m_xᵀ`` (centered covariance), likewise ``c_yx``.
        ``m_yx`` is the *uncentered* cross-moment used by the Hebbian rung.
        """
        if self._n == 0.0:
            raise RuntimeError("CoActAccumulator is empty — call update() first")
        c_xx = self.m_xx - torch.outer(self.m_x, self.m_x)
        c_yx = self.m_yx - torch.outer(self.m_y, self.m_x)
        return {
            "m_x": self.m_x,
            "m_y": self.m_y,
            "m_xx": self.m_xx,
            "m_yx": self.m_yx,
            "c_xx": c_xx,
            "c_yx": c_yx,
        }


# ---- the three closed-form rungs (CCM-1 사다리) --------------------------


def fit_W(
    moments: dict,
    method: str,
    *,
    ridge_lambda: Optional[float] = None,
    procrustes_scale: bool = True,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fit the correspondence ``(W, b)`` from accumulated moments.

    Args:
        moments: dict from :meth:`CoActAccumulator.moments`.
        method: ``"hebbian"`` (rung1) | ``"ridge"`` (rung2) | ``"procrustes"``
            (rung3) | ``"centering_only"`` | ``"whitening_only"`` (whitening
            ablation 2×2 corners — PLAN §10.1 화이트닝 ablation).
        ridge_lambda: ridge for rung2 / whitening_only. ``None`` → scale-aware
            default ``1e-2 · trace(M)/d_x`` where ``M`` is the matrix being
            inverted (``c_xx`` for ridge, ``m_xx`` for whitening_only), so it
            tracks the feature scale of each corner independently.
        procrustes_scale: rung3 — include the optimal global scale (else pure
            semi-orthogonal).
        dtype: output dtype (model dtype; moments stay float64 internally).

    Returns:
        ``(W, b)`` with ``W`` (d_y, d_x) and ``b`` (d_y,), both ``dtype``.
        ``generate`` uses ``ĥ = F.linear(LN(h_agent), W, b) = LN·Wᵀ + b``.
    """
    m_x = moments["m_x"]
    m_y = moments["m_y"]
    c_xx = moments["c_xx"]
    c_yx = moments["c_yx"]
    d_x = m_x.shape[0]

    if method == "hebbian":
        # rung1 — raw mean outer product E[y xᵀ]; no centering, no bias.
        W = moments["m_yx"].clone()
        b = torch.zeros_like(m_y)

    elif method == "ridge":
        # rung2 — W = c_yx (c_xx + λI)⁻¹ ; b = m_y - W m_x.
        if ridge_lambda is None:
            ridge_lambda = 1e-2 * (torch.diagonal(c_xx).sum() / d_x).item()
        A = c_xx + ridge_lambda * torch.eye(
            d_x, dtype=c_xx.dtype, device=c_xx.device
        )
        # Solve (c_xx+λI) Xᵀ = c_yxᵀ  →  W = X = c_yx (c_xx+λI)⁻¹.
        Wt = torch.linalg.solve(A, c_yx.t())
        W = Wt.t()
        b = m_y - W @ m_x

    elif method == "procrustes":
        # rung3 — semi-orthogonal W from SVD of the cross-covariance.
        U, S, Vh = torch.linalg.svd(c_yx, full_matrices=False)
        W = U @ Vh                              # (d_y, d_x), semi-orthogonal
        if procrustes_scale:
            denom = torch.diagonal(c_xx).sum()
            scale = S.sum() / (denom + 1e-12)
            W = scale * W
        b = m_y - W @ m_x

    elif method == "centering_only":
        # ablation corner — centering, NO whitening: the *raw* centered
        # cross-covariance with a bias (no inverse-covariance term). Isolates
        # whether subtracting the means alone suffices (PLAN §10.1 화이트닝 ablation).
        W = c_yx.clone()
        b = m_y - W @ m_x

    elif method == "whitening_only":
        # ablation corner — whitening, NO centering: un-centered ridge using the
        # *raw* second moments (m_xx, m_yx), no bias. Isolates the inverse-
        # covariance (inhibitory-normalization) term alone.
        m_xx = moments["m_xx"]
        m_yx = moments["m_yx"]
        if ridge_lambda is None:
            ridge_lambda = 1e-2 * (torch.diagonal(m_xx).sum() / d_x).item()
        A = m_xx + ridge_lambda * torch.eye(
            d_x, dtype=m_xx.dtype, device=m_xx.device
        )
        Wt = torch.linalg.solve(A, m_yx.t())
        W = Wt.t()
        b = torch.zeros_like(m_y)

    else:
        raise ValueError(f"unknown method {method!r}")

    return W.to(dtype), b.to(dtype)


# ---- the CCM bridge build -----------------------------------------------


class CCMBridge(Build):
    """Co-activation memory bridge — the recorded-W twin of :class:`B4Thin`.

    Identical interface and generation path to ``B4Thin`` (non-affine LN +
    single bridged vector ``ĥ_lm = W·LN(h_agent) + b`` fed to
    ``lm.generate``/``lm.decode_logits``), but ``W``/``b`` are **buffers filled
    from recorded co-activation statistics** (see :func:`fit_W`), never trained
    by gradient. The eval/swap harness (``eval_builds``/``swap_test``) scores any
    ``Build`` with a ``.generate(h_agent)`` method, so CCM drops in unchanged.

    Args:
        lm:               a Phase-2 :class:`MazeLM` (frozen).
        d_agent:          IMPALA-CNN embedding width (256).
        layernorm_affine: agent-side LN affine. Default ``False`` to match the
                          ACC / B4Thin interface exactly.
    """

    def __init__(
        self,
        lm: MazeLM,
        *,
        d_agent: int = 256,
        layernorm_affine: bool = False,
    ):
        super().__init__()
        self.lm = lm
        d_model = lm.config.d_model
        self.d_agent = d_agent
        self.d_model = d_model

        # Freeze the LM core ((C-thin) LM 코어 보호; matches V2/B4/B4Thin).
        for p in self.lm.parameters():
            p.requires_grad_(False)
        self.lm.eval()

        self.ln_agent = nn.LayerNorm(d_agent, elementwise_affine=layernorm_affine)
        # Recorded correspondence — *buffers*, not Parameters (memory, not
        # learning). Filled by :meth:`set_W` from a fitted (W, b).
        self.register_buffer("W", torch.zeros(d_model, d_agent))
        self.register_buffer("b", torch.zeros(d_model))
        self.pad_id = lm.config.pad_id

    def train(self, mode: bool = True) -> "CCMBridge":
        """Keep the frozen base LM in eval (deterministic decode path)."""
        super().train(mode)
        self.lm.eval()
        return self

    # ---- recorded W management ----

    @torch.no_grad()
    def set_W(self, W: torch.Tensor, b: Optional[torch.Tensor] = None) -> None:
        """Install a recorded ``W`` (d_model, d_agent) and optional bias."""
        if tuple(W.shape) != (self.d_model, self.d_agent):
            raise ValueError(
                f"W shape {tuple(W.shape)} != ({self.d_model}, {self.d_agent})"
            )
        self.W.copy_(W.to(self.W.dtype, copy=False))
        if b is None:
            self.b.zero_()
        else:
            if tuple(b.shape) != (self.d_model,):
                raise ValueError(f"b shape {tuple(b.shape)} != ({self.d_model},)")
            self.b.copy_(b.to(self.b.dtype, copy=False))

    def set_lm_trainable_interface(self, flag: bool = True) -> None:
        """step2 helper — unfreeze *only* the LM ``interface_proj`` so the
        closed-loop bridge loss can reshape the LM interface (the rest of the LM
        stays frozen). No-op for step0/step1.

        NOTE (lm.py confirmed): ``interface_proj`` is on the *encode* path only;
        the bridge generate/decode path injects ĥ as the position-0 hidden and
        does NOT use it. So this handle does not reshape the bridge generation —
        step3 instead unfreezes a decoder block (see :meth:`set_lm_trainable_block`)."""
        for p in self.lm.interface_proj.parameters():
            p.requires_grad_(flag)

    def set_lm_trainable_block(self, idx: int = 0, flag: bool = True) -> None:
        """step3 (A2) — unfreeze a single LM decoder block ``blocks[idx]`` so the
        LM core can *meet the agent halfway* on the bridge generate path (which
        reads ĥ through the decoder). Kept minimal + paired with a language
        anchor by the caller; the rest of the LM core stays frozen."""
        for p in self.lm.blocks[idx].parameters():
            p.requires_grad_(flag)

    def set_plastic(self, flag: bool = True) -> None:
        """step3 — make the recorded ``W``/``b`` *plastic* (trainable
        ``nn.Parameter``, warm-started from the current recorded values) or
        revert to buffers. When plastic, :meth:`interpreter_parameters` yields
        ``W``/``b`` so an optimizer can *grow* the bridge by gradient ("기억이
        씨앗": memory → learned). Idempotent; preserves the tensor values/device."""
        is_param = isinstance(self.W, nn.Parameter)
        if flag == is_param:
            return
        W0 = self.W.detach().clone()
        b0 = self.b.detach().clone()
        if flag:
            del self._buffers["W"]
            del self._buffers["b"]
            self.W = nn.Parameter(W0)   # warm-start from recorded values
            self.b = nn.Parameter(b0)
        else:
            del self._parameters["W"]
            del self._parameters["b"]
            self.register_buffer("W", W0)
            self.register_buffer("b", b0)

    # ---- bridge ----

    def bridge(self, h_agent: torch.Tensor) -> torch.Tensor:
        """ĥ_lm = W · LN(h_agent) + b. (B, d_a) → (B, d_lm).

        Does *not* detach ``h_agent`` — the caller controls (C-thin): step0/step1
        feed a frozen agent; step2 feeds a grad-carrying ``h_agent`` to push the
        bridge loss into the agent (pre-registered closed-loop departure)."""
        return F.linear(self.ln_agent(h_agent), self.W, self.b)

    def reconstruct(self, h_agent: torch.Tensor) -> torch.Tensor:
        """Alias for :meth:`bridge` — the bridged vector that should match
        ``lm.encode(ids)``. Used by tests / MSE diagnostics."""
        return self.bridge(h_agent)

    # ---- update (step2 closed-loop loss; no trainable bridge params) ----

    def update(
        self,
        h_agent: torch.Tensor,
        ids: torch.Tensor,
        lengths: torch.Tensor,
    ) -> dict:
        """Next-token CE through the recorded bridge, via the handle-B decode
        path (identical to ``B4Thin.update`` but ``W`` is fixed/recorded).

        ``h_agent`` is **not** detached here: in step2 the gradient flows into
        the agent (and the LM ``interface_proj`` if unfrozen) so the two nets
        adapt to the recorded bridge; ``W`` itself carries no gradient (buffer).
        For step0/step1 the backbones are frozen so this just reports the loss.
        """
        h_lm = self.bridge(h_agent)                           # (B, d_lm)
        prefix = ids[:, :-1]
        logits = self.lm.decode_logits(h_lm, prefix)          # (B, T, vocab)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            ids.reshape(-1),
            ignore_index=self.pad_id,
        )
        return {"loss": loss}

    @torch.no_grad()
    def generate(self, h_agent: torch.Tensor, max_len: int = 16) -> torch.Tensor:
        """Greedy decode from the bridged vector — identical path to V2/B4Thin."""
        return self.lm.generate(self.bridge(h_agent), max_len=max_len)

    def interpreter_parameters(self) -> Iterator[nn.Parameter]:
        """step0/1/2: empty — ``W``/``b`` are recorded buffers (memory, not
        learning). step3 (after :meth:`set_plastic`): yields the now-trainable
        ``W``/``b`` so the bridge can *grow* by gradient."""
        if isinstance(self.W, nn.Parameter):
            yield self.W
            yield self.b


__all__ = [
    "CoActAccumulator",
    "fit_W",
    "CCMBridge",
]
